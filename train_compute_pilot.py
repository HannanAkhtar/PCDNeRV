#!/usr/bin/env python3
"""Equal-wall-clock, decoder-compute-targeted HNeRV/Bunny pilot trainer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from evaluate_saved_models import evaluate_quality_and_embeddings
from hnerv_utils import loss_fn, psnr_fn_single
from model_all import TransformInput, VideoDataSet
from shared.accounting import (
    benchmark_decoder_cuda,
    decoder_layer_widths,
    measure_decoder_compute,
    param_accounting,
)
from shared.compute_pruning import (
    build_random_architecture_from_pruned_artifact,
    compute_group_mac_costs,
    compute_target_prune_plan,
    derive_hnerv_config_for_modelsize,
    find_compute_matched_hnerv,
    find_parameter_matched_hnerv,
    normalized_layer_costs,
)
from shared.groups import apply_group_prox, weighted_group_lasso_loss
from shared.nerv_targets import get_channel_group_layers
from shared.pcd_solver import PCDSolver
from shared.progressive_pruning import (
    apply_progressive_prune_plan,
    load_pilot_checkpoint,
    save_pilot_checkpoint,
)
from shared.runtime import validate_cuda_environment
from shared.wallclock import (
    ActiveTrainingBudget,
    PILOT_METHODS,
    gradual_kappa_target,
    hard_prune_due,
    fraction_checkpoint_due,
    next_fraction_checkpoint,
    pilot_phase,
    set_learning_rate,
    synchronization_callback_for,
    threshold_removal_active,
)
from shared.physical_pruning import build_model_from_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=PILOT_METHODS, required=True)
    parser.add_argument("--budget_seconds", type=float, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--outf", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--replay_artifact", default="")
    parser.add_argument("--target_params", type=int, default=0)
    parser.add_argument("--candidate_sizes", default="")
    parser.add_argument("--compute_match_tolerance", type=float, default=0.02)

    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--lambda_gl", type=float, default=1e-5)
    parser.add_argument("--removal_eps", type=float, default=1e-4)
    parser.add_argument("--min_keep", type=int, default=1)
    parser.add_argument("--m4_use_prox", action="store_true")
    parser.add_argument("--m4_lambda_prox", type=float, default=0.0)
    parser.add_argument("--beta_ema", type=float, default=0.999)
    parser.add_argument("--solver_eps", type=float, default=1e-8)

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--vid", default="bunny")
    parser.add_argument("--data_split", default="1_1_1")
    parser.add_argument("--crop_list", default="640_1280")
    parser.add_argument("--resize_list", default="-1")
    parser.add_argument("--max_frames", type=int, default=0, help="engineering smoke only")
    parser.add_argument("--embed", default="")
    parser.add_argument("--ks", default="0_1_5")
    parser.add_argument("--enc_strds", type=int, nargs="+", default=[5, 4, 4, 2, 2])
    parser.add_argument("--enc_dim", default="64_16")
    parser.add_argument("--dec_strds", type=int, nargs="+", default=[5, 4, 4, 2, 2])
    parser.add_argument("--fc_hw", default="9_16")
    parser.add_argument("--modelsize", type=float, default=1.5)
    parser.add_argument("--saturate_stages", type=int, default=-1)
    parser.add_argument("--reduce", type=float, default=1.2)
    parser.add_argument("--lower_width", type=int, default=12)
    parser.add_argument("--num_blks", default="1_1")
    parser.add_argument("--conv_type", nargs=2, default=["convnext", "pshuffel"])
    parser.add_argument("--norm", default="none")
    parser.add_argument("--act", default="gelu")
    parser.add_argument("--out_bias", default="tanh")

    parser.add_argument("--batchSize", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_type", default="cosine_0.1_1_0.1")
    parser.add_argument("--loss", default="L2")
    parser.add_argument("--max_epochs", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--monitor_every_fraction", type=float, default=0.0)
    parser.add_argument("--monitor_frames", type=int, default=8)
    parser.add_argument("--checkpoint_every", type=int, default=0)
    parser.add_argument("--checkpoint_every_fraction", type=float, default=0.10)
    parser.add_argument(
        "--msssim_device", choices=("cpu", "cuda", "auto"), default="cpu"
    )
    parser.add_argument("--manualSeed", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--fps_warmup", type=int, default=100)
    parser.add_argument("--fps_runs", type=int, default=300)
    parser.add_argument("--no_final_fps", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.kappa < 1:
        parser.error("--kappa must be in [0, 1)")
    if args.compute_match_tolerance < 0:
        parser.error("--compute_match_tolerance must be non-negative")
    if args.monitor_every_fraction < 0:
        parser.error("--monitor_every_fraction must be non-negative")
    if args.monitor_frames < 1:
        parser.error("--monitor_frames must be positive")
    if args.checkpoint_every < 0 or args.checkpoint_every_fraction < 0:
        parser.error("checkpoint cadences must be non-negative")
    if args.method == "d_replay" and not args.replay_artifact:
        parser.error("d_replay requires --replay_artifact")
    if args.method == "d_small_params" and args.target_params <= 0:
        parser.error("d_small_params requires --target_params")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu for a smoke run")
    return args


def _write_csv(path, rows):
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_if_changed(path, rows, previously_written):
    """Rewrite a recovery CSV only when its in-memory row count changed."""
    path = Path(path)
    if len(rows) == previously_written and path.is_file():
        return previously_written, False
    if rows:
        _write_csv(path, rows)
    return len(rows), bool(rows)


def _evenly_spaced_indices(frame_count, requested=8):
    count = min(int(frame_count), int(requested))
    if count <= 0:
        return []
    return sorted({int(round(value)) for value in np.linspace(0, frame_count - 1, count)})


@torch.no_grad()
def _evaluate_psnr_monitor(model, dataset, transform, config, device, indices):
    """PSNR-only deterministic monitor; embeddings are never retained."""
    was_training = model.training
    model.eval()
    values = []
    for index in indices:
        sample = dataset[index]
        image = sample["img"].unsqueeze(0).to(device=device, dtype=torch.float32)
        image_in, image_gt, _ = transform(image)
        if "pe" in str(config.get("embed", "")):
            model_input = torch.as_tensor(
                [sample["norm_idx"]], device=device, dtype=torch.float32
            )
        else:
            model_input = image_in
        output, _, _ = model(model_input)
        values.extend(float(value) for value in psnr_fn_single(output, image_gt).flatten())
    if was_training:
        model.train()
    return sum(values) / len(values)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _base_config(args, frame_count, output_hw):
    base = {
        "method": args.method,
        "arch": "hnerv",
        "vid": args.vid,
        "data_path": args.data_path,
        "data_split": args.data_split,
        "shuffle_data": True,
        "crop_list": args.crop_list,
        "resize_list": args.resize_list,
        "embed": args.embed,
        "enc_strds": list(args.enc_strds),
        "enc_dim": args.enc_dim,
        "dec_strds": list(args.dec_strds),
        "fc_hw": args.fc_hw,
        "fc_dim": 1,
        "ks": args.ks,
        "reduce": args.reduce,
        "lower_width": args.lower_width,
        "num_blks": args.num_blks,
        "conv_type": list(args.conv_type),
        "norm": args.norm,
        "act": args.act,
        "out_bias": args.out_bias,
        "modelsize": args.modelsize,
        "saturate_stages": args.saturate_stages,
        "seed": args.manualSeed,
    }
    return derive_hnerv_config_for_modelsize(
        base, args.modelsize, frame_count=frame_count, output_hw=output_hw
    )


_ARCHITECTURE_KEYS = (
    "arch", "crop_list", "resize_list", "embed", "enc_strds", "enc_dim",
    "dec_strds", "fc_hw", "fc_dim", "ks", "reduce", "lower_width",
    "num_blks", "conv_type", "norm", "act", "out_bias", "modelsize",
    "saturate_stages",
)


def _architecture_signature(config):
    return {key: config.get(key) for key in _ARCHITECTURE_KEYS}


def _validate_resume_configuration(payload, args, base_config):
    """Reject resume commands that would alter the checkpointed experiment."""
    errors = []
    checks = (
        ("method", payload.get("method"), args.method),
        ("kappa", payload.get("kappa"), args.kappa),
        ("seed", payload.get("seed"), args.manualSeed),
        ("budget_seconds", payload.get("budget_seconds"), args.budget_seconds),
    )
    for name, checkpoint_value, command_value in checks:
        if checkpoint_value != command_value:
            errors.append(
                f"{name}: checkpoint={checkpoint_value!r}, command={command_value!r}"
            )

    selection = payload.get("selection")
    if not isinstance(selection, dict) or "base_config" not in selection:
        errors.append("checkpoint does not contain original selection/base architecture metadata")
    elif _architecture_signature(selection["base_config"]) != _architecture_signature(base_config):
        errors.append(
            "base/original architecture differs from the checkpoint: "
            f"checkpoint={_architecture_signature(selection['base_config'])!r}, "
            f"command={_architecture_signature(base_config)!r}"
        )
    if "history" not in payload:
        errors.append("checkpoint does not contain complete training history")
    elif (
        not isinstance(payload["history"], list)
        or not isinstance(payload.get("epoch"), int)
        or len(payload["history"]) != payload["epoch"]
    ):
        rows = len(payload["history"]) if isinstance(payload["history"], list) else "invalid"
        errors.append(
            "checkpoint history is incomplete: "
            f"epoch={payload.get('epoch')!r}, rows={rows}"
        )
    if errors:
        raise ValueError("resume configuration mismatch:\n  - " + "\n  - ".join(errors))


def _sample_embedding(model, sample_input):
    with torch.no_grad():
        _, embeddings, _ = model(sample_input)
    return embeddings[0].detach()


def _clone_grads(model):
    return [
        parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter)
        for parameter in model.parameters()
    ]


def _set_grads(model, gradients):
    for parameter, gradient in zip(model.parameters(), gradients):
        parameter.grad = gradient


def _training_step(
    model, optimizer, batch, transform, args, phase, layers, frozen_costs, solver
):
    images = batch["img"].to(next(model.parameters()).device, non_blocking=True)
    images_in, images_gt, mask = transform(images)
    inputs = batch["norm_idx"].to(images.device) if "pe" in args.embed else images_in
    optimizer.zero_grad(set_to_none=False)
    output, _, _ = model(inputs)
    primary = loss_fn(output * mask, images_gt * mask, args.loss)
    diagnostics = {}
    structured = torch.zeros((), device=images.device)
    if phase == "weighted_group_lasso":
        structured = weighted_group_lasso_loss(layers, frozen_costs)
        (primary + args.lambda_gl * structured).backward()
    elif phase == "pcd_weighted":
        primary.backward()
        primary_grads = _clone_grads(model)
        optimizer.zero_grad(set_to_none=False)
        structured = weighted_group_lasso_loss(layers, frozen_costs)
        structured.backward()
        secondary_grads = _clone_grads(model)
        combined, raw = solver.step(primary_grads, secondary_grads)
        _set_grads(model, combined)
        rename = {
            "ce_efficiency": "primary_efficiency",
            "g_ce_norm_raw": "g_primary_norm_raw",
            "g_ce_norm_normed": "g_primary_norm_normed",
            "g_gl_norm_raw": "g_structured_norm_raw",
            "g_gl_norm_normed": "g_structured_norm_normed",
        }
        diagnostics = {rename.get(key, key): value for key, value in raw.items()}
    else:
        primary.backward()
    optimizer.step()
    if phase == "pcd_weighted" and args.m4_use_prox:
        apply_group_prox(layers, args.m4_lambda_prox * optimizer.param_groups[0]["lr"])
    return {
        "primary_loss": float(primary.detach()),
        "structured_loss_R": float(structured.detach()),
        **{key: float(value) for key, value in diagnostics.items()},
    }


def _current_metrics(model, embedding, start_macs, output_hw):
    compute = measure_decoder_compute(model, embedding)
    accounting = param_accounting(model)
    pixels = int(output_hw[0]) * int(output_hw[1])
    current_macs = int(compute["total_MACs"])
    return {
        "decoder_head_params": int(accounting["decoder_head_params"]),
        "GMACs_per_frame": current_macs / 1e9,
        "GFLOPs_per_frame": 2 * current_macs / 1e9,
        "kMACs_per_pixel": current_macs / pixels / 1000,
        "current_MACs": current_macs,
        "achieved_kappa": 1.0 - current_macs / float(start_macs),
        "decoder_widths": json.dumps(decoder_layer_widths(model), separators=(",", ":")),
    }


def _prune_event(
    model,
    optimizer,
    embedding,
    *,
    start_macs,
    target_macs,
    min_keep,
    eligible_eps,
    allow_overshoot,
):
    plans, head_keep, info = compute_target_prune_plan(
        model,
        embedding,
        target_macs=target_macs,
        start_macs=start_macs,
        min_keep=min_keep,
        eligible_eps=eligible_eps,
        allow_target_overshoot=allow_overshoot,
    )
    if info["groups_removed"]:
        surgery = apply_progressive_prune_plan(
            model, optimizer, plans, head_keep,
            verify_embedding=embedding, equivalence_tol=1e-5,
        )
        info["max_equivalence_error"] = surgery["max_equivalence_error"]
    else:
        info["max_equivalence_error"] = 0.0
    return info


def _select_initial_model(args, base_config, sample_image, frame_count):
    _seed_everything(args.manualSeed)
    start_model = build_model_from_config(base_config).to(sample_image.device).eval()
    start_embedding = _sample_embedding(start_model, sample_image)
    start_macs = int(measure_decoder_compute(start_model, start_embedding)["total_MACs"])
    target_macs = start_macs * (1.0 - args.kappa)
    selection = {"start_MACs": start_macs, "target_MACs": target_macs}
    selected_config = dict(base_config)

    sizes = [float(value) for value in args.candidate_sizes.split(",") if value.strip()]
    if args.method == "d_small_compute":
        _, selected_config, details = find_compute_matched_hnerv(
            base_config,
            target_macs,
            sample_image,
            frame_count=frame_count,
            candidate_model_sizes=sizes or None,
            tolerance=args.compute_match_tolerance,
        )
        _seed_everything(args.manualSeed)
        model = build_model_from_config(selected_config).to(sample_image.device)
        selection["dense_search"] = details
    elif args.method == "d_small_params":
        _, selected_config, details = find_parameter_matched_hnerv(
            base_config,
            args.target_params,
            frame_count=frame_count,
            output_hw=sample_image.shape[-2:],
            candidate_model_sizes=sizes or None,
        )
        _seed_everything(args.manualSeed)
        model = build_model_from_config(selected_config).to(sample_image.device)
        selection["dense_search"] = details
    elif args.method == "d_replay":
        model, selected_config, details = build_random_architecture_from_pruned_artifact(
            args.replay_artifact, args.manualSeed
        )
        artifact_hw = tuple(int(value) for value in str(selected_config["crop_list"]).split("_")[:2])
        if artifact_hw != tuple(int(value) for value in sample_image.shape[-2:]):
            raise ValueError(
                f"replay artifact resolution {artifact_hw} does not match pilot input "
                f"{tuple(sample_image.shape[-2:])}"
            )
        source_dense = build_model_from_config(selected_config).to(sample_image.device).eval()
        source_embedding = _sample_embedding(source_dense, sample_image)
        start_macs = int(measure_decoder_compute(source_dense, source_embedding)["total_MACs"])
        target_macs = start_macs * (1.0 - args.kappa)
        selection.update({"start_MACs": start_macs, "target_MACs": target_macs})
        model = model.to(sample_image.device)
        selection["replay"] = details
    else:
        model = start_model
    model.train()
    selection["base_config"] = dict(base_config)
    selection["selected_config"] = selected_config
    return model, selected_config, start_macs, target_macs, selection


def run(
    args,
    *,
    quality_evaluator=evaluate_quality_and_embeddings,
    monitor_evaluator=None,
):
    output_dir = Path(args.outf).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    runtime_environment = validate_cuda_environment(
        require_cuda=device.type == "cuda", report=True
    )
    synchronize_fn = synchronization_callback_for(device)
    _seed_everything(args.manualSeed)

    dataset_args = argparse.Namespace(
        data_path=args.data_path, crop_list=args.crop_list,
        resize_list=args.resize_list, vid=args.vid,
    )
    full_dataset = VideoDataSet(dataset_args)
    if args.max_frames > 0:
        dataset = Subset(full_dataset, list(range(min(args.max_frames, len(full_dataset)))))
    else:
        dataset = full_dataset
    loader = DataLoader(
        dataset, batch_size=args.batchSize, shuffle=True,
        num_workers=args.workers, drop_last=False,
    )
    transform = TransformInput(dataset_args).to(device)
    first_image = full_dataset[0]["img"].unsqueeze(0).to(device)
    output_hw = first_image.shape[-2:]
    architecture_frame_count = len(dataset) if args.max_frames > 0 else len(full_dataset)
    base_config = _base_config(args, architecture_frame_count, output_hw)

    checkpoint_path = Path(args.resume) if args.resume else output_dir / "pilot_latest.pth"
    resume = checkpoint_path.is_file() and not args.no_resume
    if resume:
        model, optimizer, solver, payload = load_pilot_checkpoint(checkpoint_path, device=device)
        _validate_resume_configuration(payload, args, base_config)
        original_config = payload["original_config"]
        selected_config = dict(original_config)
        start_macs = int(payload["start_MACs"])
        target_macs = start_macs * (1.0 - float(payload["kappa"]))
        budget = ActiveTrainingBudget(
            args.budget_seconds,
            elapsed=payload["counted_training_seconds"],
            synchronize_fn=synchronize_fn,
        )
        epoch = int(payload["epoch"])
        optimizer_steps = int(payload["optimizer_steps"])
        removal_events = list(payload["removal_history"])
        frozen_costs = dict(payload["frozen_layer_costs"])
        selection = dict(payload["selection"])
        selection["resumed_from"] = str(checkpoint_path)
        history = list(payload["history"])
        monitor_history = list(payload.get("monitor_history", []))
    else:
        model, selected_config, start_macs, target_macs, selection = _select_initial_model(
            args, base_config, first_image, architecture_frame_count
        )
        original_config = dict(selected_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
        solver = (
            PCDSolver(tau=args.tau, beta=args.beta_ema, eps=args.solver_eps)
            if args.method == "m4_pcd" else None
        )
        budget = ActiveTrainingBudget(
            args.budget_seconds, synchronize_fn=synchronize_fn
        )
        epoch = 0
        optimizer_steps = 0
        removal_events = []
        frozen_costs = normalized_layer_costs(
            compute_group_mac_costs(model, _sample_embedding(model, first_image))
        )
        history = []
        monitor_history = []

    hard_done = any(row.get("event") == "hard_0.9W" for row in removal_events)
    time_to_kappa = next(
        (row.get("counted_training_seconds") for row in removal_events if row.get("target_reached")),
        None,
    )
    cumulative_removed = sum(int(row.get("groups_removed", 0)) for row in removal_events)
    live_current_macs = int(
        measure_decoder_compute(model, _sample_embedding(model, first_image))["total_MACs"]
    )
    history_path = output_dir / "history.csv"
    removal_path = output_dir / "removal_events.csv"
    monitor_path = output_dir / "monitor.csv"
    history_csv_rows = len(history) if history_path.is_file() else -1
    removal_csv_rows = len(removal_events) if removal_path.is_file() else -1
    monitor_csv_rows = len(monitor_history) if monitor_path.is_file() else -1
    next_checkpoint = next_fraction_checkpoint(
        budget.schedule_fraction, args.checkpoint_every_fraction
    )
    next_monitor = next_fraction_checkpoint(
        budget.schedule_fraction, args.monitor_every_fraction
    )
    monitor_indices = _evenly_spaced_indices(len(full_dataset), args.monitor_frames)

    def save_recovery_checkpoint():
        nonlocal history_csv_rows, removal_csv_rows, monitor_csv_rows
        save_pilot_checkpoint(
            checkpoint_path,
            original_config=original_config,
            model=model,
            optimizer=optimizer,
            method=args.method,
            kappa=args.kappa,
            start_macs=start_macs,
            current_macs=live_current_macs,
            counted_training_seconds=budget.counted_training_seconds,
            budget_seconds=args.budget_seconds,
            epoch=len(history),
            optimizer_steps=optimizer_steps,
            removal_history=removal_events,
            frozen_layer_costs=frozen_costs,
            history=history,
            selection=selection,
            seed=args.manualSeed,
            monitor_history=monitor_history,
            solver=solver,
            metrics=history[-1] if history else {},
        )
        history_csv_rows, _ = _write_csv_if_changed(
            history_path, history, history_csv_rows
        )
        removal_csv_rows, _ = _write_csv_if_changed(
            removal_path, removal_events, removal_csv_rows
        )
        monitor_csv_rows, _ = _write_csv_if_changed(
            monitor_path, monitor_history, monitor_csv_rows
        )

    while not budget.exhausted and epoch < args.max_epochs:
        epoch += 1
        model.train()
        aggregate = defaultdict(list)
        step_phases = []
        completed_epoch = True
        for batch in loader:
            fraction_before = budget.schedule_fraction
            target_reached = live_current_macs <= target_macs
            phase = pilot_phase(args.method, fraction_before, target_reached)
            step_phases.append(phase)
            lr = set_learning_rate(optimizer, args.lr, args.lr_type, fraction_before)
            layers = get_channel_group_layers(model, "conv")

            step_metrics, stop = budget.completed_optimizer_step(
                lambda: _training_step(
                    model, optimizer, batch, transform, args, phase,
                    layers, frozen_costs, solver,
                )
            )
            optimizer_steps += 1
            for key, value in step_metrics.items():
                aggregate[key].append(value)
            pruned_in_step = False
            if hard_prune_due(args.method, budget.fraction, hard_done):
                with budget.measure():
                    embedding = _sample_embedding(model, first_image)
                    event = _prune_event(
                        model, optimizer, embedding,
                        start_macs=start_macs, target_macs=target_macs,
                        min_keep=args.min_keep, eligible_eps=None,
                        allow_overshoot=True,
                    )
                hard_done = True
                cumulative_removed += event["groups_removed"]
                event.update({
                    "event": "hard_0.9W", "epoch": epoch,
                    "optimizer_steps": optimizer_steps,
                    "counted_training_seconds": budget.counted_training_seconds,
                    "budget_fraction": budget.fraction,
                    "cumulative_groups_removed": cumulative_removed,
                })
                removal_events.append(event)
                pruned_in_step = True
                live_current_macs = int(event["achieved_MACs"])
                if event["target_reached"] and time_to_kappa is None:
                    time_to_kappa = budget.counted_training_seconds

            checkpoint_due = fraction_checkpoint_due(budget.fraction, next_checkpoint)
            monitor_due = fraction_checkpoint_due(budget.fraction, next_monitor)
            if pruned_in_step or checkpoint_due or monitor_due or budget.exhausted:
                save_recovery_checkpoint()
            if checkpoint_due:
                next_checkpoint = next_fraction_checkpoint(
                    budget.fraction, args.checkpoint_every_fraction
                )
            if monitor_due:
                evaluator = monitor_evaluator or _evaluate_psnr_monitor
                monitor_psnr = evaluator(
                    model, full_dataset, transform, selected_config,
                    device, monitor_indices,
                )
                monitor_history.append({
                    "budget_fraction": budget.fraction,
                    "counted_training_seconds": budget.counted_training_seconds,
                    "optimizer_steps": optimizer_steps,
                    "monitor_PSNR": float(monitor_psnr),
                    "frame_indices": json.dumps(monitor_indices),
                })
                next_monitor = next_fraction_checkpoint(
                    budget.fraction, args.monitor_every_fraction
                )
            if stop or budget.exhausted:
                completed_epoch = False
                break

        if not budget.exhausted and completed_epoch:
            scheduled_target = None
            eligible_eps = None
            event_name = None
            allow_overshoot = True
            if args.method == "m2_gradual" and 0.1 <= budget.fraction <= 0.8:
                scheduled_kappa = gradual_kappa_target(args.kappa, budget.fraction)
                scheduled_target = start_macs * (1.0 - scheduled_kappa)
                event_name = "m2_cubic_epoch_boundary"
            elif threshold_removal_active(args.method, budget.fraction):
                scheduled_target = target_macs
                eligible_eps = args.removal_eps
                allow_overshoot = False
                event_name = "threshold_epoch_boundary"
            if event_name is not None:
                with budget.measure():
                    embedding = _sample_embedding(model, first_image)
                    event = _prune_event(
                        model, optimizer, embedding,
                        start_macs=start_macs, target_macs=scheduled_target,
                        min_keep=args.min_keep, eligible_eps=eligible_eps,
                        allow_overshoot=allow_overshoot,
                    )
                if event["groups_removed"]:
                    cumulative_removed += event["groups_removed"]
                    event.update({
                        "event": event_name, "epoch": epoch,
                        "optimizer_steps": optimizer_steps,
                        "counted_training_seconds": budget.counted_training_seconds,
                        "budget_fraction": budget.fraction,
                        "cumulative_groups_removed": cumulative_removed,
                    })
                    removal_events.append(event)
                    live_current_macs = int(event["achieved_MACs"])
                    if event["achieved_MACs"] <= target_macs and time_to_kappa is None:
                        time_to_kappa = budget.counted_training_seconds
                    save_recovery_checkpoint()

            # Selection/surgery above is active training work.  If that work
            # crosses 0.9W, perform the shared hard event at this same epoch
            # boundary, before any reconstruction-only fine-tuning step.
            if hard_prune_due(args.method, budget.fraction, hard_done):
                with budget.measure():
                    embedding = _sample_embedding(model, first_image)
                    event = _prune_event(
                        model, optimizer, embedding,
                        start_macs=start_macs, target_macs=target_macs,
                        min_keep=args.min_keep, eligible_eps=None,
                        allow_overshoot=True,
                    )
                hard_done = True
                cumulative_removed += event["groups_removed"]
                event.update({
                    "event": "hard_0.9W", "epoch": epoch,
                    "optimizer_steps": optimizer_steps,
                    "counted_training_seconds": budget.counted_training_seconds,
                    "budget_fraction": budget.fraction,
                    "cumulative_groups_removed": cumulative_removed,
                })
                removal_events.append(event)
                live_current_macs = int(event["achieved_MACs"])
                if event["target_reached"] and time_to_kappa is None:
                    time_to_kappa = budget.counted_training_seconds
                save_recovery_checkpoint()

        embedding = _sample_embedding(model, first_image)
        architecture_metrics = _current_metrics(model, embedding, start_macs, output_hw)
        live_current_macs = int(architecture_metrics["current_MACs"])
        row = {
            "method": args.method,
            "seed": args.manualSeed,
            "kappa": args.kappa,
            "budget_seconds": args.budget_seconds,
            "counted_training_seconds": budget.counted_training_seconds,
            "budget_fraction": budget.fraction,
            "epoch": epoch,
            "completed_epoch": completed_epoch,
            "optimizer_steps": optimizer_steps,
            "effective_training_mode": "->".join(dict.fromkeys(step_phases)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "groups_removed_this_event": (
                removal_events[-1]["groups_removed"]
                if removal_events and removal_events[-1].get("epoch") == epoch else 0
            ),
            "cumulative_groups_removed": cumulative_removed,
            "time_target_kappa_first_reached": time_to_kappa if time_to_kappa is not None else "",
            **{key: float(np.mean(values)) for key, values in aggregate.items() if values},
            **architecture_metrics,
        }
        if args.method == "m3_group_lasso":
            row["lambda"] = args.lambda_gl
        if args.method == "m4_pcd":
            row["tau"] = args.tau
        history.append(row)

        periodic_evaluation_due = args.eval_every > 0 and epoch % args.eval_every == 0
        epoch_checkpoint_due = (
            args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0
        )
        fraction_due = fraction_checkpoint_due(budget.fraction, next_checkpoint)
        if (
            periodic_evaluation_due or epoch_checkpoint_due or fraction_due
            or budget.exhausted
        ):
            # Recovery state is durable before any evaluator is allowed to run.
            save_recovery_checkpoint()
        if fraction_due:
            next_checkpoint = next_fraction_checkpoint(
                budget.fraction, args.checkpoint_every_fraction
            )

        if periodic_evaluation_due:
            quality_result = quality_evaluator(
                model,
                {**selected_config, "data_path": args.data_path},
                args.data_path,
                device=device,
                expected_frames=132,
                max_frames=0,
                compute_msssim=True,
                msssim_device=args.msssim_device,
            )
            row["periodic_PSNR"] = quality_result["PSNR_dB"]
            row["periodic_MS_SSIM"] = quality_result["MS_SSIM"]
            history_csv_rows = -1  # Row contents changed after the pre-eval checkpoint.
            model.train()

    # Always make the exact final training state durable before final quality.
    save_recovery_checkpoint()
    quality_result = quality_evaluator(
        model,
        {**selected_config, "data_path": args.data_path},
        args.data_path,
        device=device,
        expected_frames=132,
        max_frames=0,
        compute_msssim=True,
        msssim_device=args.msssim_device,
    )
    final_quality = {
        "PSNR": quality_result["PSNR_dB"],
        "MS_SSIM": quality_result["MS_SSIM"],
        "MS_SSIM_device": quality_result["MS_SSIM_device"],
    }
    final_embeddings = quality_result["embeddings"]

    embedding = _sample_embedding(model, first_image)
    architecture_metrics = _current_metrics(model, embedding, start_macs, output_hw)
    final_macs = int(architecture_metrics["current_MACs"])
    hard_events = [row for row in removal_events if row.get("event") == "hard_0.9W"]
    compute_matched_method = args.method == "d_small_compute" or args.method.startswith("m")
    relative_compute_target_error = (
        abs(final_macs - target_macs) / float(target_macs)
        if compute_matched_method and target_macs > 0 else None
    )
    within_compute_tolerance = (
        relative_compute_target_error <= args.compute_match_tolerance
        if relative_compute_target_error is not None else None
    )
    fps = ""
    timing = None
    if not args.no_final_fps and device.type == "cuda":
        timing = benchmark_decoder_cuda(
            model, final_embeddings, warmup=args.fps_warmup, iterations=args.fps_runs
        )
        fps = timing["FPS"]
    final = {
        "format": "pcd-nerv-compute-pilot-results-v1",
        "method": args.method,
        "seed": args.manualSeed,
        "kappa": args.kappa,
        "budget_seconds": args.budget_seconds,
        "counted_training_seconds": budget.counted_training_seconds,
        "budget_fraction": budget.fraction,
        "optimizer_steps": optimizer_steps,
        "completed_epochs": sum(bool(row["completed_epoch"]) for row in history),
        "time_to_kappa": time_to_kappa,
        "target_MACs": target_macs,
        "start_MACs": start_macs,
        "achieved_MACs": final_macs,
        "target_error_MACs": final_macs - target_macs,
        "overshoot_MACs": max(0.0, target_macs - final_macs),
        "undershoot_MACs": max(0.0, final_macs - target_macs),
        "target_reached": final_macs <= target_macs,
        "target_reachable": hard_events[-1].get("target_reachable") if hard_events else None,
        "compute_match_tolerance": args.compute_match_tolerance,
        "relative_compute_target_error": relative_compute_target_error,
        "within_compute_tolerance": within_compute_tolerance,
        "final_evaluation_frames": int(quality_result["frame_count"]),
        "MS_SSIM_evaluation_device": quality_result["MS_SSIM_device"],
        "final_FPS": fps,
        "timing": timing,
        "final_widths": decoder_layer_widths(model),
        "selection": selection,
        "selected_config": selected_config,
        **architecture_metrics,
        **final_quality,
    }
    with open(output_dir / "final_results.json", "w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": runtime_environment["cuda_available"],
        "device": str(device),
        "gpu": runtime_environment["gpu_name"],
        "MS_SSIM_evaluation_device": quality_result["MS_SSIM_device"],
        "command": " ".join(sys.argv),
        "wallclock_definition": "explicit active training intervals measured by time.perf_counter",
        "excluded_from_W": [
            "quality evaluation", "lightweight PSNR monitoring", "checkpoint I/O",
            "CSV/JSON writing", "final FPS",
        ],
    }
    with open(output_dir / "environment.json", "w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2)
    print(json.dumps(final, indent=2))
    return final


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
