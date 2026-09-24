#!/usr/bin/env python3
"""Profile compute-matched HNeRV pilot feasibility without doing any training."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from model_all import VideoDataSet
from shared.accounting import decoder_layer_widths, measure_decoder_compute, param_accounting
from shared.compute_pruning import compute_target_prune_plan, find_compute_matched_hnerv
from shared.physical_pruning import apply_prune_plan, build_model_from_config
from shared.wallclock import synchronization_callback_for
from train_compute_pilot import _base_config, _sample_embedding, _seed_everything


def _timed(callable_, synchronize_fn=None):
    if synchronize_fn is not None:
        synchronize_fn()
    started = time.perf_counter()
    result = callable_()
    if synchronize_fn is not None:
        synchronize_fn()
    return result, time.perf_counter() - started


@torch.no_grad()
def profile_configuration(
    base_config,
    sample_image,
    *,
    frame_count,
    kappas=(0.50, 0.70),
    min_keep=1,
    compute_match_tolerance=0.02,
    candidate_model_sizes=None,
    synchronize_fn=None,
):
    """Return hard-pruning and dense-search feasibility for one architecture."""
    dense = build_model_from_config(base_config).to(sample_image.device).eval()
    dense_embedding = _sample_embedding(dense, sample_image)
    dense_compute = measure_decoder_compute(dense, dense_embedding)
    dense_macs = int(dense_compute["total_MACs"])
    dense_accounting = param_accounting(dense)
    report = {
        "format": "pcd-nerv-compute-preflight-v1",
        "training_performed": False,
        "frame_count": int(frame_count),
        "compute_match_tolerance": float(compute_match_tolerance),
        "dense": {
            "decoder_MACs": dense_macs,
            "decoder_GMACs": dense_macs / 1e9,
            "decoder_GFLOPs": 2.0 * dense_macs / 1e9,
            "decoder_head_params": int(dense_accounting["decoder_head_params"]),
            "widths": decoder_layer_widths(dense),
        },
        "targets": [],
    }

    for kappa in kappas:
        target_macs = dense_macs * (1.0 - float(kappa))

        def plan_hard_pruning():
            return compute_target_prune_plan(
                dense,
                dense_embedding,
                target_macs=target_macs,
                start_macs=dense_macs,
                min_keep=min_keep,
                allow_target_overshoot=True,
            )

        (plans, head_keep, planning_info), planning_seconds = _timed(
            plan_hard_pruning, synchronize_fn
        )
        pruned = copy.deepcopy(dense)
        apply_prune_plan(pruned, plans, head_keep)
        pruned_compute = measure_decoder_compute(pruned, dense_embedding)
        achieved_macs = int(pruned_compute["total_MACs"])
        hard_relative_error = abs(achieved_macs - target_macs) / max(target_macs, 1.0)

        def search_dense():
            return find_compute_matched_hnerv(
                base_config,
                target_macs,
                sample_image,
                frame_count=frame_count,
                candidate_model_sizes=candidate_model_sizes,
                tolerance=compute_match_tolerance,
            )

        (small, small_config, search_info), search_seconds = _timed(
            search_dense, synchronize_fn
        )
        small_embedding = _sample_embedding(small, sample_image)
        small_compute = measure_decoder_compute(small, small_embedding)
        small_accounting = param_accounting(small)
        small_macs = int(small_compute["total_MACs"])
        small_relative_error = abs(small_macs - target_macs) / max(target_macs, 1.0)

        report["targets"].append({
            "kappa": float(kappa),
            "target_MACs": float(target_macs),
            "target_GFLOPs": 2.0 * target_macs / 1e9,
            "hard_compute_pruning": {
                "planning_wallclock_seconds": float(planning_seconds),
                "groups_removed": int(planning_info["groups_removed"]),
                "achieved_MACs": achieved_macs,
                "achieved_GFLOPs": 2.0 * achieved_macs / 1e9,
                "achieved_kappa": 1.0 - achieved_macs / float(dense_macs),
                "target_error_MACs": float(achieved_macs - target_macs),
                "relative_compute_target_error": float(hard_relative_error),
                "within_compute_tolerance": bool(
                    hard_relative_error <= compute_match_tolerance
                ),
                "target_reachable": bool(planning_info["target_reachable"]),
                "blocked_by_discreteness": bool(
                    planning_info["blocked_by_discreteness"]
                ),
                "final_widths": decoder_layer_widths(pruned),
            },
            "d_small_compute": {
                "search_wallclock_seconds": float(search_seconds),
                "requested_modelsize_M": float(small_config["modelsize"]),
                "actual_decoder_head_params": int(
                    small_accounting["decoder_head_params"]
                ),
                "actual_model_size_Mparams": float(small_accounting["params_M"]),
                "decoder_state_MB": float(small_accounting["decoder_state_MB"]),
                "widths": decoder_layer_widths(small),
                "decoder_MACs": small_macs,
                "decoder_GFLOPs": 2.0 * small_macs / 1e9,
                "relative_compute_target_error": float(small_relative_error),
                "within_compute_tolerance": bool(
                    small_relative_error <= compute_match_tolerance
                ),
                "candidates_measured": int(search_info["candidates_measured"]),
            },
        })
        del pruned, small
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", default="compute_pilot_preflight.json")
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.50, 0.70])
    parser.add_argument("--compute_match_tolerance", type=float, default=0.02)
    parser.add_argument("--candidate_sizes", default="")
    parser.add_argument("--min_keep", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--manualSeed", type=int, default=1)
    parser.add_argument("--vid", default="bunny")
    parser.add_argument("--data_split", default="1_1_1")
    parser.add_argument("--crop_list", default="640_1280")
    parser.add_argument("--resize_list", default="-1")
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
    args = parser.parse_args(argv)
    args.method = "d_start"
    if args.compute_match_tolerance < 0:
        parser.error("--compute_match_tolerance must be non-negative")
    if args.min_keep < 1:
        parser.error("--min_keep must be >= 1")
    if any(not 0 <= kappa < 1 for kappa in args.kappas):
        parser.error("every --kappas value must be in [0, 1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu")
    return args


def run(args):
    device = torch.device(args.device)
    _seed_everything(args.manualSeed)
    dataset_args = argparse.Namespace(
        data_path=args.data_path,
        crop_list=args.crop_list,
        resize_list=args.resize_list,
        vid=args.vid,
    )
    dataset = VideoDataSet(dataset_args)
    sample_image = dataset[0]["img"].unsqueeze(0).to(device)
    base_config = _base_config(args, len(dataset), sample_image.shape[-2:])
    candidate_sizes = [
        float(value) for value in args.candidate_sizes.split(",") if value.strip()
    ]
    report = profile_configuration(
        base_config,
        sample_image,
        frame_count=len(dataset),
        kappas=args.kappas,
        min_keep=args.min_keep,
        compute_match_tolerance=args.compute_match_tolerance,
        candidate_model_sizes=candidate_sizes or None,
        synchronize_fn=synchronization_callback_for(device),
    )
    report["base_config"] = base_config
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
