"""Decoder-compute accounting and model selection for the wall-clock pilot.

The legacy parameter-budget planner remains in ``shared.physical_pruning``.
This module adds an independent MAC-budget planner.  One removable group is
still exactly one bias-inclusive post-PixelShuffle output channel.
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from model_all import HNeRVDecoder
from shared.accounting import decoder_layer_widths, measure_decoder_compute, param_accounting
from shared.nerv_targets import get_channel_group_layers
from shared.physical_pruning import (
    BlockPlan,
    apply_prune_plan,
    build_model_from_config,
    build_prune_plan_from_keep_sets,
    load_pruned_artifact,
)


def synthetic_embedding_from_config(config: Mapping, device="cpu", dtype=torch.float32):
    """Create a batch-one embedding with the shape expected by an HNeRV decoder."""
    if "pe" in str(config.get("embed", "")):
        levels = int(str(config["embed"]).split("_")[-1])
        return torch.zeros(1, 2 * levels, 1, 1, device=device, dtype=dtype)
    crop = str(config.get("crop_list", "640_1280")).split("_")
    height, width = int(crop[0]), int(crop[1])
    stride = int(np.prod(config["enc_strds"]))
    embed_dim = int(str(config["enc_dim"]).split("_")[-1])
    return torch.zeros(
        1, embed_dim, height // stride, width // stride,
        device=device, dtype=dtype,
    )


@torch.no_grad()
def compute_group_mac_costs(model, sample_embedding, normalize=True):
    """Return analytic current decoder MAC savings for each removable group.

    Removing a group saves its own ``r^2`` convolution output rows and one
    input channel in the following convolution (or RGB head).  Costs are
    measured at the actual current spatial sizes.  Normalization uses the
    group-count-weighted mean across the starting model.
    """
    layers = get_channel_group_layers(model, "conv")
    decoder = HNeRVDecoder(model).eval()
    shapes = {}
    hooks = []

    def capture(key):
        def hook(_module, inputs, output):
            shapes[key] = {
                "input": tuple(inputs[0].shape),
                "output": tuple(output.shape),
            }
        return hook

    for layer in layers:
        hooks.append(layer.conv.register_forward_hook(capture(layer.conv_path)))
    hooks.append(model.head_layer.register_forward_hook(capture("head_layer")))
    try:
        decoder(sample_embedding)
    finally:
        for handle in hooks:
            handle.remove()

    raw = {}
    for index, layer in enumerate(layers):
        own_shape = shapes[layer.conv_path]["output"]
        own_pixels = int(own_shape[-2]) * int(own_shape[-1])
        kh, kw = layer.conv.kernel_size
        own = (
            own_pixels * (layer.r ** 2) * int(layer.conv.in_channels)
            * int(kh) * int(kw)
        )
        if index + 1 < len(layers):
            consumer = layers[index + 1].conv
            consumer_shape = shapes[layers[index + 1].conv_path]["output"]
        else:
            consumer = model.head_layer
            consumer_shape = shapes["head_layer"]["output"]
        consumer_pixels = int(consumer_shape[-2]) * int(consumer_shape[-1])
        ckh, ckw = consumer.kernel_size
        downstream = (
            consumer_pixels * int(consumer.out_channels) * int(ckh) * int(ckw)
        )
        raw[layer.conv_path] = int(own + downstream)

    total_groups = sum(layer.channels for layer in layers)
    mean_cost = (
        sum(raw[layer.conv_path] * layer.channels for layer in layers) / total_groups
        if total_groups else 1.0
    )
    result = {}
    for layer in layers:
        cost = raw[layer.conv_path]
        result[layer.conv_path] = {
            "raw_MACs": int(cost),
            "normalized": float(cost / mean_cost) if normalize else float(cost),
            "groups": int(layer.channels),
        }
    return result


def normalized_layer_costs(cost_rows):
    return {name: float(row["normalized"]) for name, row in cost_rows.items()}


@torch.no_grad()
def verify_one_group_cost(model, sample_embedding, layer_index=0):
    """Compare analytic saving with an actual representative one-group prune."""
    work = copy.deepcopy(model)
    layers = get_channel_group_layers(work, "conv")
    if layers[layer_index].channels <= 1:
        raise ValueError("selected layer has no removable group")
    costs = compute_group_mac_costs(work, sample_embedding, normalize=False)
    before = measure_decoder_compute(work, sample_embedding)["total_MACs"]
    keep_sets = [list(range(layer.channels)) for layer in layers]
    keep_sets[layer_index].pop(0)
    plans, head_keep = build_prune_plan_from_keep_sets(work, keep_sets)
    apply_prune_plan(work, plans, head_keep)
    after = measure_decoder_compute(work, sample_embedding)["total_MACs"]
    predicted = costs[layers[layer_index].conv_path]["raw_MACs"]
    return {
        "predicted_MACs": int(predicted),
        "actual_MACs": int(before - after),
        "matches": int(predicted) == int(before - after),
    }


def _choose_candidate(work, embedding, min_keep, eligible_eps=None):
    layers = get_channel_group_layers(work, "conv")
    costs = compute_group_mac_costs(work, embedding, normalize=False)
    candidates = []
    for layer_index, layer in enumerate(layers):
        if layer.channels <= min_keep:
            continue
        norms = layer.group_norms().detach().cpu()
        protected = set(torch.argsort(norms, descending=True)[:min_keep].tolist())
        cost = int(costs[layer.conv_path]["raw_MACs"])
        for channel in range(layer.channels):
            norm = float(norms[channel])
            if channel in protected or (eligible_eps is not None and norm >= eligible_eps):
                continue
            candidates.append((norm / cost, layer_index, channel, norm, cost))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    return candidates


@torch.no_grad()
def compute_target_prune_plan(
    model,
    sample_embedding,
    *,
    kappa=None,
    target_macs=None,
    start_macs=None,
    min_keep=1,
    eligible_eps=None,
    allow_target_overshoot=True,
):
    """Plan hard pruning by repeatedly minimizing ``||theta_g|| / c_g``.

    Costs are recomputed after every physical one-group removal on a working
    copy.  Returned plans remain relative to the caller's current model.
    ``eligible_eps`` restricts candidates for M3/M4 threshold events.
    """
    if kappa is None and target_macs is None:
        raise ValueError("provide kappa or target_macs")
    if min_keep < 1:
        raise ValueError("min_keep must be >= 1")
    work = copy.deepcopy(model)
    embedding = sample_embedding.detach().to(next(work.parameters()).device)
    current = int(measure_decoder_compute(work, embedding)["total_MACs"])
    original_current = current
    if start_macs is None:
        start_macs = original_current
    if target_macs is None:
        target_macs = float(start_macs) * (1.0 - float(kappa))
    target_macs = float(target_macs)

    original_layers = get_channel_group_layers(model, "conv")
    identities = [list(range(layer.channels)) for layer in original_layers]
    history = []
    blocked_by_discreteness = False

    while current > target_macs:
        candidates = _choose_candidate(work, embedding, min_keep, eligible_eps)
        if not candidates:
            break
        selected = None
        for candidate in candidates:
            _, layer_index, channel, norm, cost = candidate
            trial = copy.deepcopy(work)
            trial_layers = get_channel_group_layers(trial, "conv")
            keep_sets = [list(range(layer.channels)) for layer in trial_layers]
            keep_sets[layer_index].remove(channel)
            plans, head_keep = build_prune_plan_from_keep_sets(trial, keep_sets)
            apply_prune_plan(trial, plans, head_keep)
            after = int(measure_decoder_compute(trial, embedding)["total_MACs"])
            if allow_target_overshoot or after >= target_macs:
                selected = (candidate, trial, after)
                break
        if selected is None:
            blocked_by_discreteness = True
            break
        candidate, work, after = selected
        score, layer_index, channel, norm, cost = candidate
        original_channel = identities[layer_index].pop(channel)
        history.append({
            "layer_index": int(layer_index),
            "channel_current": int(channel),
            "channel_original": int(original_channel),
            "group_norm": float(norm),
            "group_cost_MACs": int(cost),
            "score": float(score),
            "MACs_before": int(current),
            "MACs_after": int(after),
        })
        current = after

    plans, head_keep = build_prune_plan_from_keep_sets(model, identities)
    floor_reached = not _choose_candidate(work, embedding, min_keep, eligible_eps)
    achieved_kappa = 1.0 - current / float(start_macs)
    requested_kappa = 1.0 - target_macs / float(start_macs)
    info = {
        "start_MACs": int(start_macs),
        "input_current_MACs": int(original_current),
        "target_MACs": float(target_macs),
        "achieved_MACs": int(current),
        "requested_kappa": float(requested_kappa),
        "achieved_kappa": float(achieved_kappa),
        "target_error_MACs": float(current - target_macs),
        "overshoot_MACs": float(max(0.0, target_macs - current)),
        "undershoot_MACs": float(max(0.0, current - target_macs)),
        "target_reached": bool(current <= target_macs),
        "target_reachable": bool(current <= target_macs or not floor_reached),
        "blocked_by_discreteness": bool(blocked_by_discreteness),
        "groups_removed": len(history),
        "history": history,
        "MACs_monotonic": all(
            row["MACs_after"] < row["MACs_before"] for row in history
        ),
    }
    return plans, head_keep, info


def derive_hnerv_config_for_modelsize(
    base_config: Mapping,
    modelsize: float,
    *,
    frame_count: int,
    output_hw: Sequence[int],
):
    """Apply the legacy HNeRV width equation to a candidate model size."""
    config = dict(base_config)
    config["modelsize"] = float(modelsize)
    enc_strides = list(config["enc_strds"])
    if not enc_strides:
        raise ValueError("compute pilot currently supports HNeRV only")
    total_enc_stride = int(np.prod(enc_strides))
    height, width = int(output_hw[0]), int(output_hw[1])
    embed_hw = (height * width) / (total_enc_stride ** 2)
    enc_dim1, second = [float(x) for x in str(base_config["enc_dim"]).split("_")]
    ratio = base_config.get("embed_ratio")
    if ratio is None and 0 < second < 1:
        ratio = second
    if ratio is not None:
        embed_dim = int(float(ratio) * modelsize * 1e6 / frame_count / embed_hw)
    else:
        embed_dim = int(second)
    embed_dim = max(embed_dim, 1)
    config["enc_dim"] = f"{int(enc_dim1)}_{embed_dim}"
    embed_params = embed_dim / (total_enc_stride ** 2) * height * width * frame_count

    decoder_size = modelsize * 1e6 - embed_params
    if decoder_size <= 0:
        raise ValueError("candidate model size is smaller than embedding storage")
    reduce = float(config["reduce"])
    ch_reduce = 1.0 / reduce
    dec_strides = list(config["dec_strds"])
    dec_ks1, dec_ks2 = [int(x) for x in str(config["ks"]).split("_")[1:]]
    saturate = int(config.get("saturate_stages", -1))
    fixed = len(dec_strides) if saturate == -1 else saturate
    a = ch_reduce * sum(
        ch_reduce ** (2 * i) * stride ** 2 * min(2 * i + dec_ks1, dec_ks2) ** 2
        for i, stride in enumerate(dec_strides[:fixed])
    )
    fc_param = (int(np.prod(enc_strides)) // int(np.prod(dec_strides))) ** 2 * 9
    b = embed_dim * fc_param
    c = int(config["lower_width"]) ** 2 * sum(
        stride ** 2 * min(2 * (fixed + i) + dec_ks1, dec_ks2) ** 2
        for i, stride in enumerate(dec_strides[fixed:])
    )
    roots = np.roots([a, b, c - decoder_size])
    positive = [float(root.real) for root in roots if abs(root.imag) < 1e-6 and root.real > 0]
    if not positive:
        raise ValueError("candidate model size has no valid positive fc_dim")
    config["fc_dim"] = max(1, int(max(positive)))
    return config


def _candidate_sizes(start_modelsize, candidate_model_sizes=None):
    if candidate_model_sizes is not None:
        return sorted({float(x) for x in candidate_model_sizes if 0 < float(x) < start_modelsize})
    low = max(start_modelsize * 0.03, 0.002)
    return sorted(set(float(x) for x in np.geomspace(low, start_modelsize * 0.995, 80)))


@torch.no_grad()
def find_compute_matched_hnerv(
    base_config,
    target_macs,
    sample_image,
    *,
    frame_count,
    candidate_model_sizes=None,
    tolerance=0.02,
):
    """Search ordinary dense HNeRV candidates using measured decoder MACs."""
    start_size = float(base_config["modelsize"])
    output_hw = sample_image.shape[-2:]
    candidates = []
    for size in _candidate_sizes(start_size, candidate_model_sizes):
        try:
            config = derive_hnerv_config_for_modelsize(
                base_config, size, frame_count=frame_count, output_hw=output_hw
            )
            model = build_model_from_config(config).to(sample_image.device).eval()
            _, embeddings, _ = model(sample_image)
            macs = int(measure_decoder_compute(model, embeddings[0])["total_MACs"])
            candidates.append((abs(macs - target_macs), macs, config))
            del model
        except (ValueError, RuntimeError):
            continue
    if not candidates:
        raise RuntimeError("no valid dense HNeRV candidates were constructed")
    _, macs, config = min(candidates, key=lambda row: (row[0], row[1]))
    model = build_model_from_config(config).to(sample_image.device).eval()
    relative_error = abs(macs - target_macs) / max(float(target_macs), 1.0)
    if relative_error > tolerance:
        warnings.warn(
            f"closest compute-matched dense model differs by {relative_error:.2%} "
            f"(target={target_macs:.0f}, actual={macs})",
            RuntimeWarning,
        )
    return model, config, {
        "target_MACs": float(target_macs),
        "actual_MACs": int(macs),
        "relative_error": float(relative_error),
        "within_tolerance": bool(relative_error <= tolerance),
        "candidates_measured": len(candidates),
    }


def find_parameter_matched_hnerv(
    base_config,
    target_params,
    *,
    frame_count,
    output_hw,
    candidate_model_sizes=None,
):
    """Search ordinary dense HNeRV candidates using actual decoder parameters."""
    candidates = []
    for size in _candidate_sizes(float(base_config["modelsize"]), candidate_model_sizes):
        try:
            config = derive_hnerv_config_for_modelsize(
                base_config, size, frame_count=frame_count, output_hw=output_hw
            )
            model = build_model_from_config(config)
            params = int(param_accounting(model)["decoder_head_params"])
            candidates.append((abs(params - target_params), params, config))
        except (ValueError, RuntimeError):
            continue
    if not candidates:
        raise RuntimeError("no valid parameter-matched HNeRV candidates were constructed")
    _, params, config = min(candidates, key=lambda row: (row[0], row[1]))
    model = build_model_from_config(config)
    return model, config, {
        "target_params": int(target_params),
        "actual_params": int(params),
        "relative_error": abs(params - target_params) / max(float(target_params), 1.0),
        "candidates_measured": len(candidates),
    }


@torch.no_grad()
def build_random_architecture_from_pruned_artifact(artifact_path, seed, sample_embedding=None):
    """Build the artifact's exact architecture with fresh random weights."""
    source_model, payload = load_pruned_artifact(str(artifact_path), map_location="cpu")
    torch.manual_seed(int(seed))
    random_model = build_model_from_config(payload["config"])
    plans = [BlockPlan(**row) for row in payload["plans"]]
    apply_prune_plan(random_model, plans, payload["head_keep_in"])
    if sample_embedding is None:
        sample_embedding = synthetic_embedding_from_config(payload["config"])
    source_compute = measure_decoder_compute(source_model, sample_embedding)["total_MACs"]
    replay_compute = measure_decoder_compute(random_model, sample_embedding)["total_MACs"]
    source_params = param_accounting(source_model)["decoder_head_params"]
    replay_params = param_accounting(random_model)["decoder_head_params"]
    source_widths = decoder_layer_widths(source_model)
    replay_widths = decoder_layer_widths(random_model)
    if (source_compute, source_params, source_widths) != (
        replay_compute, replay_params, replay_widths
    ):
        raise AssertionError("replay architecture does not exactly match source artifact")
    copied = all(
        torch.equal(value, random_model.state_dict()[name])
        for name, value in source_model.state_dict().items()
    )
    if copied:
        raise AssertionError("replay unexpectedly copied all source weights")
    return random_model, dict(payload["config"]), {
        "decoder_head_params": int(replay_params),
        "decoder_MACs": int(replay_compute),
        "widths": replay_widths,
        "source_artifact": str(Path(artifact_path).resolve()),
    }
