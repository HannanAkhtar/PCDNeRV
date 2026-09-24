"""Device-safe in-training channel surgery with Adam-state preservation."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from model_all import HNeRVDecoder
from shared.nerv_targets import find_block_conv, get_channel_group_layers
from shared.physical_pruning import (
    BlockPlan,
    build_model_from_config,
    build_prune_plan_from_keep_sets,
)


PILOT_CHECKPOINT_FORMAT = "pcd-nerv-compute-pilot-v1"


def _rows_for_channels(channels, r):
    return [row for channel in channels for row in range(channel * r * r, (channel + 1) * r * r)]


@dataclass
class ParameterReplacement:
    name: str
    old_parameter: nn.Parameter
    new_parameter: nn.Parameter
    slice_tensor: Callable[[torch.Tensor], torch.Tensor]


def _new_conv_like(conv, in_channels, out_channels):
    if conv.groups != 1:
        raise NotImplementedError("progressive pruning currently requires groups=1 Conv2d")
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)


def _replace_optimizer_parameters(optimizer, replacements):
    by_old = {item.old_parameter: item for item in replacements}
    for group in optimizer.param_groups:
        group["params"] = [
            by_old[param].new_parameter if param in by_old else param
            for param in group["params"]
        ]
    for item in replacements:
        old_state = optimizer.state.pop(item.old_parameter, {})
        new_state = {}
        for key, value in old_state.items():
            if not torch.is_tensor(value):
                new_state[key] = copy.deepcopy(value)
            elif tuple(value.shape) == tuple(item.old_parameter.shape):
                new_state[key] = item.slice_tensor(value).clone()
            elif value.numel() == 1:
                new_state[key] = value.clone()
            elif tuple(value.shape) == tuple(item.new_parameter.shape):
                new_state[key] = value.clone()
            else:
                raise AssertionError(
                    f"cannot map Adam state {key} for {item.name}: "
                    f"{tuple(value.shape)} -> {tuple(item.new_parameter.shape)}"
                )
        optimizer.state[item.new_parameter] = new_state


def assert_optimizer_state_shapes(optimizer):
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            for key, value in optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value) and value.numel() != 1:
                    if tuple(value.shape) != tuple(parameter.shape):
                        raise AssertionError(
                            f"optimizer state {key} shape {tuple(value.shape)} "
                            f"does not match parameter {tuple(parameter.shape)}"
                        )


@torch.no_grad()
def zero_removed_groups(model, plans):
    """Temporarily zero exactly the groups a plan will physically remove."""
    core = model.module if hasattr(model, "module") else model
    for plan in plans:
        conv, _, r = find_block_conv(core.decoder[plan.block_idx])
        keep = set(plan.keep_channels)
        removed = [channel for channel in range(plan.channels_before) if channel not in keep]
        rows = _rows_for_channels(removed, r)
        if rows:
            conv.weight.data[rows] = 0
            if conv.bias is not None:
                conv.bias.data[rows] = 0


@torch.no_grad()
def apply_progressive_prune_plan(
    model,
    optimizer,
    plans,
    head_keep_in,
    *,
    verify_embedding=None,
    equivalence_tol=1e-5,
):
    """Zero, verify, physically rebuild, and migrate optimizer state."""
    core = model.module if hasattr(model, "module") else model
    zero_removed_groups(core, plans)
    zeroed_output = None
    if verify_embedding is not None:
        zeroed_output = HNeRVDecoder(core)(verify_embedding).detach().clone()

    replacements = []
    previous_keep = None
    for plan in plans:
        conv, _, r = find_block_conv(core.decoder[plan.block_idx])
        if r != plan.r:
            raise AssertionError(f"PixelShuffle factor drift at decoder.{plan.block_idx}")
        keep_rows = _rows_for_channels(plan.keep_channels, r)
        keep_in = list(range(conv.in_channels)) if previous_keep is None else list(previous_keep)
        rows_tensor = torch.as_tensor(keep_rows, dtype=torch.long, device=conv.weight.device)
        in_tensor = torch.as_tensor(keep_in, dtype=torch.long, device=conv.weight.device)
        new_conv = _new_conv_like(conv, len(keep_in), len(keep_rows))
        new_conv.weight.copy_(conv.weight.index_select(0, rows_tensor).index_select(1, in_tensor))
        old_weight, new_weight = conv.weight, new_conv.weight

        def slice_weight(value, rows=rows_tensor, inputs=in_tensor):
            return value.index_select(0, rows.to(value.device)).index_select(1, inputs.to(value.device))

        replacements.append(ParameterReplacement(
            f"{plan.conv_path}.weight", old_weight, new_weight, slice_weight
        ))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias.index_select(0, rows_tensor))
            old_bias, new_bias = conv.bias, new_conv.bias

            def slice_bias(value, rows=rows_tensor):
                return value.index_select(0, rows.to(value.device))

            replacements.append(ParameterReplacement(
                f"{plan.conv_path}.bias", old_bias, new_bias, slice_bias
            ))

        upconv = core.decoder[plan.block_idx].conv.upconv
        if isinstance(upconv, nn.Conv2d):
            core.decoder[plan.block_idx].conv.upconv = new_conv
        else:
            for module_index, module in enumerate(upconv):
                if isinstance(module, nn.Conv2d):
                    upconv[module_index] = new_conv
                    break
        previous_keep = plan.keep_channels

    if head_keep_in is not None:
        head = core.head_layer
        keep = torch.as_tensor(head_keep_in, dtype=torch.long, device=head.weight.device)
        new_head = _new_conv_like(head, len(head_keep_in), head.out_channels)
        new_head.weight.copy_(head.weight.index_select(1, keep))
        if head.bias is not None:
            new_head.bias.copy_(head.bias)

        def slice_head_weight(value, inputs=keep):
            return value.index_select(1, inputs.to(value.device))

        replacements.append(ParameterReplacement(
            "head_layer.weight", head.weight, new_head.weight, slice_head_weight
        ))
        if head.bias is not None:
            replacements.append(ParameterReplacement(
                "head_layer.bias", head.bias, new_head.bias, lambda value: value.clone()
            ))
        core.head_layer = new_head

    _replace_optimizer_parameters(optimizer, replacements)
    assert_optimizer_state_shapes(optimizer)
    max_difference = 0.0
    if verify_embedding is not None:
        rebuilt_output = HNeRVDecoder(core)(verify_embedding)
        max_difference = float((zeroed_output - rebuilt_output).abs().max())
        if max_difference >= equivalence_tol:
            raise AssertionError(
                f"zeroed/physical prune mismatch {max_difference:.3g} >= {equivalence_tol}"
            )
    return {
        "replacements": replacements,
        "max_equivalence_error": max_difference,
        "layers": get_channel_group_layers(core, "conv"),
        "groups_removed": sum(
            plan.channels_before - len(plan.keep_channels) for plan in plans
        ),
    }


def canonical_architecture_plan(original_config, current_model):
    """Represent current widths as a backwards-compatible plan from dense."""
    dense = build_model_from_config(original_config)
    current_layers = get_channel_group_layers(current_model, "conv")
    dense_layers = get_channel_group_layers(dense, "conv")
    if len(current_layers) != len(dense_layers):
        raise AssertionError("decoder depth changed during progressive pruning")
    keep_sets = []
    for dense_layer, current_layer in zip(dense_layers, current_layers):
        if current_layer.channels > dense_layer.channels:
            raise AssertionError("current architecture is wider than original")
        keep_sets.append(list(range(current_layer.channels)))
    return build_prune_plan_from_keep_sets(dense, keep_sets)


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_pilot_checkpoint(
    path,
    *,
    original_config,
    model,
    optimizer,
    method,
    kappa,
    start_macs,
    current_macs,
    counted_training_seconds,
    budget_seconds,
    epoch,
    optimizer_steps,
    removal_history,
    frozen_layer_costs,
    history,
    selection,
    seed,
    monitor_history=None,
    solver=None,
    metrics=None,
):
    plans, head_keep = canonical_architecture_plan(original_config, model)
    solver_state = None
    if solver is not None:
        solver_state = {
            "t": int(solver.t), "v": list(solver.v), "tau": float(solver.tau),
            "beta": float(solver.beta), "eps": float(solver.eps),
        }
    payload = {
        "format": PILOT_CHECKPOINT_FORMAT,
        "version": 2,
        "original_config": dict(original_config),
        "current_plans": [asdict(plan) for plan in plans],
        "head_keep_in": head_keep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_defaults": dict(optimizer.defaults),
        "method": method,
        "kappa": float(kappa),
        "start_MACs": int(start_macs),
        "current_MACs": int(current_macs),
        "counted_training_seconds": float(counted_training_seconds),
        "budget_seconds": float(budget_seconds),
        "budget_fraction": float(counted_training_seconds / budget_seconds),
        "epoch": int(epoch),
        "optimizer_steps": int(optimizer_steps),
        "removal_history": list(removal_history),
        "frozen_layer_costs": dict(frozen_layer_costs),
        "history": list(history),
        "selection": copy.deepcopy(selection),
        "seed": int(seed),
        "monitor_history": list(monitor_history or []),
        "solver_state": solver_state,
        "metrics": dict(metrics or {}),
        "rng_state": _rng_state(),
    }
    torch.save(payload, path)
    return payload


def load_pilot_checkpoint(path, device="cpu", restore_rng=True):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format") != PILOT_CHECKPOINT_FORMAT:
        raise ValueError("not a compute-pilot checkpoint")
    model = build_model_from_config(payload["original_config"])
    plans = [BlockPlan(**row) for row in payload["current_plans"]]
    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Architecture reconstruction does not need optimizer migration because no
    # state exists yet; use the same surgery to guarantee identical modules.
    apply_progressive_prune_plan(
        model, dummy_optimizer, plans, payload["head_keep_in"], verify_embedding=None
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    defaults = dict(payload.get("optimizer_defaults", {}))
    defaults.pop("params", None)
    optimizer = torch.optim.Adam(model.parameters(), **defaults)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    assert_optimizer_state_shapes(optimizer)
    solver = None
    if payload.get("solver_state") is not None:
        from shared.pcd_solver import PCDSolver

        state = payload["solver_state"]
        solver = PCDSolver(tau=state["tau"], beta=state["beta"], eps=state["eps"])
        solver.t = int(state["t"])
        solver.v = list(state["v"])
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    return model, optimizer, solver, payload
