"""Active-training wall-clock budget and pilot phase schedules."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager

import torch


PILOT_METHODS = (
    "d_start", "d_small_compute", "d_small_params", "d_replay",
    "m1_posthoc", "m2_gradual", "m3_group_lasso", "m4_pcd",
)


class ActiveTrainingBudget:
    """Accumulate only explicitly measured active-training intervals."""

    def __init__(
        self,
        budget_seconds,
        time_fn=time.perf_counter,
        elapsed=0.0,
        synchronize_fn=None,
    ):
        if budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive")
        self.budget_seconds = float(budget_seconds)
        self.time_fn = time_fn
        self.synchronize_fn = synchronize_fn
        self.counted_training_seconds = float(elapsed)

    @property
    def fraction(self):
        return self.counted_training_seconds / self.budget_seconds

    @property
    def schedule_fraction(self):
        return min(max(self.fraction, 0.0), 1.0)

    @property
    def exhausted(self):
        return self.counted_training_seconds >= self.budget_seconds

    def add_duration(self, seconds):
        if seconds < 0:
            raise ValueError("duration cannot be negative")
        self.counted_training_seconds += float(seconds)

    @contextmanager
    def measure(self):
        if self.synchronize_fn is not None:
            self.synchronize_fn()
        start = self.time_fn()
        try:
            yield
        finally:
            if self.synchronize_fn is not None:
                self.synchronize_fn()
            self.add_duration(self.time_fn() - start)

    def completed_optimizer_step(self, step_fn):
        """Time one complete step and report whether it crossed W."""
        with self.measure():
            result = step_fn()
        return result, self.exhausted


def synchronization_callback_for(device):
    """Return CUDA synchronization only for a CUDA execution device."""
    if torch.device(device).type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but CUDA is not available")
        return torch.cuda.synchronize
    return None


def next_fraction_checkpoint(current_fraction, cadence):
    """Return the next W-fraction checkpoint boundary, or infinity if disabled."""
    cadence = float(cadence)
    if cadence <= 0:
        return math.inf
    current_fraction = max(float(current_fraction), 0.0)
    return (math.floor(current_fraction / cadence) + 1) * cadence


def fraction_checkpoint_due(current_fraction, next_fraction):
    return float(current_fraction) + 1e-12 >= float(next_fraction)


def learning_rate_at_fraction(base_lr, lr_type, fraction):
    """Legacy HNeRV schedule shape parameterized by wall-clock progress."""
    progress = min(max(float(fraction), 0.0), 1.0)
    if "hybrid" in lr_type:
        up_ratio, up_pow, down_pow, min_lr, final_lr = [
            float(x) for x in lr_type.split("_")[1:]
        ]
        if progress < up_ratio:
            multiplier = min_lr + (1.0 - min_lr) * (progress / up_ratio) ** up_pow
        else:
            multiplier = 1.0 - (1.0 - final_lr) * (
                (progress - up_ratio) / (1.0 - up_ratio)
            ) ** down_pow
    elif "cosine" in lr_type:
        up_ratio, up_pow, min_lr = [float(x) for x in lr_type.split("_")[1:]]
        if progress < up_ratio:
            multiplier = min_lr + (1.0 - min_lr) * (progress / up_ratio) ** up_pow
        else:
            multiplier = 0.5 * (
                math.cos(math.pi * (progress - up_ratio) / (1.0 - up_ratio)) + 1.0
            )
    else:
        raise NotImplementedError(f"unsupported lr_type: {lr_type}")
    return float(base_lr) * multiplier


def set_learning_rate(optimizer, base_lr, lr_type, fraction):
    lr = learning_rate_at_fraction(base_lr, lr_type, fraction)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def gradual_kappa_target(final_kappa, budget_fraction):
    """M2 cubic schedule: ``kappa(u)=kappa_final*(1-(1-u)^3)``.

    ``u=(fraction-0.1)/0.7`` is clipped to [0,1], so pruning starts at 0.1W
    and reaches the requested decoder-MAC reduction at 0.8W.
    """
    u = min(max((float(budget_fraction) - 0.1) / 0.7, 0.0), 1.0)
    return float(final_kappa) * (1.0 - (1.0 - u) ** 3)


def pilot_phase(method, budget_fraction, target_reached=False):
    fraction = float(budget_fraction)
    if method in ("d_start", "d_small_compute", "d_small_params", "d_replay"):
        return "reconstruction"
    if fraction < 0.1:
        return "reconstruction_warmup"
    if method == "m1_posthoc":
        return "reconstruction_dense" if fraction < 0.9 else "reconstruction_finetune"
    if method == "m2_gradual":
        if fraction < 0.8 and not target_reached:
            return "gradual_pruning"
        return "reconstruction_finetune"
    if method == "m3_group_lasso":
        return "weighted_group_lasso" if fraction < 0.9 else "reconstruction_finetune"
    if method == "m4_pcd":
        return "pcd_weighted" if fraction < 0.9 else "reconstruction_finetune"
    raise ValueError(method)


def threshold_removal_active(method, budget_fraction):
    return method in ("m3_group_lasso", "m4_pcd") and 0.1 <= float(budget_fraction) < 0.9


def hard_prune_due(method, budget_fraction, already_done=False):
    return (
        method in ("m1_posthoc", "m2_gradual", "m3_group_lasso", "m4_pcd")
        and float(budget_fraction) >= 0.9
        and not already_done
    )
