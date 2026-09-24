#!/usr/bin/env python3
"""Calibrate compute-pilot W from six dense HNeRV Bunny epochs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_all import TransformInput, VideoDataSet
from shared.nerv_targets import get_channel_group_layers
from shared.physical_pruning import build_model_from_config
from shared.runtime import validate_cuda_environment
from shared.wallclock import ActiveTrainingBudget, synchronization_callback_for
from train_compute_pilot import _base_config, _seed_everything, _training_step


def summarize_calibration(measured_epoch_seconds, target_epochs=300):
    values = [float(value) for value in measured_epoch_seconds]
    if not values:
        raise ValueError("at least one measured epoch is required")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    recommended = mean * int(target_epochs)
    return {
        "measured_epoch_seconds": values,
        "mean_active_epoch_seconds": mean,
        "std_active_epoch_seconds": std,
        "target_epochs": int(target_epochs),
        "recommended_W_seconds": recommended,
        "estimated_11_run_counted_training_hours": 11.0 * recommended / 3600.0,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", default="compute_pilot_calibration.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batchSize", type=int, default=2)
    parser.add_argument("--manualSeed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", default="L2")
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
    args.m4_use_prox = False
    args.m4_lambda_prox = 0.0
    args.lambda_gl = 0.0
    return args


def run(args):
    device = torch.device(args.device)
    runtime = validate_cuda_environment(require_cuda=device.type == "cuda")
    _seed_everything(args.manualSeed)
    dataset_args = argparse.Namespace(
        data_path=args.data_path,
        crop_list=args.crop_list,
        resize_list=args.resize_list,
        vid=args.vid,
    )
    dataset = VideoDataSet(dataset_args)
    if len(dataset) != 132:
        raise AssertionError(f"expected 132 Bunny frames, found {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
    )
    transform = TransformInput(dataset_args).to(device)
    first_image = dataset[0]["img"].unsqueeze(0).to(device)
    config = _base_config(args, len(dataset), first_image.shape[-2:])
    model = build_model_from_config(config).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    timer = ActiveTrainingBudget(
        1e30, synchronize_fn=synchronization_callback_for(device)
    )
    epoch_times = []
    for epoch in range(1, 7):
        before = timer.counted_training_seconds
        for batch in loader:
            layers = get_channel_group_layers(model, "conv")
            timer.completed_optimizer_step(
                lambda batch=batch, layers=layers: _training_step(
                    model,
                    optimizer,
                    batch,
                    transform,
                    args,
                    "reconstruction",
                    layers,
                    {},
                    None,
                )
            )
        active_seconds = timer.counted_training_seconds - before
        if epoch >= 2:
            epoch_times.append(active_seconds)
        print(f"epoch {epoch}: active_training_seconds={active_seconds:.6f}")

    result = {
        "format": "pcd-nerv-compute-pilot-calibration-v1",
        "protocol": {"warmup_epochs": 1, "measured_epochs": [2, 3, 4, 5, 6]},
        "GPU": runtime["gpu_name"],
        "torch_version": runtime["torch_version"],
        "CUDA_version": runtime["torch_cuda_version"],
        "configuration": config,
        **summarize_calibration(epoch_times, target_epochs=300),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
