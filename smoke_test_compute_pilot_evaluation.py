#!/usr/bin/env python3
"""Exercise GPU PSNR, CPU MS-SSIM, embedding caching, and optional FPS."""

from __future__ import annotations

import argparse
import json
import math

import torch

from evaluate_saved_models import evaluate_quality_and_embeddings
from model_all import VideoDataSet
from shared.accounting import benchmark_decoder_cuda
from shared.physical_pruning import build_model_from_config
from shared.runtime import validate_cuda_environment
from train_compute_pilot import _base_config, _seed_everything


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--frames", type=int, choices=(1, 2), default=2)
    parser.add_argument("--fps_runs", type=int, default=0)
    parser.add_argument("--fps_warmup", type=int, default=2)
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
    return args


def run(args):
    runtime = validate_cuda_environment(require_cuda=True)
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
    sample = dataset[0]["img"].unsqueeze(0).cuda()
    config = _base_config(args, len(dataset), sample.shape[-2:])
    model = build_model_from_config(config).cuda().eval()
    quality = evaluate_quality_and_embeddings(
        model,
        config,
        args.data_path,
        device="cuda",
        expected_frames=132,
        max_frames=args.frames,
        compute_msssim=True,
        msssim_device="cpu",
    )
    if not math.isfinite(quality["PSNR_dB"]):
        raise FloatingPointError("PSNR is not finite")
    if not math.isfinite(quality["MS_SSIM"]):
        raise FloatingPointError("MS-SSIM is not finite")
    if len(quality["embeddings"]) != args.frames:
        raise AssertionError("embedding cache count does not match evaluated frames")
    timing = None
    if args.fps_runs > 0:
        timing = benchmark_decoder_cuda(
            model,
            quality["embeddings"],
            warmup=args.fps_warmup,
            iterations=args.fps_runs,
        )
    result = {
        "format": "pcd-nerv-compute-pilot-evaluation-smoke-v1",
        "GPU": runtime["gpu_name"],
        "torch_version": runtime["torch_version"],
        "CUDA_version": runtime["torch_cuda_version"],
        "evaluated_frames": quality["frame_count"],
        "PSNR_dB": quality["PSNR_dB"],
        "PSNR_device": "cuda",
        "MS_SSIM": quality["MS_SSIM"],
        "MS_SSIM_device": quality["MS_SSIM_device"],
        "cached_embeddings": len(quality["embeddings"]),
        "timing": timing,
        "passed": True,
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
