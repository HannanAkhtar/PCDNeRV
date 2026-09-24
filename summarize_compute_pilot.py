#!/usr/bin/env python3
"""Aggregate completed compute-pilot runs without inventing missing values."""

import argparse
import csv
import json
from pathlib import Path


FULL_COLUMNS = [
    "method", "seed", "kappa", "W_seconds", "final_PSNR", "final_MS_SSIM",
    "decoder_head_params", "GMACs", "GFLOPs", "kMACs_per_pixel",
    "achieved_kappa", "final_FPS", "optimizer_steps", "completed_epochs",
    "time_to_kappa", "compute_match_tolerance",
    "relative_compute_target_error", "within_compute_tolerance",
    "final_widths", "source_run",
]
MEETING_COLUMNS = [
    "Method", "kappa", "Params_M", "GFLOPs", "PSNR_dB", "FPS", "Steps",
    "Time_to_kappa_s",
]


def _write(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, output_dir=None):
    root = Path(root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(root.rglob("final_results.json")):
        with open(path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("format") != "pcd-nerv-compute-pilot-results-v1":
            continue
        rows.append({
            "method": result.get("method", ""),
            "seed": result.get("seed", ""),
            "kappa": result.get("kappa", ""),
            "W_seconds": result.get("budget_seconds", ""),
            "final_PSNR": result.get("PSNR", ""),
            "final_MS_SSIM": result.get("MS_SSIM", ""),
            "decoder_head_params": result.get("decoder_head_params", ""),
            "GMACs": result.get("GMACs_per_frame", ""),
            "GFLOPs": result.get("GFLOPs_per_frame", ""),
            "kMACs_per_pixel": result.get("kMACs_per_pixel", ""),
            "achieved_kappa": result.get("achieved_kappa", ""),
            "final_FPS": result.get("final_FPS", ""),
            "optimizer_steps": result.get("optimizer_steps", ""),
            "completed_epochs": result.get("completed_epochs", ""),
            "time_to_kappa": result.get("time_to_kappa", ""),
            "compute_match_tolerance": result.get("compute_match_tolerance", ""),
            "relative_compute_target_error": result.get(
                "relative_compute_target_error", ""
            ),
            "within_compute_tolerance": result.get("within_compute_tolerance", ""),
            "final_widths": json.dumps(result.get("final_widths", []), separators=(",", ":")),
            "source_run": str(path.parent),
        })
    _write(output / "compute_pilot_summary.csv", rows, FULL_COLUMNS)
    meeting = [{
        "Method": row["method"],
        "kappa": row["kappa"],
        "Params_M": (float(row["decoder_head_params"]) / 1e6) if row["decoder_head_params"] != "" else "",
        "GFLOPs": row["GFLOPs"],
        "PSNR_dB": row["final_PSNR"],
        "FPS": row["final_FPS"],
        "Steps": row["optimizer_steps"],
        "Time_to_kappa_s": row["time_to_kappa"] if row["time_to_kappa"] is not None else "",
    } for row in rows]
    _write(output / "compute_pilot_meeting_table.csv", meeting, MEETING_COLUMNS)
    print(f"Aggregated {len(rows)} runs under {output}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    summarize(args.root, args.output_dir or None)


if __name__ == "__main__":
    main()
