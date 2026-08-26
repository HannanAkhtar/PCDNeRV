#!/usr/bin/env python3
"""Unified evaluation/profiling for already-trained PCD-NeRV/HNeRV models.

The script never trains or modifies a model.  It reconstructs the physically
saved architecture, evaluates all requested Bunny frames, and writes a new set
of reports below ``<study_root>/unified_inference``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from hnerv_utils import msssim_fn_single, psnr_fn_single
from model_all import VideoDataSet, TransformInput
from shared.accounting import (
    benchmark_decoder_cuda,
    decoder_layer_widths,
    measure_decoder_compute,
    model_device,
    param_accounting,
)
from shared.physical_pruning import build_model_from_config, load_pruned_artifact


DEFAULT_STUDY_ROOT = "/content/drive/MyDrive/PCDNeRV_HNeRV_Structured_Pruning_Study"
FULL_COLUMNS = [
    "model_label", "model_type", "method", "tau", "lambda_prox", "ws_weight",
    "source_checkpoint", "source_artifact", "decoder_head_params", "params_M",
    "encoder_params", "embedding_storage_values", "total_stored_representation_values",
    "decoder_state_bytes", "decoder_state_MB", "GMACs_per_frame", "GFLOPs_per_frame",
    "PSNR_dB", "MS_SSIM", "latency_mean_ms", "latency_median_ms", "latency_std_ms",
    "latency_p90_ms", "FPS", "GPU", "precision", "batch_size", "resolution",
    "warmup", "timing_runs",
]
EXTRA_COLUMNS = [
    "status", "error", "actual_model_device", "frame_count", "result_json",
    "old_PSNR_dB", "PSNR_delta_dB", "old_decoder_head_params", "old_GFLOPs_per_frame",
    "warnings",
]


@dataclass
class ModelSpec:
    model_label: str = ""
    model_type: str = ""
    checkpoint: str = ""
    artifact: str = ""
    result_json: str = ""
    tag: str = ""


@dataclass
class LoadedModel:
    model: torch.nn.Module
    config: Dict[str, object]
    model_type: str
    source_checkpoint: str = ""
    source_artifact: str = ""
    result_json: str = ""
    old_log: Optional[Dict[str, object]] = None


def _torch_load(path, map_location="cpu"):
    """Load trusted local study artifacts across PyTorch 2.0-2.6+."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before weights_only was accepted
        return torch.load(path, map_location=map_location)


def _read_json(path: str | Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _state_dict(payload) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, Mapping) and value and all(
                isinstance(v, torch.Tensor) for v in value.values()
            ):
                return value
        if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload
    raise ValueError("checkpoint does not contain a tensor state_dict")


def _strict_load(model: torch.nn.Module, state: Mapping[str, torch.Tensor], source: str) -> None:
    state = dict(state)
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError as first_error:
        if state and all(name.startswith("module.") for name in state):
            stripped = {name[len("module."):]: value for name, value in state.items()}
            try:
                model.load_state_dict(stripped, strict=True)
                return
            except RuntimeError:
                pass
        raise AssertionError(
            f"loaded architecture does not match state_dict from {source}: {first_error}"
        ) from first_error


def _config_from(payload, old_log) -> Dict[str, object]:
    config: Dict[str, object] = {}
    if isinstance(payload, Mapping) and isinstance(payload.get("config"), Mapping):
        config.update(payload["config"])
    if isinstance(old_log, Mapping) and isinstance(old_log.get("config"), Mapping):
        config.update(old_log["config"])
    return config


def _resolve_reference(value: str, study_root: Path, anchors: Sequence[Path]) -> Path:
    raw = Path(os.path.expandvars(os.path.expanduser(str(value))))
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(anchor / raw for anchor in anchors)
        candidates.append(study_root / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = sorted(study_root.rglob(raw.name)) if study_root.is_dir() else []
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        suffix_parts = tuple(part.lower() for part in raw.parts[-4:])
        ranked = [
            match for match in matches
            if tuple(part.lower() for part in match.parts[-len(suffix_parts):]) == suffix_parts
        ]
        if len(ranked) == 1:
            return ranked[0].resolve()
        raise FileNotFoundError(
            f"ambiguous reference {value!r}; found {len(matches)} files named {raw.name}"
        )
    raise FileNotFoundError(f"could not resolve referenced artifact: {value}")


def build_run_model(spec: ModelSpec, study_root: str | Path, map_location="cpu") -> LoadedModel:
    """Strictly reconstruct a dense, artifact, or recovery model."""
    root = Path(study_root).resolve()
    result_path = Path(spec.result_json).resolve() if spec.result_json else None
    old_log = _read_json(result_path) if result_path and result_path.is_file() else None

    if spec.artifact and not spec.checkpoint:
        artifact_path = Path(spec.artifact).resolve()
        model, payload = load_pruned_artifact(str(artifact_path), map_location=map_location)
        config = dict(payload.get("config", {}))
        if old_log and isinstance(old_log.get("config"), Mapping):
            config.update(old_log["config"])
        model.eval()
        return LoadedModel(
            model=model,
            config=config,
            model_type=spec.model_type or "pruned_artifact",
            source_artifact=str(artifact_path),
            result_json=str(result_path) if result_path else "",
            old_log=old_log,
        )

    if not spec.checkpoint:
        raise ValueError("model specification has neither checkpoint nor artifact")

    checkpoint_path = Path(spec.checkpoint).resolve()
    payload = _torch_load(checkpoint_path, map_location=map_location)
    config = _config_from(payload, old_log)
    init_artifact = config.get("init_artifact")
    if init_artifact:
        anchors = [checkpoint_path.parent]
        if result_path:
            anchors.extend([result_path.parent, result_path.parent.parent])
        artifact_path = _resolve_reference(str(init_artifact), root, anchors)
        model, artifact_payload = load_pruned_artifact(str(artifact_path), map_location=map_location)
        merged = dict(artifact_payload.get("config", {}))
        merged.update(config)
        config = merged
        _strict_load(model, _state_dict(payload), str(checkpoint_path))
        model.eval()
        return LoadedModel(
            model=model,
            config=config,
            model_type="recovery_checkpoint",
            source_checkpoint=str(checkpoint_path),
            source_artifact=str(artifact_path),
            result_json=str(result_path) if result_path else "",
            old_log=old_log,
        )

    if not config:
        raise ValueError(
            f"dense checkpoint {checkpoint_path} has no config; provide its results/<tag>.json"
        )
    model = build_model_from_config(config)
    _strict_load(model, _state_dict(payload), str(checkpoint_path))
    model.eval()
    return LoadedModel(
        model=model,
        config=config,
        model_type=spec.model_type or "dense_checkpoint",
        source_checkpoint=str(checkpoint_path),
        result_json=str(result_path) if result_path else "",
        old_log=old_log,
    )


def _tag_from_checkpoint(path: Path) -> str:
    name = path.name
    for suffix in ("_model_latest.pth", "_model_best.pth", ".pth"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def _tag_from_artifact(path: Path) -> str:
    name = path.stem
    for suffix in ("_exact_pruned", "_budget_pruned", "_pruned"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _find_result_json(path: Path, tag: str) -> str:
    run_dir = path.parent.parent if path.parent.name.lower() == "pruned" else path.parent
    result_dir = run_dir / "results"
    exact = result_dir / f"{tag}.json"
    if exact.is_file():
        return str(exact.resolve())
    if result_dir.is_dir():
        choices = [p for p in result_dir.glob("*.json") if "physical_pruning" not in p.stem]
        starts = sorted((p for p in choices if tag.startswith(p.stem)), key=lambda p: len(p.stem), reverse=True)
        if starts:
            return str(starts[0].resolve())
        if len(choices) == 1:
            return str(choices[0].resolve())
    return ""


def discover_models(study_root: str | Path) -> List[ModelSpec]:
    root = Path(study_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"study root does not exist: {root}")
    specs: List[ModelSpec] = []

    for path in sorted(root.rglob("*_model_latest.pth")):
        tag = _tag_from_checkpoint(path)
        result_json = _find_result_json(path, tag)
        config = {}
        if result_json:
            try:
                config = _read_json(result_json).get("config", {})
            except Exception:
                config = {}
        recovery = "recovery" in {part.lower() for part in path.parts} or bool(config.get("init_artifact"))
        specs.append(ModelSpec(
            model_type="recovery_checkpoint" if recovery else "dense_checkpoint",
            checkpoint=str(path.resolve()), result_json=result_json, tag=tag,
        ))

    artifact_paths = []
    for path in sorted(root.rglob("*.pth")):
        lower = path.name.lower()
        if "pruned" not in lower:
            continue
        if "decoder_dense" in lower or "decoder_pruned" in lower or "decoder_head" in lower:
            continue
        if lower.endswith("_model_latest.pth") or lower.endswith("_model_best.pth"):
            continue
        if lower.endswith("_pruned.pth"):
            artifact_paths.append(path)
    for path in artifact_paths:
        tag = _tag_from_artifact(path)
        specs.append(ModelSpec(
            model_type="pruned_artifact", artifact=str(path.resolve()),
            result_json=_find_result_json(path, tag), tag=tag,
        ))

    unique = {}
    for spec in specs:
        key = spec.checkpoint or spec.artifact
        unique[str(Path(key).resolve())] = spec
    return [unique[key] for key in sorted(unique)]


def load_manifest(path: str | Path, study_root: str | Path) -> List[ModelSpec]:
    data = _read_json(path)
    entries = data.get("models", []) if isinstance(data, Mapping) else data
    if not isinstance(entries, list):
        raise ValueError("manifest must be a JSON list or an object with a 'models' list")
    root = Path(study_root)
    specs = []
    for entry in entries:
        values = dict(entry)
        for key in ("checkpoint", "artifact", "result_json"):
            if values.get(key):
                p = Path(values[key])
                values[key] = str((root / p).resolve() if not p.is_absolute() else p.resolve())
        specs.append(ModelSpec(**{k: values.get(k, "") for k in ModelSpec.__dataclass_fields__}))
    return specs


def _number_from_text(text: str, key: str) -> Optional[float]:
    match = re.search(rf"(?:^|[_/\\-]){re.escape(key)}(?:=)?([0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)", text, re.I)
    return float(match.group(1)) if match else None


def _metadata(tag: str, config: Mapping[str, object], source: str, model_type: str) -> Dict[str, object]:
    source_path = Path(source)
    # Limit filename inference to the study-side suffix.  The repository itself
    # is named PCD-NeRV, so using the entire absolute path would incorrectly
    # classify every baseline as PCD.
    informative_parts = [
        part for part in source_path.parts[-8:]
        if "pcdnerv" not in re.sub(r"[^a-z0-9]", "", part.lower())
    ]
    text = f"{tag} {' '.join(informative_parts)}".lower()
    method_raw = str(config.get("method", "")).lower()
    if "pcd" in method_raw:
        method = "pcd"
    elif "prox" in method_raw:
        method = "prox_gl"
    elif method_raw in ("ws", "weighted_sum") or "weighted" in method_raw:
        method = "weighted_sum"
    elif "baseline" in method_raw or "reconstruction" in method_raw:
        method = "baseline"
    elif "pcd" in text:
        method = "pcd"
    elif "proxgl" in text or "prox_gl" in text or "prox-gl" in text:
        method = "prox_gl"
    elif "weighted" in text or re.search(r"(?:^|_)ws_", text):
        method = "weighted_sum"
    elif "baseline" in text:
        method = "baseline"
    else:
        method = method_raw or "unknown"

    def cfg_number(*keys):
        for key in keys:
            value = config.get(key)
            if value is not None and value != "":
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        return None

    tau = cfg_number("tau")
    if tau is None:
        tau = _number_from_text(text, "tau")
    lambda_prox = cfg_number("lambda_prox", "lambda_gl", "lprox")
    if lambda_prox is None:
        lambda_prox = _number_from_text(text, "lprox") or _number_from_text(text, "lam")
    ws_weight = cfg_number("ws_weight", "weight", "w")
    if ws_weight is None:
        ws_weight = _number_from_text(text, "w")
    budget_match = re.search(r"budget[_-]?(\d+)", text)
    budget = int(budget_match.group(1)) if budget_match else None
    seed_match = re.search(r"seed[_-]?(\d+)", text)
    seed_value = config.get("seed")
    seed = int(seed_value if seed_value not in (None, "") else (seed_match.group(1) if seed_match else 1))
    size = cfg_number("modelsize")
    return {
        "method": method, "tau": tau, "lambda_prox": lambda_prox,
        "ws_weight": ws_weight, "budget": budget, "seed": seed, "modelsize": size,
        "model_type": model_type,
    }


def _readable_label(meta: Mapping[str, object], tag: str, source: str) -> str:
    method = meta["method"]
    size = meta.get("modelsize")
    if method == "pcd":
        base = f"PCD tau={meta['tau']:g}" if meta.get("tau") is not None else "PCD"
    elif method == "prox_gl":
        base = "Prox-GL"
    elif method == "weighted_sum":
        base = "Weighted-sum"
    elif method == "baseline":
        base = f"Dense HNeRV {size:g}M baseline" if size is not None else "Dense HNeRV baseline"
    else:
        base = tag.replace("_", " ")

    budget = meta.get("budget")
    model_type = meta.get("model_type")
    if model_type == "recovery_checkpoint":
        suffix = f"recovered {budget}% budget" if budget is not None else "recovered"
    elif model_type == "pruned_artifact":
        suffix = f"{budget}% budget pruned" if budget is not None else "exact pruned"
    else:
        suffix = "dense"
    return f"{base} - {suffix} (seed {meta.get('seed', 1)})"


def _dataset_path(override: str, config: Mapping[str, object], study_root: Path, result_json: str) -> Path:
    if override:
        path = Path(override).expanduser()
        if path.is_dir() or path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"--data_path does not exist: {path}")

    raw = str(config.get("data_path", ""))
    candidates = []
    if raw:
        p = Path(raw).expanduser()
        candidates.append(p)
        if not p.is_absolute():
            candidates.extend([Path.cwd() / p, study_root / p])
            if result_json:
                candidates.append(Path(result_json).parent.parent / p)
    candidates.extend([
        study_root / "data" / "bunny",
        Path(__file__).resolve().parent / "data" / "bunny",
    ])
    for path in candidates:
        if path.is_dir() or path.is_file():
            return path.resolve()
    raise FileNotFoundError("Bunny data not found; pass --data_path explicitly")


@torch.no_grad()
def evaluate_quality_and_embeddings(
    model: torch.nn.Module,
    config: Mapping[str, object],
    data_path: str | Path,
    device: str | torch.device = "cpu",
    expected_frames: int = 132,
    max_frames: int = 0,
) -> Dict[str, object]:
    """Recompute all-frame quality and cache one detached embedding per frame."""
    args = argparse.Namespace(
        data_path=str(data_path),
        crop_list=str(config.get("crop_list", "640_1280")),
        resize_list=str(config.get("resize_list", "-1")),
        vid=str(config.get("vid", "bunny")),
    )
    dataset = VideoDataSet(args)
    if expected_frames > 0 and len(dataset) != expected_frames:
        raise AssertionError(
            f"expected {expected_frames} Bunny frames, found {len(dataset)} at {data_path}"
        )
    transform = TransformInput(args)
    model.to(device=device, dtype=torch.float32).eval()
    psnr_values, msssim_values, embeddings = [], [], []
    resolution = ""
    count = len(dataset) if max_frames <= 0 else min(len(dataset), max_frames)
    for index in range(count):
        sample = dataset[index]
        img = sample["img"].unsqueeze(0).to(device=device, dtype=torch.float32)
        img_in, img_gt, _ = transform(img)
        if "pe" in str(config.get("embed", "")):
            cur_input = torch.tensor([sample["norm_idx"]], device=device, dtype=torch.float32)
        else:
            cur_input = img_in
        output, embed_list, _ = model(cur_input)
        if not torch.isfinite(output).all():
            raise FloatingPointError(f"model output contains NaN/Inf at frame {index + 1}")
        embedding = embed_list[0].detach().cpu().float().contiguous()
        if not torch.isfinite(embedding).all():
            raise FloatingPointError(f"embedding contains NaN/Inf at frame {index + 1}")
        embeddings.append(embedding)
        psnr_values.extend(float(x) for x in psnr_fn_single(output, img_gt).flatten())
        msssim_values.extend(float(x) for x in msssim_fn_single(output, img_gt).flatten())
        resolution = f"{int(img_gt.shape[-2])}x{int(img_gt.shape[-1])}"

    return {
        "PSNR_dB": sum(psnr_values) / len(psnr_values),
        "MS_SSIM": sum(msssim_values) / len(msssim_values),
        "embeddings": embeddings,
        "frame_count": count,
        "resolution": resolution,
    }


def _find_number(obj, keys: Iterable[str]) -> Optional[float]:
    wanted = {key.lower() for key in keys}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key).lower() in wanted and isinstance(value, (int, float)):
                return float(value)
        for value in obj.values():
            found = _find_number(value, wanted)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_number(value, wanted)
            if found is not None:
                return found
    return None


def _old_metrics(loaded: LoadedModel) -> Dict[str, Optional[float]]:
    log = loaded.old_log or {}
    old_psnr = _find_number(log.get("final", log), ["pred_seen_psnr", "PSNR_dB", "psnr"])
    old_params = _find_number(log, ["decoder_head_params", "dh_params_after", "dec_params_after", "pruned_decoder_params"])
    old_gflops = _find_number(log, ["GFLOPs_per_frame", "decoder_gflops", "gflops_after", "pruned_gflops"])
    if loaded.result_json:
        result_dir = Path(loaded.result_json).parent
        tag = Path(loaded.result_json).stem
        candidates = [result_dir / f"{tag}_physical_pruning.json"]
        candidates.extend(sorted(result_dir.glob(f"{tag}*pruning*.json")))
        for candidate in candidates:
            if candidate.is_file():
                payload = _read_json(candidate)
                sizes = payload.get("sizes", {})
                flops = payload.get("flops", {})
                if loaded.model_type == "pruned_artifact":
                    old_params = old_params or sizes.get("dh_params_after")
                    if old_gflops is None and isinstance(flops.get("after"), (int, float)):
                        old_gflops = float(flops["after"]) / 1e9
                else:
                    old_params = old_params or sizes.get("dh_params_before")
                    if old_gflops is None and isinstance(flops.get("before"), (int, float)):
                        old_gflops = float(flops["before"]) / 1e9
                break
    return {"psnr": old_psnr, "params": old_params, "gflops": old_gflops}


def _warn(message: str, bucket: List[str]) -> None:
    bucket.append(message)
    print(f"WARNING: {message}", flush=True)


def _blank_row(label: str, status: str, error: str = "") -> Dict[str, object]:
    row = {key: "" for key in FULL_COLUMNS + EXTRA_COLUMNS}
    row.update({"model_label": label, "model_type": "missing", "status": status, "error": error})
    return row


EXPECTED_MODELS = [
    ("Dense HNeRV 1.5M baseline", "baseline", "dense_checkpoint", None, None),
    ("Prox-GL dense", "prox_gl", "dense_checkpoint", None, None),
    ("Prox-GL exact", "prox_gl", "pruned_artifact", None, None),
    ("Weighted-sum dense", "weighted_sum", "dense_checkpoint", None, None),
    ("Weighted-sum exact", "weighted_sum", "pruned_artifact", None, None),
    *[(f"PCD tau={tau:g} exact", "pcd", "pruned_artifact", tau, None)
      for tau in (0.001, 0.005, 0.01, 0.05, 0.1)],
    ("PCD tau=0.005 recovered 50% budget", "pcd", "recovery_checkpoint", 0.005, 50),
    ("PCD tau=0.005 recovered 75% budget", "pcd", "recovery_checkpoint", 0.005, 75),
    ("PCD tau=0.01 recovered 50% budget", "pcd", "recovery_checkpoint", 0.01, 50),
    ("Weighted-sum recovered 50% budget", "weighted_sum", "recovery_checkpoint", None, 50),
    ("Prox-GL recovered 50% budget", "prox_gl", "recovery_checkpoint", None, 50),
    ("Direct HNeRV 0.75M baseline", "baseline", "dense_checkpoint", None, None),
]


def _expectation_matches(row: Mapping[str, object], expected) -> bool:
    label, method, model_type, tau, budget = expected
    if row.get("status") != "ok" or row.get("method") != method or row.get("model_type") != model_type:
        return False
    if tau is not None and (row.get("tau") == "" or abs(float(row["tau"]) - tau) > 1e-10):
        return False
    source = f"{row.get('source_checkpoint', '')} {row.get('source_artifact', '')}".lower()
    if budget is not None and not re.search(rf"budget[_-]?{budget}(?:\D|$)", source):
        return False
    if label.startswith("Dense HNeRV 1.5M"):
        return "direct_baselines" not in source and abs(float(row.get("nominal_modelsize", 0) or 0) - 1.5) < 0.01
    if label.startswith("Direct HNeRV 0.75M"):
        return "direct_baselines" in source and abs(float(row.get("nominal_modelsize", 0) or 0) - 0.75) < 0.01
    if model_type == "pruned_artifact" and budget is None and "budget" in source:
        return False
    return True


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def environment_info(study_root: Path, args) -> Dict[str, object]:
    cuda_available = torch.cuda.is_available()
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "study_root": str(study_root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()) if cuda_available else None,
        "requested_device": args.device,
        "precision": "FP32",
        "batch_size": 1,
        "warmup": args.warmup,
        "timing_runs": args.timing_runs,
        "flop_convention": "FLOPs = 2 * MACs",
        "command": " ".join(sys.argv),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified saved-model evaluation for PCD-NeRV/HNeRV")
    parser.add_argument("--study_root", default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--manifest", default="", help="optional JSON model list; paths may be relative to study_root")
    parser.add_argument("--data_path", default="", help="Bunny frame directory; overrides saved config paths")
    parser.add_argument("--output_dir", default="", help="default: <study_root>/unified_inference")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--timing_runs", type=int, default=300)
    parser.add_argument("--expected_frames", type=int, default=132)
    parser.add_argument("--max_models", type=int, default=0, help="smoke/debug only; 0 evaluates all")
    parser.add_argument("--max_frames", type=int, default=0, help="smoke/debug only; 0 evaluates all frames")
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args(argv)


def run(args) -> Dict[str, object]:
    root = Path(args.study_root).expanduser().resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda was requested, but torch.cuda.is_available() is False. "
            "Use a CUDA runtime or pass --device cpu for quality/FLOP evaluation without FPS."
        )
    specs = load_manifest(args.manifest, root) if args.manifest else discover_models(root)
    if args.max_models > 0:
        specs = specs[:args.max_models]
        print(f"WARNING: --max_models={args.max_models}; this is a partial smoke evaluation")
    if args.max_frames > 0:
        print(f"WARNING: --max_frames={args.max_frames}; quality metrics are not full-study metrics")
    if not specs:
        print(f"WARNING: no saved models discovered below {root}")

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "unified_inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, compute_rows, width_rows = [], [], []
    used_labels = set()

    for index, spec in enumerate(specs, 1):
        source = spec.checkpoint or spec.artifact
        print(f"\n[{index}/{len(specs)}] Loading {source}", flush=True)
        row = {key: "" for key in FULL_COLUMNS + EXTRA_COLUMNS}
        row.update({"status": "error", "source_checkpoint": spec.checkpoint, "source_artifact": spec.artifact})
        model_warnings: List[str] = []
        try:
            loaded = build_run_model(spec, root, map_location="cpu")
            meta = _metadata(spec.tag or Path(source).stem, loaded.config, source, loaded.model_type)
            label = spec.model_label or _readable_label(meta, spec.tag or Path(source).stem, source)
            if label in used_labels:
                suffix = Path(source).relative_to(root).as_posix() if Path(source).is_relative_to(root) else source
                label = f"{label} [{suffix}]"
            used_labels.add(label)

            data_path = _dataset_path(args.data_path, loaded.config, root, loaded.result_json)
            quality = evaluate_quality_and_embeddings(
                loaded.model, loaded.config, data_path, device=args.device,
                expected_frames=args.expected_frames, max_frames=args.max_frames,
            )
            actual_device = str(model_device(loaded.model))
            accounting = param_accounting(loaded.model, quality["embeddings"])
            sample_embedding = quality["embeddings"][0].to(model_device(loaded.model))
            compute = measure_decoder_compute(loaded.model, sample_embedding)
            widths = decoder_layer_widths(loaded.model)

            timing = {
                "latency_mean_ms": "", "latency_median_ms": "", "latency_std_ms": "",
                "latency_p90_ms": "", "FPS": "", "GPU": "",
                "precision": "FP32", "batch_size": 1, "warmup": args.warmup,
                "timing_runs": args.timing_runs,
            }
            if args.device == "cuda":
                timing = benchmark_decoder_cuda(
                    loaded.model, quality["embeddings"], warmup=args.warmup,
                    iterations=args.timing_runs,
                )
                actual_device = str(model_device(loaded.model))

            old = _old_metrics(loaded)
            psnr_delta = "" if old["psnr"] is None else quality["PSNR_dB"] - old["psnr"]
            if old["psnr"] is not None and abs(float(psnr_delta)) > 0.02:
                _warn(f"{label}: recomputed PSNR differs from old JSON by {psnr_delta:+.4f} dB", model_warnings)
            if old["params"] is not None and int(old["params"]) != int(accounting["decoder_head_params"]):
                _warn(
                    f"{label}: decoder/head params recomputed={accounting['decoder_head_params']} old={old['params']:g}",
                    model_warnings,
                )
            if old["gflops"] is not None:
                denom = max(abs(float(old["gflops"])), 1e-12)
                if abs(compute["total_GFLOPs"] - float(old["gflops"])) / denom > 0.01:
                    _warn(
                        f"{label}: GFLOPs recomputed={compute['total_GFLOPs']:.6g} old={old['gflops']:.6g}",
                        model_warnings,
                    )

            row.update({
                "model_label": label, "model_type": loaded.model_type,
                "method": meta["method"], "tau": meta["tau"] if meta["tau"] is not None else "",
                "lambda_prox": meta["lambda_prox"] if meta["lambda_prox"] is not None else "",
                "ws_weight": meta["ws_weight"] if meta["ws_weight"] is not None else "",
                "source_checkpoint": loaded.source_checkpoint, "source_artifact": loaded.source_artifact,
                **accounting,
                "GMACs_per_frame": compute["total_GMACs"],
                "GFLOPs_per_frame": compute["total_GFLOPs"],
                "PSNR_dB": quality["PSNR_dB"], "MS_SSIM": quality["MS_SSIM"],
                **timing,
                "resolution": quality["resolution"], "actual_model_device": actual_device,
                "frame_count": quality["frame_count"], "result_json": loaded.result_json,
                "old_PSNR_dB": old["psnr"] if old["psnr"] is not None else "",
                "PSNR_delta_dB": psnr_delta,
                "old_decoder_head_params": old["params"] if old["params"] is not None else "",
                "old_GFLOPs_per_frame": old["gflops"] if old["gflops"] is not None else "",
                "warnings": " | ".join(model_warnings), "status": "ok", "error": "",
                "nominal_modelsize": meta.get("modelsize") if meta.get("modelsize") is not None else "",
            })
            for layer in compute["per_layer"]:
                compute_rows.append({"model_label": label, "model_type": loaded.model_type, **layer})
            for layer in widths:
                width_rows.append({"model_label": label, "model_type": loaded.model_type, **layer})
            print(
                f"  {label}: {accounting['params_M']:.6f} M params | "
                f"{compute['total_GFLOPs']:.6f} GFLOPs | {quality['PSNR_dB']:.4f} dB | "
                f"FPS {timing['FPS'] if timing['FPS'] != '' else 'not timed'}",
                flush=True,
            )
        except Exception as error:
            label = spec.model_label or spec.tag or Path(source).stem
            row.update({"model_label": label, "model_type": spec.model_type, "error": f"{type(error).__name__}: {error}"})
            print(f"ERROR: {label}: {row['error']}", flush=True)
            if args.fail_fast:
                raise
        rows.append(row)

    for expected in EXPECTED_MODELS:
        if not any(_expectation_matches(row, expected) for row in rows):
            message = f"expected study model not found: {expected[0]}"
            print(f"WARNING: {message}")
            rows.append(_blank_row(expected[0], "missing", message))

    env = environment_info(root, args)
    env["evaluated_model_count"] = sum(row.get("status") == "ok" for row in rows)
    env["resolutions"] = sorted({str(row["resolution"]) for row in rows if row.get("resolution")})
    full_columns = FULL_COLUMNS + EXTRA_COLUMNS
    _write_csv(out_dir / "PCDNeRV_unified_inference_metrics.csv", rows, full_columns)
    meeting_columns = ["model_label", "params_M", "GFLOPs_per_frame", "PSNR_dB", "FPS"]
    _write_csv(out_dir / "PCDNeRV_meeting_table.csv", rows, meeting_columns)
    compute_columns = [
        "model_label", "model_type", "layer_name", "module_type", "input_shape", "output_shape",
        "trainable_params", "MACs", "GMACs", "FLOPs", "GFLOPs",
    ]
    width_columns = [
        "model_label", "model_type", "layer_name", "layer_kind", "input_width",
        "conv_output_rows", "pixelshuffle_factor", "post_output_width",
    ]
    _write_csv(out_dir / "per_layer_compute.csv", compute_rows, compute_columns)
    _write_csv(out_dir / "per_layer_widths.csv", width_rows, width_columns)
    with open(out_dir / "environment.json", "w", encoding="utf-8") as handle:
        json.dump(env, handle, indent=2)
    payload = {
        "environment": env,
        "models": rows,
        "per_layer_compute": compute_rows,
        "per_layer_widths": width_rows,
    }
    with open(out_dir / "PCDNeRV_unified_inference_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nSaved unified reports under {out_dir}")
    return payload


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
