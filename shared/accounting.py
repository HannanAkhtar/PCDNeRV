"""Deployment accounting and timing helpers for saved (H)NeRV models.

This module is intentionally evaluation-only.  It does not alter model
weights, pruning decisions, or training behavior.  Decoder compute follows the
repository convention used by the pilot study:

    FLOPs = 2 * MACs

Only arithmetic in Conv2d, ConvTranspose2d, and Linear modules is counted.
PixelShuffle, reshapes, activations, and normalization are not assigned MACs.
"""

from __future__ import annotations

import io
import math
import statistics
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn as nn

from model_all import HNeRVDecoder


DECODER_PREFIXES = ("decoder.", "head_layer.")
ENCODER_PREFIXES = ("encoder.", "pe_embed.")


def model_device(model: nn.Module) -> torch.device:
    """Return the actual device of a model parameter (or buffer)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def decoder_head_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return the decoder + RGB-head state from the physically loaded model."""
    return {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(DECODER_PREFIXES)
    }


def serialized_fp32_state_bytes(state: Mapping[str, torch.Tensor]) -> int:
    """Size of a torch-serialized CPU FP32 copy of ``state`` in bytes."""
    fp32_state = {}
    for name, value in state.items():
        tensor = value.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.float()
        fp32_state[name] = tensor.contiguous()
    buf = io.BytesIO()
    torch.save(fp32_state, buf)
    return len(buf.getbuffer())


def _embedding_values(embeddings) -> int:
    if embeddings is None:
        return 0
    if isinstance(embeddings, (int, float)):
        return int(embeddings)
    if isinstance(embeddings, torch.Tensor):
        return int(embeddings.numel())
    return int(sum(int(x.numel()) for x in embeddings))


def param_accounting(
    model: nn.Module, embeddings=None, *, embedding_storage=None
) -> Dict[str, int | float]:
    """Count deployment parameters and stored per-frame embedding values.

    ``decoder_head_params`` and ``encoder_params`` count trainable parameters
    from the actual loaded architecture.  The stored representation contains
    decoder/head parameters plus cached HNeRV embeddings; the encoder is
    reported separately because it is not used during HNeRV playback.
    """
    if embedding_storage is not None:
        if embeddings is not None:
            raise ValueError("pass embeddings or embedding_storage, not both")
        embeddings = embedding_storage

    decoder_head_params = 0
    encoder_params = 0
    other_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(DECODER_PREFIXES):
            decoder_head_params += int(param.numel())
        elif name.startswith(ENCODER_PREFIXES):
            encoder_params += int(param.numel())
        else:
            other_params += int(param.numel())

    embedding_storage_values = _embedding_values(embeddings)
    decoder_state_bytes = serialized_fp32_state_bytes(decoder_head_state(model))
    return {
        "decoder_head_params": decoder_head_params,
        "params_M": decoder_head_params / 1e6,
        "encoder_params": encoder_params,
        # Keep the v2 accounting API used by training and pruning reports.
        "other_params": other_params,
        "trainable_params": decoder_head_params + encoder_params + other_params,
        "embedding_storage": embedding_storage_values,
        "total_stored_representation": decoder_head_params + embedding_storage_values,
        # More explicit names used by the unified saved-model evaluator.
        "other_trainable_params": other_params,
        "embedding_storage_values": embedding_storage_values,
        "total_stored_representation_values": decoder_head_params + embedding_storage_values,
        "decoder_state_bytes": decoder_state_bytes,
        "decoder_state_MB": decoder_state_bytes / 1e6,
    }


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


@torch.no_grad()
def measure_decoder_compute(model: nn.Module, sample_embedding: torch.Tensor) -> Dict[str, object]:
    """Measure decoder/head MACs with forward hooks on the loaded architecture.

    The input must represent one frame (batch size 1).  The model is not moved;
    the embedding must already be on the same device as the model.
    """
    if int(sample_embedding.shape[0]) != 1:
        raise ValueError("measure_decoder_compute requires a batch-size-1 embedding")
    device = model_device(model)
    if sample_embedding.device != device:
        raise AssertionError(
            f"embedding is on {sample_embedding.device}, model is on {device}"
        )

    decoder = HNeRVDecoder(model).eval()
    rows: List[Dict[str, object]] = []
    hooks = []

    def make_hook(layer_name: str):
        def hook(module, inputs, output):
            inp = _first_tensor(inputs)
            out = _first_tensor(output)
            if inp is None or out is None:
                return
            if isinstance(module, nn.Conv2d):
                kernel_ops = (
                    int(module.kernel_size[0])
                    * int(module.kernel_size[1])
                    * (int(module.in_channels) // int(module.groups))
                )
                macs = int(out.numel()) * kernel_ops
            elif isinstance(module, nn.ConvTranspose2d):
                # Each input element contributes one kernel-sized patch to
                # out_channels/groups outputs.  Counting from the output shape
                # would over-count padded border positions.
                kernel_ops = (
                    int(module.kernel_size[0])
                    * int(module.kernel_size[1])
                    * (int(module.out_channels) // int(module.groups))
                )
                macs = int(inp.numel()) * kernel_ops
            elif isinstance(module, nn.Linear):
                macs = int(out.numel()) * int(module.in_features)
            else:  # pragma: no cover - hooks are only installed on supported types
                return
            rows.append({
                "layer_name": layer_name,
                "module_type": type(module).__name__,
                "input_shape": list(inp.shape),
                "output_shape": list(out.shape),
                "trainable_params": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
                "MACs": macs,
                "GMACs": macs / 1e9,
                "FLOPs": 2 * macs,
                "GFLOPs": 2 * macs / 1e9,
            })
        return hook

    for name, module in decoder.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        output = decoder(sample_embedding)
        if not torch.isfinite(output).all():
            raise FloatingPointError("decoder FLOP probe produced NaN/Inf")
    finally:
        for handle in hooks:
            handle.remove()

    total_macs = int(sum(int(row["MACs"]) for row in rows))
    return {
        "total_MACs": total_macs,
        "total_GMACs": total_macs / 1e9,
        "total_FLOPs": 2 * total_macs,
        "total_GFLOPs": 2 * total_macs / 1e9,
        "per_layer": rows,
    }


@torch.no_grad()
def measure_decoder_flops(model, sample_input, input_embed=None, return_details=False):
    """Backward-compatible decoder/head FLOP probe used by v2 pipelines.

    The historical API accepts a normal model input and returns
    ``(total_flops, per_layer_flops)``.  Passing ``return_details=True`` with
    an already-computed embedding uses the decoder-only detailed accounting.
    """
    if return_details:
        return measure_decoder_compute(model, sample_input)

    flops = {"total": 0, "per_layer": {}}
    hooks = []

    def make_hook(name):
        def hook(mod, inputs, output):
            out = _first_tensor(output)
            kh, kw = mod.kernel_size
            cin = mod.in_channels // mod.groups
            macs = int(out.numel()) * int(cin) * int(kh) * int(kw)
            layer_flops = 2 * macs
            flops["per_layer"][name] = layer_flops
            flops["total"] += layer_flops
        return hook

    core = model.module if hasattr(model, "module") else model
    for name, mod in core.named_modules():
        if isinstance(mod, nn.Conv2d) and (
            name.startswith("decoder.") or name.startswith("head_layer")
        ):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    try:
        core.eval()
        core(sample_input, input_embed)
    finally:
        for handle in hooks:
            handle.remove()
    return flops["total"], flops["per_layer"]


def state_bytes(state_dict):
    """Compatibility alias: serialized size without changing tensor dtype."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return len(buf.getbuffer())


def count_params(state_dict):
    return sum(value.numel() for value in state_dict.values())


def _block_conv_and_shuffle(block):
    conv = None
    shuffle = 1
    for module in block.modules():
        if module is block:
            continue
        if conv is None and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            conv = module
        if isinstance(module, nn.PixelShuffle):
            shuffle = int(module.upscale_factor)
    return conv, shuffle


def decoder_layer_widths(model: nn.Module) -> List[Dict[str, object]]:
    """Describe actual decoder widths, including post-PixelShuffle channels."""
    rows: List[Dict[str, object]] = []
    for index, block in enumerate(model.decoder):
        conv, shuffle = _block_conv_and_shuffle(block)
        if conv is None:
            continue
        if isinstance(conv, nn.ConvTranspose2d):
            in_width, conv_rows = int(conv.in_channels), int(conv.out_channels)
        else:
            in_width, conv_rows = int(conv.in_channels), int(conv.out_channels)

        if index == 0:
            spatial_fold = int(model.fc_h) * int(model.fc_w)
            post_width = conv_rows // spatial_fold
            kind = "fc_stem_after_spatial_reshape"
        else:
            post_width = conv_rows // (shuffle * shuffle)
            kind = "post_pixelshuffle" if shuffle > 1 else "decoder_block_output"

        rows.append({
            "layer_name": f"decoder.{index}",
            "layer_kind": kind,
            "input_width": in_width,
            "conv_output_rows": conv_rows,
            "pixelshuffle_factor": shuffle,
            "post_output_width": int(post_width),
        })

    head = model.head_layer
    rows.append({
        "layer_name": "head_layer",
        "layer_kind": "rgb_head",
        "input_width": int(head.in_channels),
        "conv_output_rows": int(head.out_channels),
        "pixelshuffle_factor": 1,
        "post_output_width": int(head.out_channels),
    })
    return rows


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@torch.no_grad()
def benchmark_decoder_cuda(
    model: nn.Module,
    cached_embeddings: Sequence[torch.Tensor],
    warmup: int = 100,
    iterations: int = 300,
) -> Dict[str, float | int | str]:
    """Benchmark FP32, batch-1 HNeRV playback with CUDA events.

    The benchmark cycles through the real cached frame embeddings.  Encoder
    execution is deliberately excluded from every timed interval.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA decoder timing requested, but CUDA is unavailable")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")
    if not cached_embeddings:
        raise ValueError("at least one cached embedding is required")

    cuda = torch.device("cuda")
    model.to(device=cuda, dtype=torch.float32).eval()
    decoder = HNeRVDecoder(model).to(device=cuda, dtype=torch.float32).eval()
    embeddings = [x.detach().to(device=cuda, dtype=torch.float32) for x in cached_embeddings]

    non_cuda_params = [name for name, p in decoder.named_parameters() if p.device.type != "cuda"]
    if non_cuda_params:
        raise AssertionError(
            "timed decoder has non-CUDA parameters: " + ", ".join(non_cuda_params[:5])
        )
    non_fp32_params = [
        name for name, p in decoder.named_parameters()
        if p.is_floating_point() and p.dtype != torch.float32
    ]
    if non_fp32_params:
        raise AssertionError(
            "timed decoder has non-FP32 parameters: " + ", ".join(non_fp32_params[:5])
        )
    if any(x.device.type != "cuda" for x in embeddings):
        raise AssertionError("one or more cached embeddings are not on CUDA")
    if any(int(x.shape[0]) != 1 for x in embeddings):
        raise AssertionError("decoder timing requires batch-size-1 embeddings")

    for i in range(warmup):
        output = decoder(embeddings[i % len(embeddings)])
    torch.cuda.synchronize()
    if not torch.isfinite(output if warmup else decoder(embeddings[0])).all():
        raise FloatingPointError("timed decoder produced NaN/Inf")
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for i in range(iterations):
        starts[i].record()
        decoder(embeddings[i % len(embeddings)])
        ends[i].record()
    torch.cuda.synchronize()

    elapsed = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    mean_ms = statistics.fmean(elapsed)
    return {
        "latency_mean_ms": mean_ms,
        "latency_median_ms": statistics.median(elapsed),
        "latency_std_ms": statistics.pstdev(elapsed),
        "latency_p90_ms": _percentile(elapsed, 0.90),
        "FPS": 1000.0 / mean_ms,
        "GPU": torch.cuda.get_device_name(torch.cuda.current_device()),
        "precision": "FP32",
        "batch_size": 1,
        "warmup": int(warmup),
        "timing_runs": int(iterations),
    }


@torch.no_grad()
def measure_decode_latency(model, sample_input, runs=20, warmup=3):
    """Preserve the physical-pruning report's existing latency API."""
    import numpy as np

    core = model.module if hasattr(model, "module") else model
    core.eval()
    times = []
    for index in range(warmup + runs):
        _, _, dec_time = core(sample_input)
        if index >= warmup:
            times.append(dec_time)
    values = np.asarray(times)
    batch = sample_input.shape[0] if sample_input.dim() == 4 else 1
    return {
        "latency_mean_s": float(values.mean()),
        "latency_p50_s": float(np.percentile(values, 50)),
        "latency_p90_s": float(np.percentile(values, 90)),
        "fps": float(batch / values.mean()) if values.mean() > 0 else float("inf"),
        "runs": runs,
    }
