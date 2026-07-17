"""
Consistent parameter accounting for (H)NeRV (deployment-team issue #9).

Four distinct storage/parameter concepts, reported separately everywhere:

  encoder_params               training-only; discarded at decode time
  decoder_head_params          decoder + RGB head — the MAIN budget axis
  embedding_storage            stored per-frame embedding values
                               (HNeRV: embed_dim x embed_h x embed_w x frames;
                                NeRV-PE: 0 — the index is free)
  total_stored_representation  decoder_head_params + embedding_storage

The physically rebuilt model is the source of truth for compression numbers;
these helpers are used identically by the trainer, the reports, and the
physical-pruning pipeline so the scopes cannot drift.
"""

import io

import torch

from shared.nerv_targets import param_scope, DECODER_HEAD_SCOPES


def param_accounting(model, embedding_storage=0):
    enc = dh = other = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scope = param_scope(name)
        if scope == 'encoder':
            enc += p.numel()
        elif scope in DECODER_HEAD_SCOPES:
            dh += p.numel()
        else:
            other += p.numel()
    return {
        'encoder_params': enc,
        'decoder_head_params': dh,
        'other_params': other,
        'trainable_params': enc + dh + other,
        'embedding_storage': int(embedding_storage),
        'total_stored_representation': dh + int(embedding_storage),
    }


def decoder_head_state(model):
    """state_dict restricted to decoder + RGB head (the video representation)."""
    return {
        k: v for k, v in model.state_dict().items()
        if k.startswith('decoder.') or k.startswith('head_layer.')
    }


def state_bytes(state_dict):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return len(buf.getbuffer())


def count_params(state_dict):
    return sum(v.numel() for v in state_dict.values())


@torch.no_grad()
def measure_decoder_flops(model, sample_input, input_embed=None):
    """
    FLOPs (2 x MACs) of the decoder + head for one frame, measured with
    forward hooks on the actual modules, so pruned models report their true
    cost. Conv MACs = out_elems x (C_in/groups) x kh x kw.
    """
    flops = {'total': 0, 'per_layer': {}}
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            kh, kw = mod.kernel_size
            cin = mod.in_channels // mod.groups
            macs = out.numel() * cin * kh * kw
            flops['per_layer'][name] = 2 * macs
            flops['total'] += 2 * macs
        return hook

    core = model.module if hasattr(model, 'module') else model
    for name, mod in core.named_modules():
        if isinstance(mod, torch.nn.Conv2d) and (
                name.startswith('decoder.') or name.startswith('head_layer')):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    try:
        core.eval()
        core(sample_input, input_embed)
    finally:
        for h in hooks:
            h.remove()
    return flops['total'], flops['per_layer']


@torch.no_grad()
def measure_decode_latency(model, sample_input, runs=20, warmup=3):
    """
    Decode latency stats (seconds) using the model's own dec_time measurement
    (decoder + head only, excludes the encoder). Returns mean/p50/p90 and fps.
    """
    import numpy as np

    core = model.module if hasattr(model, 'module') else model
    core.eval()
    times = []
    for i in range(warmup + runs):
        _, _, dec_time = core(sample_input)
        if i >= warmup:
            times.append(dec_time)
    times = np.array(times)
    batch = sample_input.shape[0] if sample_input.dim() == 4 else 1
    return {
        'latency_mean_s': float(times.mean()),
        'latency_p50_s': float(np.percentile(times, 50)),
        'latency_p90_s': float(np.percentile(times, 90)),
        'fps': float(batch / times.mean()) if times.mean() > 0 else float('inf'),
        'runs': runs,
    }
