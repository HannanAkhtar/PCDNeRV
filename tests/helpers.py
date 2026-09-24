"""Shared helpers for the PCD-NeRV v2 unit tests: a tiny NeRV-PE model."""

import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from model_all import HNeRV  # noqa: E402
from shared.physical_pruning import build_model_from_config  # noqa: E402


def tiny_args(**overrides):
    """
    Minimal NeRV-PE config: no encoder, two PixelShuffle(2) decoder blocks,
    8x8 output. Small enough that every test runs in milliseconds on CPU.
    """
    args = types.SimpleNamespace(
        embed='pe_1.25_4',        # ch_in = 8
        ks='0_3_3',               # all decoder convs 3x3
        num_blks='1_1',
        enc_strds=[],
        enc_dim='64_16',
        dec_strds=[2, 2],
        fc_hw='2_2',              # first feature map 2x2 -> output 8x8
        reduce=1.2,
        lower_width=4,
        conv_type=['convnext', 'pshuffel'],
        norm='none',
        act='gelu',
        out_bias='tanh',
        fc_dim=8,
        modelsize=0.01,
        saturate_stages=-1,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def tiny_model(seed=0, **overrides):
    torch.manual_seed(seed)
    return HNeRV(tiny_args(**overrides))


def tiny_input(batch=1):
    """NeRV-PE input: normalized frame index."""
    return torch.linspace(0.1, 0.9, batch)


def zero_channel_group(layer, ch):
    """Manually kill one post-shuffle channel (its r^2 rows AND biases)."""
    r2 = layer.r ** 2
    with torch.no_grad():
        layer.conv.weight.data[ch * r2:(ch + 1) * r2] = 0
        if layer.conv.bias is not None:
            layer.conv.bias.data[ch * r2:(ch + 1) * r2] = 0


def tiny_hnerv_config(fc_dim=8):
    return {
        'method': 'd_start', 'arch': 'hnerv', 'vid': 'tiny',
        'data_path': '', 'data_split': '1_1_1', 'shuffle_data': False,
        'crop_list': '8_8', 'resize_list': '-1', 'embed': '',
        'enc_strds': [2], 'enc_dim': '4_2', 'dec_strds': [2],
        'fc_hw': '1_1', 'fc_dim': int(fc_dim), 'ks': '1_1_3',
        'reduce': 2.0, 'lower_width': 2, 'num_blks': '1_1',
        'conv_type': ['conv', 'pshuffel'], 'norm': 'none',
        'act': 'gelu', 'out_bias': 'tanh', 'modelsize': 0.01,
        'saturate_stages': -1,
    }


def tiny_hnerv(seed=0, fc_dim=8, device='cpu', dtype=torch.float32):
    torch.manual_seed(seed)
    model = build_model_from_config(tiny_hnerv_config(fc_dim)).to(device=device, dtype=dtype)
    image = torch.randn(1, 3, 8, 8, device=device, dtype=dtype)
    with torch.no_grad():
        _, embeddings, _ = model(image)
    return model, image, embeddings[0].detach()
