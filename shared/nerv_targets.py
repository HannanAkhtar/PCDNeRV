"""
Target selection and parameter scoping for NeRV / HNeRV models (v2).

v2 targets the CONVOLUTIONAL decoder only (`decoder.1..N`), grouped as
post-PixelShuffle channels via shared.groups.ChannelGroupLayer.

The fc target from v1 is DISABLED (deployment-team issue #8): the FC stem's
output rows are reshaped through a `fc_h x fc_w` spatial view before reaching
decoder.1, so individual rows are not independently removable channels, and no
physical surgery exists for them. Reshape-aware FC grouping is a planned
extension, not a supported experiment.

Never targeted: `decoder.0` (the FC stem), `head_layer` (its 3 output filters
are the RGB channels), and the HNeRV encoder (auxiliary; discarded at decode
time). The head's INPUT is sliced by pruning when decoder.N shrinks — that is
dependency bookkeeping, not head targeting.
"""

import torch.nn as nn

from shared.groups import ChannelGroupLayer

VALID_TARGETS = ('conv',)


def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


def find_block_conv(block):
    """
    Return (conv_module, attr_path, pixelshuffle_r) for one decoder NeRVBlock.

    pshuffel    : conv.upconv = Sequential(Conv2d, PixelShuffle(r)) -> r = strd
    interpolate : conv.upconv = Sequential(Upsample, Conv2d)        -> r = 1
    strd == 1   : Sequential(Conv2d, Identity)                      -> r = 1
    """
    upconv = block.conv.upconv
    if isinstance(upconv, nn.Conv2d):
        return upconv, 'conv.upconv', 1
    if isinstance(upconv, nn.ConvTranspose2d):
        raise NotImplementedError(
            "ConvTranspose2d decoders (conv_type='conv') are not supported: "
            "their dim-0 weight groups are input channels, not output filters. "
            "Use decoder conv_type 'pshuffel' or 'interpolate'."
        )
    conv = None
    conv_idx = None
    r = 1
    for idx, m in enumerate(upconv):
        if isinstance(m, nn.Conv2d):
            conv, conv_idx = m, idx
        elif isinstance(m, nn.PixelShuffle):
            r = m.upscale_factor
    if conv is None:
        raise RuntimeError('No Conv2d found inside decoder block upconv.')
    return conv, f'conv.upconv.{conv_idx}', r


def get_channel_group_layers(model, target='conv'):
    """Build the list of ChannelGroupLayer targeted by the group lasso."""
    if target != 'conv':
        raise NotImplementedError(
            f"gl_target='{target}' is not supported in v2. Only 'conv' "
            f"(decoder post-shuffle channel groups) is a physically valid "
            f"channel-pruning experiment; fc-stem grouping needs reshape-aware "
            f"groups and surgery (deferred — see README 'Scope')."
        )
    core = _unwrap(model)
    layers = []
    for i in range(1, len(core.decoder)):
        conv, path, r = find_block_conv(core.decoder[i])
        rows = conv.weight.shape[0]
        assert rows % (r * r) == 0, (
            f'decoder.{i}: {rows} conv rows not divisible by r^2={r * r}'
        )
        layers.append(ChannelGroupLayer(
            block_idx=i, conv=conv, conv_path=f'decoder.{i}.{path}',
            r=r, channels=rows // (r * r),
        ))
    return layers


# ── parameter scoping ─────────────────────────────────────────────────────────

def param_scope(name):
    """encoder / fc_stem / conv_blocks / head / other for a parameter name."""
    if name.startswith('encoder.') or name.startswith('pe_embed.'):
        return 'encoder'
    if name.startswith('decoder.0.'):
        return 'fc_stem'
    if name.startswith('decoder.'):
        return 'conv_blocks'
    if name.startswith('head_layer.'):
        return 'head'
    return 'other'


# decoder + RGB head = the decoded video representation and the main
# parameter-budget axis (deployment-team issue #9)
DECODER_HEAD_SCOPES = ('fc_stem', 'conv_blocks', 'head')


def describe_targets(model, target='conv'):
    """Printable description of the channel-group targets."""
    layers = get_channel_group_layers(model, target)
    lines = [f"Group-lasso target = '{target}'  (unit: post-shuffle channel = r^2 rows + biases)"]
    total_ch = 0
    total_params = 0
    for layer in layers:
        w, b = layer.conv.weight, layer.conv.bias
        n_params = w.numel() + (b.numel() if b is not None else 0)
        total_ch += layer.channels
        total_params += n_params
        lines.append(
            f'  {layer.weight_name:<44} shape={list(w.shape)}  r={layer.r}  '
            f'channel-groups={layer.channels}  group_size={w[0].numel() * layer.r ** 2 + layer.r ** 2}'
        )
    lines.append(f'  -> {total_ch} channel groups over {total_params} targeted parameters (incl. biases)')
    return '\n'.join(lines)
