"""
The channel-group abstraction shared by training, reporting, and pruning.

v2 fixes (deployment-team issues #1, #2, #3):

  #1  One structured group = ONE post-PixelShuffle feature channel = the r^2
      consecutive convolution output rows that PixelShuffle(r) rearranges into
      that channel. NOT one group per conv row (v1 behaviour). `group_total`
      therefore equals the number of physically removable channels.

  #2  The group norm INCLUDES the r^2 bias entries of those rows. The proximal
      operator scales weights and biases by the same factor, so a dead group
      is a genuinely dead channel: with act(0) == 0 its output is identically
      zero and it is removable with zero output change — no bias/consumer
      special-casing needed anywhere downstream.

  #3  Everything that talks about groups — the group-lasso loss, the proximal
      shrinkage, sparsity reports, and physical pruning — imports THIS module,
      so the training objective and the pruning unit cannot drift apart again.

Weight layout note: an UpConv weight is [C*r^2, C_in, kh, kw] with the r^2
rows of each post-shuffle channel stored consecutively, so
`weight.reshape(C, -1)` puts one channel-group per row; `bias.reshape(C, r^2)`
does the same for biases. Group norms are computed with `Tensor.norm`, whose
subgradient at an exactly-zero group is 0 (safe once prox zeroes a group).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ChannelGroupLayer:
    """One targeted decoder conv, viewed as post-shuffle channel groups."""
    block_idx: int          # index into model.decoder (1..N)
    conv: nn.Conv2d         # live module reference
    conv_path: str          # e.g. 'decoder.3.conv.upconv.0'
    r: int                  # PixelShuffle upscale factor (1 when absent)
    channels: int           # post-shuffle channels = out_rows // r^2

    @property
    def weight_name(self):
        return f'{self.conv_path}.weight'

    @property
    def bias_name(self):
        return f'{self.conv_path}.bias'

    @property
    def has_bias(self):
        return self.conv.bias is not None

    def group_view(self):
        """[channels, group_size] view concatenating weight rows and biases."""
        w = self.conv.weight.reshape(self.channels, -1)
        if self.has_bias:
            b = self.conv.bias.reshape(self.channels, -1)
            return torch.cat([w, b], dim=1)
        return w

    def group_norms(self):
        return self.group_view().norm(dim=1)


# ── loss / prox / sparsity (single source of truth) ──────────────────────────

def group_lasso_loss(layers):
    """Sum of bias-inclusive channel-group L2 norms over all targeted layers."""
    device = layers[0].conv.weight.device if layers else 'cpu'
    loss = torch.tensor(0.0, device=device)
    for layer in layers:
        loss = loss + layer.group_norms().sum()
    return loss


@torch.no_grad()
def apply_group_prox(layers, threshold):
    """
    Proximal operator of threshold * sum-of-group-norms: scale each channel
    group (its r^2 weight rows AND their biases) by max(0, 1 - thr/||g||).
    Groups with norm <= threshold become exactly zero.
    """
    if threshold <= 0:
        return
    for layer in layers:
        norms = layer.group_norms().clamp(min=1e-12)
        scale = (1 - threshold / norms).clamp(min=0)
        layer.conv.weight.data.view(layer.channels, -1).mul_(scale[:, None])
        if layer.has_bias:
            layer.conv.bias.data.view(layer.channels, -1).mul_(scale[:, None])


@torch.no_grad()
def group_sparsity(layers, thr=1e-4):
    """(dead_channel_groups, total_channel_groups) across targeted layers."""
    dead = total = 0
    for layer in layers:
        n = layer.group_norms()
        dead += int((n < thr).sum())
        total += n.numel()
    return dead, total


# ── reporting ─────────────────────────────────────────────────────────────────

GROUP_UNIT = 'post-shuffle channel (r^2 conv rows + their biases)'


@torch.no_grad()
def compute_sparsity_report(model, layers, scope_fn, decoder_head_scopes,
                            group_thr=1e-4, param_thr=1e-8):
    """
    Unified per-epoch sparsity report.

    Group counts use the channel-group unit (physically removable channels).
    Parameter counts are reported for three scopes: all trainable, decoder +
    RGB head (the video representation and main budget axis), and targeted
    tensors. Group sparsity is a diagnostic; the physically rebuilt model is
    the source of truth for compression (see physical_pruning).
    """
    dead, total = group_sparsity(layers, thr=group_thr)

    targeted_names = set()
    for layer in layers:
        targeted_names.add(layer.weight_name)
        if layer.has_bias:
            targeted_names.add(layer.bias_name)

    counts = {
        'trainable': [0, 0], 'decoder_head': [0, 0],
        'targeted': [0, 0], 'non_target': [0, 0],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_total = p.numel()
        n_sparse = int((p.abs() < param_thr).sum())
        counts['trainable'][0] += n_total
        counts['trainable'][1] += n_sparse
        if scope_fn(name) in decoder_head_scopes:
            counts['decoder_head'][0] += n_total
            counts['decoder_head'][1] += n_sparse
        key = 'targeted' if name in targeted_names else 'non_target'
        counts[key][0] += n_total
        counts[key][1] += n_sparse

    def pct(num, den):
        return (num / den * 100.0) if den else 0.0

    tr_t, tr_s = counts['trainable']
    dh_t, dh_s = counts['decoder_head']
    tg_t, tg_s = counts['targeted']
    nt_t, nt_s = counts['non_target']
    return {
        'group_unit': GROUP_UNIT,
        'group_total': total,
        'group_sparse': dead,
        'group_sparsity': pct(dead, total),
        'group_sparse_fraction': f'{dead}/{total}' if total else '0/0',
        'trainable_param_total': tr_t,
        'trainable_param_sparse': tr_s,
        'model_param_sparsity': pct(tr_s, tr_t),
        'decoder_head_param_total': dh_t,
        'decoder_head_param_sparse': dh_s,
        'decoder_head_param_sparsity': pct(dh_s, dh_t),
        'targeted_param_total': tg_t,
        'targeted_param_sparse': tg_s,
        'targeted_param_sparsity': pct(tg_s, tg_t),
        'non_target_param_total': nt_t,
        'non_target_param_sparse': nt_s,
        'non_target_param_sparsity': pct(nt_s, nt_t),
        'sparse_from_targeted_pct': pct(tg_s, tr_s),
        'sparse_from_non_target_pct': pct(nt_s, tr_s),
    }


@torch.no_grad()
def build_layer_sparsity_breakdown(model, layers, scope_fn,
                                   group_thr=1e-4, param_thr=1e-8):
    """
    Per-tensor sparsity rows. Channel-group columns appear on the targeted
    conv WEIGHT rows; the matching bias tensors are marked as grouped with
    their weight (their entries are inside the same group norms).
    """
    by_weight = {layer.weight_name: layer for layer in layers}
    bias_names = {layer.bias_name for layer in layers if layer.has_bias}

    rows = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total = p.numel()
        sparse = int((p.abs() < param_thr).sum())
        row = {
            'layer': name,
            'scope': scope_fn(name),
            'shape': list(p.shape),
            'is_gl_targeted': name in by_weight or name in bias_names,
            'grouping': '',
            'param_total': total,
            'param_sparse': sparse,
            'param_sparsity': (sparse / total * 100.0) if total else 0.0,
            'group_total': 0,
            'group_sparse': 0,
            'group_sparsity': 0.0,
        }
        if name in by_weight:
            layer = by_weight[name]
            norms = layer.group_norms()
            gs = int((norms < group_thr).sum())
            row['grouping'] = f'r={layer.r}, {layer.channels} ch-groups'
            row['group_total'] = layer.channels
            row['group_sparse'] = gs
            row['group_sparsity'] = gs / layer.channels * 100.0 if layer.channels else 0.0
        elif name in bias_names:
            row['grouping'] = 'in weight groups'
        rows.append(row)
    return rows
