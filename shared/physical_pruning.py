"""
Physical pruning for PCD-NeRV v2: exact mode and matched-budget mode, both on
the shared bias-inclusive channel-group abstraction (shared.groups).

Exact mode (deployment-team issue #13: "did training create genuinely dead
channels?"):
    remove channel groups whose bias-inclusive norm < group_thr. Because v2
    groups include biases and the proximal operator zeroes whole groups, a
    dead group's post-activation output is identically zero (act(0)=0 is
    checked on the actual module), so removal preserves the output exactly —
    no consumer analysis or kept-for-exactness channels are needed anymore.

Budget mode (issue #4/#13: "which method reconstructs best at the same final
size?"):
    rank ALL channel groups by norm ascending (each layer's strongest
    min_keep channels are protected), then remove the globally weakest k
    groups where k is binary-searched so the rebuilt decoder + RGB head
    parameter count is as large as possible while <= the requested budget.
    Quality may drop; report it, optionally fine-tune afterwards
    (train_pcd_nerv.py --init_artifact).

Both modes rebuild a physically smaller dense model (the source of truth for
all compression numbers — issue #3) and save a self-contained artifact that
load_pruned_artifact() can reload for evaluation, fine-tuning, or deployment.
"""

import types
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn

from model_all import HNeRV
from shared.nerv_targets import get_channel_group_layers, find_block_conv


# ── plans ─────────────────────────────────────────────────────────────────────

@dataclass
class BlockPlan:
    block_idx: int
    conv_path: str
    r: int
    channels_before: int
    groups_dead: int                 # channel groups under group_thr (diagnostic)
    keep_channels: list = field(default_factory=list)
    in_ch_before: int = 0
    in_ch_after: int = 0
    rows_before: int = 0
    rows_after: int = 0
    params_before: int = 0
    params_after: int = 0


def _act_maps_zero_to_zero(block):
    with torch.no_grad():
        v = block.act(block.norm(torch.zeros(1)))
    return float(v.abs().max()) == 0.0


def _decoder_head_params_for_keeps(model, layers, keeps):
    """
    Exact decoder + head parameter count for a hypothetical keep configuration
    (list of kept-channel counts per targeted layer), by width propagation.
    """
    core = model.module if hasattr(model, 'module') else model
    total = sum(p.numel() for n, p in core.named_parameters() if n.startswith('decoder.0.'))
    in_ch = layers[0].conv.in_channels
    for layer, keep_n in zip(layers, keeps):
        kh, kw = layer.conv.kernel_size
        rows = keep_n * layer.r ** 2
        total += rows * in_ch * kh * kw + (rows if layer.has_bias else 0)
        in_ch = keep_n
    head = core.head_layer
    kh, kw = head.kernel_size
    total += head.out_channels * in_ch * kh * kw
    if head.bias is not None:
        total += head.out_channels
    return total


def compute_exact_plan(model, group_thr=1e-4, min_keep=1):
    """Keep every channel group whose bias-inclusive norm >= group_thr."""
    layers = get_channel_group_layers(model, 'conv')
    core = model.module if hasattr(model, 'module') else model
    keep_sets = []
    for layer in layers:
        if not _act_maps_zero_to_zero(core.decoder[layer.block_idx]):
            raise NotImplementedError(
                f'decoder.{layer.block_idx}: activation does not map 0 to 0 '
                f'(e.g. softplus) — a zero group is not a dead channel, so '
                f'exact pruning is unavailable for this activation.'
            )
        norms = layer.group_norms()
        keep = norms >= group_thr
        if int(keep.sum()) < min_keep:
            for idx in torch.argsort(norms, descending=True)[:min_keep]:
                keep[idx] = True
        keep_sets.append(torch.nonzero(keep, as_tuple=False).flatten().tolist())
    return _build_plans(model, layers, keep_sets)


def compute_budget_plan(model, budget_params=None, budget_reduce=None, min_keep=1):
    """
    Magnitude ranking to a matched decoder+head parameter budget.

    budget_params : target decoder+head parameter count, OR
    budget_reduce : target fractional reduction (e.g. 0.5 -> keep <= 50%).
    """
    layers = get_channel_group_layers(model, 'conv')
    full_keeps = [layer.channels for layer in layers]
    full_params = _decoder_head_params_for_keeps(model, layers, full_keeps)
    if budget_params is None:
        if budget_reduce is None:
            raise ValueError('budget mode needs --budget_params or --budget_reduce')
        budget_params = int(round(full_params * (1.0 - budget_reduce)))

    # rank candidates globally by norm; protect each layer's top-min_keep
    candidates = []
    for li, layer in enumerate(layers):
        norms = layer.group_norms()
        order = torch.argsort(norms, descending=True)
        protected = set(order[:min_keep].tolist())
        for ch in range(layer.channels):
            if ch not in protected:
                candidates.append((float(norms[ch]), li, ch))
    candidates.sort(key=lambda t: t[0])

    def params_after_removing(k):
        removed = [0] * len(layers)
        for _, li, _ in candidates[:k]:
            removed[li] += 1
        keeps = [c - r for c, r in zip(full_keeps, removed)]
        return _decoder_head_params_for_keeps(model, layers, keeps)

    # smallest k whose params <= budget (params_after_removing is monotone in k)
    lo, hi = 0, len(candidates)
    if params_after_removing(hi) > budget_params:
        k = hi  # budget unreachable under min_keep; prune to the floor
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if params_after_removing(mid) <= budget_params:
                hi = mid
            else:
                lo = mid + 1
        k = lo

    removed_sets = [set() for _ in layers]
    for _, li, ch in candidates[:k]:
        removed_sets[li].add(ch)
    keep_sets = [
        [ch for ch in range(layer.channels) if ch not in removed_sets[li]]
        for li, layer in enumerate(layers)
    ]
    plans, head_keep_in = _build_plans(model, layers, keep_sets)
    budget_info = {
        'requested_budget_params': int(budget_params),
        'full_decoder_head_params': int(full_params),
        'achieved_decoder_head_params': _decoder_head_params_for_keeps(
            model, layers, [len(ks) for ks in keep_sets]),
        'groups_removed': k,
        'groups_total_candidates': len(candidates),
        'budget_reached': params_after_removing(k) <= budget_params,
    }
    return plans, head_keep_in, budget_info


def _build_plans(model, layers, keep_sets, group_thr=1e-4):
    plans = []
    in_keep = None  # None = all inputs kept (layer 1's input is the fc stem output)
    for layer, keep in zip(layers, keep_sets):
        w, b = layer.conv.weight, layer.conv.bias
        norms = layer.group_norms()
        in_before = layer.conv.in_channels
        in_after = in_before if in_keep is None else len(in_keep)
        rows_after = len(keep) * layer.r ** 2
        kh, kw = layer.conv.kernel_size
        plans.append(BlockPlan(
            block_idx=layer.block_idx,
            conv_path=layer.conv_path,
            r=layer.r,
            channels_before=layer.channels,
            groups_dead=int((norms < group_thr).sum()),
            keep_channels=list(keep),
            in_ch_before=in_before,
            in_ch_after=in_after,
            rows_before=w.shape[0],
            rows_after=rows_after,
            params_before=w.numel() + (b.numel() if b is not None else 0),
            params_after=rows_after * in_after * kh * kw + (rows_after if b is not None else 0),
        ))
        in_keep = keep
    return plans, in_keep  # in_keep of the last block = head input keep


# ── surgery ───────────────────────────────────────────────────────────────────

def _rows_for_channels(channels, r):
    rows = []
    for ch in channels:
        rows.extend(range(ch * r * r, (ch + 1) * r * r))
    return rows


def apply_prune_plan(model, plans, head_keep_in):
    """Swap each targeted conv (and the head input) for sliced smaller Conv2d."""
    core = model.module if hasattr(model, 'module') else model
    prev_keep = None
    for plan in plans:
        conv, _, r = find_block_conv(core.decoder[plan.block_idx])
        assert r == plan.r
        keep_rows = _rows_for_channels(plan.keep_channels, r)
        keep_in = list(range(conv.in_channels)) if prev_keep is None else prev_keep

        new_conv = nn.Conv2d(
            len(keep_in), len(keep_rows),
            kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, bias=conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight.data[keep_rows][:, keep_in])
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias.data[keep_rows])

        upconv = core.decoder[plan.block_idx].conv.upconv
        if isinstance(upconv, nn.Conv2d):
            core.decoder[plan.block_idx].conv.upconv = new_conv
        else:
            for idx, m in enumerate(upconv):
                if isinstance(m, nn.Conv2d):
                    upconv[idx] = new_conv
                    break
        prev_keep = plan.keep_channels

    if head_keep_in is not None:
        head = core.head_layer
        new_head = nn.Conv2d(
            len(head_keep_in), head.out_channels,
            kernel_size=head.kernel_size, stride=head.stride,
            padding=head.padding, bias=head.bias is not None,
        )
        with torch.no_grad():
            new_head.weight.copy_(head.weight.data[:, head_keep_in])
            if head.bias is not None:
                new_head.bias.copy_(head.bias.data)
        core.head_layer = new_head
    return model


# ── artifact round-trip ───────────────────────────────────────────────────────

ARTIFACT_FORMAT = 'pcd-nerv-pruned-v2'


def build_model_from_config(config):
    args = types.SimpleNamespace(
        embed=config['embed'], ks=config['ks'], num_blks=config['num_blks'],
        enc_strds=list(config['enc_strds']), enc_dim=config['enc_dim'],
        dec_strds=list(config['dec_strds']), fc_hw=config['fc_hw'],
        reduce=config['reduce'], lower_width=config['lower_width'],
        conv_type=list(config['conv_type']), norm=config['norm'],
        act=config['act'], out_bias=config['out_bias'], fc_dim=config['fc_dim'],
        modelsize=config['modelsize'], saturate_stages=-1,
    )
    return HNeRV(args)


def save_pruned_artifact(path, model, plans, head_keep_in, config, extra=None):
    torch.save(
        {
            'format': ARTIFACT_FORMAT,
            'config': config,
            'plans': [asdict(p) for p in plans],
            'head_keep_in': head_keep_in,
            'state_dict': (model.module if hasattr(model, 'module') else model).state_dict(),
            'extra': extra or {},
        },
        path,
    )


def load_pruned_artifact(path, map_location='cpu'):
    payload = torch.load(path, map_location=map_location)
    assert payload.get('format') == ARTIFACT_FORMAT, 'not a PCD-NeRV v2 pruned artifact'
    model = build_model_from_config(payload['config'])
    plans = [BlockPlan(**p) for p in payload['plans']]
    apply_prune_plan(model, plans, payload['head_keep_in'])
    model.load_state_dict(payload['state_dict'])
    return model, payload
