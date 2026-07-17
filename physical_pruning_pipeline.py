#!/usr/bin/env python3
"""
physical_pruning_pipeline.py (v2) — exact and matched-budget physical pruning
for PCD-NeRV conv-target runs, with the physically rebuilt model as the source
of truth for every compression number (deployment-team issues #3/#4/#13).

Modes
-----
  --mode exact   remove channel groups whose bias-inclusive norm < group_thr.
                 Answers: "did training create genuinely dead channels?"
                 Verification asserts output equivalence (PSNR gap <= tol).

  --mode budget  rank all channel groups by norm and rebuild the smallest
                 model with decoder+head params <= the requested budget
                 (--budget_reduce 0.5 or --budget_params N). Answers: "which
                 method reconstructs best at the same final size?" Quality may
                 drop; fine-tune afterwards with
                 train_pcd_nerv.py --init_artifact <artifact> --method baseline.

Reported for BOTH modes (dense vs pruned): decoder+head params, serialized
bytes, FLOPs/frame, per-layer widths, decode latency & FPS, PSNR/MS-SSIM over
the full video, and the training report's numbers for cross-checking. Channel-
group sparsity remains a diagnostic only.

Works on fine-tuned pruned runs too: if the run's config records an
init_artifact, the model is rebuilt from that artifact before loading weights,
so prune -> fine-tune -> prune-again chains are supported.

Usage
-----
  python physical_pruning_pipeline.py --run_dir <dir> --tag <tag> --mode exact
  python physical_pruning_pipeline.py --run_dir <dir> --tag <tag> --mode budget --budget_reduce 0.5
"""

import argparse
import csv
import json
import os
import sys
import types
from copy import deepcopy
from datetime import datetime

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_all import VideoDataSet, TransformInput
from hnerv_utils import data_split, psnr_fn_single, msssim_fn_single
from shared.physical_pruning import (
    compute_exact_plan, compute_budget_plan, apply_prune_plan,
    build_model_from_config, save_pruned_artifact, load_pruned_artifact,
)
from shared.accounting import (
    param_accounting, decoder_head_state, state_bytes, count_params,
    measure_decoder_flops, measure_decode_latency,
)


def parse_args():
    p = argparse.ArgumentParser(description='PCD-NeRV v2 physical pruning (exact | budget)')
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--ckpt', type=str, default='', help='default: <run_dir>/<tag>_model_latest.pth')
    p.add_argument('--mode', type=str, default='exact', choices=['exact', 'budget'])
    p.add_argument('--budget_reduce', type=float, default=None,
                   help='[budget] fractional decoder+head parameter reduction, e.g. 0.5')
    p.add_argument('--budget_params', type=int, default=None,
                   help='[budget] absolute decoder+head parameter target')
    p.add_argument('--group_thr', type=float, default=None,
                   help='dead threshold on bias-inclusive channel-group norms (default: run config value)')
    p.add_argument('--min_keep', type=int, default=1, help='minimum surviving channels per block')
    p.add_argument('--no_verify', action='store_true')
    p.add_argument('--psnr_tol', type=float, default=1e-3, help='[exact] max PSNR gap in dB')
    p.add_argument('--bench_runs', type=int, default=20, help='latency benchmark forward passes')
    return p.parse_args()


# ── model (re)construction ────────────────────────────────────────────────────

def build_run_model(config, ckpt_path):
    """Rebuild the run's model (dense, or from its init artifact) and load weights."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if config.get('init_artifact'):
        model, _ = load_pruned_artifact(config['init_artifact'])
    else:
        model = build_model_from_config(config)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def probe_input(config):
    ns = types.SimpleNamespace(data_path=config['data_path'], crop_list=config['crop_list'],
                               resize_list=config['resize_list'], vid=config['vid'])
    frame = VideoDataSet(ns)[0]['img'].unsqueeze(0)
    if 'pe' in config['embed']:
        return torch.tensor([0.0])
    return frame


# ── verification ──────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_pair(dense, pruned, config):
    ns = types.SimpleNamespace(
        data_path=config['data_path'], crop_list=config['crop_list'],
        resize_list=config['resize_list'], vid=config['vid'],
    )
    dataset = VideoDataSet(ns)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    split = [int(x) for x in config['data_split'].split('_')]
    _, val_ind = data_split(list(range(len(dataset))), split, config.get('shuffle_data', False), 0)
    transform = TransformInput(ns)

    dense.eval()
    pruned.eval()
    use_ssim = dataset[0]['img'].shape[-2] > 160 and dataset[0]['img'].shape[-1] > 160

    stats = {m: {'seen_psnr': [], 'unseen_psnr': [], 'seen_msssim': [], 'unseen_msssim': []}
             for m in ('dense', 'pruned')}
    max_diff = 0.0
    for sample in loader:
        img, idx = sample['img'], int(sample['idx'])
        img_in, img_gt, _ = transform(img)
        cur_input = sample['norm_idx'].float() if 'pe' in config['embed'] else img_in
        out_d, _, _ = dense(cur_input)
        out_p, _, _ = pruned(cur_input)
        max_diff = max(max_diff, float((out_d - out_p).abs().max()))
        bucket = 'unseen' if idx in val_ind else 'seen'
        for m, out in (('dense', out_d), ('pruned', out_p)):
            stats[m][f'{bucket}_psnr'].append(float(psnr_fn_single(out, img_gt).mean()))
            if use_ssim:
                stats[m][f'{bucket}_msssim'].append(float(msssim_fn_single(out, img_gt).mean()))

    res = {}
    for m in ('dense', 'pruned'):
        for k, v in stats[m].items():
            res[f'{m}_{k}'] = float(np.mean(v)) if v else float('nan')
    res['max_abs_output_diff'] = max_diff
    return res


# ── report ────────────────────────────────────────────────────────────────────

def build_report_lines(tag, mode, config, plans, crosscheck_rows, sizes, widths,
                       flops, latency, verify, budget_info, verdict):
    L = []
    L.append('=' * 156)
    L.append(f'Physical Pruning Report ({mode} mode): {tag}')
    L.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | vid={config["vid"]} | '
             f'method={config.get("method", "?")} | seed={config.get("seed", "?")} | '
             f'group_thr={sizes["group_thr"]:g} | unit=post-shuffle channel (r^2 rows + biases)')
    L.append('=' * 156)

    L.append('Cross-check vs training sparsity report (dead channel-groups per targeted layer)')
    L.append('-' * 156)
    L.append(f'{"Layer":<44} {"Report dead/total":>18} {"Recomputed dead/total":>22} {"Match":>6}')
    for r in crosscheck_rows:
        L.append(f'{r["layer"]:<44} {r["report"]:>18} {r["recomputed"]:>22} {"OK" if r["match"] else "MISMATCH":>6}')
    L.append('-' * 156)

    if budget_info:
        L.append('Budget')
        L.append('-' * 156)
        L.append(f'  requested decoder+head params : {budget_info["requested_budget_params"]:,} '
                 f'(full: {budget_info["full_decoder_head_params"]:,})')
        L.append(f'  achieved  decoder+head params : {budget_info["achieved_decoder_head_params"]:,} '
                 f'| groups removed {budget_info["groups_removed"]}/{budget_info["groups_total_candidates"]} '
                 f'| budget_reached={budget_info["budget_reached"]}')
        L.append('-' * 156)

    L.append('Prune plan and per-layer widths (post-shuffle channel granularity)')
    L.append('-' * 156)
    L.append(f'{"Block":<38} {"r":>3} {"dead":>6} {"ch before->after":>18} {"in_ch before->after":>20} '
             f'{"rows before->after":>20} {"params before->after":>22}')
    for p in plans:
        L.append(
            f'{p.conv_path:<38} {p.r:>3} {p.groups_dead:>6} {p.channels_before:>8} -> {len(p.keep_channels):<7} '
            f'{p.in_ch_before:>9} -> {p.in_ch_after:<8} {p.rows_before:>9} -> {p.rows_after:<8} '
            f'{p.params_before:>11} -> {p.params_after:<9}'
        )
    L.append(f'{"head_layer":<38} {"-":>3} {"-":>6} {"3 (RGB, fixed)":>18} '
             f'{widths["head_in_before"]:>9} -> {widths["head_in_after"]:<8}')
    L.append('-' * 156)

    L.append('Size / cost accounting (physically rebuilt model = source of truth)')
    L.append('-' * 156)
    L.append(f'  decoder+head params    : {sizes["dh_params_before"]:>12,} -> {sizes["dh_params_after"]:>12,}   '
             f'({sizes["dh_params_reduction_pct"]:.2f}% reduction)')
    L.append(f'  decoder+head bytes     : {sizes["dh_bytes_before"]:>12,} -> {sizes["dh_bytes_after"]:>12,}   '
             f'({sizes["dh_bytes_reduction_pct"]:.2f}% reduction)  '
             f'[{sizes["dh_bytes_before"]/1e6:.3f} MB -> {sizes["dh_bytes_after"]/1e6:.3f} MB]')
    L.append(f'  decoder+head FLOPs/frame: {flops["before"]:,} -> {flops["after"]:,}   '
             f'({flops["reduction_pct"]:.2f}% reduction)')
    L.append(f'  decode latency (mean)  : {latency["before"]["latency_mean_s"]*1e3:.2f} ms -> '
             f'{latency["after"]["latency_mean_s"]*1e3:.2f} ms   '
             f'(fps {latency["before"]["fps"]:.1f} -> {latency["after"]["fps"]:.1f}, '
             f'{latency["runs"]} runs, {latency["device"]})')
    L.append(f'  embedding storage      : {sizes["embedding_storage"]:,} values (unchanged by conv pruning)')
    L.append(f'  total stored repr      : {sizes["total_stored_before"]:,} -> {sizes["total_stored_after"]:,}')
    L.append(f'  encoder params (aux)   : {sizes["encoder_params"]:,} (training-only; untouched)')
    L.append('-' * 156)

    if verify:
        L.append('Verification (full video, dense vs physically pruned)')
        L.append('-' * 156)
        L.append(f'  max |out_dense - out_pruned| : {verify["max_abs_output_diff"]:.3e}')
        L.append(f'  PSNR seen   : dense {verify["dense_seen_psnr"]:.4f} | pruned {verify["pruned_seen_psnr"]:.4f} | '
                 f'training report {verify.get("reported_seen_psnr", float("nan")):.4f}')
        L.append(f'  PSNR unseen : dense {verify["dense_unseen_psnr"]:.4f} | pruned {verify["pruned_unseen_psnr"]:.4f}')
        if not np.isnan(verify.get('dense_seen_msssim', float('nan'))):
            L.append(f'  MS-SSIM seen: dense {verify["dense_seen_msssim"]:.6f} | pruned {verify["pruned_seen_msssim"]:.6f}')
        L.append(f'  round-trip artifact reload   : {"OK" if verify.get("roundtrip_ok") else "FAILED"}')
        L.append('-' * 156)

    L.append(f'VERDICT: {verdict}')
    L.append('=' * 156)
    return L


def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
    args = parse_args()

    run_dir, tag, mode = args.run_dir, args.tag, args.mode
    with open(os.path.join(run_dir, 'results', f'{tag}.json'), encoding='utf-8') as f:
        log = json.load(f)
    config = log['config']
    if config.get('sparsity_target', 'conv') != 'conv':
        raise SystemExit('v2 physical pruning supports conv-target runs only.')

    ckpt_path = args.ckpt or os.path.join(run_dir, f'{tag}_model_latest.pth')
    print(f'Loading run    : {tag} ({mode} mode)')
    print(f'Loading weights: {ckpt_path}')
    dense = build_run_model(config, ckpt_path)

    group_thr = args.group_thr if args.group_thr is not None else config.get('group_thr', 1e-4)

    # ── plan ──────────────────────────────────────────────────────────────────
    budget_info = None
    if mode == 'exact':
        plans, head_keep_in = compute_exact_plan(dense, group_thr=group_thr, min_keep=args.min_keep)
    else:
        plans, head_keep_in, budget_info = compute_budget_plan(
            dense, budget_params=args.budget_params,
            budget_reduce=args.budget_reduce, min_keep=args.min_keep)

    # ── cross-check vs the training report (channel-group granularity) ───────
    breakdown = {r['layer']: r for r in log.get('final_layer_breakdown', [])}
    crosscheck_rows, all_match = [], True
    for p in plans:
        rep = breakdown.get(f'{p.conv_path}.weight')
        rep_str = f"{rep['group_sparse']}/{rep['group_total']}" if rep else 'n/a'
        match = bool(rep) and rep['group_sparse'] == p.groups_dead and rep['group_total'] == p.channels_before
        all_match &= match
        crosscheck_rows.append({'layer': f'{p.conv_path}.weight', 'report': rep_str,
                                'recomputed': f'{p.groups_dead}/{p.channels_before}', 'match': match})
    if not all_match:
        print('WARNING: recomputed dead-group counts do not match the training report '
              '(different checkpoint epoch or threshold?). Proceeding with recomputed values.')

    # ── surgery ───────────────────────────────────────────────────────────────
    pruned = deepcopy(dense)
    apply_prune_plan(pruned, plans, head_keep_in)
    pruned.eval()

    # ── measurements: params, bytes, flops, latency, widths ─────────────────
    emb = int(config.get('accounting', {}).get('embedding_storage', 0))
    acct_before = param_accounting(dense, embedding_storage=emb)
    acct_after = param_accounting(pruned, embedding_storage=emb)
    dh_dense, dh_pruned = decoder_head_state(dense), decoder_head_state(pruned)

    pruned_dir = os.path.join(run_dir, 'pruned')
    os.makedirs(pruned_dir, exist_ok=True)
    dense_dh_path = os.path.join(pruned_dir, f'{tag}_decoder_head_dense.pth')
    pruned_dh_path = os.path.join(pruned_dir, f'{tag}_{mode}_decoder_head_pruned.pth')
    torch.save(dh_dense, dense_dh_path)
    torch.save(dh_pruned, pruned_dh_path)

    def red(b, a):
        return (1.0 - a / b) * 100.0 if b else 0.0

    sizes = {
        'group_thr': group_thr,
        'dh_params_before': acct_before['decoder_head_params'],
        'dh_params_after': acct_after['decoder_head_params'],
        'dh_bytes_before': state_bytes(dh_dense),
        'dh_bytes_after': state_bytes(dh_pruned),
        'embedding_storage': emb,
        'total_stored_before': acct_before['total_stored_representation'],
        'total_stored_after': acct_after['total_stored_representation'],
        'encoder_params': acct_before['encoder_params'],
    }
    sizes['dh_params_reduction_pct'] = red(sizes['dh_params_before'], sizes['dh_params_after'])
    sizes['dh_bytes_reduction_pct'] = red(sizes['dh_bytes_before'], sizes['dh_bytes_after'])

    probe = probe_input(config)
    flops_before, _ = measure_decoder_flops(dense, probe)
    flops_after, _ = measure_decoder_flops(pruned, probe)
    flops = {'before': flops_before, 'after': flops_after,
             'reduction_pct': red(flops_before, flops_after)}
    lat_before = measure_decode_latency(dense, probe, runs=args.bench_runs)
    lat_after = measure_decode_latency(pruned, probe, runs=args.bench_runs)
    latency = {'before': lat_before, 'after': lat_after,
               'runs': args.bench_runs, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

    widths = {
        'head_in_before': dense.head_layer.in_channels,
        'head_in_after': pruned.head_layer.in_channels,
    }

    # ── artifact + round-trip ─────────────────────────────────────────────────
    artifact_path = os.path.join(pruned_dir, f'{tag}_{mode}_pruned.pth')
    save_pruned_artifact(artifact_path, pruned, plans, head_keep_in, config,
                         extra={'mode': mode, 'budget_info': budget_info,
                                'source_checkpoint': ckpt_path})

    # ── verification ──────────────────────────────────────────────────────────
    verify = None
    if mode == 'exact':
        verdict = 'NOT VERIFIED (--no_verify)'
    else:
        verdict = f'BUDGET PRUNE (unverified quality): see budget block'
    if not args.no_verify:
        print('Evaluating dense vs pruned over the full video...')
        verify = eval_pair(dense, pruned, config)
        verify['reported_seen_psnr'] = float(log['final'].get('pred_seen_psnr', float('nan')))

        reloaded, _ = load_pruned_artifact(artifact_path)
        reloaded.eval()
        with torch.no_grad():
            o1, _, _ = pruned(probe)
            o2, _, _ = reloaded(probe)
        verify['roundtrip_ok'] = bool(float((o1 - o2).abs().max()) == 0.0)

        psnr_gap = verify['pruned_seen_psnr'] - verify['dense_seen_psnr']
        if mode == 'exact':
            ok = abs(psnr_gap) <= args.psnr_tol and verify['roundtrip_ok']
            verdict = (
                f'EXACT-EQUIVALENT PRUNE: PSNR gap {abs(psnr_gap):.2e} dB (tol {args.psnr_tol}), '
                f'max output diff {verify["max_abs_output_diff"]:.3e}, '
                f'decoder+head {sizes["dh_params_reduction_pct"]:.2f}% smaller'
            ) if ok else (
                f'NOT EXACT: PSNR gap {abs(psnr_gap):.4f} dB exceeds tol {args.psnr_tol} or round-trip failed'
            )
        else:
            verdict = (
                f'BUDGET PRUNE: decoder+head {sizes["dh_params_before"]:,} -> {sizes["dh_params_after"]:,} '
                f'({sizes["dh_params_reduction_pct"]:.2f}% reduction, '
                f'requested {budget_info["requested_budget_params"]:,}, reached={budget_info["budget_reached"]}), '
                f'PSNR {verify["dense_seen_psnr"]:.2f} -> {verify["pruned_seen_psnr"]:.2f} dB '
                f'({psnr_gap:+.2f}); fine-tune with --init_artifact to recover'
            )

    # ── outputs ───────────────────────────────────────────────────────────────
    lines = build_report_lines(tag, mode, config, plans, crosscheck_rows, sizes,
                               widths, flops, latency, verify, budget_info, verdict)
    reports_dir = os.path.join(run_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f'{tag}_physical_pruning_{mode}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    out = {
        'tag': tag, 'mode': mode, 'config': config, 'group_thr': group_thr,
        'crosscheck': crosscheck_rows, 'crosscheck_all_match': all_match,
        'plans': [vars(p) for p in plans], 'head_keep_in': head_keep_in,
        'sizes': sizes, 'widths': widths, 'flops': flops, 'latency': latency,
        'budget_info': budget_info, 'verify': verify, 'verdict': verdict,
        'artifact': artifact_path,
    }
    json_out = os.path.join(run_dir, 'results', f'{tag}_physical_pruning_{mode}.json')
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    csv_path = os.path.join(pruned_dir, 'physical_pruning_summary.csv')
    row = {
        'tag': tag, 'mode': mode, 'vid': config['vid'], 'method': config.get('method'),
        'seed': config.get('seed'), 'tau': config.get('tau'),
        'lambda_prox': config.get('lambda_prox'), 'ws_weight': config.get('ws_weight'),
        'source_checkpoint': ckpt_path, 'pruned_artifact': artifact_path,
        'dh_params_before': sizes['dh_params_before'], 'dh_params_after': sizes['dh_params_after'],
        'dh_params_reduction_pct': f"{sizes['dh_params_reduction_pct']:.4f}",
        'dh_bytes_before': sizes['dh_bytes_before'], 'dh_bytes_after': sizes['dh_bytes_after'],
        'flops_before': flops['before'], 'flops_after': flops['after'],
        'latency_ms_before': f"{lat_before['latency_mean_s']*1e3:.3f}",
        'latency_ms_after': f"{lat_after['latency_mean_s']*1e3:.3f}",
        'requested_budget_params': budget_info['requested_budget_params'] if budget_info else '',
        'psnr_seen_dense': f"{verify['dense_seen_psnr']:.4f}" if verify else '',
        'psnr_seen_pruned': f"{verify['pruned_seen_psnr']:.4f}" if verify else '',
        'max_abs_output_diff': f"{verify['max_abs_output_diff']:.3e}" if verify else '',
        'verdict': verdict,
    }
    exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

    print('\n'.join(lines))
    for pth in (report_path, json_out, artifact_path, csv_path):
        print(f'Saved {pth}')


if __name__ == '__main__':
    main()
