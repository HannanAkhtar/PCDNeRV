"""
train_pcd_nerv.py (v2) — Training Natively Sparse Neural Representations for
Videos with Priority-Constrained Descent, plus the baselines needed for a fair
comparison.

v2 training methods (--method; deployment-team issues #5/#12 — all methods
share the same model, data, optimizer, lr schedule, reconstruction loss,
sparsity accounting, and evaluation pipeline):

  baseline       reconstruction only — ordinary HNeRV/NeRV training.
                 (post-training structured pruning = baseline + the budget
                  mode of physical_pruning_pipeline.py)
  prox_gl        reconstruction gradient + proximal group shrinkage
                 (conventional proximal group lasso; no PCD correction)
  weighted_sum   single backward on  L_rec + ws_weight * L_group  (no prox)
  pcd            PCD: reconstruction primary, group-lasso secondary,
                 solver-combined direction + proximal shrinkage

Hyperparameter names are separated per issue #6:
  --tau          PCD directional pressure (pcd only)
  --lambda_prox  proximal shrinkage strength; per-step threshold = lambda_prox * lr
                 (pcd and prox_gl)
  --ws_weight    weighted-sum loss coefficient (weighted_sum only)

Group unit (issues #1/#2): one group = one post-PixelShuffle channel = its r^2
conv rows + their biases (shared/groups.py) — identical in the loss, the prox,
the reports, and physical pruning. `--gl_target` accepts only 'conv' (issue #8).

Run tags are method/seed-safe (issue #11), e.g.
  baseline_seed1, proxgl_conv_lprox1e-03_seed1, ws_conv_w1e-05_seed1,
  pcd_conv_tau0.05_lprox1e-03_seed1        (+ '_ft' for fine-tuning runs)

Sweeps thread the run-specific values into the update itself (issue #7 fix).
`--loss` defaults to L2 so method comparisons share one reconstruction
objective (issue #10). `--init_artifact` fine-tunes a physically pruned model
(issue #4's post-pruning recovery schedule).
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

# Anaconda on Windows ships multiple OpenMP runtimes (torch + MKL/matplotlib);
# without this the process aborts with "OMP: Error #15" at plot generation.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_all import VideoDataSet, HNeRV, HNeRVDecoder, TransformInput
from hnerv_utils import *
from shared.pcd_solver import PCDSolver
from shared.groups import (
    group_lasso_loss, apply_group_prox,
    compute_sparsity_report, build_layer_sparsity_breakdown, GROUP_UNIT,
)
from shared.nerv_targets import (
    get_channel_group_layers, describe_targets, param_scope,
    DECODER_HEAD_SCOPES, VALID_TARGETS,
)
from shared.accounting import param_accounting
from shared.physical_pruning import load_pruned_artifact
from shared.analysis import run_analysis

METHODS = ('baseline', 'prox_gl', 'weighted_sum', 'pcd')


def parse_args():
    parser = argparse.ArgumentParser(description='PCD-NeRV v2: sparse NeRV/HNeRV training with explicit baselines')
    # ── Method selection (v2) ─────────────────────────────────────────────────
    parser.add_argument('--method', type=str, default='pcd', choices=list(METHODS),
        help='training method: baseline | prox_gl | weighted_sum | pcd')
    parser.add_argument('--gl_target', type=str, default='conv', choices=list(VALID_TARGETS),
        help="group target: 'conv' only (fc grouping is not physically valid yet — see README)")
    parser.add_argument('--tau', type=float, default=0.05, help='[pcd] constraint strength tau')
    parser.add_argument('--lambda_prox', type=float, default=1e-3,
        help='[pcd, prox_gl] proximal group-shrinkage strength; per-step threshold = lambda_prox * lr')
    parser.add_argument('--ws_weight', type=float, default=1e-5,
        help='[weighted_sum] coefficient of the group-lasso term in L_rec + w * L_group')
    parser.add_argument('--beta_ema', type=float, default=0.999, help='[pcd] EMA beta for gradient-norm history')
    parser.add_argument('--solver_eps', type=float, default=1e-8, help='[pcd] numerical epsilon in the solver')
    parser.add_argument('--use_warmup', action='store_true', help='reconstruction-only warmup before the chosen method')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup duration (requires --use_warmup)')
    parser.add_argument('--group_thr', type=float, default=1e-4, help='channel-group norm below which a group counts as sparse')
    parser.add_argument('--param_thr', type=float, default=1e-8, help='|w| below which a parameter counts as sparse')
    parser.add_argument('--tau_values', type=str, default='', help='[pcd] comma-separated tau sweep')
    parser.add_argument('--lambda_values', type=str, default='', help='[pcd, prox_gl] comma-separated lambda_prox sweep')
    parser.add_argument('--ws_values', type=str, default='', help='[weighted_sum] comma-separated ws_weight sweep')
    parser.add_argument('--ckpt_every', type=int, default=100, help='save a numbered checkpoint every N epochs')
    parser.add_argument('--init_artifact', type=str, default='',
        help='fine-tune from a physically pruned artifact (pruned/<tag>_<mode>_pruned.pth); arch comes from the artifact')

    # ── Dataset parameters (HNeRV) ────────────────────────────────────────────
    parser.add_argument('--data_path', type=str, default='', help='data path for vid')
    parser.add_argument('--vid', type=str, default='k400_train0', help='video id',)
    parser.add_argument('--shuffle_data', action='store_true', help='randomly shuffle the frame idx')
    parser.add_argument('--data_split', type=str, default='1_1_1',
        help='Valid_train/total_train/all data split, e.g., 18_19_20 means for every 20 samples, the first 19 samples is full train set, and the first 18 samples is chose currently')
    parser.add_argument('--crop_list', type=str, default='640_1280', help='video crop size',)
    parser.add_argument('--resize_list', type=str, default='-1', help='video resize size',)

    # ── NeRV architecture parameters (HNeRV) ──────────────────────────────────
    parser.add_argument('--embed', type=str, default='', help='empty for HNeRV; pe_<base>_<levels> for NeRV positional encoding')
    parser.add_argument('--ks', type=str, default='0_3_3', help='kernel size for encoder and decoder')
    parser.add_argument('--enc_strds', type=int, nargs='+', default=[], help='stride list for encoder (HNeRV mode)')
    parser.add_argument('--enc_dim', type=str, default='64_16', help='enc latent dim and embedding ratio')
    parser.add_argument('--modelsize', type=float,  default=1.5, help='model parameters size: model size + embedding parameters')
    parser.add_argument('--saturate_stages', type=int, default=-1, help='saturate stages for model size computation')
    parser.add_argument('--fc_hw', type=str, default='9_16', help='out size (h,w) for mlp (NeRV-PE mode)')
    parser.add_argument('--reduce', type=float, default=1.2, help='channel reduction for next stage')
    parser.add_argument('--lower_width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list for decoder')
    parser.add_argument('--num_blks', type=str, default='1_1', help='block number for encoder and decoder')
    parser.add_argument('--conv_type', default=['convnext', 'pshuffel'], type=str, nargs='+',
        help='conv type for encoder/decoder', choices=['pshuffel', 'conv', 'convnext', 'interpolate'])
    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use',
        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])

    # ── General training setups (HNeRV) ───────────────────────────────────────
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--not_resume', action='store_true', help='not resume from latest checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=300, help='Epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='learning rate schedule')
    parser.add_argument('--loss', type=str, default='L2',
        help='reconstruction loss (default L2; keep it fixed across methods being compared)')
    parser.add_argument('--out_bias', default='tanh', type=str, help='using sigmoid/tanh/0.5 for output prediction')

    # ── Evaluation parameters (HNeRV) ─────────────────────────────────────────
    parser.add_argument('--eval_freq', type=int, default=1, help='evaluation frequency (default 1: full metrics every epoch)')
    parser.add_argument('--quant_model_bit', type=int, default=8, help='bit length for model quantization')
    parser.add_argument('--quant_embed_bit', type=int, default=6, help='bit length for embedding quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--dump_videos', action='store_true', default=False, help='concat the prediction images into video')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')

    # ── Logging / output (HNeRV) ──────────────────────────────────────────────
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--debug', action='store_true', help='debug status: shortens epochs/steps')
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained dense weights for initialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output logs, results and model checkpoints')
    parser.add_argument('--suffix', default='', help='suffix str for outf')

    args = parser.parse_args()
    torch.set_printoptions(precision=4)

    args.enc_strd_str = ','.join([str(x) for x in args.enc_strds])
    args.dec_strd_str = ','.join([str(x) for x in args.dec_strds])
    args.quant_str = f'quant_M{args.quant_model_bit}_E{args.quant_embed_bit}'

    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    # Short experiment id; hyperparameters live in the per-run JSON config.
    exp_id = f'{args.vid}/{args.data_split}_e{args.epochs}_size{args.modelsize}M_{args.loss}{args.suffix}'
    args.exp_id = exp_id
    args.outf = os.path.join(args.outf, exp_id)

    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)
    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    return args


def parse_csv_floats(value):
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    return [float(p.strip()) for p in value.split(',') if p.strip()]


def data_to_gpu(x, device):
    return x.to(device)


def make_tag(method, target, tau, lambda_prox, ws_weight, seed, finetune=False):
    """Method/seed-safe run tags (issue #11)."""
    if method == 'baseline':
        core = 'baseline'
    elif method == 'prox_gl':
        core = f'proxgl_{target}_lprox{lambda_prox:.0e}'
    elif method == 'weighted_sum':
        core = f'ws_{target}_w{ws_weight:.0e}'
    elif method == 'pcd':
        core = f'pcd_{target}_tau{tau:.4g}_lprox{lambda_prox:.0e}'
    else:
        raise ValueError(method)
    return f'{core}_seed{seed}' + ('_ft' if finetune else '')


# ── PCD gradient plumbing (from the reference shared/training.py) ─────────────

def clone_grads(model):
    return [
        p.grad.data.clone() if p.grad is not None else torch.zeros_like(p.data)
        for p in model.parameters() if p.requires_grad
    ]


def set_grads(model, grads):
    i = 0
    for p in model.parameters():
        if p.requires_grad:
            p.grad = grads[i]
            i += 1


_DIAG_RENAME = {
    'ce_efficiency':    'primary_efficiency',
    'g_ce_norm_raw':    'g_primary_norm_raw',
    'g_ce_norm_normed': 'g_primary_norm_normed',
}


def train_one_epoch(model, train_dataloader, optimizer, args, method, gl_layers,
                    solver, lambda_prox, ws_weight, epoch, is_warmup, device, plog):
    """
    One epoch of the selected method. `lambda_prox` / `ws_weight` are the
    RUN-SPECIFIC values threaded in by the sweep runner (issue #7 fix) — the
    update never reads them from args.
    """
    model.train()
    agg = defaultdict(list)
    pred_psnr_list = []
    lr = args.lr
    effective = 'baseline' if (is_warmup or method == 'baseline') else method

    for i, sample in enumerate(train_dataloader):
        img_data, norm_idx, img_idx = (data_to_gpu(sample['img'], device),
            data_to_gpu(sample['norm_idx'], device), data_to_gpu(sample['idx'], device))
        if i > 10 and args.debug:
            break

        img_data, img_gt, inpaint_mask = args.transform_func(img_data)
        cur_input = norm_idx if 'pe' in args.embed else img_data
        cur_epoch = (epoch + float(i) / len(train_dataloader)) / args.epochs
        lr = adjust_lr(optimizer, cur_epoch, args)

        if effective == 'baseline':
            optimizer.zero_grad(set_to_none=False)
            img_out, _, _ = model(cur_input)
            primary_loss = loss_fn(img_out * inpaint_mask, img_gt * inpaint_mask, args.loss)
            primary_loss.backward()
            optimizer.step()
            agg['primary_loss'].append(primary_loss.item())
            with torch.no_grad():
                agg['gl_loss'].append(float(group_lasso_loss(gl_layers)))

        elif effective == 'prox_gl':
            optimizer.zero_grad(set_to_none=False)
            img_out, _, _ = model(cur_input)
            primary_loss = loss_fn(img_out * inpaint_mask, img_gt * inpaint_mask, args.loss)
            primary_loss.backward()
            optimizer.step()
            apply_group_prox(gl_layers, lambda_prox * lr)
            agg['primary_loss'].append(primary_loss.item())
            with torch.no_grad():
                agg['gl_loss'].append(float(group_lasso_loss(gl_layers)))

        elif effective == 'weighted_sum':
            optimizer.zero_grad(set_to_none=False)
            img_out, _, _ = model(cur_input)
            primary_loss = loss_fn(img_out * inpaint_mask, img_gt * inpaint_mask, args.loss)
            lgl = group_lasso_loss(gl_layers)
            total = primary_loss + ws_weight * lgl
            total.backward()
            optimizer.step()
            agg['primary_loss'].append(primary_loss.item())
            agg['gl_loss'].append(lgl.item())
            agg['total_loss'].append(total.item())

        elif effective == 'pcd':
            optimizer.zero_grad(set_to_none=False)
            img_out, _, _ = model(cur_input)
            primary_loss = loss_fn(img_out * inpaint_mask, img_gt * inpaint_mask, args.loss)
            primary_loss.backward()
            g_primary = clone_grads(model)

            optimizer.zero_grad(set_to_none=False)
            lgl = group_lasso_loss(gl_layers)
            lgl.backward()
            g_gl = clone_grads(model)

            combined, diag = solver.step(g_primary, g_gl)
            set_grads(model, combined)
            optimizer.step()
            apply_group_prox(gl_layers, lambda_prox * lr)

            agg['primary_loss'].append(primary_loss.item())
            agg['gl_loss'].append(lgl.item())
            for k, v in diag.items():
                if isinstance(v, (int, float)):
                    agg[_DIAG_RENAME.get(k, k)].append(float(v))
        else:
            raise ValueError(effective)

        pred_psnr_list.append(psnr_fn_single(img_out.detach(), img_gt))
        if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
            pred_psnr = torch.cat(pred_psnr_list).mean()
            print_str = '[{}] Epoch[{}/{}], Step [{}/{}], lr:{:.2e} pred_PSNR: {} | primary: {:.5f} | gl: {:.2f}'.format(
                datetime.now().strftime('%Y/%m/%d %H:%M:%S'), epoch+1, args.epochs, i+1, len(train_dataloader), lr,
                RoundTensor(pred_psnr, 2), np.mean(agg['primary_loss']), np.mean(agg['gl_loss']))
            if effective == 'pcd' and len(agg.get('mu', [])):
                print_str += ' | tau={:.3f} mu={:.4f} eff={:.3f} cos={:.3f}'.format(
                    agg['tau'][-1], agg['mu'][-1], agg['primary_efficiency'][-1], agg['cosine_sim'][-1])
            plog(print_str)

    train_metrics = {k: float(np.mean(v)) for k, v in agg.items() if len(v)}
    train_metrics['train_pred_psnr'] = float(torch.cat(pred_psnr_list).mean().item())
    train_metrics['lr'] = float(lr)
    train_metrics['effective_step'] = effective
    return train_metrics


# ── Checkpoint-trend analysis (PSNR-based) ────────────────────────────────────

def _dominates(a, b):
    return (
        a.get('pred_seen_psnr', -1e9) >= b.get('pred_seen_psnr', -1e9)
        and a.get('group_sparsity', -1e9) >= b.get('group_sparsity', -1e9)
        and (
            a.get('pred_seen_psnr', -1e9) > b.get('pred_seen_psnr', -1e9)
            or a.get('group_sparsity', -1e9) > b.get('group_sparsity', -1e9)
        )
    )


def _checkpoint_progress_summary(ck_rows):
    if len(ck_rows) < 2:
        msg = 'Insufficient checkpoints for comparison (need at least two).'
        return {'status': 'insufficient', 'message': msg}, [msg]

    rows = sorted(ck_rows, key=lambda r: r['epoch'])
    lines = []
    lines.append('Checkpoint comparison (Pareto + stationarity proxies)')
    lines.append('-' * 92)
    lines.append(f"{'Epoch':>5} | {'PSNR':>7} | {'GroupS%':>8} | {'DecHeadS%':>9} | {'mu':>8} | {'Prim_eff':>8}")
    lines.append('-' * 92)
    for r in rows:
        lines.append(
            f"{r['epoch']:>5} | {r.get('pred_seen_psnr', 0.0):>7.2f} | {r.get('group_sparsity', 0.0):>8.2f} | "
            f"{r.get('decoder_head_param_sparsity', 0.0):>9.2f} | {r.get('mu', float('nan')):>8.4f} | "
            f"{r.get('primary_efficiency', float('nan')):>8.3f}"
        )

    d = []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        d.append({
            'from': prev['epoch'], 'to': cur['epoch'],
            'delta_psnr': cur.get('pred_seen_psnr', 0.0) - prev.get('pred_seen_psnr', 0.0),
            'delta_group': cur.get('group_sparsity', 0.0) - prev.get('group_sparsity', 0.0),
            'delta_dh': cur.get('decoder_head_param_sparsity', 0.0) - prev.get('decoder_head_param_sparsity', 0.0),
            'delta_mu': cur.get('mu', 0.0) - prev.get('mu', 0.0),
            'dominates': _dominates(cur, prev),
        })

    lines.append('-' * 92)
    for dd in d:
        lines.append(
            f"{dd['from']} -> {dd['to']}: dPSNR={dd['delta_psnr']:+.3f}, dGroup={dd['delta_group']:+.3f}, "
            f"dDecHead={dd['delta_dh']:+.3f}, dMu={dd['delta_mu']:+.4f}, dominates_prev={'Y' if dd['dominates'] else 'N'}"
        )

    psnr_eps, spar_eps, mu_eps = 0.05, 0.25, 0.01
    last = d[-1]
    last_plateau = (abs(last['delta_psnr']) < psnr_eps and abs(last['delta_group']) < spar_eps
                    and abs(last['delta_dh']) < spar_eps and abs(last['delta_mu']) < mu_eps)
    if not last_plateau or last['dominates']:
        verdict = 'continues_improving_until_last_checkpoint'
        verdict_msg = 'Training is still moving in a Pareto-improving direction through the final checkpoint.'
    else:
        verdict = 'converged_before_last_checkpoint'
        verdict_msg = 'Trajectory appears to have converged before the final checkpoint (late-stage plateau).'
    lines.append('-' * 92)
    lines.append(f'Verdict: {verdict_msg}')
    return {'status': 'ok', 'verdict': verdict, 'verdict_message': verdict_msg,
            'checkpoints': rows, 'transitions': d}, lines


# ── Sparse-structure report ───────────────────────────────────────────────────

def write_sparsity_breakdown(log, layer_rows, path, tag):
    lines = []
    lines.append('=' * 160)
    lines.append(f'Sparse Structure Report: {tag}')
    lines.append(f'Group unit: {GROUP_UNIT} — group sparsity is a DIAGNOSTIC; '
                 f'physical pruning reports are the source of truth for compression.')
    lines.append('=' * 160)
    f = log['final']
    lines.append(
        'Global summary | '
        f"targeted channel-groups: {f.get('group_sparse', 0)}/{f.get('group_total', 0)} "
        f"({f.get('group_sparsity', 0.0):.2f}%) | "
        f"decoder+head params: {f.get('decoder_head_param_sparse', 0)}/{f.get('decoder_head_param_total', 0)} "
        f"({f.get('decoder_head_param_sparsity', 0.0):.2f}%) | "
        f"trainable params: {f.get('trainable_param_sparse', 0)}/{f.get('trainable_param_total', 0)} "
        f"({f.get('model_param_sparsity', 0.0):.2f}%)"
    )
    lines.append(
        'Sparse param composition | '
        f"targeted: {f.get('targeted_param_sparse', 0)}/{f.get('trainable_param_sparse', 0)} "
        f"({f.get('sparse_from_targeted_pct', 0.0):.2f}%) | "
        f"non-targeted: {f.get('non_target_param_sparse', 0)}/{f.get('trainable_param_sparse', 0)} "
        f"({f.get('sparse_from_non_target_pct', 0.0):.2f}%)"
    )
    lines.append('-' * 160)
    lines.append(
        f"{'Layer':<48} {'Scope':<12} {'Targeted':<9} {'Grouping':<26} {'ParamSparse/Total':<22} {'ParamSpar%':>10} "
        f"{'GroupSparse/Total':<18} {'GroupSpar%':>10}"
    )
    lines.append('-' * 160)
    for row in layer_rows:
        gp_frac = f"{row.get('group_sparse', 0)}/{row.get('group_total', 0)}" if row.get('group_total') else '-'
        pp_frac = f"{row.get('param_sparse', 0)}/{row.get('param_total', 0)}"
        lines.append(
            f"{row.get('layer', ''):<48} {row.get('scope', ''):<12} {str(row.get('is_gl_targeted', False)):<9} "
            f"{row.get('grouping', ''):<26} {pp_frac:<22} {row.get('param_sparsity', 0.0):>10.2f} "
            f"{gp_frac:<18} {row.get('group_sparsity', 0.0):>10.2f}"
        )
    lines.append('=' * 160)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


# ── HNeRV evaluation (from upstream; distributed removed) ─────────────────────

@torch.no_grad()
def evaluate(model, full_dataloader, args, plog, dump_vis=False, huffman_coding=False):
    import imageio

    img_embed_list = []
    model_list, quant_ckt = quant_model(model, args)
    metric_list = [[] for _ in range(len(args.metric_names))]
    for model_ind, cur_model in enumerate(model_list):
        time_list = []
        cur_model.eval()
        device = next(cur_model.parameters()).device
        if dump_vis:
            visual_dir = f'{args.outf}/visualize_model' + ('_quant' if model_ind else '_orig')
            print(f'Saving predictions to {visual_dir}...')
            if not os.path.isdir(visual_dir):
                os.makedirs(visual_dir)

        for i, sample in enumerate(full_dataloader):
            img_data, norm_idx, img_idx = (data_to_gpu(sample['img'], device),
                data_to_gpu(sample['norm_idx'], device), data_to_gpu(sample['idx'], device))
            if i > 10 and args.debug:
                break
            img_data, img_gt, inpaint_mask = args.transform_func(img_data)
            cur_input = norm_idx if 'pe' in args.embed else img_data
            img_out, embed_list, dec_time = cur_model(cur_input, dequant_vid_embed[i] if model_ind else None)
            if model_ind == 0:
                img_embed_list.append(embed_list[0])

            time_list.append(dec_time)
            if args.eval_fps:
                time_list.pop()
                for _ in range(100):
                    img_out, embed_list, dec_time = cur_model(cur_input, embed_list[0])
                    time_list.append(dec_time)

            pred_psnr, pred_ssim = psnr_fn_batch([img_out], img_gt), msssim_fn_batch([img_out], img_gt)
            for metric_idx, cur_v in  enumerate([pred_psnr, pred_ssim]):
                for batch_i, cur_img_idx in enumerate(img_idx):
                    metric_idx_start = 2 if cur_img_idx in args.val_ind_list else 0
                    metric_list[metric_idx_start+metric_idx+4*model_ind].append(cur_v[:,batch_i])

            if dump_vis:
                for batch_ind, cur_img_idx in enumerate(img_idx):
                    full_ind = i * args.batchSize + batch_ind
                    dump_img_list = [img_data[batch_ind], img_out[batch_ind]]
                    temp_psnr_list = ','.join([str(round(x[batch_ind].item(), 2)) for x in pred_psnr])
                    concat_img = torch.cat(dump_img_list, dim=2)
                    save_image(concat_img, f'{visual_dir}/pred_{full_ind:04d}_{temp_psnr_list}.png')

            if i % args.print_freq == 0 or i == len(full_dataloader) - 1:
                avg_time = sum(time_list) / len(time_list)
                fps = args.batchSize / avg_time
                print_str = '[{}] Eval at Step [{}/{}] , FPS {}, '.format(
                    datetime.now().strftime('%Y/%m/%d %H:%M:%S'), i+1, len(full_dataloader), round(fps, 1))
                metric_name = ('quant' if model_ind else 'pred') + '_seen_psnr'
                for v_name, v_list in zip(args.metric_names, metric_list):
                    if metric_name in v_name:
                        cur_value = torch.stack(v_list, dim=-1).mean(-1) if len(v_list) else torch.zeros(1)
                        print_str += f'{v_name}: {RoundTensor(cur_value, 2)} | '
                plog(print_str)

        if model_ind == 0:
            vid_embed = torch.cat(img_embed_list, 0)
            quant_embed, dequant_emved = quant_tensor(vid_embed, args.quant_embed_bit)
            dequant_vid_embed = dequant_emved.split(args.batchSize, dim=0)

        results_list = [torch.stack(v_list, dim=1).mean(1).cpu() if len(v_list) else torch.zeros(1) for v_list in metric_list]
        args.fps = fps
        h,w = img_data.shape[-2:]
        cur_model.train()

        if dump_vis and args.dump_videos:
            gif_file = os.path.join(args.outf, 'gt_pred' + ('_quant.gif' if model_ind else '.gif'))
            with imageio.get_writer(gif_file, mode='I') as writer:
                for filename in sorted(os.listdir(visual_dir)):
                    image = imageio.v2.imread(os.path.join(visual_dir, filename))
                    writer.append_data(image)
            if not args.dump_images:
                shutil.rmtree(visual_dir)

    if quant_ckt != None:
        try:
            from dahuffman import HuffmanCodec
            quant_vid = {'embed': quant_embed, 'model': quant_ckt}
            torch.save(quant_vid, f'{args.outf}/{args.run_tag}_quant_vid.pth')
            torch.jit.save(torch.jit.trace(HNeRVDecoder(model), (vid_embed[:2])), f'{args.outf}/{args.run_tag}_img_decoder.pth')
            if huffman_coding:
                quant_v_list = quant_embed['quant'].flatten().tolist()
                tmin_scale_len = quant_embed['min'].nelement() + quant_embed['scale'].nelement()
                for k, layer_wt in quant_ckt.items():
                    quant_v_list.extend(layer_wt['quant'].flatten().tolist())
                    tmin_scale_len += layer_wt['min'].nelement() + layer_wt['scale'].nelement()
                unique, counts = np.unique(quant_v_list, return_counts=True)
                num_freq = dict(zip(unique, counts))
                codec = HuffmanCodec.from_data(quant_v_list)
                sym_bit_dict = {}
                for k, v in codec.get_code_table().items():
                    sym_bit_dict[k] = v[0]
                total_bits = 0
                for num, freq in num_freq.items():
                    total_bits += freq * sym_bit_dict[num]
                args.bits_per_param = total_bits / len(quant_v_list)
                total_bits += tmin_scale_len * 16               #(16bits for float16)
                args.full_bits_per_param = total_bits / len(quant_v_list)
                args.total_bpp = total_bits / args.final_size / args.full_data_length
                plog(f'After quantization and encoding: \n bits per parameter: {round(args.full_bits_per_param, 2)}, bits per pixel: {round(args.total_bpp, 4)}')
        except Exception as e:
            plog(f'WARNING: quantized-decoder export / huffman coding failed: {type(e).__name__}: {e}')

    return results_list, (h,w)


def quant_model(model, args):
    model_list = [deepcopy(model)]
    if args.quant_model_bit == -1:
        return model_list, None
    else:
        cur_model = deepcopy(model)
        quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]
        encoder_k_list = []
        for k,v in cur_ckt.items():
            if 'encoder' in k:
                encoder_k_list.append(k)
            else:
                quant_v, new_v = quant_tensor(v, args.quant_model_bit)
                quant_ckt[k] = quant_v
                cur_ckt[k] = new_v
        for encoder_k in encoder_k_list:
            del quant_ckt[encoder_k]
        cur_model.load_state_dict(cur_ckt)
        model_list.append(cur_model)
        return model_list, quant_ckt


# ── CSV dump ──────────────────────────────────────────────────────────────────

def Dump2CSV(args, best_results_list, results_list, psnr_list, final_row, acct, filename='results.csv'):
    result_dict = {'Vid':args.vid, 'Method':args.method_run, 'GLTarget':args.gl_target,
        'Tau':args.tau_run, 'LambdaProx':args.lambda_prox_run, 'WsWeight':args.ws_weight_run,
        'Seed':args.manualSeed, 'CurEpoch':args.cur_epoch, 'Time':args.train_time,
        'FPS':args.fps, 'Split':args.data_split, 'Embed':args.embed, 'Crop': args.crop_list,
        'Resize':args.resize_list, 'Lr_type':args.lr_type, 'LR (E-3)': args.lr*1e3, 'Batch':args.batchSize,
        'EncoderParams':acct['encoder_params'], 'DecoderHeadParams':acct['decoder_head_params'],
        'EmbeddingStorage':acct['embedding_storage'], 'TotalStored':acct['total_stored_representation'],
        'ModelSize': args.modelsize, 'Epoch':args.epochs, 'Loss':args.loss, 'Act':args.act, 'Norm':args.norm,
        'FC':args.fc_hw, 'Reduce':args.reduce, 'ENC_type':args.conv_type[0], 'ENC_strds':args.enc_strd_str, 'KS':args.ks,
        'enc_dim':args.enc_dim, 'DEC':args.conv_type[1], 'DEC_strds':args.dec_strd_str, 'lower_width':args.lower_width,
        'Quant':args.quant_str, 'bits/param':args.bits_per_param, 'bits/param w/ overhead':args.full_bits_per_param,
        'bits/pixel':args.total_bpp, f'PSNR_list_{args.eval_freq}':','.join([RoundTensor(v, 2) for v in psnr_list]),
        'ChannelGroupSparsity%': round(final_row.get('group_sparsity', 0.0), 4),
        'ChannelGroupSparse/Total': final_row.get('group_sparse_fraction', '0/0'),
        'DecoderHeadParamSparsity%': round(final_row.get('decoder_head_param_sparsity', 0.0), 4),
        'ModelParamSparsity%': round(final_row.get('model_param_sparsity', 0.0), 4),}
    result_dict.update({f'best_{k}':RoundTensor(v, 4 if 'ssim' in k else 2) for k,v in zip(args.metric_names, best_results_list)})
    result_dict.update({f'{k}':RoundTensor(v, 4 if 'ssim' in k else 2) for k,v in zip(args.metric_names, results_list) if 'pred' in k})
    csv_path = os.path.join(args.outf, filename)
    print(f'results dumped to {csv_path}')
    pd.DataFrame(result_dict,index=[0]).to_csv(csv_path)


# ── One full run (fixed method/target/hyperparameters) ────────────────────────

def run_single(args, full_dataloader, train_dataloader, method, tau, lambda_prox, ws_weight):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    finetune = bool(args.init_artifact)
    tag = make_tag(method, args.gl_target, tau, lambda_prox, ws_weight, args.manualSeed, finetune)
    args.run_tag = tag
    args.method_run = method
    args.tau_run = tau if method == 'pcd' else None
    args.lambda_prox_run = lambda_prox if method in ('pcd', 'prox_gl') else None
    args.ws_weight_run = ws_weight if method == 'weighted_sum' else None
    wu = args.warmup_epochs if (args.use_warmup and method != 'baseline') else 0

    results_dir = os.path.join(args.outf, 'results')
    reports_dir = os.path.join(args.outf, 'reports')
    ckpt_dir    = os.path.join(args.outf, 'checkpoints')
    logs_dir    = os.path.join(args.outf, 'logs')
    for d in (results_dir, reports_dir, ckpt_dir, logs_dir):
        os.makedirs(d, exist_ok=True)
    log_txt_path = os.path.join(logs_dir, f'{tag}_train.txt')

    def plog(msg):
        print(msg, flush=True)
        with open(log_txt_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    plog(f"\n{'=' * 96}")
    plog(f'  PCD-NeRV v2 run: {tag}')
    plog(f'  method={method}  target={args.gl_target}  tau={args.tau_run}  '
         f'lambda_prox={args.lambda_prox_run}  ws_weight={args.ws_weight_run}  '
         f'seed={args.manualSeed}  warmup={wu}  epochs={args.epochs}  device={device}')
    if finetune:
        plog(f'  fine-tuning from pruned artifact: {args.init_artifact}')
    plog(f"{'=' * 96}")

    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    if finetune:
        model, artifact_payload = load_pruned_artifact(args.init_artifact)
    else:
        model = HNeRV(args)
        artifact_payload = None

    acct = param_accounting(model, embedding_storage=args.embed_param)
    plog('Param accounting | ' + ' | '.join(f'{k}={v:,}' for k, v in acct.items()))

    gl_layers = get_channel_group_layers(model, args.gl_target)
    plog(describe_targets(model, args.gl_target))

    if torch.cuda.is_available():
        model = model.cuda()
        gl_layers = get_channel_group_layers(model, args.gl_target)  # re-bind module refs on device

    def sparsity_report_fn():
        return compute_sparsity_report(
            model, gl_layers, param_scope, DECODER_HEAD_SCOPES,
            group_thr=args.group_thr, param_thr=args.param_thr)

    optimizer = optim.Adam(model.parameters(), weight_decay=0.)
    solver = PCDSolver(tau=tau, beta=args.beta_ema, eps=args.solver_eps) if method == 'pcd' else None

    # initialization from pretrained dense weights (HNeRV --weight)
    if args.weight != 'None' and not finetune:
        plog(f"=> loading checkpoint '{args.weight}'")
        checkpoint = torch.load(args.weight, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        new_ckt = {k.replace('blocks.0.', ''): v for k, v in orig_ckt.items()}
        new_ckt = {k.replace('module.', ''): v for k, v in new_ckt.items()}
        model.load_state_dict(new_ckt, strict=False)
        plog(f"=> loaded checkpoint '{args.weight}' (epoch {checkpoint.get('epoch', '?')})")

    # auto-resume from the tag-scoped latest checkpoint
    start_epoch = 0
    out_json_path = os.path.join(results_dir, f'{tag}.json')
    preserved_epochs = []
    latest_path = os.path.join(args.outf, f'{tag}_model_latest.pth')
    if not args.not_resume and os.path.isfile(latest_path):
        checkpoint = torch.load(latest_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if solver is not None and 'solver_state' in checkpoint:
            solver.v = list(checkpoint['solver_state']['v'])
            solver.t = int(checkpoint['solver_state']['t'])
            solver.tau = float(checkpoint['solver_state']['tau'])
        start_epoch = checkpoint['epoch']
        plog(f"=> Auto resume loaded checkpoint '{latest_path}' (epoch {start_epoch})")
        if os.path.isfile(out_json_path):
            try:
                with open(out_json_path, 'r', encoding='utf-8') as f:
                    old_log = json.load(f)
                preserved_epochs = [r for r in old_log.get('epochs', [])
                                    if isinstance(r, dict) and int(r.get('epoch', 0)) <= start_epoch]
            except Exception:
                preserved_epochs = []

    if start_epoch >= args.epochs:
        plog(f'  Run {tag} already completed ({start_epoch} epochs); skipping training.')

    config = {
        'method': method,
        'format_version': 2,
        'arch': 'hnerv' if len(args.enc_strds) else 'nerv-pe',
        'sparsity_target': args.gl_target,
        'group_unit': GROUP_UNIT,
        'vid': args.vid, 'data_path': args.data_path, 'data_split': args.data_split,
        'shuffle_data': bool(args.shuffle_data),
        'crop_list': args.crop_list, 'resize_list': args.resize_list,
        'embed': args.embed, 'enc_strds': list(args.enc_strds), 'enc_dim': args.enc_dim,
        'dec_strds': list(args.dec_strds), 'fc_hw': args.fc_hw, 'fc_dim': int(args.fc_dim),
        'ks': args.ks, 'reduce': args.reduce, 'lower_width': args.lower_width,
        'num_blks': args.num_blks, 'conv_type': list(args.conv_type), 'norm': args.norm,
        'act': args.act, 'out_bias': args.out_bias, 'modelsize': args.modelsize,
        'loss': args.loss, 'lr': args.lr, 'lr_type': args.lr_type,
        'batchSize': args.batchSize, 'epochs': args.epochs, 'warmup': wu,
        'seed': args.manualSeed,
        'tau': args.tau_run, 'lambda_prox': args.lambda_prox_run, 'ws_weight': args.ws_weight_run,
        'beta_ema': args.beta_ema, 'solver_eps': args.solver_eps,
        'group_thr': args.group_thr, 'param_thr': args.param_thr,
        'quant_model_bit': args.quant_model_bit, 'quant_embed_bit': args.quant_embed_bit,
        'eval_freq': args.eval_freq, 'ckpt_every': args.ckpt_every,
        'init_artifact': args.init_artifact or None,
        'accounting': {k: int(v) for k, v in acct.items()},
    }
    log = {'config': config, 'epochs': preserved_epochs}

    try:
        writer = SummaryWriter(os.path.join(args.outf, 'tensorboard', tag))
    except Exception as e:
        plog(f'WARNING: tensorboard unavailable ({type(e).__name__}: {e}); continuing without it.')
        class _NullWriter:
            def add_scalar(self, *a, **k): pass
            def close(self): pass
        writer = _NullWriter()

    best_metric_list = [torch.tensor(0) for _ in range(len(args.metric_names))]
    psnr_list = []
    results_list = [torch.zeros(1) for _ in range(len(args.metric_names))]
    saved_ckpt_epochs = []
    start = datetime.now()

    for epoch in range(start_epoch, args.epochs):
        is_warmup = (epoch + 1) <= wu
        epoch_start_time = datetime.now()

        trm = train_one_epoch(model, train_dataloader, optimizer, args, method, gl_layers,
                              solver, lambda_prox, ws_weight, epoch, is_warmup, device, plog)

        sparsity = sparsity_report_fn()

        eval_metrics = {}
        if (epoch + 1) % args.eval_freq == 0 or (args.epochs - epoch) in [1, 3, 5]:
            is_final = (epoch == args.epochs - 1)
            results_list, hw = evaluate(model, full_dataloader, args, plog,
                dump_vis=(args.dump_images or args.dump_videos) if is_final else False,
                huffman_coding=is_final)
            for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
                best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
                if metric_name == 'pred_seen_psnr':
                    psnr_list.append(metric_value.max())
                best_metric_list[i] = best_metric_value
                eval_metrics[metric_name] = float(metric_value.max().item())
                writer.add_scalar(f'Val/{metric_name}', eval_metrics[metric_name], epoch+1)

        ep_time = (datetime.now() - epoch_start_time).total_seconds()
        head = f'Epoch[{epoch+1}/{args.epochs}]' + (' [WARMUP]' if is_warmup else '')
        plog(f'{head} TRAIN | method {method} | primary({args.loss}) {trm.get("primary_loss", 0):.5f} | '
             f'gl {trm.get("gl_loss", 0):.2f} | train_PSNR {trm.get("train_pred_psnr", 0):.2f} | '
             f'lr {trm.get("lr", 0):.2e} | {ep_time:.1f}s')
        if method == 'pcd' and 'mu' in trm:
            plog(f'{" " * len(head)} PCD   | tau={trm.get("tau", tau):.3f} | mu={trm.get("mu", 0):.4f} | '
                 f'eff={trm.get("primary_efficiency", 1.0):.3f} | cos={trm.get("cosine_sim", 0):.3f} | '
                 f'conflict={100 * trm.get("conflict", 0):.0f}%')
        if method == 'weighted_sum' and 'total_loss' in trm:
            plog(f'{" " * len(head)} WS    | total {trm["total_loss"]:.5f} = primary + {ws_weight:g} * gl')
        plog(f'{" " * len(head)} SPARSE| ch-groups {sparsity["group_sparse_fraction"]} '
             f'({sparsity["group_sparsity"]:.2f}%) | '
             f'dec+head {sparsity["decoder_head_param_sparse"]}/{sparsity["decoder_head_param_total"]} '
             f'({sparsity["decoder_head_param_sparsity"]:.2f}%) | '
             f'all {sparsity["trainable_param_sparse"]}/{sparsity["trainable_param_total"]} '
             f'({sparsity["model_param_sparsity"]:.2f}%)')
        if eval_metrics:
            plog(f'{" " * len(head)} EVAL  | seen PSNR {eval_metrics.get("pred_seen_psnr", 0):.2f} '
                 f'SSIM {eval_metrics.get("pred_seen_ssim", 0):.4f} | '
                 f'unseen PSNR {eval_metrics.get("pred_unseen_psnr", 0):.2f} '
                 f'SSIM {eval_metrics.get("pred_unseen_ssim", 0):.4f} | '
                 f'quant seen PSNR {eval_metrics.get("quant_seen_psnr", 0):.2f} '
                 f'SSIM {eval_metrics.get("quant_seen_ssim", 0):.4f}')

        writer.add_scalar('Train/pred_PSNR', trm.get('train_pred_psnr', 0), epoch+1)
        writer.add_scalar('Train/primary_loss', trm.get('primary_loss', 0), epoch+1)
        writer.add_scalar('Train/gl_loss', trm.get('gl_loss', 0), epoch+1)
        writer.add_scalar('Train/lr', trm.get('lr', 0), epoch+1)
        for k in ('mu', 'primary_efficiency', 'cosine_sim', 'conflict', 'tau', 'total_loss'):
            if k in trm:
                writer.add_scalar(f'PCD/{k}', trm[k], epoch+1)
        writer.add_scalar('Sparsity/channel_group_sparsity', sparsity['group_sparsity'], epoch+1)
        writer.add_scalar('Sparsity/decoder_head_param_sparsity', sparsity['decoder_head_param_sparsity'], epoch+1)
        writer.add_scalar('Sparsity/model_param_sparsity', sparsity['model_param_sparsity'], epoch+1)

        row = {'epoch': epoch + 1, 'warmup': bool(is_warmup), 'epoch_seconds': ep_time,
               **trm, **sparsity, **eval_metrics}
        log['epochs'] = [r for r in log['epochs'] if int(r.get('epoch', 0)) != epoch + 1]
        log['epochs'].append(row)
        log['final'] = log['epochs'][-1]
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)

        save_checkpoint = {
            'epoch': epoch + 1,
            'tag': tag,
            'config': config,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if solver is not None:
            save_checkpoint['solver_state'] = {'v': list(solver.v), 't': solver.t, 'tau': solver.tau}
        torch.save(save_checkpoint, latest_path)
        if (epoch + 1) % args.ckpt_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(ckpt_dir, f'{tag}_ep{epoch+1:03d}.pth')
            torch.save(save_checkpoint, ckpt_path)
            saved_ckpt_epochs.append(epoch + 1)
            plog(f'  Saved checkpoint {ckpt_path}')
            if best_metric_list[0] == results_list[0].max():
                torch.save(save_checkpoint, os.path.join(args.outf, f'{tag}_model_best.pth'))

    train_time = str(datetime.now() - start)
    plog(f'Training complete in: {train_time}')

    if not log['epochs']:
        results_list, hw = evaluate(model, full_dataloader, args, plog)
        sparsity = sparsity_report_fn()
        row = {'epoch': start_epoch, 'warmup': False, **sparsity,
               **{name: float(v.max().item()) for name, v in zip(args.metric_names, results_list)}}
        log['epochs'].append(row)

    log['final'] = log['epochs'][-1]

    layer_rows = build_layer_sparsity_breakdown(
        model, gl_layers, param_scope,
        group_thr=args.group_thr, param_thr=args.param_thr)
    log['final_layer_breakdown'] = layer_rows

    breakdown_path = os.path.join(reports_dir, f'{tag}_sparsity_breakdown.txt')
    write_sparsity_breakdown(log, layer_rows, breakdown_path, tag)
    plog(f'  Saved {breakdown_path}')

    ck_epochs = sorted(set(saved_ckpt_epochs) | ({args.epochs} if log['epochs'] else set()))
    ck_rows = []
    for row in log['epochs']:
        if row.get('epoch') in ck_epochs and 'pred_seen_psnr' in row:
            ck_rows.append({
                'epoch': row['epoch'],
                'pred_seen_psnr': row.get('pred_seen_psnr', 0.0),
                'group_sparsity': row.get('group_sparsity', 0.0),
                'decoder_head_param_sparsity': row.get('decoder_head_param_sparsity', 0.0),
                'mu': row.get('mu', 0.0),
                'primary_efficiency': row.get('primary_efficiency', 0.0),
            })
    ck_summary, ck_lines = _checkpoint_progress_summary(ck_rows)
    log['checkpoint_comparison'] = ck_summary
    ck_report_path = os.path.join(reports_dir, f'{tag}_checkpoint_progress.txt')
    with open(ck_report_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(ck_lines) + '\n')
    plog(f'  Saved {ck_report_path}')

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)
    plog(f'  Saved {out_json_path}')

    args.cur_epoch = args.epochs
    args.train_time = train_time
    Dump2CSV(args, best_metric_list, results_list, psnr_list, log['final'], acct, f'{tag}_epoch{args.epochs}.csv')

    writer.close()
    return tag, log


def train(args):
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    args.metric_names = ['pred_seen_psnr', 'pred_seen_ssim', 'pred_unseen_psnr', 'pred_unseen_ssim',
        'quant_seen_psnr', 'quant_seen_ssim', 'quant_unseen_psnr', 'quant_unseen_ssim']
    args.fps, args.bits_per_param, args.full_bits_per_param, args.total_bpp = 0., -1., -1., -1.

    # fine-tuning: architecture comes from the pruned artifact's config
    if args.init_artifact:
        payload = torch.load(args.init_artifact, map_location='cpu')
        art_cfg = payload['config']
        for k_args, k_cfg in [('embed', 'embed'), ('enc_strds', 'enc_strds'), ('enc_dim', 'enc_dim'),
                              ('dec_strds', 'dec_strds'), ('fc_hw', 'fc_hw'), ('ks', 'ks'),
                              ('reduce', 'reduce'), ('lower_width', 'lower_width'),
                              ('num_blks', 'num_blks'), ('conv_type', 'conv_type'), ('norm', 'norm'),
                              ('act', 'act'), ('out_bias', 'out_bias'), ('modelsize', 'modelsize')]:
            setattr(args, k_args, art_cfg[k_cfg])
        args.fc_dim = int(art_cfg['fc_dim'])
        args.enc_strd_str = ','.join([str(x) for x in args.enc_strds])
        args.dec_strd_str = ','.join([str(x) for x in args.dec_strds])
        print(f'Architecture taken from pruned artifact: {args.init_artifact}')

    # ── dataloaders (HNeRV) ───────────────────────────────────────────────────
    full_dataset = VideoDataSet(args)
    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batchSize, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0)

    train_dataset = Subset(full_dataset, train_ind_list)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True,
         num_workers=args.workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    # ── embedding storage + model width from size budget (HNeRV) ─────────────
    if 'pe' in args.embed or 'le' in args.embed:
        embed_param = 0
        embed_dim = int(args.embed.split('_')[-1]) * 2
        fc_param = np.prod([int(x) for x in args.fc_hw.split('_')])
    else:
        total_enc_strds = np.prod(args.enc_strds)
        embed_hw = args.final_size / total_enc_strds**2
        enc_dim1, embed_ratio = [float(x) for x in args.enc_dim.split('_')]
        embed_dim = int(embed_ratio * args.modelsize * 1e6 / args.full_data_length / embed_hw) if embed_ratio < 1 else int(embed_ratio)
        embed_param = float(embed_dim) / total_enc_strds**2 * args.final_size * args.full_data_length
        args.enc_dim = f'{int(enc_dim1)}_{embed_dim}'
        fc_param = (np.prod(args.enc_strds) // np.prod(args.dec_strds))**2 * 9

    args.embed_param = int(embed_param)

    if not args.init_artifact:
        decoder_size = args.modelsize * 1e6 - embed_param
        ch_reduce = 1. / args.reduce
        dec_ks1, dec_ks2 = [int(x) for x in args.ks.split('_')[1:]]
        fix_ch_stages = len(args.dec_strds) if args.saturate_stages == -1 else args.saturate_stages
        a =  ch_reduce * sum([ch_reduce**(2*i) * s**2 * min((2*i + dec_ks1), dec_ks2)**2 for i,s in enumerate(args.dec_strds[:fix_ch_stages])])
        b =  embed_dim * fc_param
        c =  args.lower_width **2 * sum([s**2 * min(2*(fix_ch_stages + i) + dec_ks1, dec_ks2)  **2 for i, s in enumerate(args.dec_strds[fix_ch_stages:])])
        args.fc_dim = int(np.roots([a,b,c - decoder_size]).max())

    args.transform_func = TransformInput(args)

    # ── run grid for the chosen method (issue #7: values threaded per run) ────
    method = args.method
    if method == 'pcd':
        tau_vals = parse_csv_floats(args.tau_values) or [args.tau]
        lam_vals = parse_csv_floats(args.lambda_values) or [args.lambda_prox]
        grid = [(t, l, args.ws_weight) for t in tau_vals for l in lam_vals]
    elif method == 'prox_gl':
        lam_vals = parse_csv_floats(args.lambda_values) or [args.lambda_prox]
        grid = [(args.tau, l, args.ws_weight) for l in lam_vals]
    elif method == 'weighted_sum':
        ws_vals = parse_csv_floats(args.ws_values) or [args.ws_weight]
        grid = [(args.tau, args.lambda_prox, w) for w in ws_vals]
    else:  # baseline
        grid = [(args.tau, args.lambda_prox, args.ws_weight)]

    for tau, lam, ws in grid:
        run_single(args, full_dataloader, train_dataloader, method, tau, lam, ws)

    run_analysis(args.outf)


def main():
    try:
        sys.stdout.reconfigure(errors='replace')
        sys.stderr.reconfigure(errors='replace')
    except Exception:
        pass
    args = parse_args()
    print(args, flush=True)
    train(args)


if __name__ == '__main__':
    main()
