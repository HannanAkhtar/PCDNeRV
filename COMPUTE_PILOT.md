# HNeRV compute-target pilot

This pilot is additive. The legacy experiment remains fixed-epoch training,
mostly post-training physical pruning, and optional decoder-parameter budgets.
The new experiment gives every method the same counted active-training
wall-clock budget `W` and, for M1–M4, the same final decoder-MAC reduction
target `kappa`, with physical channel removal during training.

`kappa=0.50` means a 50% reduction in decoder-plus-RGB-head MACs per frame. It
does **not** mean a 50% parameter reduction. If the start model costs
`start_MACs`, the target is `start_MACs * (1-kappa)`. Results report actual
GMACs/frame, `GFLOPs=2*GMACs`, kMACs/pixel, achieved kappa, discrete-target
overshoot/undershoot, and reachability.

## Compared methods

- `d_start`: the large dense HNeRV, trained without pruning for all of W.
- `d_small_compute`: an ordinary dense HNeRV selected by building candidates
  and measuring their decoder MACs, then trained from scratch for W.
- `d_small_params`: optional ordinary dense control matched by measured
  decoder-plus-head parameters.
- `d_replay`: the exact physical widths of a saved PCD artifact, randomly
  initialized and trained from scratch. No PCD weights are copied.
- `m1_posthoc`: large and dense through 0.9W, hard compute pruning, then
  reconstruction-only fine-tuning.
- `m2_gradual`: physical pruning at epoch boundaries from 0.1W to 0.8W using
  `kappa(u)=kappa_final*(1-(1-u)^3)`, where `u=(t/W-0.1)/0.7` clipped to [0,1].
- `m3_group_lasso`: reconstruction plus compute-weighted structured loss from
  0.1W to 0.9W, threshold removal at epoch boundaries, hard completion at
  0.9W, then reconstruction-only fine-tuning.
- `m4_pcd`: reconstruction is primary and compute-weighted structured loss is
  secondary for the unchanged PCD solver. Tau controls the direction. The
  main mode does not use `lambda_prox`; `--m4_use_prox --m4_lambda_prox ...`
  exists only as an explicit ablation.

The structured objective is `R=sum_g c_g*||theta_g||_2`. Start-model costs are
normalized to group-weighted mean 1 and frozen for M3/M4 training. Hard
pruning recomputes current costs after every removal and ranks groups by
`||theta_g||_2/c_g`. Every method uses the repository's existing
PixelShuffle-aware, bias-inclusive groups and the same minimum-width rule.

## Wall-clock definition

`budget_fraction = counted_training_seconds / budget_seconds` using
`time.perf_counter()`. On CUDA, every counted interval is bracketed by
`torch.cuda.synchronize()` so W measures completed device work rather than
queued launches. Forward/backward/update, structured objectives, pruning
selection, physical surgery, and Adam-state surgery count inside W. Quality
evaluation, checkpoint and report I/O, plotting, and final CUDA-event FPS do
not. The LR schedule keeps the legacy HNeRV shape but uses budget fraction.
The run stops after the completed optimizer step (and its immediately due
pruning work) that exhausts W.

Progressive checkpoints are versioned and contain the current architecture,
strict model state, Adam state, elapsed W, complete training history, original
selection metadata, RNG state, frozen compute weights, and PCD solver state.
Resume reconstructs the smaller architecture before loading tensors and
rejects changes to method, kappa, seed, base architecture, or W.

## No-training preflight

Run the feasibility profiler before committing GPU time. It builds the full
requested HNeRV, plans hard pruning at both targets, and searches the ordinary
dense compute-matched controls; it performs no optimizer steps or training.

```bash
python preflight_compute_pilot.py --data_path data/bunny --device cuda \
  --output output/compute_pilot/preflight.json
```

The report includes dense MACs/GFLOPs, planning time, groups removed, achieved
kappa, target mismatch and widths, plus each `d_small_compute` candidate's
requested model size, actual parameters, widths, GFLOPs, and compute mismatch.

## Engineering smoke

```bash
python train_compute_pilot.py --method m4_pcd --budget_seconds 1 --kappa 0.50 \
  --outf output/compute_pilot_smoke --data_path data/bunny --vid bunny \
  --device cuda --max_frames 2 --crop_list 192_384 --batchSize 1 --workers 0 \
  --enc_strds 2 --enc_dim 4_2 --dec_strds 2 --ks 1_1_3 --reduce 2 \
  --modelsize 0.35 --lower_width 2 --conv_type conv pshuffel \
  --num_blks 1_1 --max_epochs 20 --eval_every 1 --fps_warmup 2 --fps_runs 5
```

## Bunny seed-1 commands

Set W explicitly from your chosen calibration; it is intentionally not
hard-coded:

The `--lambda_gl 1e-5` and `--tau 0.05` values below are example engineering
values only. They are not frozen scientific hyperparameters and must be chosen
or calibrated under the study protocol before the real pilot.

`d_start` depends on seed and W but not on the compute target. Run it once per
seed/W and use that same result as the reference for both kappa targets.

```bash
W=<active-training-seconds>
COMMON="--budget_seconds $W --data_path data/bunny --vid bunny --manualSeed 1 \
 --device cuda --conv_type convnext pshuffel --act gelu --norm none \
 --crop_list 640_1280 --resize_list -1 --enc_strds 5 4 4 2 2 \
 --enc_dim 64_16 --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2 \
 --modelsize 1.5 --lower_width 12 --batchSize 2 --lr 0.001"

python train_compute_pilot.py $COMMON --method d_start         --kappa 0.50 --outf output/compute_pilot/k050/d_start/seed1
python train_compute_pilot.py $COMMON --method d_small_compute --kappa 0.50 --outf output/compute_pilot/k050/d_small_compute/seed1
python train_compute_pilot.py $COMMON --method m1_posthoc      --kappa 0.50 --outf output/compute_pilot/k050/m1_posthoc/seed1
python train_compute_pilot.py $COMMON --method m2_gradual      --kappa 0.50 --outf output/compute_pilot/k050/m2_gradual/seed1
python train_compute_pilot.py $COMMON --method m3_group_lasso  --kappa 0.50 --lambda_gl 1e-5 --outf output/compute_pilot/k050/m3_group_lasso/seed1
python train_compute_pilot.py $COMMON --method m4_pcd          --kappa 0.50 --tau 0.05 --outf output/compute_pilot/k050/m4_pcd/seed1

# Reuse the k050/d_start/seed1 result above as the kappa=.70 reference.
python train_compute_pilot.py $COMMON --method d_small_compute --kappa 0.70 --outf output/compute_pilot/k070/d_small_compute/seed1
python train_compute_pilot.py $COMMON --method m1_posthoc      --kappa 0.70 --outf output/compute_pilot/k070/m1_posthoc/seed1
python train_compute_pilot.py $COMMON --method m2_gradual      --kappa 0.70 --outf output/compute_pilot/k070/m2_gradual/seed1
python train_compute_pilot.py $COMMON --method m3_group_lasso  --kappa 0.70 --lambda_gl 1e-5 --outf output/compute_pilot/k070/m3_group_lasso/seed1
python train_compute_pilot.py $COMMON --method m4_pcd          --kappa 0.70 --tau 0.05 --outf output/compute_pilot/k070/m4_pcd/seed1
```

Aggregate completed runs with:

```bash
python summarize_compute_pilot.py output/compute_pilot
```
