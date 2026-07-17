# PCD-NeRV v2

**Training Natively Sparse Neural Representations for Videos with Priority-Constrained Descent — with the baselines and pruning modes needed for fair structured-pruning experiments.**

v2 is a revision of PCD-NeRV that addresses every issue in the deployment
team's review (`PCD-NeRV_CODEBASE_ISSUES.md`). The headline changes:

| # | Issue | v2 fix |
|---|---|---|
| 1 | groups didn't match PixelShuffle channels | one group = one **post-shuffle channel** (its r² conv rows), everywhere: loss, prox, reports, pruning (`shared/groups.py`) |
| 2 | biases excluded from groups | group norms and proximal shrinkage **include the r² biases**; a dead group is a genuinely dead channel |
| 3 | group sparsity ≠ physical reduction | the **physically rebuilt model is the source of truth**: params, serialized bytes, FLOPs/frame, per-layer widths, decode latency/FPS are all measured on it; group sparsity is labeled a diagnostic |
| 4 | only exact-threshold pruning | added **matched-budget pruning** (`--mode budget --budget_reduce 0.25/0.5/0.75/0.9`) + fine-tuning from pruned artifacts (`--init_artifact`) |
| 5 | missing baselines | explicit `--method {baseline, prox_gl, weighted_sum, pcd}` sharing model/data/optimizer/schedule/loss/eval |
| 6 | conflated hyperparameter names | `--tau` (PCD pressure), `--lambda_prox` (proximal strength), `--ws_weight` (weighted-sum coefficient) |
| 7 | λ-sweep used the wrong value | run-specific values are threaded into the update function; regression-tested |
| 8 | fc grouping not physically valid | fc target **disabled** (clear error); conv decoder target only |
| 9 | inconsistent parameter accounting | `encoder_params` / `decoder_head_params` / `embedding_storage` / `total_stored_representation` reported everywhere (`shared/accounting.py`); decoder+head is the budget axis |
| 10 | inconsistent reconstruction losses | default `--loss L2`; all provided scripts use L2 |
| 11 | tags not seed/method-safe | tags like `pcd_conv_tau0.05_lprox1e-03_seed1`, `baseline_seed1`, `ws_conv_w1e-05_seed1` (+ `_ft`) |
| 12 | warm-up-only mislabeled as PCD | `--method baseline` is a first-class mode |
| 13 | exact vs budget answer different questions | both modes exist, reported separately (`_physical_pruning_exact` / `_budget` outputs) |
| 14 | no unit tests | `tests/` + `run_tests.py`: 29 tests covering grouping, bias-inclusive norms, prox, solver constraints, λ-sweep, method steps, exact & budget pruning, accounting, tags |
| 15 | over-claiming | see "What this repository can and cannot show" below |

---

## 1. Method overview

All four training methods share the same model, data pipeline, Adam optimizer,
per-step cosine schedule, reconstruction loss, and evaluation:

| `--method` | Update per batch | Sparsity mechanism |
|---|---|---|
| `baseline` | backprop `L_rec`, step | none (post-training pruning only) |
| `prox_gl` | backprop `L_rec`, step, then proximal group shrinkage (threshold `lambda_prox · lr`) | prox |
| `weighted_sum` | backprop `L_rec + ws_weight · L_group`, step | loss pressure only (no exact zeros) |
| `pcd` | backprop `L_rec` and `L_group` separately → `PCDSolver` combines directions (τ) → step → prox | PCD + prox |

`L_group` = Σ over post-shuffle channels of ‖[r² weight rows, r² biases]‖₂.
With `--use_warmup`, non-baseline methods run reconstruction-only for
`--warmup_epochs` first.

**Group unit.** `PixelShuffle(r)` bundles r² consecutive conv output rows into
one feature channel, so v2 defines one group per channel: `group_total` in
every report equals the number of physically removable channels, and the
proximal step zeroes whole channels (weights *and* biases) — with `act(0)=0`
a dead group's output is identically zero.

**Never targeted:** `decoder.0` (the FC stem), `head_layer` (RGB output), the
HNeRV encoder. The head's *input* shrinks when decoder.N loses channels
(dependency bookkeeping, not targeting).

## 2. Installation & tests

```bash
pip install -r requirements.txt
python run_tests.py          # 29 unit tests, < 1 s on CPU
```

## 3. Quick start

```bash
bash scripts/smoke_test.sh
```

runs the unit tests, one 5-epoch run per method on the bundled 12-frame bunny
subset, exact pruning on the pcd run, 50%-budget pruning on the baseline run,
and a 2-epoch fine-tune from the budget-pruned artifact (~4 min on CPU).

## 4. Training commands

HNeRV-1.5M bunny recipe (640×1280), one line per method:

```bash
COMMON="--outf bunny_v2 --data_path data/bunny --vid bunny \
  --conv_type convnext pshuffel --act gelu --norm none --crop_list 640_1280 \
  --resize_list -1 --loss L2 --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
  --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2 --modelsize 1.5 \
  -e 300 --eval_freq 1 --lower_width 12 -b 2 --lr 0.001"

python train_pcd_nerv.py $COMMON --method baseline
python train_pcd_nerv.py $COMMON --method prox_gl      --lambda_prox 1e-3
python train_pcd_nerv.py $COMMON --method weighted_sum --ws_weight 1e-5
python train_pcd_nerv.py $COMMON --method pcd --tau 0.05 --lambda_prox 1e-3
```

(or `bash scripts/run_bunny_all_methods.sh [--manualSeed N]`).

NeRV (positional-encoding) baseline: replace the encoder flags with
`--embed pe_1.25_80 --fc_hw 8_16 --dec_strds 5 4 2 2 --ks 0_3_3 --reduce 2`.

Sweeps (values are threaded per-run; issue #7 is fixed and regression-tested):

```bash
python train_pcd_nerv.py $COMMON --method pcd --tau_values 0.01,0.05,0.1 --lambda_values 1e-4,1e-3
python train_pcd_nerv.py $COMMON --method prox_gl --lambda_values 1e-4,1e-3,1e-2
python train_pcd_nerv.py $COMMON --method weighted_sum --ws_values 1e-6,1e-5,1e-4
```

Multi-seed: `--manualSeed N` (the seed is part of the run tag, so seeds never
collide or cross-resume).

## 5. Physical pruning

```bash
# exact: remove only genuinely dead channels; asserts output equivalence
python physical_pruning_pipeline.py --run_dir <dir> --tag <tag> --mode exact

# budget: rebuild to <= a requested decoder+head parameter count
python physical_pruning_pipeline.py --run_dir <dir> --tag <tag> \
    --mode budget --budget_reduce 0.5        # or --budget_params N

# fine-tune the budget-pruned model (identical reconstruction-only schedule)
python train_pcd_nerv.py $COMMON --method baseline -e 30 \
    --init_artifact <dir>/pruned/<tag>_budget_pruned.pth
```

`scripts/run_budget_grid.sh` automates the 25/50/75/90% grid + fine-tuning.

Both modes report, for dense vs pruned: decoder+head params and serialized
bytes, FLOPs/frame (hook-measured), per-layer widths, decode latency & FPS,
and full-video PSNR/MS-SSIM. Exact mode additionally asserts
`|ΔPSNR| ≤ --psnr_tol` and max-abs output difference; budget mode reports the
quality drop and whether the budget was reached (per-layer `--min_keep`
floors can make extreme budgets unreachable). Artifacts are self-contained;
`shared.physical_pruning.load_pruned_artifact()` rebuilds the model, and
prune → fine-tune → prune-again chains are supported.

## 6. Flag reference (v2-specific)

| Flag | Default | Applies to | Description |
|---|---|---|---|
| `--method` | `pcd` | — | `baseline` / `prox_gl` / `weighted_sum` / `pcd` |
| `--gl_target` | `conv` | prox_gl, ws, pcd | `conv` only (fc rejected — issue #8) |
| `--tau` | `0.05` | pcd | PCD directional pressure |
| `--lambda_prox` | `1e-3` | pcd, prox_gl | proximal shrinkage; per-step threshold `lambda_prox · lr` |
| `--ws_weight` | `1e-5` | weighted_sum | coefficient of `L_group` in the summed loss |
| `--tau_values` / `--lambda_values` / `--ws_values` | — | per method | comma-separated sweeps |
| `--use_warmup` / `--warmup_epochs` | off / 10 | non-baseline | reconstruction-only warmup |
| `--group_thr` | `1e-4` | — | dead threshold on bias-inclusive channel-group norms |
| `--param_thr` | `1e-8` | — | per-weight sparsity threshold (diagnostic) |
| `--init_artifact` | — | any | fine-tune from a pruned artifact (architecture comes from it) |
| `--ckpt_every` | `100` | — | periodic checkpoints |
| `--eval_freq` | `1` | — | full quality eval every N epochs (all metrics every epoch by default) |
| `--manualSeed` | `1` | — | seed (recorded in the run tag) |

All HNeRV data/architecture/training flags are inherited unchanged from v1
(`--data_path`, `--crop_list`, `--data_split`, `--embed`, `--enc_strds`,
`--dec_strds`, `--modelsize`, `-e`, `-b`, `--lr`, quantization flags, …); the
decoder `conv_type` must be `pshuffel` or `interpolate`.

## 7. Per-epoch logging & outputs

Every epoch prints/logs: reconstruction loss, group-lasso loss, train PSNR,
lr, epoch time; PCD diagnostics (τ, μ, primary efficiency, cosine, conflict
rate — pcd runs); channel-group sparsity, decoder+head parameter sparsity, and
all-parameter sparsity; seen/unseen/quantized PSNR & MS-SSIM. Everything lands
in `results/<tag>.json` (per-epoch rows + config + parameter accounting),
`logs/<tag>_train.txt`, tensorboard, and per-run CSVs; cross-run `summary.txt`
and plots aggregate every run in the same `--outf` with method-aware coloring.

Reports per run:
- `reports/<tag>_sparsity_breakdown.txt` — per-tensor table with scope,
  grouping (`r=…, N ch-groups` / `in weight groups`), param and channel-group
  sparsity; header states the group unit and that it is a diagnostic.
- `reports/<tag>_checkpoint_progress.txt` — checkpoint Pareto trend.
- `reports/<tag>_physical_pruning_{exact,budget}.txt` — the compression
  source of truth (params/bytes/FLOPs/widths/latency + verification).

## 8. What this repository can and cannot show

Supported claims (with the included verification):
- PCD (and prox-GL) can drive whole post-shuffle channels of an (H)NeRV
  decoder to exact zero during training;
- exact pruning physically removes them with **identical** outputs;
- at matched decoder+head budgets, the four methods can be compared fairly
  (same loss, schedule, data, seeds, pruning, and evaluation).

Not yet supported:
- fc-stem structured pruning (needs reshape-aware grouping and surgery);
- any claim that PCD is *superior* — that requires the full experiment grid
  (multiple videos, multiple seeds, budget-matched comparisons with fine-tune
  recovery) which this codebase now enables but does not pre-suppose.

## 9. Repository layout

```
train_pcd_nerv.py             training entry point (4 methods, sweeps, fine-tuning)
physical_pruning_pipeline.py  exact | budget pruning + measurements + verification
run_tests.py, tests/          unit-test suite (29 tests)
shared/groups.py              THE channel-group abstraction (loss/prox/reports)
shared/nerv_targets.py        target selection (conv), scoping, ConvTranspose guard
shared/accounting.py          4-scope parameter accounting, FLOPs, latency
shared/physical_pruning.py    exact/budget plans, surgery, artifacts
shared/pcd_solver.py          PCD solver (math verbatim from the reference)
shared/analysis.py            method-aware plots + summary
model_all.py, hnerv_utils.py  HNeRV model/utils (network math unchanged)
scripts/                      smoke test, all-methods bunny run, budget grid
data/bunny_smoke/             12-frame subset for smoke tests
examples/                     sample generated reports
```

Platform notes (Windows/Anaconda quirks, GPU auto-use, CPU timing) are
unchanged from v1: `KMP_DUPLICATE_LIB_OK` is set by the entry points, console
output is ASCII-safe, reports are UTF-8, tensorboard failures never kill a
run, and CUDA is used automatically when available.
