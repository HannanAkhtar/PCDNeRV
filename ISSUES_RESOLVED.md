# PCD-NeRV v2 — Resolution of the Codebase Issues

## Purpose

This document responds point by point to `PCD-NeRV_CODEBASE_ISSUES.md` (the
deployment team's review of the v1 repository). For each of the 15 issues it
restates the required fix and records exactly what was changed in **PCD-NeRV
v2**, where the change lives, and how it is verified (unit test and/or smoke
evidence). v1 is preserved unchanged alongside v2 for comparison.

Verification baseline referenced throughout:

- **Unit tests** — `python run_tests.py`, 29 tests, all passing.
- **Smoke matrix** — `bash scripts/smoke_test.sh`: unit tests, one 5-epoch run
  per training method on the bundled 12-frame bunny subset (192×384, 0.35M
  decoder), exact pruning on the `pcd` run, 50%-budget pruning on the
  `baseline` run, and a 2-epoch fine-tune from the budget-pruned artifact.
  All green.

---

## 1. Group-lasso unit did not match a real PixelShuffle output channel — FIXED

**Required:** one structured group = the r² consecutive convolution output
rows that `PixelShuffle(r)` rearranges into one post-shuffle feature channel;
the same definition in loss, prox, reporting, and pruning.

**What was done:**

- New module `shared/groups.py` defines the single group abstraction,
  `ChannelGroupLayer`: for each targeted decoder conv it records the live
  module, the PixelShuffle factor `r` (1 for `interpolate` blocks and
  stride-1 blocks), and `channels = out_rows / r²`. `group_view()` reshapes
  the weight to `[channels, r²·C_in·kh·kw]` (the r² rows per channel are
  stored consecutively, so this is exact) and concatenates the r² biases.
- `group_lasso_loss`, `apply_group_prox`, `group_sparsity`,
  `compute_sparsity_report`, and `build_layer_sparsity_breakdown` all operate
  on these channel groups and live in that one module. The trainer
  (`train_pcd_nerv.py`), the reports, and the physical pruner
  (`shared/physical_pruning.py`) import them from there — the training
  objective and the pruning unit are structurally the same object and cannot
  drift apart again.
- `shared/nerv_targets.get_channel_group_layers()` builds the specs by
  walking `decoder.1..N`, locating the Conv2d and PixelShuffle inside each
  block (ConvTranspose2d decoders are rejected with an explanatory error).
- Effect on reported numbers: the smoke decoder that v1 reported as **1,274
  row-groups** is now **114 channel groups** — `group_total` equals the number
  of physically removable channels, and PCD/prox pressure is spent only on
  units that can actually shrink the architecture.

**Verified by:** `tests/test_groups.py::TestPixelShuffleGrouping`
(`channels × r² == conv rows`; the group view covers every weight and bias
entry exactly once), plus every pruning test operating on the same unit.

---

## 2. Biases were not part of the group — FIXED

**Required:** include the r² bias entries in the same group norm and apply the
same proximal scaling to weights and biases.

**What was done:**

- `ChannelGroupLayer.group_view()` concatenates `weight.reshape(C, -1)` with
  `bias.reshape(C, r²)`; `group_norms()` is therefore bias-inclusive.
- `apply_group_prox()` computes one scale factor
  `max(0, 1 − thr/‖g‖)` per channel and multiplies **both** the r² weight rows
  and the r² bias entries in place. Groups at or below the threshold become
  exactly zero, weights and biases alike.
- The group-lasso loss is differentiable through both (its gradient now
  reaches the biases, so PCD's secondary gradient and the weighted-sum term
  also press on biases).
- Downstream consequence: a dead group's post-activation output is
  identically zero (the pruner checks `act(0) == 0` on the actual block and
  refuses activations like softplus), so v1's bias/consumer special-casing
  and "kept-for-exactness" channels are gone entirely — weight-group sparsity
  now *is* removable-channel sparsity.

**Verified by:** `tests/test_groups.py::TestBiasInclusiveNorms`
(manual √(‖W‖²+‖b‖²) match; **the v1 failure mode — zero weights with live
bias — is asserted NOT dead**; loss gradient reaches biases) and
`TestProximalShrinkage` (identical scale factor on weights and biases;
below-threshold groups exactly zero).

---

## 3. Reported group sparsity ≠ physical model reduction — FIXED

**Required:** treat the physically rebuilt model as the source of truth;
report actual parameters, serialized size, FLOPs, per-layer widths, and
measured latency/FPS; demote group sparsity to a diagnostic.

**What was done:**

- `physical_pruning_pipeline.py` measures everything on the physically
  rebuilt model: decoder+head parameter counts (before/after, with reduction
  %), serialized state bytes, **FLOPs/frame** measured with forward hooks on
  the actual modules (`shared/accounting.measure_decoder_flops`), a
  **per-layer widths table** (channels/rows/in-channels before → after,
  including the head input), and **decode latency mean/p50/p90 + FPS**
  (`measure_decode_latency`, using the model's own decoder-only timing).
- Group sparsity is explicitly labeled: the sparsity-breakdown header states
  the group unit and that it is "a DIAGNOSTIC; physical pruning reports are
  the source of truth for compression", and the analysis Pareto plot is
  titled "diagnostic".
- Why this matters is already visible in the smoke run: the exact prune of
  the 5-epoch `pcd` run removes **56.19% of decoder+head parameters** but
  **98.54% of FLOPs** (11.40 ms → 2.80 ms, 87.7 → 357.4 fps on CPU), because
  the dead channels concentrate in late, high-resolution layers. Neither
  number is derivable from the 48.25% group-sparsity diagnostic alone.

**Verified by:** `tests/test_physical_pruning.py::test_param_count_matches_width_propagation`
(hook-measured accounting equals width-propagation math) and the smoke
reports (`examples/example_pruning_exact.txt`).

---

## 4. Only exact-threshold pruning; no matched budgets — FIXED

**Required:** a budget-pruning mode that ranks real channel groups and
rebuilds a model near a requested decoder+head parameter count; report quality
immediately after pruning and after an identical reconstruction-only
fine-tuning schedule.

**What was done:**

- `shared/physical_pruning.compute_budget_plan()`: ranks **all** channel
  groups across layers by bias-inclusive norm (each layer's strongest
  `--min_keep` channels are protected), then binary-searches the removal
  count so the rebuilt decoder+head parameter count is as large as possible
  while ≤ the budget. Budget is given as `--budget_reduce 0.25/0.5/0.75/0.9`
  or `--budget_params N`. Unreachable budgets (below the min-keep floor)
  prune to the floor and report `budget_reached=False`.
- `physical_pruning_pipeline.py --mode budget` runs it, reports the quality
  drop over the full video, and saves a self-contained artifact.
- Post-pruning recovery: `train_pcd_nerv.py --init_artifact <artifact>`
  fine-tunes the pruned model — the architecture is rebuilt from the
  artifact, the run is tagged `..._ft`, and any method/schedule can be used
  (reconstruction-only `--method baseline` is the intended identical
  recovery schedule). Prune → fine-tune → prune-again chains work (the
  pipeline rebuilds fine-tuned models from their recorded artifact).
- `scripts/run_budget_grid.sh` automates the 25/50/75/90% grid with
  fine-tuning.

**Verified by:** `tests/test_physical_pruning.py::TestBudgetPruning` (budget
met and equal to the plan's accounting; min-keep respected; the weakest group
is removed first; unreachable budgets hit the floor and say so) and the smoke
matrix: 50% request → **50.83% achieved** (170,336 ≤ 173,225 requested),
PSNR 16.30 → 11.17 dB before recovery, and the fine-tune run training the
65-channel-group pruned architecture.

---

## 5. Missing explicit training baselines — FIXED

**Required:** distinct modes for reconstruction-only, post-training pruning,
conventional proximal group lasso, weighted-sum, and PCD, all sharing model,
data, optimizer, schedule, loss, pruning, and evaluation.

**What was done:**

- `train_pcd_nerv.py --method {baseline, prox_gl, weighted_sum, pcd}`, all
  implemented in one `train_one_epoch()` that shares the model construction,
  dataloaders, Adam optimizer, per-step cosine schedule, reconstruction loss,
  sparsity reporting, evaluation, checkpointing, and analysis:
  - `baseline` — reconstruction backward + step (group-lasso value logged
    under `no_grad` as a diagnostic only);
  - `prox_gl` — reconstruction step, then proximal group shrinkage;
  - `weighted_sum` — single backward on `L_rec + ws_weight · L_group`, no prox;
  - `pcd` — dual backward, `PCDSolver` direction, step, prox.
- "Post-training structured pruning" is `--method baseline` followed by
  `--mode budget` pruning (+ optional `--init_artifact` fine-tune) — same
  pruning and evaluation code as every other method.

**Verified by:** `tests/test_training_steps.py::TestMethodSteps` — baseline
never shrinks groups even with a huge λ; prox_gl zeroes all groups with a
huge λ; **weighted-sum gradients equal `∇L_rec + w·∇L_group` parameter-wise**;
pcd produces solver diagnostics and applies prox. The smoke matrix runs all
four into one method-labeled `summary.txt` (baseline 16.30 dB / prox_gl 15.77
/ weighted_sum 16.20 / pcd 11.19 with 55/114 groups dead at aggressive
τ=0.5, λ_prox=50).

---

## 6. `tau`, proximal lambda, and weighted-sum weight conflated — FIXED

**Required:** separate names — `tau` (PCD pressure), `lambda_prox` (proximal
strength), `ws_weight` (weighted-sum coefficient).

**What was done:** exactly those flags: `--tau` (pcd only), `--lambda_prox`
(pcd and prox_gl; per-step threshold `lambda_prox · lr`), `--ws_weight`
(weighted_sum only), with per-method sweep flags `--tau_values`,
`--lambda_values`, `--ws_values`. The run config JSON records each value only
for the methods it applies to (`null` otherwise), so ablations are
unambiguous. `lambda_gl` no longer exists anywhere in v2.

**Verified by:** `tests/test_accounting.py::TestRunTags` (tag formats) and the
config blocks in every smoke run JSON.

---

## 7. Lambda sweep did not use the run-specific value — CONFIRMED AS A REAL BUG, FIXED

**Required:** pass the run-specific proximal lambda into the training function
and use it for every proximal update.

**What was done:**

- Confirmed the v1 defect: `run_single()` recorded the per-run λ in the tag
  and config, but the epoch loop called `apply_prox_fn(model, args.lambda_gl
  * lr)` — every entry of a `--lambda_values` sweep trained with the CLI
  default while being labeled differently. (v1 runs launched as separate
  processes with `--lambda_gl`, including all v1 smoke results, are
  unaffected.)
- In v2, `run_single(...)` receives `(tau, lambda_prox, ws_weight)` from the
  sweep grid and passes them as **function arguments** to
  `train_one_epoch(...)`; the update reads only those arguments, never
  `args`.

**Verified by:**
`tests/test_training_steps.py::TestLambdaSweepCorrectness` — a regression
test that plants a *contradictory decoy* value on `args.lambda_prox` and
asserts the passed argument wins in both directions (passed 0 + decoy 10⁹ →
nothing dies; passed 10⁹ + decoy 0 → every group dies).

---

## 8. FC-target grouping not a physically valid channel-pruning experiment — DISABLED AS REQUESTED

**Required:** disable FC structured pruning for the initial work; support only
convolutional decoder targets; add reshape-aware FC grouping later as an
extension.

**What was done:** `VALID_TARGETS = ('conv',)`;
`get_channel_group_layers(model, 'fc')` raises `NotImplementedError` with an
explanation (FC-stem rows pass through the `fc_h × fc_w` reshape before
decoder.1, so rows are not independently removable channels and no surgery
exists); the trainer's `--gl_target` accepts only `conv`. The v1 fc-target
code was removed rather than left half-supported. The README's scope section
documents reshape-aware FC grouping + surgery as a planned extension.

**Verified by:** `tests/test_groups.py::test_fc_target_is_rejected`.

---

## 9. Inconsistent parameter accounting — FIXED

**Required:** report `encoder_params`, `decoder_head_params`,
`embedding_storage`, and `total_stored_representation` separately and
consistently; use decoder + RGB head as the main budget axis.

**What was done:**

- New `shared/accounting.py::param_accounting()` returns exactly those four
  quantities (plus `other_params`/`trainable_params`), derived from one shared
  scope function (`shared/nerv_targets.param_scope`, with
  `DECODER_HEAD_SCOPES = fc_stem + conv_blocks + head`).
- Used identically by: the trainer (printed at run start, stored in the
  config JSON's `accounting` block, CSV columns), the sparsity reports
  (`decoder_head_param_*` keys replace v1's ambiguous "decoder" scope, which
  had *included* the head in reports but *excluded* it in the HNeRV-style
  parameter printout), and the pruning pipeline (before/after accounting,
  including `total_stored_representation` with the unchanged embedding
  storage).
- The budget axis for pruning (`--budget_reduce`/`--budget_params`) is
  decoder+head parameters, as specified.

**Verified by:** `tests/test_accounting.py` (scopes partition all trainable
parameters; `decoder_head_state()` matches the scope count; embedding
storage passthrough; NeRV-PE has zero encoder params) — plus the smoke `pcd`
run where all-params sparsity (32.48%) vs decoder+head sparsity (57.90%)
shows the encoder separation working.

---

## 10. Inconsistent reconstruction losses across scripts — FIXED

**Required:** one reconstruction objective for every initial comparison; L2
as the default.

**What was done:** the trainer's `--loss` default changed from `Fusion6` to
`L2`; every provided script (`smoke_test.sh`, `run_bunny_all_methods.sh`,
`run_budget_grid.sh`) uses L2; the flag's help text and README both state the
loss must be held fixed across methods being compared, with other losses as a
later ablation.

---

## 11. Run names not seed- and method-safe — FIXED

**Required:** tags containing at least method, target, seed, and relevant
hyperparameters.

**What was done:** `train_pcd_nerv.make_tag()` produces
`baseline_seed1`, `proxgl_conv_lprox1e-03_seed1`, `ws_conv_w1e-05_seed1`,
`pcd_conv_tau0.05_lprox1e-03_seed1` (+ `_ft` for fine-tuning runs) — matching
the format requested in the review (`pcd_conv_tau0.01_lprox1e-03_seed1` is
asserted literally in the tests). Only method-relevant hyperparameters appear
in the tag. Every per-run file (JSON, reports, checkpoints, latest/best,
tensorboard dir, CSV) is tag-scoped, so different seeds or methods can never
overwrite or auto-resume one another.

**Verified by:** `tests/test_accounting.py::TestRunTags` (six distinct tags
across methods/seeds/ft; exact format match).

---

## 12. Warm-up-only runs mislabeled as PCD — FIXED

**Required:** a true reconstruction-only method mode.

**What was done:** `--method baseline` is that mode (issue 5) — no group-lasso
gradient, no solver, no prox, tagged `baseline_seedN`. Warmup remains
available for the non-baseline methods (`--use_warmup --warmup_epochs N`) and
is now honestly logged: warmup epochs print `[WARMUP]`, record
`effective_step: baseline` in their JSON rows, and produce no PCD
diagnostics.

**Verified by:**
`tests/test_training_steps.py::test_warmup_turns_any_method_into_baseline`
(a warmup epoch of `pcd` performs a pure baseline step: no shrinkage even
with huge λ, no solver keys).

---

## 13. Exact pruning and budget pruning answer different questions — BOTH KEPT, SEPARATED

**Required:** keep both modes and report them separately.

**What was done:** `physical_pruning_pipeline.py --mode {exact, budget}`.
Outputs are mode-suffixed and coexist per run:
`reports/<tag>_physical_pruning_exact.txt` vs `..._budget.txt`,
`results/<tag>_physical_pruning_{exact,budget}.json`,
`pruned/<tag>_{exact,budget}_pruned.pth`, and a `mode` column in the summary
CSV. The verdict semantics differ deliberately: exact mode **asserts** output
equivalence (PSNR gap ≤ `--psnr_tol`, max-abs output diff reported,
round-trip reload check); budget mode reports budget attainment and the
quality drop, and points to `--init_artifact` for recovery. Both were
exercised in the smoke matrix (exact: gap 0.00e+00; budget: 50.83% at −5.13 dB
pre-recovery).

---

## 14. No unit-test suite protecting the structural assumptions — FIXED

**Required (their list → where it is covered):**

| Requested test | Where |
|---|---|
| PixelShuffle grouping | `test_groups.py::TestPixelShuffleGrouping` |
| bias-inclusive group norm | `test_groups.py::TestBiasInclusiveNorms` |
| proximal whole-group shrinkage | `test_groups.py::TestProximalShrinkage` |
| exact physical-pruning equivalence | `test_physical_pruning.py::TestExactPruning` (dead channel removed, `max|Δout| < 1e-6`; dense no-op is bit-exact) |
| budget-pruning parameter count | `test_physical_pruning.py::TestBudgetPruning` |
| PCD solver active/inactive constraints | `test_solver.py` (aligned → pure primary step, μ=0; opposed → μ matches the closed form, the normalized constraint holds with equality, the returned direction preserves the raw primary magnitude; zero-GL degenerate case) |
| lambda-sweep correctness | `test_training_steps.py::TestLambdaSweepCorrectness` |
| method-specific training steps | `test_training_steps.py::TestMethodSteps` |
| *(additional)* accounting scopes, run tags, artifact round-trip | `test_accounting.py`, `test_physical_pruning.py::TestArtifactRoundTrip` |

29 tests total, stdlib `unittest` (no new dependency), `python run_tests.py`,
< 1 s on CPU. `scripts/smoke_test.sh` runs the suite before any training, so
a grouping error can no longer make it into a long run unnoticed.

---

## 15. Current results cannot establish PCD superiority — ACKNOWLEDGED AND ENCODED

**What was done:** README §8 ("What this repository can and cannot show")
states the supported feasibility claims (channel-exact native sparsity;
exact removal with identical outputs; fair budget-matched comparison
*machinery*) and explicitly lists what is **not** claimed: fc-stem pruning,
and any PCD-superiority conclusion — the latter requires the multi-video,
multi-seed, budget-matched grid that v2 now enables (issues 1–14) but does
not presuppose. The analysis/summary outputs are method-labeled so those
comparisons read correctly once the full grid is run.

---

## Recommended fix order — followed

The review's 10-step fix order maps onto the v2 work as: (1) channel grouping
→ `shared/groups.py`; (2) bias inclusion → same module; (3) aligned
reporting/pruning → single shared abstraction + source-of-truth reports;
(4) explicit methods → `--method`; (5) budget pruning → `--mode budget` +
`--init_artifact`; (6) loss/accounting standardization → L2 default +
`shared/accounting.py`; (7) sweep fix + seed-safe tags → threaded values +
`make_tag`; (8) fc disabled; (9) unit + smoke tests → `tests/` +
`scripts/smoke_test.sh`; (10) full from-scratch experiments — **not launched
from this machine** (CPU-only laptop); the codebase is ready for the
resources team.

## The v1 "what can still be done" section — superseded

The interim workarounds it prescribed are no longer needed: pre-PixelShuffle
group sparsity no longer exists (groups *are* channels), per-lambda separate
processes are unnecessary (sweep fixed and regression-tested), and the
warm-up-only control is replaced by `--method baseline`.
