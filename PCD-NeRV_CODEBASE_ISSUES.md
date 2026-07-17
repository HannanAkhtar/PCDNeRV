# PCD-NeRV Codebase Issues Before Full Structured-Pruning Experiments

## Purpose

This document records the issues identified in the current **PCD-NeRV** repository that should be addressed before using it for a full, publication-quality comparison of structured pruning methods on NeRV/HNeRV.

The current repository is still useful for a **limited proof of concept**. It can show that:

- PCD can be integrated into HNeRV training or fine-tuning;
- group shrinkage can drive some decoder weights to exact zero;
- the existing physical-pruning pipeline can sometimes build a smaller dense decoder with exactly the same output as the dense PCD model.

However, the current results should not yet be interpreted as a fair comparison against ordinary HNeRV, weighted-sum regularization, or properly channel-aware structured pruning.

---

## 1. The current group-lasso unit does not match a real PixelShuffle output channel

### Current behavior

For each decoder convolution before `PixelShuffle(r)`, the code treats every convolution output row/filter as a separate group.

### Why this is incorrect for NeRV/HNeRV

A PixelShuffle block rearranges `r^2` convolution output rows into **one post-shuffle feature channel**. For example:

- `r = 2`: 4 convolution rows form one feature channel;
- `r = 4`: 16 convolution rows form one feature channel;
- `r = 5`: 25 convolution rows form one feature channel.

Therefore, one individually zero convolution row is not independently removable. All `r^2` rows belonging to a post-shuffle channel must be removed together.

### Consequences

- Reported `group_total` is much larger than the number of physically removable channels.
- Reported group sparsity can overstate actual structured sparsity.
- PCD and the proximal step may spend optimization effort zeroing individual rows that do not lead to a smaller architecture.
- The training objective and the physical-pruning unit are misaligned.

### Required fix

Create one structured group per post-PixelShuffle feature channel:

```text
one group = r^2 consecutive convolution output rows
```

All group-lasso loss, proximal shrinkage, sparsity reporting, and pruning code should use this same channel-group definition.

---

## 2. Biases are not included in the group-lasso/proximal group

### Current behavior

The group-lasso loss and proximal shrinkage target convolution weights, while the corresponding biases are not shrunk as part of the same group.

### Why this matters

A channel with:

```text
weights = 0
bias != 0
```

can still produce a constant nonzero feature map after its activation. It is not necessarily a dead channel.

### Consequences

- A weight group may be reported as zero while the channel is still active.
- The physical pruner must retain some apparently sparse channels for exactness.
- Weight sparsity is not equivalent to removable channel sparsity.

### Required fix

Include both the grouped convolution rows and their associated bias entries in a single group norm. Apply the same proximal scaling factor to both weights and biases.

---

## 3. Reported group sparsity is not equivalent to physical model reduction

### Current behavior

Reports emphasize the fraction of targeted groups whose norms are below a threshold.

### Why this can be misleading

Even after fixing PixelShuffle grouping, different channels can save different numbers of parameters. Removing a channel affects:

1. the current layer's output rows;
2. the next layer's corresponding input channel;
3. possibly the RGB head input.

Therefore:

```text
percentage of groups removed != percentage of parameters removed
```

and neither is automatically equal to the FLOP reduction.

### Required fix

Treat the physically rebuilt model as the source of truth. Report:

- actual decoder + head parameters before and after pruning;
- actual serialized model size;
- actual FLOPs;
- actual per-layer widths;
- measured FPS and latency.

Group sparsity should remain a diagnostic, not the main compression result.

---

## 4. The current pruning pipeline mainly supports exact threshold pruning, not matched budgets

### Current behavior

The physical-pruning pipeline removes naturally dead channels according to `group_thr` and exactness checks.

### Why this is insufficient for a fair experiment

Different training methods may naturally produce different amounts of exact sparsity. Comparing their resulting models directly would mix together:

- how aggressively each method compresses;
- how well it preserves reconstruction at a given size.

A fair comparison needs matched final budgets, such as:

- 25% parameter reduction;
- 50% parameter reduction;
- 75% parameter reduction;
- 90% parameter reduction.

### Required fix

Add a budget-pruning mode that ranks real channel groups and physically rebuilds a model near a requested final decoder + head parameter count.

Report quality both:

- immediately after pruning;
- after an identical reconstruction-only fine-tuning schedule.

---

## 5. The repository does not yet provide all necessary training baselines

### Current behavior

The main trainer is designed around PCD, with a primary-only warm-up option.

### Missing formal modes

A full experiment needs distinct, explicit modes for:

1. **Reconstruction-only baseline**
   - ordinary HNeRV/NeRV training;
   - no group-lasso gradient;
   - no proximal shrinkage.

2. **Post-training structured pruning**
   - reconstruction-only training followed by channel ranking and pruning.

3. **Conventional proximal group-lasso training**
   - reconstruction gradient update;
   - proximal group shrinkage;
   - no PCD gradient correction.

4. **Weighted-sum training**
   - `L_rec + w * L_group`;
   - no PCD.

5. **PCD training**
   - reconstruction primary;
   - group-lasso secondary;
   - proximal shrinkage.

### Required fix

Add an explicit `--method` argument and ensure all methods use the same model, data, optimizer, schedule, reconstruction loss, pruning code, and evaluation pipeline.

---

## 6. `tau`, proximal lambda, and weighted-sum weight need separate meanings and names

### Current meaning in the repository

- `tau`: PCD's minimum normalized secondary-progress requirement;
- `lambda_gl`: proximal shrinkage strength, applied through a threshold proportional to the current learning rate.

### Potential source of confusion

In ordinary weighted-sum training, a lambda-like coefficient usually means:

```text
L_total = L_rec + lambda * L_group
```

That is not the same role as the repository's current proximal `lambda_gl`.

### Required fix

Use separate names:

- `tau` — PCD directional pressure;
- `lambda_prox` — proximal group shrinkage strength;
- `ws_weight` — weighted-sum loss coefficient.

This separation is needed for interpretable ablations.

---

## 7. The lambda sweep may not use the run-specific lambda value correctly

### Identified issue

The sweep code passes a run-specific lambda to the run setup and records it in the tag/config, but the epoch update reads `args.lambda_gl` when calculating the proximal threshold.

### Consequence

Different entries in a lambda sweep may be labeled differently while using the same effective proximal lambda during training.

### Temporary workaround

Before fixing the code, run each lambda value as a separate process using `--lambda_gl`, rather than using `--lambda_values`.

### Required fix

Pass the run-specific proximal lambda directly into the training function and use it for every proximal update.

---

## 8. FC-target grouping is not currently a physically valid channel-pruning experiment

### Current behavior

The FC stem's output rows are treated individually as groups.

### Why this is problematic

The FC-stem output is reshaped into a spatial feature map. Multiple rows may belong to one feature channel after reshaping. Individual rows are not necessarily independently removable channels.

The repository also does not physically rebuild FC-pruned models.

### Required fix

For the initial work, disable FC structured pruning and support only convolutional decoder targets. Add reshape-aware FC grouping and physical surgery later as a separate extension.

---

## 9. Parameter accounting is inconsistent

### Current issue

Some code paths count only `decoder.parameters()` and exclude the RGB `head_layer`, while reports may use broader or differently named scopes.

For HNeRV, there are also several distinct storage/parameter concepts:

- training-only encoder parameters;
- decoder + RGB head parameters;
- stored per-frame embeddings;
- total stored video representation.

### Required fix

Report these separately and consistently:

- `encoder_params`;
- `decoder_head_params`;
- `embedding_storage`;
- `total_stored_representation`.

For the first structured-pruning study, use **decoder + RGB head parameters** as the main parameter-budget axis.

---

## 10. Reconstruction losses are inconsistent across scripts

### Current behavior

Some examples use `L2`, while at least one full HNeRV script uses `Fusion6`.

### Why this matters

A comparison between methods is not valid if the reconstruction loss changes at the same time as the optimization method.

### Required fix

Use one reconstruction objective for every initial comparison. `L2`/MSE is the simplest choice for the first experiment.

Other losses can be tested later as an ablation.

---

## 11. Run names are not fully seed- and method-safe

### Current behavior

Tags contain PCD target, tau, and lambda, but do not fully identify different methods or seeds.

### Consequences

- multiple seeds can overwrite or accidentally resume one another;
- a warm-up-only control may still be labeled as PCD;
- outputs can become ambiguous during aggregation.

### Required fix

Include at least:

- method;
- target;
- seed;
- relevant hyperparameters.

Example:

```text
pcd_conv_tau0.01_lprox1e-03_seed1
```

---

## 12. A warm-up-only run can serve as a temporary baseline, but it is not cleanly labeled

### Temporary proof-of-concept workaround

The current trainer's warm-up phase performs a primary-only HNeRV update. Setting:

```text
--use_warmup --warmup_epochs equal to total epochs
```

can create a reconstruction-only control without changing the code.

### Limitation

The run is still stored/tagged as a PCD configuration, which is confusing and unsuitable for final experiments.

### Required fix

Add a true reconstruction-only method mode.

---

## 13. Exact pruning and budget pruning answer different questions

### Exact pruning

Removes only channels proven to be exactly redundant, preserving identical output.

This answers:

> Did training create genuinely dead channels?

### Budget pruning

Removes the least important channels until a requested size is reached, with a possible quality drop.

This answers:

> Which training method gives the best reconstruction at the same final model size?

### Required experimental treatment

Keep both modes and report them separately. Exact pruning is valuable as a proof of native sparsity, but budget pruning is required for fair quality-versus-size comparisons.

---

## 14. No dedicated unit-test suite currently protects the structural assumptions

### Needed tests

At minimum:

- PixelShuffle grouping test;
- bias-inclusive group-norm test;
- proximal whole-group shrinkage test;
- exact physical-pruning equivalence test;
- budget-pruning parameter-count test;
- PCD solver active/inactive constraint tests;
- lambda-sweep correctness test;
- method-specific training-step tests.

Without these tests, an architectural grouping error can invalidate long training runs.

---

## 15. Current results cannot yet establish that PCD is superior

The existing repository can support a limited feasibility claim:

> PCD plus proximal shrinkage can create some exact decoder redundancy in HNeRV, and the redundancy can be physically removed.

It cannot yet support the stronger claim:

> PCD gives better PSNR at the same final parameter count than ordinary HNeRV, post-training pruning, weighted-sum regularization, or conventional proximal group lasso.

That stronger claim requires:

- corrected physical channel groups;
- explicit baselines;
- matched parameter budgets;
- consistent reconstruction losses and training schedules;
- multiple videos and selected multi-seed runs.

---

# Recommended Fix Order

1. Correct PixelShuffle-aware channel grouping.
2. Include biases in group norms and proximal shrinkage.
3. Align reporting and physical pruning with the same group abstraction.
4. Add explicit baseline, proximal group-lasso, weighted-sum, and PCD modes.
5. Add matched parameter-budget pruning.
6. Standardize reconstruction loss and parameter accounting.
7. Fix lambda-sweep handling and seed-safe tags.
8. Disable FC pruning until reshape-aware grouping is implemented.
9. Add unit and smoke tests.
10. Only then launch full from-scratch experiments.

---

# What Can Still Be Done Before These Fixes

A small HNeRV proof of concept can still be run with the current repository, provided that the conclusions are limited.

Use:

- HNeRV only;
- convolutional target only;
- one pretrained checkpoint or one small model;
- one video and one seed;
- separate process launches for every lambda value;
- exact physical pruning after PCD fine-tuning.

Ignore the reported pre-PixelShuffle `group_sparsity` as a physical channel count. Use the physical-pruning report's actual decoder parameter reduction and exact output-equivalence check as the trustworthy outputs.
