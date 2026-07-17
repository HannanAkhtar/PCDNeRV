#!/usr/bin/env bash
# Matched-budget pruning grid (25/50/75/90% decoder+head parameter reduction)
# for one trained run, followed by identical reconstruction-only fine-tuning.
#
# Usage: bash scripts/run_budget_grid.sh <run_dir> <tag> [ft_epochs] [common training flags...]
# The training flags must match the original run's data/architecture flags.
cd "$(dirname "$0")/.."
set -e

RUN_DIR="$1"; TAG="$2"; FT_EPOCHS="${3:-30}"; shift 3 || true

for REDUCE in 0.25 0.50 0.75 0.90; do
  python physical_pruning_pipeline.py --run_dir "$RUN_DIR" --tag "$TAG" \
    --mode budget --budget_reduce $REDUCE
  # budget artifacts share one filename per tag+mode; archive per-budget copies
  cp "$RUN_DIR/pruned/${TAG}_budget_pruned.pth" \
     "$RUN_DIR/pruned/${TAG}_budget${REDUCE}_pruned.pth"
  python train_pcd_nerv.py "$@" --method baseline -e "$FT_EPOCHS" \
    --suffix "_ft${REDUCE}" \
    --init_artifact "$RUN_DIR/pruned/${TAG}_budget${REDUCE}_pruned.pth"
done
