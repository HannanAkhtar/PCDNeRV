#!/usr/bin/env bash
# Full 300-epoch bunny comparison: all four methods on the HNeRV-1.5M recipe,
# same reconstruction loss (L2), same schedule, same seed. Pass extra flags
# through, e.g.:  bash scripts/run_bunny_all_methods.sh --manualSeed 42
cd "$(dirname "$0")/.."
set -e

COMMON="--outf bunny_v2 --data_path data/bunny --vid bunny \
  --conv_type convnext pshuffel --act gelu --norm none --crop_list 640_1280 \
  --resize_list -1 --loss L2 --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
  --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2 --modelsize 1.5 \
  -e 300 --eval_freq 1 --lower_width 12 -b 2 --lr 0.001"

python train_pcd_nerv.py $COMMON --method baseline                            "$@"
python train_pcd_nerv.py $COMMON --method prox_gl      --lambda_prox 1e-3     "$@"
python train_pcd_nerv.py $COMMON --method weighted_sum --ws_weight 1e-5       "$@"
python train_pcd_nerv.py $COMMON --method pcd  --tau 0.05 --lambda_prox 1e-3  "$@"
