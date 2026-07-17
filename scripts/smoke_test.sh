#!/usr/bin/env bash
# PCD-NeRV v2 smoke test: unit tests, one 5-epoch run per training method,
# exact pruning on the pcd run, budget pruning (50%) on the baseline run, and
# a 2-epoch fine-tune from the budget-pruned artifact. CPU-friendly (~4 min).
cd "$(dirname "$0")/.."
set -e

echo "=== unit tests ==="
python run_tests.py

COMMON="--outf smoke --data_path data/bunny_smoke --vid bunny_smoke \
  --conv_type convnext pshuffel --act gelu --norm none --crop_list 192_384 \
  --resize_list -1 --loss L2 --enc_strds 4 4 3 2 2 --enc_dim 64_16 \
  --dec_strds 4 4 3 2 2 --ks 0_1_5 --reduce 1.2 --modelsize 0.35 \
  --data_split 2_2_3 -e 5 --eval_freq 1 --lower_width 6 -b 1 -j 0 --lr 0.001"
RUN_DIR="output/smoke/bunny_smoke/2_2_3_e5_size0.35M_L2"

echo "=== training methods ==="
python train_pcd_nerv.py $COMMON --method baseline
python train_pcd_nerv.py $COMMON --method prox_gl      --lambda_prox 50
python train_pcd_nerv.py $COMMON --method weighted_sum --ws_weight 1e-5
python train_pcd_nerv.py $COMMON --method pcd          --tau 0.5 --lambda_prox 50

echo "=== exact pruning on the pcd run ==="
python physical_pruning_pipeline.py --run_dir "$RUN_DIR" \
  --tag pcd_conv_tau0.5_lprox5e+01_seed1 --mode exact

echo "=== budget pruning (50%) on the baseline run ==="
python physical_pruning_pipeline.py --run_dir "$RUN_DIR" \
  --tag baseline_seed1 --mode budget --budget_reduce 0.5

echo "=== fine-tune from the budget-pruned artifact ==="
python train_pcd_nerv.py $COMMON --method baseline -e 2 \
  --init_artifact "$RUN_DIR/pruned/baseline_seed1_budget_pruned.pth"

echo "SMOKE OK"
