#!/usr/bin/env bash
set -euo pipefail

# Round 2 — embb
# A) DESIGN primary baseline (gat_head_merge=mean / gat_final_head_merge=mean),
#    sequence_length=15, mse, lambda_smooth=0.
# embb 第一輪 val=0.026 已接近 noise floor，不跑 weighted/smooth ablation。

BASE_ARGS=(
  --train_dirs tr0-9 tr11-14 tr16-27
  --val_dirs tr10
  --test_dirs tr15
  --slice_type embb
  --epochs 100
  --patience 30
  --batch_size 1024
  --learning_rate 1e-4
  --scheduler cosine
  --rnn_type bilstm
  --include_target_history true
  --readout mean
  --adj_type binary_selfloop
  --chebyshev_k 3.0
  --anomaly_error_mode both
  --loss_type mse
  --lambda_smooth 0.0
)

echo "[INFO] Starting Round 2 embb sweep in $(pwd)"

echo
echo "============================================================"
echo "[RUN A] embb | primary mean/mean | sequence_length 30"
echo "============================================================"
python train.py "${BASE_ARGS[@]}" \
  --gat_head_merge mean \
  --gat_final_head_merge mean \
  --sequence_length 30

echo
echo "============================================================"
echo "[RUN B] embb | concat/mean | sequence_length 15"
echo "============================================================"
python train.py "${BASE_ARGS[@]}" \
  --gat_head_merge concat \
  --gat_final_head_merge mean \
  --sequence_length 30

echo
echo "[INFO] Round 2 embb completed."
