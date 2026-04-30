#!/usr/bin/env bash
set -euo pipefail

# Round 2 — urllc
# A) DESIGN primary baseline: mean/mean, mse, lambda_smooth=0
# B) Attack the val plateau (~0.9): weighted_mse + lambda_smooth=0.1
# 共通：sequence_length=15（比 Round 1 的 30 省時）、patience=20、epochs=80

BASE_ARGS=(
  --train_dirs tr0-9 tr11-14 tr16-27
  --val_dirs tr10
  --test_dirs tr15
  --slice_type urllc
  --epochs 200
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
  --lambda_smooth 0.0
  --sequence_length 30
)

echo "[INFO] Starting Round 3 urllc sweep in $(pwd)"

echo
echo "============================================================"
echo "[RUN A] urllc | concat/mean | huber | huber_delta 1.0"
echo "============================================================"
python train.py "${BASE_ARGS[@]}" \
  --loss_type huber \
  --huber_delta 1.0 \
  --gat_head_merge concat \
  --gat_final_head_merge mean \

echo
echo "============================================================"
echo "[RUN B] urllc | concat/mean | huber | huber_delta 0.5"
echo "============================================================"
python train.py "${BASE_ARGS[@]}" \
  --loss_type huber \
  --huber_delta 0.5\
  --gat_head_merge concat \
  --gat_final_head_merge mean

echo
echo "============================================================"
echo "[RUN C] urllc | mean/mean | huber | huber_delta 1.0"
echo "============================================================"
python train.py "${BASE_ARGS[@]}" \
  --loss_type huber \
  --huber_delta 1.0 \
  --gat_head_merge mean \
  --gat_final_head_merge mean

echo
echo "[INFO] Round 2 urllc completed."
