#!/usr/bin/env bash
# ============================================================================
# Noise-only distillation: generate random noise + teacher soft labels
#
# This script replaces the full pipeline (VAE encode → cluster → decode)
# with a simple noise generator. The output format is compatible with
# run-train-distilled-student.
#
# 用法：
#   bash noise_distillate.sh                         # 默认 dermamnist IPC=100
#   DATASET=bloodmnist IPC=50 bash noise_distillate.sh
# ============================================================================
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
DATASET="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
if [[ "$DATASET" == "aptos" || "$DATASET" == "aptos2019" || "$DATASET" == "aptos-2019" ]]; then
  DATASET="aptos-2019-blindness-detection"
fi

DATA_ROOT="${DATA_ROOT:-data}"
IPC="${IPC:-100}"
TEACHER_BACKBONE="${TEACHER_BACKBONE:-resnet18}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${DATASET}_224_noise_ipc${IPC}}"
BASELINE_DIR="${BASELINE_DIR:-outputs/${DATASET}_224_distill_baseline}"
FKD_PRECOMPUTE_BATCHES="${FKD_PRECOMPUTE_BATCHES:-false}"
FKD_TRAIN_EPOCHS="${FKD_TRAIN_EPOCHS:-300}"
FKD_BATCH_SIZE="${FKD_BATCH_SIZE:-1024}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-20.0}"

if [[ ! -f "${BASELINE_DIR}/teacher_best.pt" ]]; then
  echo "[ERROR] Missing teacher checkpoint: ${BASELINE_DIR}/teacher_best.pt"
  echo "[ERROR] Run: DATASET=${DATASET} DATA_ROOT=${DATA_ROOT} OUTPUT_DIR=${BASELINE_DIR} bash baseline.sh"
  exit 1
fi

FKD_PRECOMPUTE_FLAG="--fkd-precompute-batches"
if [[ "$FKD_PRECOMPUTE_BATCHES" == "false" ]]; then
  FKD_PRECOMPUTE_FLAG="--no-fkd-precompute-batches"
fi

uv run run-noise-distillate \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --teacher-baseline-dir "$BASELINE_DIR" \
  --clusters-per-class "$IPC" \
  --teacher-backbone "$TEACHER_BACKBONE" \
  --teacher-temperature "$TEACHER_TEMPERATURE" \
  --eval-batch-size 1024 \
  "$FKD_PRECOMPUTE_FLAG" \
  --fkd-train-epochs "$FKD_TRAIN_EPOCHS" \
  --fkd-batch-size "$FKD_BATCH_SIZE" \
  --fp16 \
  "$@"

echo "[INFO] Noise distillation finished."
echo "[INFO] Train student with:"
echo "  uv run run-train-distilled-student \\"
echo "    --dataset ${DATASET} --data-root ${DATA_ROOT} \\"
echo "    --distilled-data ${OUTPUT_DIR}/distilled_data.pt \\"
echo "    --output-dir outputs/${DATASET}_224_noise_student_ipc${IPC}"
