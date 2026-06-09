#!/usr/bin/env bash
# ============================================================================
# Train student on noise-distilled data — NO augmentation, NO FKD
# ============================================================================
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
DATASET="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
if [[ "$DATASET" == "aptos" || "$DATASET" == "aptos2019" || "$DATASET" == "aptos-2019" ]]; then
  DATASET="aptos-2019-blindness-detection"
fi

DATA_ROOT="${DATA_ROOT:-data}"
BACKBONE="${BACKBONE:-resnet18}"
IPC="${IPC:-100}"
DISTILLED_DIR="${DISTILLED_DIR:-outputs/${DATASET}_224_noise_ipc${IPC}}"
DISTILLED_DATA="${DISTILLED_DATA:-${DISTILLED_DIR}/distilled_data.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${DATASET}_224_noise_student_ipc${IPC}}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
TRAIN_LR="${TRAIN_LR:-4e-4}"
KD_TEMPERATURE="${KD_TEMPERATURE:-0}"
HARD_LABEL_ALPHA="${HARD_LABEL_ALPHA:-0.0}"

if [[ "$DATASET" == "aptos-2019-blindness-detection" ]]; then
  if [[ ! -f "$DATA_ROOT/train.csv" || ! -d "$DATA_ROOT/train_images" ]]; then
    if [[ ! -f "$DATA_ROOT/APTOS_2019_Blindness_Detection/train.csv" || ! -d "$DATA_ROOT/APTOS_2019_Blindness_Detection/train_images" ]]; then
      echo "[ERROR] APTOS-2019 dataset directory not found under DATA_ROOT: $DATA_ROOT"
      exit 1
    fi
  fi
fi

uv run run-train-distilled-student \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --distilled-data "$DISTILLED_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --student-backbone "$BACKBONE" \
  --train-epochs "$TRAIN_EPOCHS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --train-lr "$TRAIN_LR" \
  --weight-decay 1e-4 \
  --kd-temperature "$KD_TEMPERATURE" \
  --weight-balance-alpha 0.0 \
  --soft-label-sharpen 1.0 \
  --hard-label-alpha "$HARD_LABEL_ALPHA" \
  --train-crop-min-scale 1.0 \
  --train-crop-max-scale 1.0 \
  --train-horizontal-flip-prob 0.0 \
  --no-use-fkd-batches \
  --num-workers 4 \
  "$@"
