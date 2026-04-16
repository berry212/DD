#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
DATASET="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
if [[ "$DATASET" == "odir5k" ]]; then
  DATASET="odir-5k"
fi
if [[ "$DATASET" == "aptos" || "$DATASET" == "aptos2019" || "$DATASET" == "aptos-2019" ]]; then
  DATASET="aptos-2019-blindness-detection"
fi

DATA_ROOT="${DATA_ROOT:-${HF_DATASETS_CACHE:-${HF_HOME:-data}}}"
BACKBONE="${BACKBONE:-resnet50}"
IPC="${IPC:-100}"
DISTILLED_DIR="${DISTILLED_DIR:-outputs/${DATASET}_224_distill_ipc${IPC}}"
DISTILLED_DATA="${DISTILLED_DATA:-${DISTILLED_DIR}/distilled_data.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${DATASET}_224_student_ipc${IPC}}"

if [[ "$DATASET" == "aptos-2019-blindness-detection" ]]; then
  if [[ ! -f "$DATA_ROOT/train.csv" || ! -d "$DATA_ROOT/train_images" ]]; then
    if [[ ! -f "$DATA_ROOT/APTOS_2019_Blindness_Detection/train.csv" || ! -d "$DATA_ROOT/APTOS_2019_Blindness_Detection/train_images" ]]; then
      echo "[ERROR] APTOS-2019 dataset directory not found under DATA_ROOT: $DATA_ROOT"
      echo "[ERROR] Expected either:"
      echo "[ERROR]   - $DATA_ROOT/train.csv and $DATA_ROOT/train_images"
      echo "[ERROR]   - $DATA_ROOT/APTOS_2019_Blindness_Detection/train.csv and train_images"
      exit 1
    fi
  fi
fi

if [[ -z "${KD_TEMPERATURE:-}" ]]; then
  if [[ "$DATASET" == "dermamnist" ]]; then
    if [[ "$IPC" -le 100 ]]; then
      KD_TEMPERATURE="1.5"
    else
      KD_TEMPERATURE="2.0"
    fi
  else
    KD_TEMPERATURE="1.0"
  fi
fi

if [[ -z "${HARD_LABEL_ALPHA:-}" ]]; then
  if [[ "$DATASET" == "dermamnist" ]]; then
    HARD_LABEL_ALPHA="0.35"
  else
    HARD_LABEL_ALPHA="0.0"
  fi
fi

if [[ -z "${WEIGHT_BALANCE_ALPHA:-}" ]]; then
  if [[ "$DATASET" == "dermamnist" ]]; then
    WEIGHT_BALANCE_ALPHA="0.35"
  else
    WEIGHT_BALANCE_ALPHA="0.0"
  fi
fi

if [[ -z "${SOFT_LABEL_SHARPEN:-}" ]]; then
  if [[ "$DATASET" == "dermamnist" ]]; then
    if [[ "$IPC" -le 100 ]]; then
      SOFT_LABEL_SHARPEN="0.9"
    else
      SOFT_LABEL_SHARPEN="0.85"
    fi
  else
    SOFT_LABEL_SHARPEN="1.0"
  fi
fi

uv run run-train-distilled-student \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --distilled-data "$DISTILLED_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --student-backbone "$BACKBONE" \
  --train-epochs 20 \
  --train-batch-size 64 \
  --eval-batch-size 128 \
  --train-lr 3e-4 \
  --weight-decay 1e-4 \
  --kd-temperature "$KD_TEMPERATURE" \
  --hard-label-alpha "$HARD_LABEL_ALPHA" \
  --weight-balance-alpha "$WEIGHT_BALANCE_ALPHA" \
  --soft-label-sharpen "$SOFT_LABEL_SHARPEN" \
  --amp \
  --num-workers 4 \
  "$@"
