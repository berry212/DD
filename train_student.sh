#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
DATA_ROOT="${DATA_ROOT:-${HF_DATASETS_CACHE:-${HF_HOME:-data}}}"
BACKBONE="${BACKBONE:-resnet50}"
IPC="${IPC:-100}"
DISTILLED_DIR="${DISTILLED_DIR:-outputs/${DATASET}_224_distill_ipc${IPC}}"
DISTILLED_DATA="${DISTILLED_DATA:-${DISTILLED_DIR}/distilled_data.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${DATASET}_224_student_ipc${IPC}}"

if [[ -z "${KD_TEMPERATURE:-}" ]]; then
  if [[ "$DATASET" == "dermamnist" ]]; then
    KD_TEMPERATURE="2.0"
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
    SOFT_LABEL_SHARPEN="0.85"
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
