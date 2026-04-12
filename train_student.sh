#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
BACKBONE="${BACKBONE:-resnet50}"
IPC="${IPC:-100}"
DISTILLED_DIR="${DISTILLED_DIR:-outputs/${DATASET}_224_distill_ipc${IPC}}"
DISTILLED_DATA="${DISTILLED_DATA:-${DISTILLED_DIR}/distilled_data.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${DATASET}_224_student_ipc${IPC}}"

uv run run-train-distilled-student \
  --dataset "$DATASET" \
  --data-root data \
  --distilled-data "$DISTILLED_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --student-backbone "$BACKBONE" \
  --train-epochs 20 \
  --train-batch-size 64 \
  --eval-batch-size 128 \
  --train-lr 3e-4 \
  --weight-decay 1e-4 \
  --amp \
  --num-workers 4 \
  "$@"
