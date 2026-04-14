#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
DATA_ROOT="${DATA_ROOT:-${HF_DATASETS_CACHE:-${HF_HOME:-data}}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${DATASET}_224_distill_baseline}"
TEACHER_BACKBONE="${TEACHER_BACKBONE:-resnet50}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-10}"

uv run run-baseline-resnet18 \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --teacher-backbone "$TEACHER_BACKBONE" \
    --teacher-epochs "$TEACHER_EPOCHS" \
    "$@"
