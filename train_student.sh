#!/usr/bin/env bash
set -euo pipefail

BACKBONE="${BACKBONE:-resnet50}"

uv run run-train-distilled-student \
  --data-root data \
  --distilled-data outputs/dermamnist_224_distill/distilled_data.pt \
  --output-dir outputs/dermamnist_224_student \
  --student-backbone "$BACKBONE" \
  --train-epochs 20 \
  --train-batch-size 64 \
  --eval-batch-size 128 \
  --train-lr 3e-4 \
  --weight-decay 1e-4 \
  --amp \
  --num-workers 4 \
  "$@"
