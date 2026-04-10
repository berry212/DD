#!/usr/bin/env bash
set -euo pipefail

uv run run-train-distilled-student \
  --data-root data \
  --distilled-data outputs/dermamnist_224_distill/distilled_data.pt \
  --output-dir outputs/dermamnist_224_student \
  --train-epochs 30 \
  --train-batch-size 64 \
  --eval-batch-size 128 \
  --train-lr 1e-3 \
  --weight-decay 1e-4 \
  "$@"
