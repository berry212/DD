#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
DATA_NPZ="${DATA_NPZ:-data/${DATASET}_224.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora_${DATASET}}"

uv run train-lora-dreammnist \
  --dataset "$DATASET" \
  --data-npz "$DATA_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --base-model-id runwayml/stable-diffusion-v1-5 \
  --resolution 224 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --epochs 20 \
  --max-train-steps 3000 \
  --rank 16 \
  --lora-alpha 16 \
  --lr 5e-5 \
  --lr-schedule cosine \
  --lr-warmup-steps 100 \
  --class-balance \
  --prompt-dropout-prob 0.1 \
  --snr-gamma 5.0 \
  --noise-offset 0.05 \
  --max-grad-norm 1.0 \
  --fp16 \
  "$@"
