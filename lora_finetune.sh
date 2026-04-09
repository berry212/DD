#!/usr/bin/env bash
set -euo pipefail

uv run train-lora-dreammnist \
  --data-npz data/dermamnist_224.npz \
  --output-dir outputs/lora_dreammnist \
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
