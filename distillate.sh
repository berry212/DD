#!/usr/bin/env bash
set -euo pipefail

LORA_PATH="${LORA_PATH:-outputs/lora_dreammnist}"

if [[ ! -d "$LORA_PATH" ]]; then
  echo "[WARN] LoRA path not found: $LORA_PATH"
  echo "[WARN] Please run: bash lora_finetune.sh"
fi

uv run run-distillation \
  --data-root data \
  --output-dir outputs/dermamnist_224_distill \
  --vae-model-id stabilityai/sd-vae-ft-mse \
  --vae-subfolder none \
  --diffusion-model-id runwayml/stable-diffusion-v1-5 \
  --lora-path "$LORA_PATH" \
  --lora-scale 0.9 \
  --prompt-conditioning \
  --guidance-scale 3.0 \
  --clusters-per-class 100 \
  --sde-steps 100 \
  --sde-noise-strength 0.2 \
  --encode-batch-size 64 \
  --decode-batch-size 32 \
  --train-epochs 30 \
  --train-batch-size 64 \
  --eval-batch-size 128 \
  --fp16 \
  "$@"
