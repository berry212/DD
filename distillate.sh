#!/usr/bin/env bash
set -euo pipefail

LORA_PATH="${LORA_PATH:-outputs/lora_dreammnist}"
CLUSTERS_PER_CLASS="${CLUSTERS_PER_CLASS:-100}"
TEACHER_BACKBONE="${TEACHER_BACKBONE:-resnet50}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"

if [[ ! -d "$LORA_PATH" ]]; then
  echo "[WARN] LoRA path not found: $LORA_PATH"
  echo "[WARN] Please run: bash lora_finetune.sh"
fi

uv run run-distillation \
  --data-root data \
  --output-dir outputs/dermamnist_224_distill \
  --vae-model-id stabilityai/sd-vae-ft-mse \
  --diffusion-model-id runwayml/stable-diffusion-v1-5 \
  --lora-path "$LORA_PATH" \
  --lora-scale 0.9 \
  --guidance-scale 3.0 \
  --clusters-per-class "$CLUSTERS_PER_CLASS" \
  --teacher-backbone "$TEACHER_BACKBONE" \
  --teacher-epochs "$TEACHER_EPOCHS" \
  --sde-steps 200 \
  --sde-noise-strength 0.2 \
  --encode-batch-size 64 \
  --decode-batch-size 32 \
  --fp16 \
  "$@"

echo "[INFO] Distillation finished."
echo "[INFO] Train student with: uv run run-train-distilled-student --data-root data --distilled-data outputs/dermamnist_224_distill/distilled_data.pt --output-dir outputs/dermamnist_224_student"
