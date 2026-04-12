#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
LORA_PATH="${LORA_PATH:-outputs/lora_dreammnist}"
IPC="${IPC:-100}"
TEACHER_BACKBONE="${TEACHER_BACKBONE:-resnet50}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${DATASET}_224_distill_ipc${IPC}}"
BASELINE_DIR="${BASELINE_DIR:-outputs/${DATASET}_224_distill_baseline}"

if [[ ! -d "$LORA_PATH" ]]; then
  echo "[WARN] LoRA path not found: $LORA_PATH"
  echo "[WARN] Please run: bash lora_finetune.sh"
fi

uv run run-distillation \
  --dataset "$DATASET" \
  --data-root data \
  --output-dir "$OUTPUT_DIR" \
  --teacher-baseline-dir "$BASELINE_DIR" \
  --vae-model-id stabilityai/sd-vae-ft-mse \
  --diffusion-model-id runwayml/stable-diffusion-v1-5 \
  --lora-path "$LORA_PATH" \
  --lora-scale 0.9 \
  --guidance-scale 3.0 \
  --clusters-per-class "$IPC" \
  --teacher-backbone "$TEACHER_BACKBONE" \
  --teacher-epochs "$TEACHER_EPOCHS" \
  --sde-steps 200 \
  --sde-noise-strength 0.2 \
  --encode-batch-size 64 \
  --decode-batch-size 32 \
  --fp16 \
  "$@"

echo "[INFO] Distillation finished."
echo "[INFO] Teacher baseline folder: ${BASELINE_DIR} (metrics: teacher_baseline_metrics.json)"
echo "[INFO] Train student with: uv run run-train-distilled-student --dataset ${DATASET} --data-root data --distilled-data ${OUTPUT_DIR}/distilled_data.pt --output-dir outputs/${DATASET}_224_student"
