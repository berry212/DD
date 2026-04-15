#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
DATA_ROOT="${DATA_ROOT:-${HF_DATASETS_CACHE:-${HF_HOME:-data}}}"
if [[ -n "${LORA_PATH:-}" ]]; then
  LORA_PATH="${LORA_PATH}"
else
  LORA_PATH="outputs/lora_${DATASET}"
  if [[ "$DATASET" == "dermamnist" && ! -d "$LORA_PATH" && -d "outputs/lora_dreammnist" ]]; then
    LORA_PATH="outputs/lora_dreammnist"
  fi
fi
IPC="${IPC:-100}"
TEACHER_BACKBONE="${TEACHER_BACKBONE:-resnet50}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${DATASET}_224_distill_ipc${IPC}}"
BASELINE_DIR="${BASELINE_DIR:-outputs/${DATASET}_224_distill_baseline}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
SDE_STEPS="${SDE_STEPS:-200}"

if [[ -z "${TEACHER_TEMPERATURE:-}" ]]; then
  if [[ "$DATASET" == "dermamnist" ]]; then
    TEACHER_TEMPERATURE="8.0"
  else
    TEACHER_TEMPERATURE="20.0"
  fi
fi

if [[ -z "${SDE_NOISE_STRENGTH:-}" ]]; then
  if [[ "$DATASET" == "dermamnist" ]]; then
    SDE_NOISE_STRENGTH="0.1"
  else
    SDE_NOISE_STRENGTH="0.2"
  fi
fi

if [[ -z "${CLVQ_MEDOID_ANCHOR:-}" ]]; then
  if [[ "$DATASET" == "dermamnist" ]]; then
    CLVQ_MEDOID_ANCHOR="0.65"
  else
    CLVQ_MEDOID_ANCHOR="0.0"
  fi
fi

if [[ ! -d "$LORA_PATH" ]]; then
  echo "[WARN] LoRA path not found: $LORA_PATH"
  echo "[WARN] Please run: bash lora_finetune.sh"
fi

uv run run-distillation \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --teacher-baseline-dir "$BASELINE_DIR" \
  --vae-model-id stabilityai/sd-vae-ft-mse \
  --diffusion-model-id runwayml/stable-diffusion-v1-5 \
  --lora-path "$LORA_PATH" \
  --lora-scale 0.9 \
  --guidance-scale "$GUIDANCE_SCALE" \
  --clusters-per-class "$IPC" \
  --clvq-medoid-anchor "$CLVQ_MEDOID_ANCHOR" \
  --teacher-backbone "$TEACHER_BACKBONE" \
  --teacher-epochs "$TEACHER_EPOCHS" \
  --teacher-temperature "$TEACHER_TEMPERATURE" \
  --sde-steps "$SDE_STEPS" \
  --sde-noise-strength "$SDE_NOISE_STRENGTH" \
  --encode-batch-size 32 \
  --decode-batch-size 32 \
  --fp16 \
  "$@"

echo "[INFO] Distillation finished."
echo "[INFO] Teacher baseline folder: ${BASELINE_DIR} (metrics: teacher_baseline_metrics.json)"
echo "[INFO] Train student with: uv run run-train-distilled-student --dataset ${DATASET} --data-root ${DATA_ROOT} --distilled-data ${OUTPUT_DIR}/distilled_data.pt --output-dir outputs/${DATASET}_224_student_ipc${IPC}"
