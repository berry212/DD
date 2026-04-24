#!/usr/bin/env bash
set -euo pipefail

# ✅ 新增：安全浮点数比较函数 (替代 Bash 原生 -le/-gt)
float_le() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !(a <= b) }'
}

DATASET="${DATASET:-dermamnist}"
DATASET="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
if [[ "$DATASET" == "odir5k" ]]; then
  DATASET="odir-5k"
fi
if [[ "$DATASET" == "aptos" || "$DATASET" == "aptos2019" || "$DATASET" == "aptos-2019" ]]; then
  DATASET="aptos-2019-blindness-detection"
fi

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
  if [[ "$DATASET" == "dermamnist" ]]; then TEACHER_TEMPERATURE="8.0"
  else TEACHER_TEMPERATURE="20.0"
  fi
fi

if [[ -z "${SDE_NOISE_STRENGTH:-}" ]]; then
  if [[ "$DATASET" == "dermamnist" ]]; then SDE_NOISE_STRENGTH="0.1"
  else SDE_NOISE_STRENGTH="0.2"
  fi
fi

if [[ -z "${CLVQ_MEDOID_ANCHOR:-}" ]]; then
  if [[ "$DATASET" == "dermamnist" ]]; then CLVQ_MEDOID_ANCHOR="0.65"
  else CLVQ_MEDOID_ANCHOR="0.0"
  fi
fi

# ✅ 修复：使用 float_le 替代 [[ "$IPC" -le 100 ]]
if [[ -z "${WEIGHT_COUNT_POWER:-}" ]]; then
  if [[ "$DATASET" == "dermamnist" ]]; then
    if float_le "$IPC" 100; then
      WEIGHT_COUNT_POWER="0.4"
    else
      WEIGHT_COUNT_POWER="0.5"
    fi
  else
    WEIGHT_COUNT_POWER="0.7"
  fi
fi

if [[ ! -d "$LORA_PATH" ]]; then
  echo "[WARN] LoRA path not found: $LORA_PATH"
  echo "[WARN] Please run: bash lora_finetune.sh"
fi

if [[ "$DATASET" == "aptos-2019-blindness-detection" ]]; then
  if [[ ! -f "$DATA_ROOT/train.csv" || ! -d "$DATA_ROOT/train_images" ]]; then
    if [[ ! -f "$DATA_ROOT/APTOS_2019_Blindness_Detection/train.csv" || ! -d "$DATA_ROOT/APTOS_2019_Blindness_Detection/train_images" ]]; then
      echo "[ERROR] APTOS-2019 dataset directory not found under DATA_ROOT: $DATA_ROOT"
      exit 1
    fi
  fi
fi

CLVQ_MEDOID_ANCHOR="0.0"
WEIGHT_COUNT_POWER="1.0"

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
  --weight-count-power "$WEIGHT_COUNT_POWER" \
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
echo "[INFO] Teacher baseline folder: ${BASELINE_DIR}"
echo "[INFO] Train student with: uv run run-train-distilled-student --dataset ${DATASET} --data-root ${DATA_ROOT} --distilled-data ${OUTPUT_DIR}/distilled_data.pt --output-dir outputs/${DATASET}_224_student_ipc${IPC}"