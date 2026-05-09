#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
DATASET="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
if [[ "$DATASET" == "odir5k" ]]; then
  DATASET="odir-5k"
fi
if [[ "$DATASET" == "aptos" || "$DATASET" == "aptos2019" || "$DATASET" == "aptos-2019" ]]; then
  DATASET="aptos-2019-blindness-detection"
fi

DATA_ROOT="${DATA_ROOT:-${HF_DATASETS_CACHE:-${HF_HOME:-data}}}"
IPC="${IPC:-100}"
DISTILL_METHOD="${DISTILL_METHOD:-clvq}"
KMEANS_MAX_ITER="${KMEANS_MAX_ITER:-300}"
CLVQ_MAX_ITER="${CLVQ_MAX_ITER:-10000}"
CLVQ_BATCH_SIZE="${CLVQ_BATCH_SIZE:-1024}"
TEACHER_BACKBONE="${TEACHER_BACKBONE:-resnet18}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${DATASET}_224_distill_ipc${IPC}}"
BASELINE_DIR="${BASELINE_DIR:-outputs/${DATASET}_224_distill_baseline}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
SDE_STEPS="${SDE_STEPS:-200}"
SDE_NOISE_STRENGTH="${SDE_NOISE_STRENGTH:-0.2}"
CLVQ_MEDOID_ANCHOR="${CLVQ_MEDOID_ANCHOR:-0.0}"
WEIGHTING_STRATEGY="${WEIGHTING_STRATEGY:-heuristic}"
MODEL_TYPE="${MODEL_TYPE:-sd}"

if [[ "$MODEL_TYPE" == "dit" ]]; then
  DIFFUSION_MODEL_ID="facebook/DiT-XL-2-256"
  MODEL_ARGS=(--model-type dit)
else
  DIFFUSION_MODEL_ID="runwayml/stable-diffusion-v1-5"
  MODEL_ARGS=()
fi

TRAIN_EPOCHS="${TRAIN_EPOCHS:-300}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TRAIN_CROP_MIN_SCALE="${TRAIN_CROP_MIN_SCALE:-0.08}"
TRAIN_CROP_MAX_SCALE="${TRAIN_CROP_MAX_SCALE:-1.0}"
TRAIN_HFLIP_PROB="${TRAIN_HFLIP_PROB:-0.5}"
FKD_PRECOMPUTE_BATCHES="${FKD_PRECOMPUTE_BATCHES:-true}"
AUTO_TRAIN_TEACHER_BASELINE="${AUTO_TRAIN_TEACHER_BASELINE:-true}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-20.0}"

if [[ -n "${LORA_PATH:-}" ]]; then
  LORA_PATH="${LORA_PATH}"
else
  LORA_PATH="outputs/lora_${DATASET}"
  if [[ "$DATASET" == "dermamnist" && ! -d "$LORA_PATH" && -d "outputs/lora_dreammnist" ]]; then
    LORA_PATH="outputs/lora_dreammnist"
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

if [[ "$TEACHER_BACKBONE" != "resnet18" ]]; then
  echo "[WARN] Paper-aligned protocol uses a ResNet-18 teacher. Current TEACHER_BACKBONE=${TEACHER_BACKBONE}"
fi

if [[ ! -f "${BASELINE_DIR}/teacher_best.pt" ]]; then
  if [[ "$AUTO_TRAIN_TEACHER_BASELINE" == "true" ]]; then
    echo "[INFO] Teacher checkpoint missing. Training baseline teacher at ${BASELINE_DIR}"
    DATASET="$DATASET" \
    DATA_ROOT="$DATA_ROOT" \
    OUTPUT_DIR="$BASELINE_DIR" \
    TEACHER_BACKBONE="$TEACHER_BACKBONE" \
    TEACHER_EPOCHS="$TEACHER_EPOCHS" \
    bash baseline.sh
  else
    echo "[ERROR] Missing teacher checkpoint: ${BASELINE_DIR}/teacher_best.pt"
    echo "[ERROR] Run: DATASET=${DATASET} DATA_ROOT=${DATA_ROOT} OUTPUT_DIR=${BASELINE_DIR} TEACHER_BACKBONE=${TEACHER_BACKBONE} bash baseline.sh"
    exit 1
  fi
fi

FKD_PRECOMPUTE_FLAG="--fkd-precompute-batches"
if [[ "$FKD_PRECOMPUTE_BATCHES" == "false" ]]; then
  FKD_PRECOMPUTE_FLAG="--no-fkd-precompute-batches"
fi

uv run run-distillation \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --teacher-baseline-dir "$BASELINE_DIR" \
  --vae-model-id stabilityai/sd-vae-ft-mse \
  --diffusion-model-id "$DIFFUSION_MODEL_ID" \
  "${MODEL_ARGS[@]}" \
  --lora-path "$LORA_PATH" \
  --lora-scale 0.9 \
  --guidance-scale "$GUIDANCE_SCALE" \
  --clusters-per-class "$IPC" \
  --distill-method "$DISTILL_METHOD" \
  --kmeans-max-iter "$KMEANS_MAX_ITER" \
  --clvq-max-iter "$CLVQ_MAX_ITER" \
  --clvq-batch-size "$CLVQ_BATCH_SIZE" \
  --clvq-medoid-anchor "$CLVQ_MEDOID_ANCHOR" \
  --weighting-strategy "$WEIGHTING_STRATEGY" \
  --teacher-backbone "$TEACHER_BACKBONE" \
  --teacher-epochs "$TEACHER_EPOCHS" \
  --teacher-temperature "$TEACHER_TEMPERATURE" \
  --no-auto-train-teacher-baseline \
  --sde-steps "$SDE_STEPS" \
  --sde-noise-strength "$SDE_NOISE_STRENGTH" \
  --encode-batch-size 32 \
  --decode-batch-size 4 \
  "$FKD_PRECOMPUTE_FLAG" \
  --fkd-train-epochs "$TRAIN_EPOCHS" \
  --fkd-batch-size "$TRAIN_BATCH_SIZE" \
  --fkd-crop-min-scale "$TRAIN_CROP_MIN_SCALE" \
  --fkd-crop-max-scale "$TRAIN_CROP_MAX_SCALE" \
  --fkd-horizontal-flip-prob "$TRAIN_HFLIP_PROB" \
  --fp16 \
  "$@"

echo "[INFO] Distillation finished."
echo "[INFO] Teacher baseline folder: ${BASELINE_DIR}"
echo "[INFO] Train student with: uv run run-train-distilled-student --dataset ${DATASET} --data-root ${DATA_ROOT} --distilled-data ${OUTPUT_DIR}/distilled_data.pt --output-dir outputs/${DATASET}_224_student_ipc${IPC}"
