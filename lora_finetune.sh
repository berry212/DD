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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora_${DATASET}}"
DIT_MODEL_ID="${DIT_MODEL_ID:-facebook/DiT-XL-2-256}"
VAE_MODEL_ID="${VAE_MODEL_ID:-}"
RESOLUTION="${RESOLUTION:-224}"

if [[ "$DATASET" == "odir-5k" ]]; then
  if [[ ! -d "$DATA_ROOT/ODIR-5K" ]]; then
    echo "[ERROR] ODIR-5K dataset directory not found: $DATA_ROOT/ODIR-5K"
    echo "[ERROR] Please place ODIR-5K under ./data/ODIR-5K (or set DATA_ROOT accordingly)."
    exit 1
  fi
fi

if [[ "$DATASET" == "aptos-2019-blindness-detection" ]]; then
  if [[ ! -f "$DATA_ROOT/train.csv" || ! -d "$DATA_ROOT/train_images" ]]; then
    if [[ ! -f "$DATA_ROOT/APTOS_2019_Blindness_Detection/train.csv" || ! -d "$DATA_ROOT/APTOS_2019_Blindness_Detection/train_images" ]]; then
      echo "[ERROR] APTOS-2019 dataset directory not found under DATA_ROOT: $DATA_ROOT"
      echo "[ERROR] Expected either:"
      echo "[ERROR]   - $DATA_ROOT/train.csv and $DATA_ROOT/train_images"
      echo "[ERROR]   - $DATA_ROOT/APTOS_2019_Blindness_Detection/train.csv and train_images"
      exit 1
    fi
  fi
fi

uv run run-train-lora-sd \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --dit-model-id "$DIT_MODEL_ID" \
  --vae-model-id "$VAE_MODEL_ID" \
  --resolution "$RESOLUTION" \
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
  --snr-gamma 5.0 \
  --noise-offset 0.05 \
  --max-grad-norm 1.0 \
  --fp16 \
  "$@"
