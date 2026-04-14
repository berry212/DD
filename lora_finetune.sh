#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
DATASET="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
if [[ "$DATASET" == "odir5k" ]]; then
  DATASET="odir-5k"
fi

DATA_ROOT="${DATA_ROOT:-${HF_DATASETS_CACHE:-${HF_HOME:-data}}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora_${DATASET}}"

if [[ "$DATASET" == "odir-5k" ]]; then
  if [[ ! -d "$DATA_ROOT/ODIR-5K" ]]; then
    echo "[ERROR] ODIR-5K dataset directory not found: $DATA_ROOT/ODIR-5K"
    echo "[ERROR] Please place ODIR-5K under ./data/ODIR-5K (or set DATA_ROOT accordingly)."
    exit 1
  fi
fi

uv run run-train-lora-sd \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
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
