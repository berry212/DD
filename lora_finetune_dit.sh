#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LoRA fine-tune a DiT (PixArt-α) model on medical datasets.
#
# Memory-optimised defaults for 12 GB VRAM (RTX 4070 Super):
#   batch_size=1  grad_accum=8  rank=8  fp16  resolution=256
#
# Usage:
#   DATASET=bloodmnist bash lora_finetune_dit.sh
#   DATASET=dermamnist DIT_MODEL_ID=PixArt-alpha/PixArt-XL-2-1024-MS bash lora_finetune_dit.sh
# ============================================================

DATASET="${DATASET:-dermamnist}"
DATASET="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
if [[ "$DATASET" == "odir5k" ]]; then
  DATASET="odir-5k"
fi
if [[ "$DATASET" == "aptos" || "$DATASET" == "aptos2019" || "$DATASET" == "aptos-2019" ]]; then
  DATASET="aptos-2019-blindness-detection"
fi

DATA_ROOT="${DATA_ROOT:-${HF_DATASETS_CACHE:-${HF_HOME:-data}}}"
DIT_MODEL_ID="${DIT_MODEL_ID:-PixArt-alpha/PixArt-XL-2-1024-MS}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora_dit_${DATASET}}"

# ---- memory-safe defaults (12 GB) ----
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
RESOLUTION="${RESOLUTION:-256}"
RANK="${RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LR="${LR:-5e-5}"
EPOCHS="${EPOCHS:-10}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1500}"
LOG_STEPS="${LOG_STEPS:-10}"

# ---- dataset validation ----
if [[ "$DATASET" == "odir-5k" ]]; then
  if [[ ! -d "$DATA_ROOT/ODIR-5K" ]]; then
    echo "[ERROR] ODIR-5K dataset directory not found: $DATA_ROOT/ODIR-5K"
    exit 1
  fi
fi

if [[ "$DATASET" == "aptos-2019-blindness-detection" ]]; then
  if [[ ! -f "$DATA_ROOT/train.csv" || ! -d "$DATA_ROOT/train_images" ]]; then
    if [[ ! -f "$DATA_ROOT/APTOS_2019_Blindness_Detection/train.csv" || ! -d "$DATA_ROOT/APTOS_2019_Blindness_Detection/train_images" ]]; then
      echo "[ERROR] APTOS-2019 dataset directory not found under DATA_ROOT: $DATA_ROOT"
      exit 1
    fi
  fi
fi

echo "============================================"
echo "[DiT LoRA] dataset       = $DATASET"
echo "[DiT LoRA] model         = $DIT_MODEL_ID"
echo "[DiT LoRA] output        = $OUTPUT_DIR"
echo "[DiT LoRA] batch_size    = $BATCH_SIZE"
echo "[DiT LoRA] grad_accum    = $GRAD_ACCUM"
echo "[DiT LoRA] resolution    = $RESOLUTION"
echo "[DiT LoRA] rank          = $RANK"
echo "[DiT LoRA] lora_alpha    = $LORA_ALPHA"
echo "[DiT LoRA] effective_bs  = $((BATCH_SIZE * GRAD_ACCUM))"
echo "============================================"

uv run run-train-lora-sd \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --diffusion-model-id "$DIT_MODEL_ID" \
  --backbone-type dit \
  --resolution "$RESOLUTION" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --epochs "$EPOCHS" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --rank "$RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lr "$LR" \
  --lr-schedule cosine \
  --lr-warmup-steps 100 \
  --class-balance \
  --prompt-dropout-prob 0.1 \
  --snr-gamma 5.0 \
  --noise-offset 0.05 \
  --max-grad-norm 1.0 \
  --log-steps "$LOG_STEPS" \
  --fp16 \
  --num-workers 2 \
  "$@"

echo "[DiT LoRA] Finished. Weights saved to $OUTPUT_DIR"
