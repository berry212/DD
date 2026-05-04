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

IPC="${IPC:-100}"
DISTILLED_DIR="${DISTILLED_DIR:-outputs/${DATASET}_224_distill_ipc${IPC}}"
DISTILLED_DATA="${DISTILLED_DATA:-${DISTILLED_DIR}/distilled_data.pt}"
METADATA="${METADATA:-${DISTILLED_DIR}/distilled_metadata.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${DISTILLED_DIR}}"

LABEL_SOURCE="${LABEL_SOURCE:-auto}"
MAX_POINTS="${MAX_POINTS:-3000}"
PCA_DIM="${PCA_DIM:-50}"
PERPLEXITY="${PERPLEXITY:-30}"
N_ITER="${N_ITER:-2000}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-128}"
DEVICE="${DEVICE:-auto}"
IMAGENET_PRETRAINED="${IMAGENET_PRETRAINED:-1}"
POINT_SIZE="${POINT_SIZE:-12}"
ALPHA="${ALPHA:-0.85}"
SEED="${SEED:-42}"

if [[ ! -f "$DISTILLED_DATA" ]]; then
  echo "[ERROR] distilled_data.pt not found: $DISTILLED_DATA"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

PRETRAINED_FLAG=(--imagenet-pretrained)
if [[ "$IMAGENET_PRETRAINED" == "0" || "$IMAGENET_PRETRAINED" == "false" || "$IMAGENET_PRETRAINED" == "False" ]]; then
  PRETRAINED_FLAG=(--no-imagenet-pretrained)
fi

uv run run-visualize-distilled-tsne \
  --distilled-data "$DISTILLED_DATA" \
  --metadata "$METADATA" \
  --output-dir "$OUTPUT_DIR" \
  --label-source "$LABEL_SOURCE" \
  --max-points "$MAX_POINTS" \
  --pca-dim "$PCA_DIM" \
  --perplexity "$PERPLEXITY" \
  --n-iter "$N_ITER" \
  --feature-batch-size "$FEATURE_BATCH_SIZE" \
  --device "$DEVICE" \
  "${PRETRAINED_FLAG[@]}" \
  --point-size "$POINT_SIZE" \
  --alpha "$ALPHA" \
  --seed "$SEED" \
  "$@"

echo "[INFO] t-SNE visualization complete."
echo "[INFO] Output dir: $OUTPUT_DIR"
