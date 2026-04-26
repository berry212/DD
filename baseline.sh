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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${DATASET}_224_distill_baseline}"
TEACHER_BACKBONE="${TEACHER_BACKBONE:-${BACKBONE:-resnet18}}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"

if [[ -z "${TEACHER_BATCH_SIZE:-}" ]]; then
    case "${TEACHER_BACKBONE}" in
        vit|vit_tiny|vit-tiny|vit_tiny_patch16_224)
            TEACHER_BATCH_SIZE="64"
            ;;
        *)
            TEACHER_BATCH_SIZE="128"
            ;;
    esac
fi

if [[ -z "${EVAL_BATCH_SIZE:-}" ]]; then
    case "${TEACHER_BACKBONE}" in
        vit|vit_tiny|vit-tiny|vit_tiny_patch16_224)
            EVAL_BATCH_SIZE="64"
            ;;
        *)
            EVAL_BATCH_SIZE="128"
            ;;
    esac
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

uv run run-baseline-resnet18 \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --teacher-backbone "$TEACHER_BACKBONE" \
    --teacher-epochs "$TEACHER_EPOCHS" \
    --teacher-batch-size "$TEACHER_BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    "$@"
