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
SEED="${SEED:-42}"

TEACHER_BACKBONE="${TEACHER_BACKBONE:-resnet50}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"
BACKBONE="${BACKBONE:-resnet50}"

ABLATION_ROOT="${ABLATION_ROOT:-outputs/${DATASET}_224_ablation_influence_ipc${IPC}}"
BASELINE_DIR="${BASELINE_DIR:-outputs/${DATASET}_224_distill_baseline}"

DISTILL_WITHOUT_DIR="${DISTILL_WITHOUT_DIR:-${ABLATION_ROOT}/distill_without_influence}"
DISTILL_WITH_DIR="${DISTILL_WITH_DIR:-${ABLATION_ROOT}/distill_with_influence}"
STUDENT_WITHOUT_DIR="${STUDENT_WITHOUT_DIR:-${ABLATION_ROOT}/student_without_influence}"
STUDENT_WITH_DIR="${STUDENT_WITH_DIR:-${ABLATION_ROOT}/student_with_influence}"

mkdir -p "$ABLATION_ROOT"

echo "[Ablation] dataset=$DATASET data_root=$DATA_ROOT ipc=$IPC seed=$SEED"
echo "[Ablation] output_root=$ABLATION_ROOT"

DISTILL_COMMON_ARGS=(
  --teacher-backbone "$TEACHER_BACKBONE"
  --teacher-epochs "$TEACHER_EPOCHS"
  --seed "$SEED"
)
if [[ -n "${TEACHER_TEMPERATURE:-}" ]]; then DISTILL_COMMON_ARGS+=(--teacher-temperature "$TEACHER_TEMPERATURE"); fi
if [[ -n "${SDE_STEPS:-}" ]]; then DISTILL_COMMON_ARGS+=(--sde-steps "$SDE_STEPS"); fi
if [[ -n "${SDE_NOISE_STRENGTH:-}" ]]; then DISTILL_COMMON_ARGS+=(--sde-noise-strength "$SDE_NOISE_STRENGTH"); fi
if [[ -n "${GUIDANCE_SCALE:-}" ]]; then DISTILL_COMMON_ARGS+=(--guidance-scale "$GUIDANCE_SCALE"); fi
if [[ -n "${CLVQ_MEDOID_ANCHOR:-}" ]]; then DISTILL_COMMON_ARGS+=(--clvq-medoid-anchor "$CLVQ_MEDOID_ANCHOR"); fi
if [[ -n "${INFLUENCE_BLEND_BETA:-}" ]]; then DISTILL_COMMON_ARGS+=(--influence-blend-beta "$INFLUENCE_BLEND_BETA"); fi
if [[ -n "${WEIGHT_COUNT_POWER:-}" ]]; then DISTILL_COMMON_ARGS+=(--weight-count-power "$WEIGHT_COUNT_POWER"); fi
if [[ -n "${WEIGHT_INFLUENCE_POWER:-}" ]]; then DISTILL_COMMON_ARGS+=(--weight-influence-power "$WEIGHT_INFLUENCE_POWER"); fi
if [[ -n "${INFLUENCE_QUANTILE:-}" ]]; then DISTILL_COMMON_ARGS+=(--influence-quantile "$INFLUENCE_QUANTILE"); fi
if [[ -n "${INFLUENCE_MIN_VALUE:-}" ]]; then DISTILL_COMMON_ARGS+=(--influence-min-value "$INFLUENCE_MIN_VALUE"); fi
if [[ -n "${INFLUENCE_MAX_VALUE:-}" ]]; then DISTILL_COMMON_ARGS+=(--influence-max-value "$INFLUENCE_MAX_VALUE"); fi

STUDENT_COMMON_ARGS=(
  --student-backbone "$BACKBONE"
  --seed "$SEED"
)
if [[ -n "${TRAIN_EPOCHS:-}" ]]; then STUDENT_COMMON_ARGS+=(--train-epochs "$TRAIN_EPOCHS"); fi
if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then STUDENT_COMMON_ARGS+=(--train-batch-size "$TRAIN_BATCH_SIZE"); fi
if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then STUDENT_COMMON_ARGS+=(--eval-batch-size "$EVAL_BATCH_SIZE"); fi
if [[ -n "${TRAIN_LR:-}" ]]; then STUDENT_COMMON_ARGS+=(--train-lr "$TRAIN_LR"); fi
if [[ -n "${WEIGHT_DECAY:-}" ]]; then STUDENT_COMMON_ARGS+=(--weight-decay "$WEIGHT_DECAY"); fi
if [[ -n "${KD_TEMPERATURE:-}" ]]; then STUDENT_COMMON_ARGS+=(--kd-temperature "$KD_TEMPERATURE"); fi
if [[ -n "${HARD_LABEL_ALPHA:-}" ]]; then STUDENT_COMMON_ARGS+=(--hard-label-alpha "$HARD_LABEL_ALPHA"); fi
if [[ -n "${WEIGHT_BALANCE_ALPHA:-}" ]]; then STUDENT_COMMON_ARGS+=(--weight-balance-alpha "$WEIGHT_BALANCE_ALPHA"); fi
if [[ -n "${SOFT_LABEL_SHARPEN:-}" ]]; then STUDENT_COMMON_ARGS+=(--soft-label-sharpen "$SOFT_LABEL_SHARPEN"); fi

echo "[Ablation] Round 1/4: distill without influence"
DATASET="$DATASET" \
DATA_ROOT="$DATA_ROOT" \
IPC="$IPC" \
BASELINE_DIR="$BASELINE_DIR" \
OUTPUT_DIR="$DISTILL_WITHOUT_DIR" \
INFLUENCE_MODE="none" \
bash distillate.sh "${DISTILL_COMMON_ARGS[@]}"

echo "[Ablation] Round 2/4: student without influence"
DATASET="$DATASET" \
DATA_ROOT="$DATA_ROOT" \
BACKBONE="$BACKBONE" \
IPC="$IPC" \
DISTILLED_DIR="$DISTILL_WITHOUT_DIR" \
OUTPUT_DIR="$STUDENT_WITHOUT_DIR" \
bash train_student.sh "${STUDENT_COMMON_ARGS[@]}"

echo "[Ablation] Round 3/4: distill with image-grad influence"
DATASET="$DATASET" \
DATA_ROOT="$DATA_ROOT" \
IPC="$IPC" \
BASELINE_DIR="$BASELINE_DIR" \
OUTPUT_DIR="$DISTILL_WITH_DIR" \
INFLUENCE_MODE="image-grad" \
bash distillate.sh "${DISTILL_COMMON_ARGS[@]}"

echo "[Ablation] Round 4/4: student with influence"
DATASET="$DATASET" \
DATA_ROOT="$DATA_ROOT" \
BACKBONE="$BACKBONE" \
IPC="$IPC" \
DISTILLED_DIR="$DISTILL_WITH_DIR" \
OUTPUT_DIR="$STUDENT_WITH_DIR" \
bash train_student.sh "${STUDENT_COMMON_ARGS[@]}"

echo "[Ablation] Generating comparison report"
uv run run-influence-ablation-report \
  --without-influence-distill-dir "$DISTILL_WITHOUT_DIR" \
  --with-influence-distill-dir "$DISTILL_WITH_DIR" \
  --without-influence-student-dir "$STUDENT_WITHOUT_DIR" \
  --with-influence-student-dir "$STUDENT_WITH_DIR" \
  --output-dir "$ABLATION_ROOT" \
  --fail-on-unfair

echo "[Ablation] Done. See:"
echo "[Ablation]   $ABLATION_ROOT/influence_ablation_report.json"
echo "[Ablation]   $ABLATION_ROOT/influence_ablation_report.md"
