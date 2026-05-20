#!/usr/bin/env bash
# ============================================================================
# 方法对比实验：D4M / MGD³ / DDOQ / Ours
#
# D4M:   guidance=off, uniform, hard label, classwise cluster (baseline)
# MGD³:  guidance=on,  uniform, soft label, classwise cluster
# DDOQ:  guidance=off, heuristic,soft label, classwise cluster
# Ours:  guidance=off, uniform, soft label, global cluster
#
# 用法：
#   bash contrast.sh                               # 默认 dermamnist IPC=100
#   DATASET=bloodmnist IPC=50 bash contrast.sh     # 指定数据集和 IPC
# ============================================================================
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
DATASET="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
if [[ "$DATASET" == "aptos" || "$DATASET" == "aptos2019" || "$DATASET" == "aptos-2019" ]]; then
  DATASET="aptos-2019-blindness-detection"
fi

DATA_ROOT="${DATA_ROOT:-data}"
IPC_LIST="${IPC:-10 50 100 200}"

TEACHER_BACKBONE="${TEACHER_BACKBONE:-resnet18}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-20}"
STUDENT_BATCH_SIZE="${STUDENT_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
STUDENT_LR="${STUDENT_LR:-4e-4}"
KD_TEMPERATURE="${KD_TEMPERATURE:-0}"
DISTILL_METHOD="clvq"

BASELINE_DIR="outputs/${DATASET}_224_distill_baseline"
LORA_PATH="outputs/lora_${DATASET}"

# ── 确保 teacher ──
if [[ ! -f "${BASELINE_DIR}/teacher_best.pt" ]]; then
  echo "[INFO] Training teacher baseline ..."
  DATASET="$DATASET" DATA_ROOT="$DATA_ROOT" OUTPUT_DIR="$BASELINE_DIR" \
    TEACHER_BACKBONE="$TEACHER_BACKBONE" TEACHER_EPOCHS="$TEACHER_EPOCHS" \
    bash baseline.sh
fi

RESULTS_DIR="outputs/contrast"
mkdir -p "$RESULTS_DIR"

# ═══════════════════════════════════════════════════════════════════
# Helper: run distillation if not exists + train student
# ═══════════════════════════════════════════════════════════════════

run_method() {
  local METHOD="$1"           # D4M / MGD3 / DDOQ / Ours
  local IPC="$2"
  local LORA_FLAG="$3"        # --lora-path value or ""
  local GUIDANCE_LAMBDA="$4"
  local WEIGHTING="$5"
  local HARD_ALPHA="$6"
  local GLOBAL_FLAG="$7"      # --use-global-cluster or --no-use-global-cluster

  local DISTILL_DIR="outputs/${DATASET}_224_distill_ipc${IPC}_contrast_${METHOD}"
  local STUDENT_DIR="outputs/${DATASET}_224_student_ipc${IPC}_contrast_${METHOD}"

  echo ""
  echo "===== Method: ${METHOD} | IPC=${IPC} ====="

  if [[ ! -f "${DISTILL_DIR}/distilled_data.pt" ]]; then
    uv run run-distillation \
      --dataset "$DATASET" --data-root "$DATA_ROOT" --output-dir "$DISTILL_DIR" \
      --teacher-baseline-dir "$BASELINE_DIR" \
      --vae-model-id stabilityai/sd-vae-ft-mse \
      --diffusion-model-id runwayml/stable-diffusion-v1-5 \
      $LORA_FLAG \
      --clusters-per-class "$IPC" --distill-method "$DISTILL_METHOD" \
      $GLOBAL_FLAG \
      --weighting-strategy "$WEIGHTING" --weight-smooth 0.0 \
      --teacher-backbone "$TEACHER_BACKBONE" --teacher-epochs "$TEACHER_EPOCHS" \
      --teacher-temperature 20.0 --no-auto-train-teacher-baseline \
      --sde-steps 200 --sde-noise-strength 0.2 --guidance-scale 3.0 \
      --mode-guidance-lambda "$GUIDANCE_LAMBDA" --mode-guidance-t-stop 80 \
      --best-of-n-candidates 1 \
      --encode-batch-size 32 --decode-batch-size 32 \
      --no-fkd-precompute-batches --fp16 --num-workers 4
  else
    echo "  [Distill] Already exists, skip: $DISTILL_DIR"
  fi

  uv run run-train-distilled-student \
    --dataset "$DATASET" --data-root "$DATA_ROOT" \
    --distilled-data "${DISTILL_DIR}/distilled_data.pt" --output-dir "$STUDENT_DIR" \
    --student-backbone "$TEACHER_BACKBONE" \
    --train-epochs "$STUDENT_EPOCHS" --train-batch-size "$STUDENT_BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" --train-lr "$STUDENT_LR" \
    --weight-decay 1e-4 --kd-temperature "$KD_TEMPERATURE" \
    --hard-label-alpha "$HARD_ALPHA" --weight-balance-alpha 0.0 --soft-label-sharpen 1.0 \
    --train-crop-min-scale 0.08 --train-crop-max-scale 1.0 --train-horizontal-flip-prob 0.5 \
    --no-use-fkd-batches --amp --num-workers 4

  # Print result
  local S="${STUDENT_DIR}/summary.json"
  python3 -c "
import json
d=json.load(open('$S'))
print(f'  {d[\"test_acc_at_best_val\"]:.4f}  (best_val={d[\"best_val_acc\"]:.4f}  auc={d.get(\"auc_macro\",\"nan\")})')
"
}

# ═══════════════════════════════════════════════════════════════════
# Run 4 methods × all IPCs
# ═══════════════════════════════════════════════════════════════════

for IPC in $IPC_LIST; do
  echo ""
  echo "############################################################"
  echo "  IPC = $IPC"
  echo "############################################################"

  # D4M: guidance=off, uniform, hard label (alpha=1.0), classwise
  run_method "D4M" "$IPC" \
    "--lora-path ${LORA_PATH} --lora-scale 0.9" \
    "0.0" "uniform" "1.0" "--no-use-global-cluster"

  # MGD³: guidance=on (λ=0.1), uniform, soft label, classwise
  run_method "MGD3" "$IPC" \
    "--lora-path ${LORA_PATH} --lora-scale 0.9" \
    "0.1" "uniform" "0.0" "--no-use-global-cluster"

  # DDOQ: guidance=off, heuristic, soft label, classwise
  run_method "DDOQ" "$IPC" \
    "--lora-path ${LORA_PATH} --lora-scale 0.9" \
    "0.0" "heuristic" "0.0" "--no-use-global-cluster"

  # Ours: guidance=off, uniform, soft label, global cluster
  run_method "Ours" "$IPC" \
    "--lora-path ${LORA_PATH} --lora-scale 0.9" \
    "0.0" "uniform" "0.0" "--use-global-cluster"
done

echo ""
echo "============================================================"
echo "  All done. Generating plots..."
echo "============================================================"

# ── 绘图：每个 IPC 两张图 ──
for IPC in $IPC_LIST; do
  uv run python3 plot_contrast.py \
    --dataset "$DATASET" --ipc "$IPC" \
    --d4m-json "outputs/${DATASET}_224_student_ipc${IPC}_contrast_D4M/student_history.json" \
    --mgd3-json "outputs/${DATASET}_224_student_ipc${IPC}_contrast_MGD3/student_history.json" \
    --ddoq-json "outputs/${DATASET}_224_student_ipc${IPC}_contrast_DDOQ/student_history.json" \
    --ours-json "outputs/${DATASET}_224_student_ipc${IPC}_contrast_Ours/student_history.json" \
    --output-acc "outputs/contrast/${DATASET}_ipc${IPC}_test_acc.png" \
    --output-loss "outputs/contrast/${DATASET}_ipc${IPC}_test_loss.png"
done

echo ""
echo "============================================================"
echo "  All plots saved → outputs/contrast/"
echo "============================================================"
ls -la outputs/contrast/${DATASET}_ipc*_test_*.png
