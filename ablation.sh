#!/usr/bin/env bash
# ============================================================================
# 消融实验：LoRA 微调 × 全局聚类 × 软标签
#
# 配置 A: 无 LoRA + 逐类聚类 + 硬标签
# 配置 B:   LoRA + 逐类聚类 + 硬标签
# 配置 C:   LoRA + 全局聚类 + 硬标签
# 配置 D:   LoRA + 全局聚类 + 软标签
#
# 用法：
#   bash ablation.sh                              # 默认 dermamnist
#   DATASET=bloodmnist bash ablation.sh           # 指定数据集
#   DATASET=aptos-2019-blindness-detection bash ablation.sh
# ============================================================================
set -euo pipefail

DATASET="${DATASET:-dermamnist}"
DATASET="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
if [[ "$DATASET" == "aptos" || "$DATASET" == "aptos2019" || "$DATASET" == "aptos-2019" ]]; then
  DATASET="aptos-2019-blindness-detection"
fi

DATA_ROOT="${DATA_ROOT:-data}"
IPC_LIST="${IPC:-10 50 100 200}"

# ── 公共参数 ──
TEACHER_BACKBONE="${TEACHER_BACKBONE:-resnet18}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-20}"
STUDENT_BATCH_SIZE="${STUDENT_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
STUDENT_LR="${STUDENT_LR:-4e-4}"
KD_TEMPERATURE="${KD_TEMPERATURE:-0}"
WEIGHTING_STRATEGY="uniform"
DISTILL_METHOD="clvq"

BASELINE_DIR="outputs/${DATASET}_224_distill_baseline"
LORA_PATH="outputs/lora_${DATASET}"

# ── 确保 teacher baseline 存在 ──
if [[ ! -f "${BASELINE_DIR}/teacher_best.pt" ]]; then
  echo "[INFO] Teacher missing, training baseline ..."
  DATASET="$DATASET" DATA_ROOT="$DATA_ROOT" OUTPUT_DIR="$BASELINE_DIR" \
    TEACHER_BACKBONE="$TEACHER_BACKBONE" TEACHER_EPOCHS="$TEACHER_EPOCHS" \
    bash baseline.sh
fi

RESULTS_DIR="outputs/ablation"
mkdir -p "$RESULTS_DIR"

SUMMARY_CSV="${RESULTS_DIR}/${DATASET}_ablation.csv"
echo "config,lora,global_cluster,soft_label,ipc,test_acc,best_val_acc,test_auc_macro,num_distilled" > "$SUMMARY_CSV"

# ═══════════════════════════════════════════════════════════════════════════
# 按配置和 IPC 遍历
# ═══════════════════════════════════════════════════════════════════════════

for IPC in $IPC_LIST; do

  # ── 配置 A: 无 LoRA, 逐类聚类, 硬标签 ──
  CONFIG="A"
  DISTILL_SUFFIX="_A_nolora_classwise_hard"
  STUDENT_SUFFIX="_A_nolora_classwise_hard"
  DISTILL_DIR="outputs/${DATASET}_224_distill_ipc${IPC}${DISTILL_SUFFIX}"
  STUDENT_DIR="outputs/${DATASET}_224_student_ipc${IPC}${STUDENT_SUFFIX}"

  echo ""
  echo "===== Config ${CONFIG}: no-LoRA + classwise + hard label | IPC=${IPC} ====="

  if [[ ! -f "${DISTILL_DIR}/distilled_data.pt" ]]; then
    uv run run-distillation \
      --dataset "$DATASET" --data-root "$DATA_ROOT" --output-dir "$DISTILL_DIR" \
      --teacher-baseline-dir "$BASELINE_DIR" \
      --vae-model-id stabilityai/sd-vae-ft-mse \
      --diffusion-model-id runwayml/stable-diffusion-v1-5 \
      --lora-path "" \
      --clusters-per-class "$IPC" --distill-method "$DISTILL_METHOD" \
      --no-use-global-cluster \
      --weighting-strategy "$WEIGHTING_STRATEGY" --weight-smooth 0.0 \
      --teacher-backbone "$TEACHER_BACKBONE" --teacher-epochs "$TEACHER_EPOCHS" \
      --teacher-temperature 20.0 --no-auto-train-teacher-baseline \
      --sde-steps 200 --sde-noise-strength 0.2 --guidance-scale 3.0 \
      --mode-guidance-lambda 0.0 \
      --encode-batch-size 32 --decode-batch-size 32 \
      --no-fkd-precompute-batches --fp16 --num-workers 4
  else
    echo "[Distill] Already exists, skip: $DISTILL_DIR"
  fi

  uv run run-train-distilled-student \
    --dataset "$DATASET" --data-root "$DATA_ROOT" \
    --distilled-data "${DISTILL_DIR}/distilled_data.pt" --output-dir "$STUDENT_DIR" \
    --student-backbone "$TEACHER_BACKBONE" \
    --train-epochs "$STUDENT_EPOCHS" --train-batch-size "$STUDENT_BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" --train-lr "$STUDENT_LR" \
    --weight-decay 1e-4 --kd-temperature "$KD_TEMPERATURE" \
    --hard-label-alpha 1.0 --weight-balance-alpha 0.0 --soft-label-sharpen 1.0 \
    --train-crop-min-scale 0.08 --train-crop-max-scale 1.0 --train-horizontal-flip-prob 0.5 \
    --no-use-fkd-batches --amp --num-workers 4

  SUMMARY_JSON="${STUDENT_DIR}/summary.json"
  TEST_ACC=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['test_acc_at_best_val'])")
  BEST_VAL=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['best_val_acc'])")
  AUC=$(python3 -c "import json; d=json.load(open('$SUMMARY_JSON')); print(d.get('auc_macro','nan'))")
  N_DIST=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['num_distilled'])")
  echo "$CONFIG,no,classwise,no,$IPC,$TEST_ACC,$BEST_VAL,$AUC,$N_DIST" >> "$SUMMARY_CSV"
  echo "  Result: test_acc=$TEST_ACC  best_val=$BEST_VAL  auc=$AUC"

  # ── 配置 B: LoRA, 逐类聚类, 硬标签 ──
  CONFIG="B"
  DISTILL_SUFFIX="_B_lora_classwise_hard"
  STUDENT_SUFFIX="_B_lora_classwise_hard"
  DISTILL_DIR="outputs/${DATASET}_224_distill_ipc${IPC}${DISTILL_SUFFIX}"
  STUDENT_DIR="outputs/${DATASET}_224_student_ipc${IPC}${STUDENT_SUFFIX}"

  echo ""
  echo "===== Config ${CONFIG}: LoRA + classwise + hard label | IPC=${IPC} ====="

  if [[ ! -f "${DISTILL_DIR}/distilled_data.pt" ]]; then
    uv run run-distillation \
      --dataset "$DATASET" --data-root "$DATA_ROOT" --output-dir "$DISTILL_DIR" \
      --teacher-baseline-dir "$BASELINE_DIR" \
      --vae-model-id stabilityai/sd-vae-ft-mse \
      --diffusion-model-id runwayml/stable-diffusion-v1-5 \
      --lora-path "$LORA_PATH" --lora-scale 0.9 \
      --clusters-per-class "$IPC" --distill-method "$DISTILL_METHOD" \
      --no-use-global-cluster \
      --weighting-strategy "$WEIGHTING_STRATEGY" --weight-smooth 0.0 \
      --teacher-backbone "$TEACHER_BACKBONE" --teacher-epochs "$TEACHER_EPOCHS" \
      --teacher-temperature 20.0 --no-auto-train-teacher-baseline \
      --sde-steps 200 --sde-noise-strength 0.2 --guidance-scale 3.0 \
      --mode-guidance-lambda 0.0 \
      --encode-batch-size 32 --decode-batch-size 32 \
      --no-fkd-precompute-batches --fp16 --num-workers 4
  else
    echo "[Distill] Already exists, skip: $DISTILL_DIR"
  fi

  uv run run-train-distilled-student \
    --dataset "$DATASET" --data-root "$DATA_ROOT" \
    --distilled-data "${DISTILL_DIR}/distilled_data.pt" --output-dir "$STUDENT_DIR" \
    --student-backbone "$TEACHER_BACKBONE" \
    --train-epochs "$STUDENT_EPOCHS" --train-batch-size "$STUDENT_BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" --train-lr "$STUDENT_LR" \
    --weight-decay 1e-4 --kd-temperature "$KD_TEMPERATURE" \
    --hard-label-alpha 1.0 --weight-balance-alpha 0.0 --soft-label-sharpen 1.0 \
    --train-crop-min-scale 0.08 --train-crop-max-scale 1.0 --train-horizontal-flip-prob 0.5 \
    --no-use-fkd-batches --amp --num-workers 4

  SUMMARY_JSON="${STUDENT_DIR}/summary.json"
  TEST_ACC=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['test_acc_at_best_val'])")
  BEST_VAL=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['best_val_acc'])")
  AUC=$(python3 -c "import json; d=json.load(open('$SUMMARY_JSON')); print(d.get('auc_macro','nan'))")
  N_DIST=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['num_distilled'])")
  echo "$CONFIG,yes,classwise,no,$IPC,$TEST_ACC,$BEST_VAL,$AUC,$N_DIST" >> "$SUMMARY_CSV"
  echo "  Result: test_acc=$TEST_ACC  best_val=$BEST_VAL  auc=$AUC"

  # ── 配置 C: LoRA, 全局聚类, 硬标签 ──
  CONFIG="C"
  DISTILL_SUFFIX="_C_lora_global_hard"
  STUDENT_SUFFIX="_C_lora_global_hard"
  DISTILL_DIR="outputs/${DATASET}_224_distill_ipc${IPC}${DISTILL_SUFFIX}"
  STUDENT_DIR="outputs/${DATASET}_224_student_ipc${IPC}${STUDENT_SUFFIX}"

  echo ""
  echo "===== Config ${CONFIG}: LoRA + global + hard label | IPC=${IPC} ====="

  if [[ ! -f "${DISTILL_DIR}/distilled_data.pt" ]]; then
    uv run run-distillation \
      --dataset "$DATASET" --data-root "$DATA_ROOT" --output-dir "$DISTILL_DIR" \
      --teacher-baseline-dir "$BASELINE_DIR" \
      --vae-model-id stabilityai/sd-vae-ft-mse \
      --diffusion-model-id runwayml/stable-diffusion-v1-5 \
      --lora-path "$LORA_PATH" --lora-scale 0.9 \
      --clusters-per-class "$IPC" --distill-method "$DISTILL_METHOD" \
      --use-global-cluster \
      --weighting-strategy "$WEIGHTING_STRATEGY" --weight-smooth 0.0 \
      --teacher-backbone "$TEACHER_BACKBONE" --teacher-epochs "$TEACHER_EPOCHS" \
      --teacher-temperature 20.0 --no-auto-train-teacher-baseline \
      --sde-steps 200 --sde-noise-strength 0.2 --guidance-scale 3.0 \
      --mode-guidance-lambda 0.0 \
      --encode-batch-size 32 --decode-batch-size 32 \
      --no-fkd-precompute-batches --fp16 --num-workers 4
  else
    echo "[Distill] Already exists, skip: $DISTILL_DIR"
  fi

  uv run run-train-distilled-student \
    --dataset "$DATASET" --data-root "$DATA_ROOT" \
    --distilled-data "${DISTILL_DIR}/distilled_data.pt" --output-dir "$STUDENT_DIR" \
    --student-backbone "$TEACHER_BACKBONE" \
    --train-epochs "$STUDENT_EPOCHS" --train-batch-size "$STUDENT_BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" --train-lr "$STUDENT_LR" \
    --weight-decay 1e-4 --kd-temperature "$KD_TEMPERATURE" \
    --hard-label-alpha 1.0 --weight-balance-alpha 0.0 --soft-label-sharpen 1.0 \
    --train-crop-min-scale 0.08 --train-crop-max-scale 1.0 --train-horizontal-flip-prob 0.5 \
    --no-use-fkd-batches --amp --num-workers 4

  SUMMARY_JSON="${STUDENT_DIR}/summary.json"
  TEST_ACC=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['test_acc_at_best_val'])")
  BEST_VAL=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['best_val_acc'])")
  AUC=$(python3 -c "import json; d=json.load(open('$SUMMARY_JSON')); print(d.get('auc_macro','nan'))")
  N_DIST=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['num_distilled'])")
  echo "$CONFIG,yes,global,no,$IPC,$TEST_ACC,$BEST_VAL,$AUC,$N_DIST" >> "$SUMMARY_CSV"
  echo "  Result: test_acc=$TEST_ACC  best_val=$BEST_VAL  auc=$AUC"

  # ── 配置 D: LoRA, 全局聚类, 软标签 ──
  CONFIG="D"
  DISTILL_SUFFIX="_D_lora_global_soft"
  STUDENT_SUFFIX="_D_lora_global_soft"
  DISTILL_DIR="outputs/${DATASET}_224_distill_ipc${IPC}${DISTILL_SUFFIX}"
  STUDENT_DIR="outputs/${DATASET}_224_student_ipc${IPC}${STUDENT_SUFFIX}"

  echo ""
  echo "===== Config ${CONFIG}: LoRA + global + soft label | IPC=${IPC} ====="

  if [[ ! -f "${DISTILL_DIR}/distilled_data.pt" ]]; then
    uv run run-distillation \
      --dataset "$DATASET" --data-root "$DATA_ROOT" --output-dir "$DISTILL_DIR" \
      --teacher-baseline-dir "$BASELINE_DIR" \
      --vae-model-id stabilityai/sd-vae-ft-mse \
      --diffusion-model-id runwayml/stable-diffusion-v1-5 \
      --lora-path "$LORA_PATH" --lora-scale 0.9 \
      --clusters-per-class "$IPC" --distill-method "$DISTILL_METHOD" \
      --use-global-cluster \
      --weighting-strategy "$WEIGHTING_STRATEGY" --weight-smooth 0.0 \
      --teacher-backbone "$TEACHER_BACKBONE" --teacher-epochs "$TEACHER_EPOCHS" \
      --teacher-temperature 20.0 --no-auto-train-teacher-baseline \
      --sde-steps 200 --sde-noise-strength 0.2 --guidance-scale 3.0 \
      --mode-guidance-lambda 0.0 \
      --encode-batch-size 32 --decode-batch-size 32 \
      --no-fkd-precompute-batches --fp16 --num-workers 4
  else
    echo "[Distill] Already exists, skip: $DISTILL_DIR"
  fi

  uv run run-train-distilled-student \
    --dataset "$DATASET" --data-root "$DATA_ROOT" \
    --distilled-data "${DISTILL_DIR}/distilled_data.pt" --output-dir "$STUDENT_DIR" \
    --student-backbone "$TEACHER_BACKBONE" \
    --train-epochs "$STUDENT_EPOCHS" --train-batch-size "$STUDENT_BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" --train-lr "$STUDENT_LR" \
    --weight-decay 1e-4 --kd-temperature "$KD_TEMPERATURE" \
    --hard-label-alpha 0.0 --weight-balance-alpha 0.0 --soft-label-sharpen 1.0 \
    --train-crop-min-scale 0.08 --train-crop-max-scale 1.0 --train-horizontal-flip-prob 0.5 \
    --no-use-fkd-batches --amp --num-workers 4

  SUMMARY_JSON="${STUDENT_DIR}/summary.json"
  TEST_ACC=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['test_acc_at_best_val'])")
  BEST_VAL=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['best_val_acc'])")
  AUC=$(python3 -c "import json; d=json.load(open('$SUMMARY_JSON')); print(d.get('auc_macro','nan'))")
  N_DIST=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['num_distilled'])")
  echo "$CONFIG,yes,global,yes,$IPC,$TEST_ACC,$BEST_VAL,$AUC,$N_DIST" >> "$SUMMARY_CSV"
  echo "  Result: test_acc=$TEST_ACC  best_val=$BEST_VAL  auc=$AUC"

done

echo ""
echo "============================================================"
echo "  Ablation complete!  Results → $SUMMARY_CSV"
echo "============================================================"
column -t -s, "$SUMMARY_CSV"
