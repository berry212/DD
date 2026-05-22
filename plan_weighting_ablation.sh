#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Weighting Strategy Ablation: 3 datasets × 4 IPC × 3 strategies
#   uniform   = 均匀权重
#   heuristic = 正相关权重 (大簇权重大)
#   inverse   = 负相关权重 (小簇权重大)
# ============================================================

DATASETS=("dermamnist" "bloodmnist" "aptos-2019-blindness-detection")
IPCS=(200 100 50 10)
WEIGHTS=("uniform" "heuristic" "inverse")
# WEIGHTS=("heuristic" "inverse")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for DATASET in "${DATASETS[@]}"; do
  echo "=============================================="
  echo "  DATASET: $DATASET"
  echo "=============================================="

  for IPC in "${IPCS[@]}"; do
    for WEIGHT in "${WEIGHTS[@]}"; do

      DISTILL_OUT="outputs/${DATASET}_224_distill_ipc${IPC}_${WEIGHT}"
      STUDENT_OUT="outputs/${DATASET}_224_student_ipc${IPC}_${WEIGHT}"

      echo ""
      echo ">>> [${DATASET}] IPC=${IPC} WEIGHT=${WEIGHT}"

      # ── Distillation ──
      if [[ -f "${DISTILL_OUT}/distilled_data.pt" ]]; then
        echo "    [SKIP] distill output exists: ${DISTILL_OUT}/distilled_data.pt"
      else
        echo "    distill → ${DISTILL_OUT}"
        DATASET="$DATASET" \
        IPC="$IPC" \
        WEIGHTING_STRATEGY="$WEIGHT" \
        DISTILL_METHOD="clvq" \
        OUTPUT_DIR="$DISTILL_OUT" \
        DATA_ROOT="data" \
        bash "$SCRIPT_DIR/distillate.sh"
      fi

      # ── Student Training ──
      if [[ -f "${STUDENT_OUT}/summary.json" ]]; then
        echo "    [SKIP] student output exists: ${STUDENT_OUT}/summary.json"
      else
        echo "    student → ${STUDENT_OUT}"
        DATASET="$DATASET" \
        IPC="$IPC" \
        DATA_ROOT="data" \
        DISTILLED_DATA="${DISTILL_OUT}/distilled_data.pt" \
        OUTPUT_DIR="$STUDENT_OUT" \
        bash "$SCRIPT_DIR/train_student.sh"
      fi

      echo "    ✓ done"
    done
  done
done

echo ""
echo "=============================================="
echo "  ALL DONE"
echo "=============================================="
