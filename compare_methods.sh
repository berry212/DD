#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Distillation Method Comparison: Random vs KMeans vs Ours (CLVQ)
#
# 在三个数据集 (dermamnist, bloodmnist, aptos-2019-blindness-detection)
# 上对三种蒸馏方法做对比实验：
#   - Random: 按类别随机选取 → 按类别聚类，学生训练用硬标签
#   - KMeans: 按类别 K-Means 聚类 → 按类别聚类，学生训练用硬标签
#   - Ours:   CLVQ (Class-aware Learned Vector Quantization) → 软标签
#
# IPC 默认: 10 50 100 200
#
# 使用方式:
#   bash compare_methods.sh                           # 全部三个数据集
#   DATASETS="dermamnist" bash compare_methods.sh     # 仅单个数据集
#   IPCS="10 50" bash compare_methods.sh              # 仅部分 IPC
#   DRY_RUN=1 bash compare_methods.sh                 # 预览将要执行的命令
# ============================================================

DATASETS=("${DATASETS:-dermamnist bloodmnist aptos-2019-blindness-detection}")
IPCS=("${IPCS:-200 100 50 10}")
# METHODS=("random" "kmeans" "clvq")          # clvq = Ours
# METHOD_LABELS=("Random" "KMeans" "Ours")
METHODS=("random" "kmeans")          # clvq = Ours
METHOD_LABELS=("Random" "KMeans")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-data}"
WEIGHTING_STRATEGY="${WEIGHTING_STRATEGY:-uniform}"   # 统一用 uniform 权重做公平对比
DRY_RUN="${DRY_RUN:-0}"

# ── 颜色 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║  Distillation Method Comparison                     ║${NC}"
echo -e "${BOLD}${CYAN}║  Random  vs  KMeans  vs  Ours (CLVQ)                ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Datasets:    ${BLUE}${DATASETS[*]}${NC}"
echo -e "  IPCs:        ${BLUE}${IPCS[*]}${NC}"
echo -e "  Methods:     ${BLUE}${METHOD_LABELS[*]}${NC}"
echo -e "  Weighting:   ${BLUE}${WEIGHTING_STRATEGY}${NC}"
echo -e "  Label mode:  ${BLUE}Random/KMeans=Hard  |  Ours=Soft${NC}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo -e "  Mode:        ${YELLOW}DRY RUN (preview only)${NC}"
fi
echo ""

# ── 统计 ──
TOTAL_TASKS=0
COMPLETED=0
SKIPPED=0
FAILED=0

for DATASET in ${DATASETS[*]}; do
  echo -e "${BOLD}${BLUE}┌─────────────────────────────────────────────────────┐${NC}"
  echo -e "${BOLD}${BLUE}│  DATASET: ${DATASET}${NC}"
  echo -e "${BOLD}${BLUE}└─────────────────────────────────────────────────────┘${NC}"

  # 检查 baseline teacher 是否存在
  BASELINE_DIR="outputs/${DATASET}_224_distill_baseline"
  if [[ ! -f "${BASELINE_DIR}/teacher_best.pt" ]]; then
    echo -e "  ${YELLOW}[WARN] Baseline teacher not found at ${BASELINE_DIR}${NC}"
    echo -e "  ${YELLOW}       distillate.sh will auto-train it if AUTO_TRAIN_TEACHER_BASELINE=true${NC}"
  fi

  for IPC in ${IPCS[*]}; do
    for METHOD in ${METHODS[*]}; do
      TOTAL_TASKS=$((TOTAL_TASKS + 1))

      # 方法标签 & 学生训练参数
      case "$METHOD" in
        random)
          METHOD_LABEL="Random"
          # Random 用硬标签: 对 teacher 输出的 argmax 做交叉熵
          HARD_LABEL_ALPHA="1.0"
          USE_FKD="false"
          ;;
        kmeans)
          METHOD_LABEL="KMeans"
          # KMeans 用硬标签
          HARD_LABEL_ALPHA="1.0"
          USE_FKD="false"
          ;;
        clvq)
          METHOD_LABEL="Ours"
          # Ours (CLVQ) 用软标签 — 知识蒸馏
          HARD_LABEL_ALPHA="0.0"
          USE_FKD="true"
          ;;
        *)
          METHOD_LABEL="$METHOD"
          HARD_LABEL_ALPHA="0.0"
          USE_FKD="true"
          ;;
      esac

      # ── 输出目录命名 ──
      # random/kmeans 现在使用按类别聚类（classwise），目录名加后缀区分
      if [[ "$METHOD" == "clvq" ]]; then
        DISTILL_OUT="outputs/${DATASET}_224_distill_ipc${IPC}_method-${METHOD}-softlabel"
        STUDENT_OUT="outputs/${DATASET}_224_student_ipc${IPC}_method-${METHOD}-softlabel"
      else
        DISTILL_OUT="outputs/${DATASET}_224_distill_ipc${IPC}_method-${METHOD}-classwise"
        STUDENT_OUT="outputs/${DATASET}_224_student_ipc${IPC}_method-${METHOD}-classwise"
      fi

      echo ""
      echo -e "  ${BOLD}[${DATASET}] IPC=${IPC}  ${METHOD_LABEL} (${METHOD})  label=${HARD_LABEL_ALPHA:0:1}hard${NC}"

      # ══════════════════════════════════════════
      # Step 1: Distillation
      # ══════════════════════════════════════════
      if [[ -f "${DISTILL_OUT}/distilled_data.pt" ]]; then
        echo -e "    ${GREEN}[SKIP]${NC} distill already exists: ${DISTILL_OUT}"
        SKIPPED=$((SKIPPED + 1))
      else
        echo -e "    ${CYAN}[RUN]${NC}  distill → ${DISTILL_OUT}"

        if [[ "$DRY_RUN" == "1" ]]; then
          echo -e "    ${YELLOW}(dry-run)${NC} DATASET=${DATASET} IPC=${IPC} DISTILL_METHOD=${METHOD} bash distillate.sh"
          echo -e "    ${YELLOW}(dry-run)${NC}   → student: HARD_LABEL_ALPHA=${HARD_LABEL_ALPHA} USE_FKD_BATCHES=${USE_FKD}"
        else
          set +e
          DATASET="$DATASET" \
          IPC="$IPC" \
          DISTILL_METHOD="$METHOD" \
          WEIGHTING_STRATEGY="$WEIGHTING_STRATEGY" \
          MODEL_TYPE="sd" \
          OUTPUT_DIR="$DISTILL_OUT" \
          DATA_ROOT="$DATA_ROOT" \
          bash "$SCRIPT_DIR/distillate.sh"
          DISTILL_EXIT=$?
          set -e

          if [[ $DISTILL_EXIT -ne 0 ]]; then
            echo -e "    ${RED}[FAIL]${NC} distill failed for ${DATASET} IPC=${IPC} METHOD=${METHOD}"
            FAILED=$((FAILED + 1))
            continue
          fi
          echo -e "    ${GREEN}[OK]${NC}   distill done"
        fi
      fi

      # ══════════════════════════════════════════
      # Step 2: Student Training
      # ══════════════════════════════════════════
      if [[ -f "${STUDENT_OUT}/summary.json" ]]; then
        echo -e "    ${GREEN}[SKIP]${NC} student already exists: ${STUDENT_OUT}"
        SKIPPED=$((SKIPPED + 1))
      else
        echo -e "    ${CYAN}[RUN]${NC}  student → ${STUDENT_OUT}"

        if [[ "$DRY_RUN" == "1" ]]; then
          echo -e "    ${YELLOW}(dry-run)${NC} DATASET=${DATASET} IPC=${IPC} DISTILLED_DATA=${DISTILL_OUT}/distilled_data.pt bash train_student.sh"
          echo -e "    ${YELLOW}(dry-run)${NC}   → student: HARD_LABEL_ALPHA=${HARD_LABEL_ALPHA} USE_FKD_BATCHES=${USE_FKD}"
        else
          set +e
          DATASET="$DATASET" \
          IPC="$IPC" \
          DATA_ROOT="$DATA_ROOT" \
          DISTILLED_DATA="${DISTILL_OUT}/distilled_data.pt" \
          OUTPUT_DIR="$STUDENT_OUT" \
          HARD_LABEL_ALPHA="$HARD_LABEL_ALPHA" \
          USE_FKD_BATCHES="$USE_FKD" \
          bash "$SCRIPT_DIR/train_student.sh"
          STUDENT_EXIT=$?
          set -e

          if [[ $STUDENT_EXIT -ne 0 ]]; then
            echo -e "    ${RED}[FAIL]${NC} student training failed for ${DATASET} IPC=${IPC} METHOD=${METHOD}"
            FAILED=$((FAILED + 1))
            continue
          fi
          echo -e "    ${GREEN}[OK]${NC}   student done"
        fi
      fi

      COMPLETED=$((COMPLETED + 1))
    done
  done
done

# ═══════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║  Summary                                             ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Total tasks:    ${BOLD}${TOTAL_TASKS}${NC}"
echo -e "  Completed/OK:   ${GREEN}${COMPLETED}${NC}"
echo -e "  Skipped:        ${YELLOW}${SKIPPED}${NC}"
echo -e "  Failed:         ${RED}${FAILED}${NC}"
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
  echo -e "${YELLOW}[INFO] This was a dry run. Remove DRY_RUN=1 to execute.${NC}"
fi

# ── 打印结果汇总表 ──
echo -e "${BOLD}Result Summary:${NC}"
echo ""
printf "  %-40s %-8s %-8s %-12s %-12s %-12s\n" \
  "Experiment" "IPC" "Method" "Test Acc" "AUC" "Macro F1"
printf "  %-40s %-8s %-8s %-12s %-12s %-12s\n" \
  "----------------------------------------" "--------" "--------" "------------" "------------" "------------"

for DATASET in ${DATASETS[*]}; do
  for IPC in ${IPCS[*]}; do
    for METHOD in ${METHODS[*]}; do
      # 匹配新命名: clvq→-softlabel, random/kmeans→-classwise
      if [[ "$METHOD" == "clvq" ]]; then
        STUDENT_OUT="outputs/${DATASET}_224_student_ipc${IPC}_method-${METHOD}-softlabel"
      else
        STUDENT_OUT="outputs/${DATASET}_224_student_ipc${IPC}_method-${METHOD}-classwise"
      fi
      SUMMARY_FILE="${STUDENT_OUT}/summary.json"

      case "$METHOD" in
        random) METHOD_LABEL="Random" ;;
        kmeans) METHOD_LABEL="KMeans" ;;
        clvq)   METHOD_LABEL="Ours"   ;;
        *)      METHOD_LABEL="$METHOD" ;;
      esac

      EXP_NAME="${DATASET}_ipc${IPC}_${METHOD_LABEL}"

      if [[ -f "$SUMMARY_FILE" ]]; then
        # 用 python 提取指标
        read -r ACC AUC F1 <<< "$(
          python3 -c "
import json
with open('${SUMMARY_FILE}') as f:
    s = json.load(f)
acc = s.get('test_acc_at_best_val', s.get('final_test_acc', 'N/A'))
auc = s.get('auc_macro', 'N/A')
f1  = s.get('macro_f1', 'N/A')
print(f'{acc} {auc} {f1}')
" 2>/dev/null || echo "ERR ERR ERR"
        )"
        printf "  ${GREEN}%-40s${NC} %-8s %-8s %-12s %-12s %-12s\n" \
          "$EXP_NAME" "$IPC" "$METHOD_LABEL" "$ACC" "$AUC" "$F1"
      else
        printf "  ${YELLOW}%-40s${NC} %-8s %-8s %-12s %-12s %-12s\n" \
          "$EXP_NAME" "$IPC" "$METHOD_LABEL" "N/A" "N/A" "N/A"
      fi
    done
  done
done

echo ""
echo -e "${GREEN}ALL DONE${NC}"
