#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Extract test_acc_at_best_val from plan_weighting_ablation.sh outputs
# ============================================================

OUTPUT_BASE="${1:-outputs}"

DATASETS=("dermamnist" "bloodmnist" "aptos-2019-blindness-detection")
IPCS=(200 100 50 10)
WEIGHTS=("uniform" "heuristic" "inverse")

echo ""
echo "============================================================"
echo "  test_acc_at_best_val 汇总"
echo "  output base: ${OUTPUT_BASE}"
echo "============================================================"
echo ""

# Print header
printf "%-40s" "DATASET"
for IPC in "${IPCS[@]}"; do
    for WEIGHT in "${WEIGHTS[@]}"; do
        printf "  ipc%-4s %-10s" "${IPC}" "${WEIGHT}"
    done
done
echo ""
printf "%-40s" "----------------------------------------"
for IPC in "${IPCS[@]}"; do
    for WEIGHT in "${WEIGHTS[@]}"; do
        printf "  %-5s %-10s" "-----" "----------"
    done
done
echo ""

for DATASET in "${DATASETS[@]}"; do
    printf "%-40s" "${DATASET}"
    for IPC in "${IPCS[@]}"; do
        for WEIGHT in "${WEIGHTS[@]}"; do
            STUDENT_OUT="${OUTPUT_BASE}/${DATASET}_224_student_ipc${IPC}_${WEIGHT}"
            SUMMARY="${STUDENT_OUT}/summary.json"
            if [[ -f "${SUMMARY}" ]]; then
                ACC=$(python3 -c "import json; d=json.load(open('${SUMMARY}')); print(f\"{d['test_acc_at_best_val']:.4f}\")" 2>/dev/null || echo "ERR")
                printf "  %-5s %-10s" "${ACC}" ""
            else
                printf "  %-5s %-10s" "N/A" ""
            fi
        done
    done
    echo ""
done

echo ""
echo "============================================================"

# Also print as a markdown table for easy copy-paste
echo ""
echo "## Markdown Table"
echo ""
echo "| Dataset | IPC | Uniform | Heuristic | Inverse |"
echo "|---------|-----|---------|-----------|---------|"

for DATASET in "${DATASETS[@]}"; do
    for IPC in "${IPCS[@]}"; do
        printf "| %-40s | %-3s |" "${DATASET}" "${IPC}"
        for WEIGHT in "${WEIGHTS[@]}"; do
            STUDENT_OUT="${OUTPUT_BASE}/${DATASET}_224_student_ipc${IPC}_${WEIGHT}"
            SUMMARY="${STUDENT_OUT}/summary.json"
            if [[ -f "${SUMMARY}" ]]; then
                ACC=$(python3 -c "import json; d=json.load(open('${SUMMARY}')); print(f\"{d['test_acc_at_best_val']:.4f}\")" 2>/dev/null || echo "ERR")
                printf " %-8s |" "${ACC}"
            else
                printf " %-8s |" "N/A"
            fi
        done
        echo ""
    done
done

echo ""
echo "Done."
