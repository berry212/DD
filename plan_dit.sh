DATASET="${DATASET:-dermamnist}"

# DATASET=$DATASET bash baseline.sh
DATASET=$DATASET bash lora_finetune_dit.sh

for IPC in 200 100 50 10; do
    DATASET=$DATASET DATA_ROOT=data IPC=$IPC BACKBONE_TYPE=dit bash distillate.sh
    DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash train_student.sh
done
