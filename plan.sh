# DATASET=APTOS_2019_Blindness_Detection DATA_ROOT=data IPC=100 bash train_student.sh
# DATASET=APTOS_2019_Blindness_Detection DATA_ROOT=data IPC=50 bash train_student.sh
# DATASET=APTOS_2019_Blindness_Detection DATA_ROOT=data IPC=10 bash train_student.sh

## dermamnist bloodmnist aptos-2019-blindness-detection

DATASET="${DATASET:-dermamnist}"

# MODEL_TYPE=dit DATASET=$DATASET bash lora_finetune.sh

# DATASET=$DATASET DATA_ROOT=data bash lora_finetune.sh

# for IPC in 0.2 0.4 0.6 0.8; do
#     DISTILL_METHOD=random DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash distillate.sh
#     DISTILL_METHOD=random DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash train_student.sh
# done

# for IPC in 0.6; do
#     DISTILL_METHOD=kmeans DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash distillate.sh
#     DISTILL_METHOD=kmeans DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash train_student.sh
# done

# for IPC in 200 100 50 10; do
#     MODEL_TYPE=dit DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash distillate.sh
#     MODEL_TYPE=dit DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash train_student.sh
# done

# DATASET="bloodmnist"

# MODEL_TYPE=dit DATASET=$DATASET bash lora_finetune.sh
# DATASET=$DATASET bash baseline.sh

# for IPC in 200 100 50 10; do
#     MODEL_TYPE=dit DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash distillate.sh
#     MODEL_TYPE=dit DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash train_student.sh
# done

DATASET="aptos-2019-blindness-detection"

# DATASET=$DATASET bash baseline.sh

MODEL_TYPE=dit DATASET=$DATASET bash lora_finetune.sh

for IPC in 200 100 50 10; do
    MODEL_TYPE=dit DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash distillate.sh
    MODEL_TYPE=dit DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash train_student.sh
done