## dermamnist bloodmnist aptos-2019-blindness-detection
## "heuristic", "direct", "uniform", "inverse"

DATASET="${DATASET:-dermamnist}"

# MODEL_TYPE=unet DATASET=$DATASET bash lora_finetune.sh

for IPC in 1000 200 100 50 10; do
    DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash noise_distillate.sh 
    DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash noise_train_student.sh
done

DATASET="bloodmnist"

# MODEL_TYPE=unet DATASET=$DATASET bash lora_finetune.sh

# for IPC in 200 100 50 10; do
#     DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash noise_distillate.sh 
#     DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash noise_train_student.sh
# done

DATASET="aptos-2019-blindness-detection"

# MODEL_TYPE=unet DATASET=$DATASET bash lora_finetune.sh

# for IPC in 200 100 50 10; do
#     DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash noise_distillate.sh 
#     DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash noise_train_student.sh
# done

# DATASET="pathmnist"

# MODEL_TYPE=unet DATASET=$DATASET bash lora_finetune.sh

# for IPC in 200 100 50 10; do
#     MODEL_TYPE=unet DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash distillate.sh
#     MODEL_TYPE=unet DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash train_student.sh
# done