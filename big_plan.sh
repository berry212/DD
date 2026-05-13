## dermamnist bloodmnist aptos-2019-blindness-detection

DATASET="${DATASET:-dermamnist}"

DATASET="dermamnist"

# MODEL_TYPE=unet DATASET=$DATASET bash lora_finetune.sh

for IPC in 0.1 0.2 0.4 0.6; do
    WEIGHTING_STRATEGY=inverse MODEL_TYPE=unet DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash distillate.sh
    WEIGHTING_STRATEGY=inverse MODEL_TYPE=unet DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash train_student.sh
done

DATASET="bloodmnist"

# MODEL_TYPE=unet DATASET=$DATASET bash lora_finetune.sh

for IPC in 0.1 0.2 0.4 0.6; do
    WEIGHTING_STRATEGY=inverse MODEL_TYPE=unet DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash distillate.sh
    WEIGHTING_STRATEGY=inverse MODEL_TYPE=unet DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash train_student.sh
done

DATASET="aptos-2019-blindness-detection"

# MODEL_TYPE=unet DATASET=$DATASET bash lora_finetune.sh

for IPC in 0.1 0.2 0.4 0.6; do
    WEIGHTING_STRATEGY=inverse MODEL_TYPE=unet DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash distillate.sh
    WEIGHTING_STRATEGY=inverse MODEL_TYPE=unet DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash train_student.sh
done

DATASET="pathmnist"

MODEL_TYPE=unet DATASET=$DATASET bash lora_finetune.sh

for IPC in 200 100 50 10; do
    MODEL_TYPE=unet DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash distillate.sh
    MODEL_TYPE=unet DATASET=$DATASET DATA_ROOT=data IPC=$IPC bash train_student.sh
done