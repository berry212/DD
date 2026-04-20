# DATASET=APTOS_2019_Blindness_Detection DATA_ROOT=data IPC=100 bash train_student.sh
# DATASET=APTOS_2019_Blindness_Detection DATA_ROOT=data IPC=50 bash train_student.sh
# DATASET=APTOS_2019_Blindness_Detection DATA_ROOT=data IPC=10 bash train_student.sh

## dermamnist bloodmnist aptos-2019-blindness-detection
## 目前不太支持的数据集 odir-5k

DATASET="${DATASET:-aptos-2019-blindness-detection}"

for IPC in 10 50 100 200; do
    DATASET=aptos-2019-blindness-detection DATA_ROOT=data IPC=$IPC bash distillate.sh
    DATASET=aptos-2019-blindness-detection DATA_ROOT=data IPC=$IPC bash train_student.sh
done