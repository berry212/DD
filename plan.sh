# DATASET=APTOS_2019_Blindness_Detection DATA_ROOT=data IPC=100 bash train_student.sh
# DATASET=APTOS_2019_Blindness_Detection DATA_ROOT=data IPC=50 bash train_student.sh
# DATASET=APTOS_2019_Blindness_Detection DATA_ROOT=data IPC=10 bash train_student.sh

DATASET=dermamnist DATA_ROOT=data IPC=100 bash distillate.sh
DATASET=dermamnist DATA_ROOT=data IPC=100 bash train_student.sh