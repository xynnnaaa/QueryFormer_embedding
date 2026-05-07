#!/bin/bash

# 学习率列表
lrs=(0.001 0.002 0.0005 0.0002)
# batch size 列表
bss=(512 256)

# 固定路径和配置
CONFIG_FILE="/home/vipuser/QueryFormer/data/ergastf1/config-join.json"
LOG_DIR="/home/vipuser/QueryFormer/data/ergastf1/runfile/join-v2"
mkdir -p ${LOG_DIR}

for bs in ${bss[@]}; do
    for lr in ${lrs[@]}; do
        # # 跳过 lr=0.001 且 bs=1024 的组合
        # if [[ "$lr" == "0.001" && "$bs" == "1024" ]]; then
        #     echo "Skipping lr=0.001, bs=1024 (already completed)"
        #     continue
        # fi

        echo "=========================================="
        echo "Starting experiment with lr = ${lr}, bs = ${bs}"
        echo "Log: ${LOG_DIR}/join-lr${lr}-bs${bs}.log"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES=1 python3 -u train.py ${CONFIG_FILE} --lr ${lr} --bs ${bs} > ${LOG_DIR}/join-lr${lr}-bs${bs}.log 2>&1

        echo "Finished experiment with lr = ${lr}, bs = ${bs}"
        echo ""
    done
done

echo "All experiments completed."


# #!/bin/bash

# # 学习率列表
# lrs=(0.001 0.002 0.0005 0.0002)
# # batch size 列表
# bss=(512 256)

# # 定义两组配置，格式："配置文件路径:日志目录路径"
# configs=(
#     "/home/vipuser/QueryFormer/data/genome/config-join.json:/home/vipuser/QueryFormer/data/genome/runfile/join-v2"
#     "/home/vipuser/QueryFormer/data/ergastf1/config-join.json:/home/vipuser/QueryFormer/data/ergastf1/runfile/join-v2"
# )

# for config_pair in "${configs[@]}"; do
#     # 按冒号拆分配置文件路径和日志目录
#     IFS=':' read -r CONFIG_FILE LOG_DIR <<< "$config_pair"
#     mkdir -p "${LOG_DIR}"

#     for lr in ${lrs[@]}; do
#         for bs in ${bss[@]}; do
#             echo "=========================================="
#             echo "Starting experiment with lr = ${lr}, bs = ${bs}"
#             echo "Config: ${CONFIG_FILE}"
#             echo "Log: ${LOG_DIR}/single-lr${lr}-bs${bs}.log"
#             echo "=========================================="

#             CUDA_VISIBLE_DEVICES=0 python3 -u train.py "${CONFIG_FILE}" --lr ${lr} --bs ${bs} > "${LOG_DIR}/join-lr${lr}-bs${bs}.log" 2>&1

#             echo "Finished experiment with lr = ${lr}, bs = ${bs}"
#             echo ""
#         done
#     done
# done

# echo "All experiments completed."