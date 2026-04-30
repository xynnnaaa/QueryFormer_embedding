#!/bin/bash

# 学习率列表
lrs=(0.001 0.002 0.0005 0.0002)
# batch size 列表
bss=(1024 256)

# 固定路径和配置
CONFIG_FILE="/home/vipuser/QueryFormer/data/stats/config-single.json"
LOG_DIR="/home/vipuser/QueryFormer/data/stats/runfile/single"
mkdir -p ${LOG_DIR}

for lr in ${lrs[@]}; do
    for bs in ${bss[@]}; do
        # 跳过 lr=0.001 且 bs=1024 的组合
        if [[ "$lr" == "0.001" && "$bs" == "1024" ]]; then
            echo "Skipping lr=0.001, bs=1024 (already completed)"
            continue
        fi

        echo "=========================================="
        echo "Starting experiment with lr = ${lr}, bs = ${bs}"
        echo "Log: ${LOG_DIR}/single-lr${lr}-bs${bs}.log"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES=1 python3 -u train.py ${CONFIG_FILE} --lr ${lr} --bs ${bs} > ${LOG_DIR}/single-lr${lr}-bs${bs}.log 2>&1

        echo "Finished experiment with lr = ${lr}, bs = ${bs}"
        echo ""
    done
done

echo "All experiments completed."