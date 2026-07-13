#!/bin/bash

# lrs=(0.001 0.002 0.0005 0.0002)
# bss=(256 512 1024)

# 学习率列表
lrs=(0.0005)
bss=(256 512)

# 配置文件列表
configs=("single")

# 固定路径
CONFIG_DIR="/home/vipuser/QueryFormer/data/imdb"
LOG_DIR="/home/vipuser/QueryFormer/data/imdb/runfile/no-join-no-pca"

mkdir -p ${LOG_DIR}

# 依次运行不同配置文件 + 不同学习率
for config in ${configs[@]}; do

    CONFIG_FILE="${CONFIG_DIR}/config-${config}.json"

    for bs in ${bss[@]}; do

        for lr in ${lrs[@]}; do

            echo "=========================================="
            echo "Starting experiment:"
            echo "Config = ${config}"
            echo "LR = ${lr}"
            echo "BS = ${bs}"
            echo "Log = ${LOG_DIR}/lr${lr}-bs${bs}.log"
            echo "=========================================="

            CUDA_VISIBLE_DEVICES=1 python3 -u train.py ${CONFIG_FILE} --lr ${lr} --bs ${bs} \
            > ${LOG_DIR}/lr${lr}-bs${bs}.log 2>&1

            echo "Finished experiment:"
            echo "Config = ${config}"
            echo "LR = ${lr}"
            echo "BS = ${bs}"
            echo ""

        done

    done

done

echo "All experiments completed."

