#!/bin/bash

# 学习率列表
lrs=(0.001 0.002 0.0005 0.0002)

# 固定路径和配置
CONFIG_FILE="/home/vipuser/QueryFormer/data/imdb/config-single.json"
LOG_DIR="/home/vipuser/QueryFormer/data/imdb/runfile"
mkdir -p ${LOG_DIR}

for lr in ${lrs[@]}; do
    echo "=========================================="
    echo "Starting experiment with lr = ${lr}"
    echo "Log: ${LOG_DIR}/single-${lr}.log"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=1 python3 -u train.py ${CONFIG_FILE} --lr ${lr} > ${LOG_DIR}/single-${lr}.log 2>&1

    echo "Finished experiment with lr = ${lr}"
    echo ""
done

echo "All experiments completed."