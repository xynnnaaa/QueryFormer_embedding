#!/bin/bash

# ===== 设置要训练的数据集 =====
# 可选值：imdb, ergastf1, genome, 等
dataset="stats"

# ===== 固定配置 =====
BASE_DIR="/home/vipuser/QueryFormer/data/${dataset}"
CONFIG_DIR="${BASE_DIR}"          # 假设配置文件也放在同一目录下
LOG_DIR="${BASE_DIR}/runfile"
mkdir -p ${LOG_DIR}

# 配置文件及其对应的日志文件名（去掉 .json 后缀）
configs=(
    "default"
    "single"
    "join"
)

# 使用的 GPU 设备
CUDA_DEVICE=1

echo "=========================================="
echo "Starting sequential training for dataset: ${dataset}"
echo "Log directory: ${LOG_DIR}"
echo "=========================================="

for cfg in ${configs[@]}; do
    CONFIG_FILE="${CONFIG_DIR}/config-${cfg}.json"
    LOG_FILE="${LOG_DIR}/${cfg}.log"

    echo "=========================================="
    echo "Running with config: ${CONFIG_FILE}"
    echo "Log: ${LOG_FILE}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python3 -u train.py ${CONFIG_FILE} > ${LOG_FILE} 2>&1

    echo "Finished: ${cfg}"
    echo ""
done

echo "All experiments completed for dataset: ${dataset}"