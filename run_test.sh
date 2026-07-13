#!/bin/bash

# ===================================================================
# 串行训练脚本 - 按顺序执行以下任务
# 1. stats   (no-join, lr=0.0002)
# 2. genome  (no-join, lr=0.001)
# 3. imdb    (no-join, lr=0.0005, bs=512)
# 4. ergastf1 (join, 多组 lr 和 bs)
# ===================================================================


echo "========== 开始执行训练任务 =========="

# ---------- 1. stats ----------
echo "[1/4] 训练 stats 数据集 (no-join, lr=0.0002)"
CUDA_VISIBLE_DEVICES=1 python3 -u train.py \
    /home/vipuser/QueryFormer/data/stats/config-single.json \
    --lr 0.0002 \
    > /home/vipuser/QueryFormer/data/stats/runfile/no-join/0.0002.log 2>&1
echo "[1/4] 完成"

# ---------- 2. genome ----------
echo "[2/4] 训练 genome 数据集 (no-join, lr=0.001)"
CUDA_VISIBLE_DEVICES=1 python3 -u train.py \
    /home/vipuser/QueryFormer/data/genome/config-single.json \
    --lr 0.001 \
    > /home/vipuser/QueryFormer/data/genome/runfile/no-join/0.001.log 2>&1
echo "[2/4] 完成"


# ---------- 4. ergastf1 (多组参数) ----------
echo "[4/4] 开始 ergastf1 数据集参数扫描 (join 模式)"

lrs=(0.002 0.0005 0.0002)
bss=(256)

CONFIG_DIR="/home/vipuser/QueryFormer/data/ergastf1"
LOG_DIR="/home/vipuser/QueryFormer/data/ergastf1/runfile/full"

mkdir -p ${LOG_DIR}

# 配置文件固定为 config-join.json（您原代码中 configs=("join")）
CONFIG_FILE="${CONFIG_DIR}/config-join.json"

for bs in ${bss[@]}; do
    for lr in ${lrs[@]}; do
        echo "=========================================="
        echo "开始: Config=join, LR=${lr}, BS=${bs}"
        echo "日志: ${LOG_DIR}/lr${lr}-bs${bs}.log"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES=1 python3 -u train.py \
            ${CONFIG_FILE} \
            --lr ${lr} \
            --bs ${bs} \
            > ${LOG_DIR}/lr${lr}-bs${bs}.log 2>&1

        echo "完成: Config=join, LR=${lr}, BS=${bs}"
        echo ""
    done
done

echo "========== 所有任务执行完毕 =========="