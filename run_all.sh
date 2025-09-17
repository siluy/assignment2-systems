#!/bin/bash

# 定义要测试的 d_model 列表
D_MODELS=(16 32 64 128)

# 循环遍历列表，为每个 d_model 启动一个独立的 Python 进程
for D_MODEL in "${D_MODELS[@]}"; do
    echo "================================================="
    echo "Starting benchmark for d_model = $D_MODEL"
    echo "================================================="
    uv run cs336_systems/pytorch_attention.py --d_model $D_MODEL
    echo "Finished benchmark for d_model = $D_MODEL"
    echo ""
done

echo "All benchmarks completed."