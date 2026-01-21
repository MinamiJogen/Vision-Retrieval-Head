#!/bin/bash

#############################################################
# 单数据集测试脚本
# 用法: bash test_single_dataset.sh <dataset_name>
#
# 运行30秒后自动停止，看看数据集是否能正常加载
#############################################################

if [ -z "$1" ]; then
    echo "用法: bash test_single_dataset.sh <dataset_name>"
    echo "例如: bash test_single_dataset.sh MMBenchVideo"
    exit 1
fi

DATASET=$1
MODEL="LongVA-7B"

export LMUData="/disk3/minami/Vision-Retrieval-Head/VLMEvalKit/dataset"

echo "=========================================="
echo "测试数据集: $DATASET"
echo "模型: $MODEL"
echo "将在30秒后自动停止..."
echo "=========================================="
echo ""

# 在后台运行，30秒后自动杀掉
timeout 30s python run.py \
    --data "$DATASET" \
    --model "$MODEL" \
    --mode infer \
    --verbose

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 124 ]; then
    echo "✓ 测试正常（30秒超时）"
    echo "  数据集 $DATASET 可以正常加载"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 测试完成"
else
    echo "❌ 测试失败（退出码: $EXIT_CODE）"
    echo "  数据集 $DATASET 有问题"
fi
echo "=========================================="
