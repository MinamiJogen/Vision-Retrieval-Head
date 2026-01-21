#!/bin/bash
#############################################################
# LongVA Video Models Evaluation with Vision Cache
#
# 使用预处理的 vision features 缓存进行评估
# 测试 Video-MME 数据集（128 frames, float16）
#
# 前置条件:
#   1. 已运行 preprocess_video_mme.py 完成预处理
#   2. 缓存目录: /disk3/minami/LMUData/vision_cache
#############################################################

set -e

# Set cache directories to disk3
export TMPDIR=/disk3/minami/tmp
export TEMP=/disk3/minami/tmp
export TMP=/disk3/minami/tmp
export TORCH_HOME=/disk3/minami/tmp/torch
export XDG_CACHE_HOME=/disk3/minami/tmp/cache
mkdir -p $TMPDIR $TORCH_HOME $XDG_CACHE_HOME

# Suppress PyTorch meta parameter warnings
export PYTHONWARNINGS="ignore::UserWarning"

# OpenAI API Key

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${SCRIPT_DIR}/results/longva_cached_${TIMESTAMP}"
LOGS_DIR="${SCRIPT_DIR}/logs"
CACHE_DIR="/disk3/minami/LMUData/vision_cache"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# LongVA Models (使用带缓存的版本)
MODELS=(
    "LongVA-Temporal-v1-Cached"
    "LongVA-Temporal-v2-Cached"
    "LongVA-7B-Cached"
)

# Dataset: Video-MME only
DATASET="Video-MME_128frame"

echo "=========================================="
echo "LongVA Video Models Evaluation (Cached)"
echo "=========================================="
echo "Start Time: $(date)"
echo "Results Directory: $RESULTS_DIR"
echo "Logs Directory: $LOGS_DIR"
echo "Cache Directory: $CACHE_DIR"
echo ""
echo "Models (${#MODELS[@]}):"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""
echo "Dataset: $DATASET"
echo "=========================================="
echo ""

# Check cache exists
if [ ! -d "$CACHE_DIR" ]; then
    echo "❌ 错误: 缓存目录不存在: $CACHE_DIR"
    echo ""
    echo "请先运行预处理脚本:"
    echo "  python preprocess_video_mme.py"
    exit 1
fi

if [ ! -f "$CACHE_DIR/cache_meta.json" ]; then
    echo "⚠️  警告: 缓存元数据文件不存在: $CACHE_DIR/cache_meta.json"
    echo "缓存可能不完整，继续执行..."
    echo ""
fi

# Check if cache contains the dataset
DATASET_CACHE_DIR="$CACHE_DIR/$DATASET"
if [ -d "$DATASET_CACHE_DIR" ]; then
    CACHE_COUNT=$(find "$DATASET_CACHE_DIR" -name "*.pt" | wc -l)
    echo "✓ 找到缓存: $DATASET (${CACHE_COUNT} 个文件)"
    echo ""
else
    echo "⚠️  警告: 数据集缓存目录不存在: $DATASET_CACHE_DIR"
    echo "缓存可能不完整，继续执行..."
    echo ""
fi

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Main evaluation function
evaluate_models() {
    local log_file="${LOGS_DIR}/longva_cached_${TIMESTAMP}.log"

    log_message "开始评估" | tee -a "$log_file"
    log_message "使用 GPU: $CUDA_VISIBLE_DEVICES" | tee -a "$log_file"
    echo "" | tee -a "$log_file"

    for model in "${MODELS[@]}"; do
        log_message "评估模型: $model" | tee -a "$log_file"
        echo "" | tee -a "$log_file"

        # Run evaluation
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run.py \
            --data "$DATASET" \
            --model "$model" \
            --mode all \
            --verbose \
            --work-dir "$RESULTS_DIR" \
            2>&1 | tee -a "$log_file"

        local exit_code=${PIPESTATUS[0]}

        echo "" | tee -a "$log_file"
        if [ $exit_code -eq 0 ]; then
            log_message "✓ 完成: $model" | tee -a "$log_file"
        else
            log_message "✗ 错误: $model (exit code: $exit_code)" | tee -a "$log_file"
        fi
        echo "" | tee -a "$log_file"

        # Small delay between models
        sleep 2
    done

    log_message "所有模型评估完成" | tee -a "$log_file"
}

# Run evaluation
evaluate_models

echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "End Time: $(date)"
echo ""
echo "Results location: $RESULTS_DIR"
echo "Logs location: $LOGS_DIR/longva_cached_${TIMESTAMP}.log"
echo "=========================================="
echo ""
echo "Quick access:"
echo "  View results: ls -lh $RESULTS_DIR/*/"
echo "  View logs: cat $LOGS_DIR/longva_cached_${TIMESTAMP}.log"
echo "  Find CSVs: find $RESULTS_DIR -name '*.csv'"
echo ""

# Show cache statistics if available
if [ -f "$CACHE_DIR/cache_meta.json" ]; then
    echo "缓存统计:"
    cat "$CACHE_DIR/cache_meta.json" | python3 -m json.tool | grep -A 10 "Video-MME" || true
    echo ""
fi

echo "评估完成! 请检查结果文件。"
echo ""
