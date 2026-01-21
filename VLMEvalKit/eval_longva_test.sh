#!/bin/bash

#############################################################
# LongVA Models Evaluation Script
#
# Evaluates 3 LongVA models on 9 benchmarks:
#   - LongVA-Temporal-v1
#   - LongVA-Temporal-v2
#   - LongVA-7B
#
# Features:
#   - Automatic resume from interruption
#   - OpenAI-based evaluation
#   - Progress logging
#############################################################

set -e

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results/longva_$(date +%Y%m%d_%H%M%S)"
LOGS_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOGS_DIR}/eval_longva_$(date +%Y%m%d_%H%M%S).log"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# LongVA Models
MODELS=(
    "LongVA-Temporal-v1"
    "LongVA-Temporal-v2"
    "LongVA-7B"
)

# 9 Benchmarks
DATASETS=(
    "MMBench_DEV_EN"
    "MME"
    "SEEDBench_IMG"
    "HallusionBench"
    "AI2D_TEST"
    "OCRBench"
    "MathVista_MINI"
    "RealWorldQA"
    "POPE"
)

echo "=========================================="
echo "LongVA Models Evaluation"
echo "=========================================="
echo "Start Time: $(date)"
echo "Results Directory: $RESULTS_DIR"
echo "Log File: $LOG_FILE"
echo ""
echo "Models (${#MODELS[@]}):"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""
echo "Datasets (${#DATASETS[@]}):"
for dataset in "${DATASETS[@]}"; do
    echo "  - $dataset"
done
echo "=========================================="
echo ""

# Check OpenAI API key
if ! grep -q "your_openai_api_key_here" .env 2>/dev/null; then
    echo "✓ OpenAI API key configured"
else
    echo "⚠️  WARNING: Please set your OpenAI API key in VLMEvalKit/.env"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

evaluate_model() {
    local model=$1
    log_message "=========================================="
    log_message "Starting evaluation for: $model"
    log_message "=========================================="

    python run.py \
        --data "${DATASETS[@]}" \
        --model "$model" \
        --mode all \
        --verbose \
        --work-dir "$RESULTS_DIR" \
        2>&1 | tee -a "$LOG_FILE"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log_message "✓ Completed evaluation for: $model"
    else
        log_message "✗ Error during evaluation for: $model (exit code: $exit_code)"
        return $exit_code
    fi

    log_message ""
}

log_message "Starting LongVA models evaluation..."
log_message "Results directory: $RESULTS_DIR"
log_message ""

failed_models=()

for model in "${MODELS[@]}"; do
    if ! evaluate_model "$model"; then
        failed_models+=("$model")
        log_message "⚠️  Evaluation failed for $model, continuing with next model..."
    fi
    sleep 5
done

log_message "=========================================="
log_message "Evaluation Summary"
log_message "=========================================="
log_message "End Time: $(date)"
log_message ""

if [ ${#failed_models[@]} -eq 0 ]; then
    log_message "✓ All LongVA models evaluated successfully!"
else
    log_message "⚠️  Some models failed:"
    for model in "${failed_models[@]}"; do
        log_message "  - $model"
    done
fi

log_message ""
log_message "Results location: $RESULTS_DIR"
log_message "Log file: $LOG_FILE"
log_message "=========================================="

echo ""
echo "Quick access:"
echo "  View results: ls -lh $RESULTS_DIR/*/"
echo "  View log: cat $LOG_FILE"
echo "  Find CSVs: find $RESULTS_DIR -name '*.csv'"
echo ""
