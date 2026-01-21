#!/bin/bash

#############################################################
# Qwen3-VL Video Models Parallel Evaluation Script (128 frames)
#
# Evaluates Qwen3-VL models on 7 video benchmarks with 128 frames
# Using 8 GPUs divided into 4 groups (2 GPUs per group)
#
# Dataset Distribution (balanced by sample count):
#   Group 0 (GPU 0-1): Video-MME, Video_Holmes (5,608 samples)
#   Group 1 (GPU 2-3): TempCompass (5,538 samples)
#   Group 2 (GPU 4-5): MLVU, VideoMMMU (4,006 samples)
#   Group 3 (GPU 6-7): MMBench_Video, LongVideoBench (3,443 samples)
#
# Total: 18,595 samples across 7 benchmarks
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
RESULTS_DIR="${SCRIPT_DIR}/results/qwen3vl_video_128f_parallel_${TIMESTAMP}"
LOGS_DIR="${SCRIPT_DIR}/logs"
CONFIG_DIR="${SCRIPT_DIR}/configs"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$CONFIG_DIR"

# Model paths (local paths to HuggingFace snapshots)
QWEN3VL_8B_PATH="/disk3/minami/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

# Create JSON config file for Qwen3-VL-8B
CONFIG_FILE="${CONFIG_DIR}/qwen3vl_8b_video_128f_${TIMESTAMP}.json"
cat > "$CONFIG_FILE" << 'EOF'
{
    "model": {
        "Qwen3-VL-8B-Instruct-Local": {
            "class": "Qwen3VLChat",
            "model_path": "MODEL_PATH_PLACEHOLDER",
            "use_custom_prompt": false,
            "use_vllm": true,
            "temperature": 0.7,
            "max_new_tokens": 16384,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "top_p": 0.8,
            "top_k": 20
        }
    },
    "data": {
        "Video-MME_128frame": {},
        "VideoMMMU_128frame": {},
        "LongVideoBench_128frame": {},
        "MLVU_128frame": {},
        "Video_Holmes_128frame": {},
        "TempCompass_128frame": {},
        "MMBench_Video_128frame_nopack": {}
    }
}
EOF

# Replace placeholder with actual model path
sed -i "s|MODEL_PATH_PLACEHOLDER|${QWEN3VL_8B_PATH}|g" "$CONFIG_FILE"

# Dataset groups (balanced by sample count)
# Group 0: 5,608 samples (2,999 + 2,609)
DATASETS_GROUP0=(
    "Video-MME_128frame"
    "Video_Holmes_128frame"
)

# Group 1: 5,538 samples
DATASETS_GROUP1=(
    "TempCompass_128frame"
)

# Group 2: 4,006 samples (2,599 + 1,407)
DATASETS_GROUP2=(
    "MLVU_128frame"
    "VideoMMMU_128frame"
)

# Group 3: 3,443 samples (2,105 + 1,338)
DATASETS_GROUP3=(
    "MMBench_Video_128frame_nopack"
    "LongVideoBench_128frame"
)

echo "=========================================="
echo "Qwen3-VL Video Models Parallel Evaluation (128 frames)"
echo "=========================================="
echo "Start Time: $(date)"
echo "Results Directory: $RESULTS_DIR"
echo "Logs Directory: $LOGS_DIR"
echo "Config File: $CONFIG_FILE"
echo ""
echo "Model:"
echo "  - Qwen3-VL-8B-Instruct (Local: ${QWEN3VL_8B_PATH})"
echo ""
echo "Dataset Groups (4 groups, 8 GPUs total):"
echo "  Group 0 (GPU 0-1, 5608 samples):"
for dataset in "${DATASETS_GROUP0[@]}"; do
    echo "    - $dataset"
done
echo "  Group 1 (GPU 2-3, 5538 samples):"
for dataset in "${DATASETS_GROUP1[@]}"; do
    echo "    - $dataset"
done
echo "  Group 2 (GPU 4-5, 4006 samples):"
for dataset in "${DATASETS_GROUP2[@]}"; do
    echo "    - $dataset"
done
echo "  Group 3 (GPU 6-7, 3443 samples):"
for dataset in "${DATASETS_GROUP3[@]}"; do
    echo "    - $dataset"
done
echo "=========================================="
echo ""

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to create a group-specific config file
create_group_config() {
    local group_id=$1
    shift
    local datasets=("$@")

    local group_config="${CONFIG_DIR}/qwen3vl_8b_group${group_id}_${TIMESTAMP}.json"

    # Start JSON
    cat > "$group_config" << EOF
{
    "model": {
        "Qwen3-VL-8B-Instruct-Local": {
            "class": "Qwen3VLChat",
            "model_path": "${QWEN3VL_8B_PATH}",
            "use_custom_prompt": false,
            "use_vllm": true,
            "temperature": 0.7,
            "max_new_tokens": 16384,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "top_p": 0.8,
            "top_k": 20
        }
    },
    "data": {
EOF

    # Add datasets
    local first=true
    for dataset in "${datasets[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$group_config"
        fi
        echo -n "        \"$dataset\": {}" >> "$group_config"
    done

    # Close JSON
    cat >> "$group_config" << EOF

    }
}
EOF

    echo "$group_config"
}

# Function to evaluate a group of datasets
evaluate_group() {
    local group_id=$1
    local gpu_ids=$2
    shift 2
    local datasets=("$@")

    local log_file="${LOGS_DIR}/qwen3vl_group${group_id}_${TIMESTAMP}.log"
    local group_config=$(create_group_config $group_id "${datasets[@]}")

    log_message "Group ${group_id} starting on GPUs ${gpu_ids}" | tee -a "$log_file"
    log_message "Group ${group_id} config: ${group_config}" | tee -a "$log_file"

    CUDA_VISIBLE_DEVICES=$gpu_ids python run.py \
        --config "$group_config" \
        --mode all \
        --verbose \
        --work-dir "$RESULTS_DIR" \
        2>&1 | tee -a "$log_file"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log_message "Group ${group_id}: Completed successfully" | tee -a "$log_file"
    else
        log_message "Group ${group_id}: Error (exit code: $exit_code)" | tee -a "$log_file"
    fi

    log_message "Group ${group_id} finished" | tee -a "$log_file"
}

# Start all 4 groups in parallel
log_message "Starting parallel evaluation across 4 groups..."
echo ""

evaluate_group 0 "0,1" "${DATASETS_GROUP0[@]}" &
PID0=$!

evaluate_group 1 "2,3" "${DATASETS_GROUP1[@]}" &
PID1=$!

evaluate_group 2 "4,5" "${DATASETS_GROUP2[@]}" &
PID2=$!

evaluate_group 3 "6,7" "${DATASETS_GROUP3[@]}" &
PID3=$!

# Wait for all groups to complete
log_message "Waiting for all groups to complete..."
wait $PID0
EXIT0=$?
log_message "Group 0 finished with exit code: $EXIT0"

wait $PID1
EXIT1=$?
log_message "Group 1 finished with exit code: $EXIT1"

wait $PID2
EXIT2=$?
log_message "Group 2 finished with exit code: $EXIT2"

wait $PID3
EXIT3=$?
log_message "Group 3 finished with exit code: $EXIT3"

echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "End Time: $(date)"
echo ""
echo "Group Exit Codes:"
echo "  Group 0 (GPU 0-1): $EXIT0"
echo "  Group 1 (GPU 2-3): $EXIT1"
echo "  Group 2 (GPU 4-5): $EXIT2"
echo "  Group 3 (GPU 6-7): $EXIT3"
echo ""

if [ $EXIT0 -eq 0 ] && [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ] && [ $EXIT3 -eq 0 ]; then
    echo "All groups completed successfully!"
else
    echo "Some groups encountered errors. Check the logs for details."
fi

echo ""
echo "Results location: $RESULTS_DIR"
echo "Logs location: $LOGS_DIR/*_${TIMESTAMP}.log"
echo "Config files: $CONFIG_DIR/*_${TIMESTAMP}.json"
echo "=========================================="
echo ""
echo "Quick access:"
echo "  View results: ls -lh $RESULTS_DIR/*/"
echo "  View logs: tail -f $LOGS_DIR/*_${TIMESTAMP}.log"
echo "  Find CSVs: find $RESULTS_DIR -name '*.csv'"
echo ""
