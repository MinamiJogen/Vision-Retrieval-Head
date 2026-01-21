#!/bin/bash

# Example script to run VLMEvalKit evaluation with LongVA model
#
# Usage:
#   bash run_vlmeval_example.sh

# Navigate to VLMEvalKit directory
cd VLMEvalKit

# Example 1: Evaluate LongVA on a single dataset (inference only)
# python run.py --data MMBench_DEV_EN --model LongVA-Temporal-v1 --mode infer

# Example 2: Evaluate LongVA on multiple datasets (inference and evaluation)
# python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model LongVA-Temporal-v1 --verbose

# Example 3: Run with specific GPU
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMBench_DEV_EN --model LongVA-Temporal-v1 --verbose

# Example 4: Run multiple models in parallel (if you have multiple GPUs)
# torchrun --nproc-per-node=2 run.py --data MMBench_DEV_EN --model LongVA-Temporal-v1 --verbose

echo "This is an example script. Please uncomment and modify the desired command above."
echo ""
echo "Common datasets:"
echo "  - MMBench_DEV_EN: Multi-modal benchmark"
echo "  - MME: Comprehensive evaluation"
echo "  - SEEDBench_IMG: Image understanding benchmark"
echo "  - MMMU_DEV_VAL: Multi-discipline understanding"
echo ""
echo "Usage example:"
echo "  python run.py --data MMBench_DEV_EN --model LongVA-Temporal-v1 --mode infer --verbose"
