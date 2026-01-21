#!/usr/bin/env bash
set -euo pipefail

# 创建日志目录（如果不存在）
mkdir -p ./mergelogs

# 设置日志文件名（包含时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL_LOG="./mergelogs/full_output_v1_${TIMESTAMP}.log"

# 使用 exec 和 tee 来同时输出到终端和文件
exec > >(tee -a "${FULL_LOG}")
exec 2>&1

echo "========================================="
echo "开始运行脚本 (Version 1) - $(date)"
echo "完整日志将保存到: ${FULL_LOG}"
echo "========================================="
echo

# 要评测的 α 列表（从 0.1 到 0.9）
alphas=(0.7 0.8 0.9 1.0)

for a in "${alphas[@]}"; do
    echo "================  α = ${a}  ================"
    echo "开始时间: $(date)"

    # 目录 / 文件名
    outdir="/disk3/minami/huggingface/hub/models--LongVA-Merge1-a${a}"
    logfile="./mergelogs/alpha1${a}.json"
    suffix="videomme_longva1_a${a}"
    
    # 为每个 alpha 创建单独的日志文件（可选）
    individual_log="./mergelogs/alpha1_${a}_${TIMESTAMP}.log"

    # 1) 合并模型
    echo ">>> 开始合并模型 (merge-givin1.py)..."
    python merge-givin1.py --alpha "${a}" --output_dir "${outdir}" 2>&1 | tee -a "${individual_log}"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ">>> 模型合并成功"
    else
        echo ">>> 模型合并失败，退出代码: ${PIPESTATUS[0]}"
        exit 1
    fi

    # 2) 评测 VideoMME
    echo ">>> 开始评测 VideoMME..."
    accelerate launch --num_processes 4 --main_process_port 12345 \
        -m lmms_eval \
        --model longva \
        --model_args "pretrained=${outdir},conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=128,model_name=llava_qwen" \
        --tasks videomme \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "${suffix}" \
        --output_path "${logfile}" 2>&1 | tee -a "${individual_log}"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ">>> 评测成功"
    else
        echo ">>> 评测失败，退出代码: ${PIPESTATUS[0]}"
        exit 1
    fi

    echo "α=${a} 评测完成 → ${logfile}"
    echo "单独日志保存到 → ${individual_log}"
    echo "结束时间: $(date)"
    echo
done

echo "===== 全部 α 运行完毕 ====="
echo "完成时间: $(date)"
echo "完整日志已保存到: ${FULL_LOG}"

# 可选：显示日志文件信息
echo
echo "日志文件信息："
ls -lh ./mergelogs/*.log 2>/dev/null | grep -E "(full_output_v1_|alpha1_)" | tail -n 10