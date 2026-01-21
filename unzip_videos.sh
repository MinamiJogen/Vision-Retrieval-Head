#!/bin/bash

# 目标解压目录
OUT_DIR="/disk3/minami/Vision-Retrieval-Head/videos"
# 源 zip 文件目录
ZIP_DIR="/disk3/minami/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b"

mkdir -p "$OUT_DIR"

for i in {01..20}; do
    ZIP_FILE="$ZIP_DIR/videos_chunked_${i}.zip"
    if [ -f "$ZIP_FILE" ]; then
        echo "📦 解压 $ZIP_FILE"
        unzip -q "$ZIP_FILE" -d "$OUT_DIR"
    else
        echo "⚠️ 缺失文件: $ZIP_FILE"
    fi
done

echo "✅ 所有视频已解压到 $OUT_DIR"
