#!/usr/bin/env python3
"""
测试 LongVA 模型的 vision encoder 输出维度
用于准确估算缓存空间需求
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

# 添加环境变量
os.environ['TMPDIR'] = '/disk3/minami/tmp'
os.environ['TORCH_HOME'] = '/disk3/minami/tmp/torch'

def test_vision_output_size():
    """测试单个视频的 vision encoder 输出大小"""

    print("=" * 60)
    print("测试 LongVA Vision Encoder 输出维度")
    print("=" * 60)

    try:
        from longva.model.builder import load_pretrained_model
        from longva.mm_utils import process_images
        print("✓ 成功导入 LongVA")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("\n请先安装 LongVA:")
        print("  git clone https://github.com/EvolvingLMMs-Lab/LongVA")
        print("  cd LongVA && pip install -e .")
        return

    # 加载模型
    print("\n正在加载模型...")
    model_path = "lmms-lab/LongVA-7B-DPO"

    try:
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, "llava_qwen", device_map="auto"
        )
        print(f"✓ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return

    # 创建模拟视频帧 (128帧)
    print("\n创建模拟视频帧 (128帧 × 336×336)...")
    nframe = 128
    dummy_frames = np.random.randint(0, 255, (nframe, 336, 336, 3), dtype=np.uint8)

    # 预处理
    print("预处理视频帧...")
    video_tensor = image_processor.preprocess(dummy_frames, return_tensors="pt")["pixel_values"]
    video_tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"\n预处理后的 video_tensor shape: {video_tensor.shape}")
    print(f"  - Batch size: {video_tensor.shape[0]}")
    print(f"  - Channels: {video_tensor.shape[1]}")
    print(f"  - Height: {video_tensor.shape[2]}")
    print(f"  - Width: {video_tensor.shape[3]}")

    # 提取 vision features
    print("\n提取 vision features...")
    with torch.no_grad():
        # 方法1: 尝试直接访问 vision tower
        if hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
            vision_tower = model.model.vision_tower
            print("  使用 model.model.vision_tower")
        elif hasattr(model, 'vision_tower'):
            vision_tower = model.vision_tower
            print("  使用 model.vision_tower")
        else:
            print("  ✗ 无法找到 vision_tower")
            print(f"  模型属性: {dir(model)}")
            return

        # 编码图像
        vision_features = vision_tower(video_tensor)
        print(f"\nVision Tower 输出 shape: {vision_features.shape}")

        # 方法2: 尝试访问 mm_projector
        if hasattr(model, 'model') and hasattr(model.model, 'mm_projector'):
            mm_projector = model.model.mm_projector
            print("  使用 model.model.mm_projector")
        elif hasattr(model, 'mm_projector'):
            mm_projector = model.mm_projector
            print("  使用 model.mm_projector")
        else:
            print("  ✗ 无法找到 mm_projector")
            return

        # 投影到 LLM 空间
        projected_features = mm_projector(vision_features)
        print(f"MM Projector 输出 shape: {projected_features.shape}")

    # 计算缓存大小
    print("\n" + "=" * 60)
    print("缓存空间估算")
    print("=" * 60)

    # 单个视频的特征大小
    num_tokens = projected_features.shape[1]
    hidden_dim = projected_features.shape[2]
    size_bytes = num_tokens * hidden_dim * 2  # float16 = 2 bytes
    size_mb = size_bytes / (1024 ** 2)

    print(f"\n单个视频 (128帧):")
    print(f"  Token 数量: {num_tokens}")
    print(f"  Hidden 维度: {hidden_dim}")
    print(f"  数据类型: float16 (2 bytes)")
    print(f"  特征大小: {size_mb:.2f} MB ({size_bytes:,} bytes)")

    # 所有数据集
    total_samples = 18595
    total_size_gb = size_mb * total_samples / 1024
    total_size_tb = total_size_gb / 1024

    print(f"\n所有数据集 ({total_samples} 个视频):")
    print(f"  总缓存大小: {total_size_gb:.2f} GB ({total_size_tb:.2f} TB)")

    # 优化建议
    print("\n" + "=" * 60)
    print("优化建议")
    print("=" * 60)

    if total_size_tb > 5:
        print("\n⚠️  缓存需求较大 (> 5 TB)")
        print("\n可选优化方案:")
        print("  1. 使用 bfloat16 量化 (无精度损失)")
        print("  2. 使用 int8 量化 (轻微精度损失，减少4倍)")
        print("  3. 使用压缩存储 (torch.save 的 compression 参数)")

        # 计算优化后的大小
        size_int8 = total_size_gb / 4
        print(f"\nINT8 量化后: {size_int8:.2f} GB ({size_int8/1024:.2f} TB)")
    else:
        print("\n✓ 缓存需求合理")

    print("\n" + "=" * 60)

    return {
        'num_tokens': num_tokens,
        'hidden_dim': hidden_dim,
        'size_mb': size_mb,
        'total_size_tb': total_size_tb
    }

if __name__ == "__main__":
    try:
        result = test_vision_output_size()
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
