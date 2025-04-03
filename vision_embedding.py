import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from longva.model.builder import load_pretrained_model
from longva.mm_utils import process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
import torch
import numpy as np
import math
import argparse

torch.manual_seed(0)
model_path = "lmms-lab/LongVA-7B-DPO"

# 输入保持不变（虽然 bbox 和 needle 暂不使用）
images_and_bboxes = [
    ["image1.JPG", None],
    ["target.JPG", (1000, 2270, 2357, 2802)],
]
question = "What is the main object in the lower part of the second picture?"
needle = "car"  # 暂不使用

def main(args):
    # 模型加载：加载 tokenizer、model、image_processor 等
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, "llava_qwen", device_map="cuda:0"
    )
    # 删除不需要的层（只保留视觉编码部分）
    del model.model.layers
    model.config.image_aspect_ratio = "pad"
    model.config.mm_patch_merge_type = "flat"

    # 加载图片（忽略 bbox 信息）
    images = [Image.open(img_path).convert("RGB") for img_path, _ in images_and_bboxes]
    print(f"加载了 {len(images)} 张图片")

    # 预处理图片，得到模型需要的 tensor 格式
    processed_images = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    # 通过模型的 encode_images 获取视觉 embedding（不使用 vision_tower）
    with torch.inference_mode():
        image_embeddings = model.encode_images(processed_images)
    # image_embeddings 的形状通常为 [B, L, F]，其中 B 为图片数，L 为每张图片生成的 token 数，F 为特征维度

    # 若设置了 pooling_size，则对空间维度进行池化
    if args.pooling_size != 0:
        B, L, F = image_embeddings.shape
        n = int(math.sqrt(L))
        image_embeddings_spatial = image_embeddings.view(B, n, n, F).permute(0, 3, 1, 2)  # [B, F, n, n]
        image_embeddings_spatial_pool = torch.nn.functional.avg_pool2d(
            image_embeddings_spatial, args.pooling_size, args.pooling_size
        )  # 池化后形状 [B, F, new_n, new_n]
        # flatten 回序列形式，形状变为 [B, new_n*new_n, F]
        image_embeddings = image_embeddings_spatial_pool.flatten(2).transpose(1, 2).contiguous()

    # 将每张图片的 embedding 拼接为一个序列（这里将批次维度 merge 到序列维度中）
    # 例如两张图片分别生成 [L1, F] 和 [L2, F]，拼接后变为 [1, L1+L2, F]
    image_embeddings = image_embeddings.view(1, -1, image_embeddings.shape[-1])
    print(f"最终拼接的视觉 embedding shape: {image_embeddings.shape}")

    # 此处仅打印视觉 embedding 拼接后的形状，你可以根据需要后续使用 final_embeddings
    return image_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling_size", type=int, default=0, help="设置池化窗口大小，0 表示不池化")
    args = parser.parse_args()
    final_embeddings = main(args)
