# -*- coding: utf-8 -*-
"""
LongVA 多模态模型——注意力可视化示例脚本（中文详尽注释版）
----------------------------------------------------------
本脚本演示如何：
1. 加载 LongVA‑7B‑DPO 多模态模型（LLaVA‑Qwen 架构）。
2. 将多张图片（可带或不带 BBox）与文本问题拼接成输入。
3. 在解码过程中捕获某个“needle”词（例如 car）生成时的自注意力张量。
4. 计算并打印每一层、每个注意力头对指定 BBox 对应视觉 token 的关注度。

⚠️ 注意：
* 脚本默认使用 flash‑attention；若硬件/驱动不支持，可改成 "attn_mode='torch'"。
* 请确保显存足够（脚本前面设置了 PYTORCH_CUDA_ALLOC_CONF 以降低碎片化）。
* 需要提前准备好 image1.JPG 与 target.JPG 两张图片，并根据需要修改 BBox 坐标。
"""

import os
# 避免 CUDA 内存碎片过大，按 128 MB 对齐分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from longva.model.builder import load_pretrained_model  # LongVA 模型加载器
from transformers import AutoTokenizer                   # 备用：若需独立调用 tokenizer
from longva.mm_utils import tokenizer_image_token, process_images  # LongVA 提供的辅助函数
from longva.constants import IMAGE_TOKEN_INDEX           # 特殊 token，用于占位 <image>
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

# 为了结果可复现，固定随机种子
torch.manual_seed(0)

# -------------------------
# 1. 配置与输入
# -------------------------
model_path = "lmms-lab/LongVA-7B-DPO"  # 🤖 预训练权重名称（HF Hub）

# 图片路径与对应 BBox（左上 x,y, 右下 x,y）；若无 BBox 则填 None
images_and_bboxes = [
    ["image1.JPG", None],
    ["target.JPG", (1000, 2270, 2357, 2802)],
]

question = "What is the main object in the lower part of the second picture?"  # 用户问题
needle = "car"  # 想在生成文本中捕获的关键词

# -------------------------
# 2. 加载模型 & 预处理图片
# -------------------------
# tokenizer / model / image_processor 均由 LongVA 封装返回
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path,
    None,                # 默认使用官方权重
    "llava_qwen",        # 模型架构标识
    device_map="auto"    # 自动把权重切到多块 GPU（若可用）
)

vision_tower = model.get_vision_tower()  # 取出视觉分支，后续可用其属性

# 读取并转换图片为 RGB
images = [Image.open(img_path).convert("RGB") for img_path, _ in images_and_bboxes]
# process_images 会做 resize / normalization，并转成 (B, C, H, W) 张量
auto_dtype = torch.float16  # 使用 fp16 节省显存
images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=auto_dtype)

# -------------------------
# 3. 构造多模态 Prompt
# -------------------------
# LongVA 的多模态格式：在文本中用 <image> 占位符指示图像位置
prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"  # 系统指令
    f"<|im_start|>user\n" + "<image>\n" * len(images_and_bboxes) + f"{question}<|im_end|>\n"
    "<|im_start|>assistant\n"  # 模型将从这里开始生成回答
)

# 将 prompt 编码为 input_ids，并把 <image> 替换为 IMAGE_TOKEN_INDEX
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
input_ids = input_ids.unsqueeze(0).to(model.device)  # (1, seq_len)

# -------------------------
# 4. 函数：BBox → 视觉 token 区间映射
# -------------------------

def map_bbox_to_visual_token_ranges():
    """根据给定 BBox 计算其对应的视觉 token 索引区间。

    假设 Vision Tower 把图像分成 N×N 个 336×336 patch，并且每个 patch
    展开为 144 个 token，则可以用 block 索引 * tokens_per_block 计算 token 区间。
    """
    block_size = 336           # Vision Tower 预设的 patch 尺寸
    tokens_per_block = 144     # 每个 patch 输出的 token 数
    visual_token_start = 1     # 注意：0 号位置是 BOS

    # 找到带 BBox 的那张图（这里只取第一张带 BBox 的）
    bbox_idx = next((i for i, (_, bbox) in enumerate(images_and_bboxes) if bbox), None)
    assert bbox_idx is not None, "必须提供至少一个带 BBox 的图片"

    image_path, bbox = images_and_bboxes[bbox_idx]
    image = Image.open(image_path).convert("RGB")

    # 用 image_processor 得到模型实际输入尺寸 (processed_h, processed_w)
    raw_tensor = image_processor(image, return_tensors="pt")["pixel_values"]
    _, _, processed_h, processed_w = raw_tensor.shape

    # 计算 patch 网格行列数
    n_rows = processed_h // block_size
    n_cols = processed_w // block_size

    # 将原图坐标缩放到 processed 尺寸
    x_scale, y_scale = processed_w / image.width, processed_h / image.height
    scaled_xmin, scaled_ymin = bbox[0] * x_scale, bbox[1] * y_scale
    scaled_xmax, scaled_ymax = bbox[2] * x_scale, bbox[3] * y_scale

    # 遍历每个 patch，计算与 BBox 的 IoU（这里只用面积占比）
    selected_blocks = []        # 满足阈值的 patch 索引 (r, c)
    block_overlap_ratios = {}   # 保存所有 patch 的重叠比例，方便后备选
    max_ratio = 0.0

    for r in range(n_rows):
        for c in range(n_cols):
            block_xmin, block_ymin = c * block_size, r * block_size
            block_xmax, block_ymax = (c + 1) * block_size, (r + 1) * block_size

            # 交集面积
            inter_w = max(0, min(block_xmax, scaled_xmax) - max(block_xmin, scaled_xmin))
            inter_h = max(0, min(block_ymax, scaled_ymax) - max(block_ymin, scaled_ymin))
            inter_area = inter_w * inter_h

            ratio = inter_area / (block_size * block_size)  # BBox 占该 patch 的比例
            block_overlap_ratios[(r, c)] = ratio
            max_ratio = max(max_ratio, ratio)

            if ratio >= 0.5:  # 如果超过 50% 重叠就直接选中
                selected_blocks.append((r, c))

    # 若没有 patch 满足 0.5 阈值，就挑重叠率最高的一批
    if not selected_blocks:
        selected_blocks = [k for k, v in block_overlap_ratios.items() if np.isclose(v, max_ratio, atol=1e-6)]

    # 计算这些 patch 对应的 token 索引区间
    selected_token_ranges = []
    for (r, c) in selected_blocks:
        block_idx = r * n_cols + c                 # flatten 后的 patch 序号
        token_start = visual_token_start + block_idx * tokens_per_block
        token_end = token_start + tokens_per_block
        selected_token_ranges.append((block_idx, token_start, token_end))

    print(f"✅ 精准映射视觉 token 范围 (Attention Index)：{selected_token_ranges}")
    return selected_token_ranges

# 预先计算 BBox 对应的 token 区间
selected_token_ranges = map_bbox_to_visual_token_ranges()

# -------------------------
# 5. 首次前向：获取 past_key_values 以及 hidden_states 长度
# -------------------------
max_new_tokens = 1000  # 解码最大长度

generated_ids = input_ids.clone()  # 初始化已生成序列 = 输入序列

eos_token_id = tokenizer.eos_token_id  # 终止 token
past = None                             # 用于增量解码
found_attention = None                  # 存储捕获到的注意力

auto_attn_mode = "flash"  # "flash" 或 "torch"

with torch.inference_mode():
    outputs = model(
        input_ids=input_ids,
        images=images_tensor,
        image_sizes=[img.size for img in images],
        modalities=["image"] * len(images),
        use_cache=True,
        output_attentions=False,   # 首次不需要 attentions
        return_dict=True,
        attn_mode=auto_attn_mode,
        output_hidden_states=True  # 方便我们知道序列总长度
    )
    past = outputs.past_key_values  # 后续增量解码要用

# ---------- 打印序列长度信息 ----------
hidden_states = outputs.hidden_states[0]  # (1, seq_len, hidden_dim)
seq_total = hidden_states.shape[1]
print(f"\n🟢 模型实际输入 embedding 序列长度（text + vision token 总长度）: {seq_total}")
print(f"🟢 hidden_states shape: {hidden_states.shape}")

# Vision Tower 每张图的 patch 数
num_patches_per_image = vision_tower.num_patches_per_side ** 2
num_image_tokens = num_patches_per_image * len(images)
print(f"🟢 图片对应的视觉 token 数量（含所有图片）: {num_image_tokens}")

text_token_len = seq_total - num_image_tokens  # 这里包含 <image> 占位符
print(f"🟢 文本 token 数量: {text_token_len}")
print(f"🟢 图片视觉 token 在 embedding 中的范围: [{text_token_len}, {seq_total - 1}]")

# -------------------------
# 6. 增量解码 & 捕获注意力
# -------------------------
temperature = 1.0

for step in range(max_new_tokens):
    with torch.inference_mode():
        current_input = generated_ids[:, -1:]  # 仅输入最后一个 token
        outputs = model(
            input_ids=current_input,
            images=images_tensor,
            image_sizes=[img.size for img in images],
            modalities=["image"] * len(images),
            use_cache=True,
            past_key_values=past,
            output_attentions=True,   # 需要 attentions！
            return_dict=True,
            attn_mode=auto_attn_mode
        )

        past = outputs.past_key_values  # 更新缓存

        # 采样下一个 token
        next_token_logits = outputs.logits[:, -1, :]
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        # 过滤掉 IMAGE_TOKEN_INDEX，只看可读文本
        filtered_ids = [tid.item() for tid in generated_ids[0] if tid.item() not in (-1, IMAGE_TOKEN_INDEX)]
        current_text = tokenizer.decode(filtered_ids, skip_special_tokens=True)

        # 一旦生成文本里出现 needle，就保存注意力
        if needle in current_text and found_attention is None:
            print(f"🎯🎯🎯 [NEEDLE FOUND at step {step}] 生成中发现 needle：'{needle}'")
            found_attention = [att.detach().cpu() for att in outputs.attentions]

        # 生成到 EOS 就停止
        if next_token_id[0] == eos_token_id:
            print(f"[Step {step}] 遇到 EOS，结束")
            break

    # 手动释放显存
    del outputs
    torch.cuda.empty_cache()

# -------------------------
# 7. Attention 分析
# -------------------------
if found_attention is not None:
    print(f"✅ Needle '{needle}' 被发现，开始 Attention 分析")
    num_layers = len(found_attention)
    num_heads = found_attention[0][0].shape[0]

    for layer_idx, layer_attn in enumerate(found_attention):
        attn_tensor = layer_attn[0]  # shape = (num_heads, 1, seq_total)

        for head_idx in range(num_heads):
            head_attn = attn_tensor[head_idx, 0]  # 取出单个 head 的注意力向量
            head_attn[0] = 0.0  # 忽略 BOS
            head_attn = torch.nan_to_num(head_attn.float(), nan=0.0)
            head_attn = torch.softmax(head_attn, dim=0)  # 归一化到概率

            # ---- (A) 计算 BBox token 的注意力总和 ----
            bbox_sum = sum(
                head_attn[start:end].sum().item()
                for _, start, end in selected_token_ranges
            )

            # ---- (B) 计算非 BBox token 的最大注意力 ----
            mask = torch.ones_like(head_attn, dtype=torch.bool)
            for _, start, end in selected_token_ranges:
                mask[start:end] = False
            non_bbox_max = head_attn[mask].max().item()

            # ---- (C) 判断该 head 是否显著关注 BBox ----
            ratio = bbox_sum / (head_attn.sum().item() + 1e-8)
            if bbox_sum > non_bbox_max and ratio >= 0.1:
                print(
                    f"✅ [L{layer_idx} H{head_idx}] BBox_sum: {bbox_sum:.4f} > max_other: {non_bbox_max:.4f} | 占比: {ratio:.2%}"
                )

            # ---- (D) 打印该 head 的 top‑k 注意力 token ----
            topk_values, topk_indices = torch.topk(head_attn[1:], k=5)  # 排除 BOS
            print(
                f"Layer {layer_idx} Head {head_idx} top‑5 attn tokens (index): "
                f"{(topk_indices + 1).tolist()} values: {topk_values.tolist()}"
            )
else:
    print(f"❌ Needle '{needle}' 未匹配到，跳过 Attention 分析")