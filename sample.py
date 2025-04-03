import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from longva.model.builder import load_pretrained_model
from transformers import AutoTokenizer
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
model_path = "lmms-lab/LongVA-7B-DPO"

images_and_bboxes = [
    ["image1.JPG", None],
    ["image3.JPG", None],
    ["image2.JPG", (1000, 2270, 2357, 2802)],
    ["target.JPG", None],
]
question = "What animal is lying on the grass?"
needle = "cat"

# === 模型加载 ===
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda")
vision_tower = model.get_vision_tower()

images = [Image.open(img_path).convert("RGB") for img_path, _ in images_and_bboxes]
images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    f"<|im_start|>user\n" + "<image>\n" * len(images_and_bboxes) + f"{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

# === 推理阶段 ===
max_new_tokens = 1000
generated_ids = input_ids.clone()
eos_token_id = tokenizer.eos_token_id
past = None
temperature = 1
found_attention = None

with torch.inference_mode():
    outputs = model(
        input_ids=input_ids,
        images=images_tensor,
        image_sizes=[img.size for img in images],
        modalities=["image"] * len(images),
        use_cache=True,
        output_attentions=False,
        return_dict=True,
        attn_mode="flash",
        output_hidden_states=True
    )
    past = outputs.past_key_values

# ✅ 提取 hidden_states 并计算 token 分布
hidden_states = outputs.hidden_states[0]  # (batch, seq_len, hidden_dim)
total_seq_len = hidden_states.shape[1]
print(f"\n🟢 模型实际输入 embedding 序列长度（text + vision token 总长度）: {total_seq_len}")
print(f"🟢 hidden_states shape: {hidden_states.shape}")

# num_image_patches = model.get_vision_tower().num_patches_per_side ** 2
# num_image_tokens = num_image_patches * len(images)
# text_token_len = total_seq_len - num_image_tokens
# print(f"🟢 图片对应的视觉 token 数量（含所有图片）: {num_image_tokens}")
# print(f"🟢 文本 token 数量: {text_token_len}")
# print(f"🟢 图片视觉 token 在 embedding 中的范围: [{text_token_len}, {total_seq_len - 1}]")

# # ✅ 获取带 bbox 图片的视觉 token 范围
# def compute_selected_token_ranges_from_output(hidden_states, bbox_idx, num_images, num_patches_per_image):
#     total_seq_len = hidden_states.shape[1]
#     text_token_len = total_seq_len - num_patches_per_image * num_images
#     token_start = text_token_len + bbox_idx * num_patches_per_image
#     token_end = token_start + num_patches_per_image
#     return [(bbox_idx, token_start, token_end)], text_token_len, total_seq_len

# # 🔍 找到带 bbox 的图片索引
# bbox_idx = next((i for i, (_, bbox) in enumerate(images_and_bboxes) if bbox is not None), None)
# assert bbox_idx is not None, "必须提供带有 bbox 的图片"

# selected_token_ranges, text_token_len, total_seq_len = compute_selected_token_ranges_from_output(
#     hidden_states, bbox_idx, len(images), num_image_patches
# )
# print(f"✅ 使用带有 bbox 的图片的视觉 token 范围 (Attention Index)：{selected_token_ranges}")

num_patches_per_grid = vision_tower.num_patches_per_side ** 2

# === 精确计算每张图片对应的视觉 token 数量 ===
grids_per_image_list = []
for img in images:
    patches_tensor = process_images([img], image_processor, model.config)
    grids_per_image_list.append(patches_tensor.shape[0])

tokens_per_image_list = [g * num_patches_per_grid for g in grids_per_image_list]

# === 计算文本 token 长度 ===
num_image_tokens = sum(tokens_per_image_list)
text_token_len = total_seq_len - num_image_tokens

print(f"🟢 图片对应的视觉 token 总数量（所有图片）: {num_image_tokens}")
print(f"🟢 文本 token 数量: {text_token_len}")
print(f"🟢 图片视觉 token 在 embedding 中的范围: [{text_token_len}, {total_seq_len - 1}]")

# === 计算带 bbox 图片的视觉 token 范围 ===
bbox_idx = next((i for i, (_, bbox) in enumerate(images_and_bboxes) if bbox is not None), None)
assert bbox_idx is not None, "必须提供带有 bbox 的图片"

token_cursor = text_token_len
for idx in range(bbox_idx):
    token_cursor += tokens_per_image_list[idx]

token_start = token_cursor
token_end = token_start + tokens_per_image_list[bbox_idx]

selected_token_ranges = [(bbox_idx, token_start, token_end)]

print(f"✅ 带有 bbox 的图片视觉 token 精确范围 (Attention Index)：{selected_token_ranges}")


# === 生成新 token ===
for step in range(max_new_tokens):
    with torch.inference_mode():
        current_input = generated_ids[:, -1:]
        outputs = model(
            input_ids=current_input,
            images=images_tensor,
            image_sizes=[img.size for img in images],
            modalities=["image"] * len(images),
            use_cache=True,
            past_key_values=past,
            output_attentions=True,
            return_dict=True,
            attn_mode="flash"
        )
        past = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        filtered_ids = [tid.item() for tid in generated_ids[0] if tid.item() >= 0 and tid.item() != IMAGE_TOKEN_INDEX]
        current_text = tokenizer.decode(filtered_ids, skip_special_tokens=True)

        if needle in current_text and found_attention is None:
            print(f"🎯🎯🎯 [NEEDLE FOUND at step {step}] 生成中发现 needle：'{needle}'")
            found_attention = [att.detach().cpu() for att in outputs.attentions]

        if next_token_id[0] == eos_token_id:
            print(f"[Step {step}] 遇到 EOS，结束")
            break
    del outputs
    torch.cuda.empty_cache()

# === 后处理 ===
clean_generated_ids = [
    token_id.item() for token_id in generated_ids[0]
    if token_id.item() >= 0 and token_id.item() != IMAGE_TOKEN_INDEX
]
final_output = tokenizer.decode(clean_generated_ids, skip_special_tokens=True)

print("\n最终生成文本：")
print(final_output)

# === 解析 attention 并打印关注度（每层打印最关注目标图片的 head）===
if found_attention is not None:
    print(f"\n✅ Needle '{needle}' 被发现，开始 Attention 分析")

    # === 选中 token 索引 ===
    selected_token_indices = []
    for _, start, end in selected_token_ranges:
        selected_token_indices.extend(range(start, end))
    selected_token_indices = torch.tensor(selected_token_indices, device="cpu")

    print(f"🎯 所选图片视觉 token 区间: {selected_token_ranges}")
    print(f"🎯 所选 token 总数: {len(selected_token_indices)}")
    print(f"🎯 found_attention 层数: {len(found_attention)}\n")

    attention_ratio_threshold = 0.08  # 可调：超过这个值才认为是高关注

    for layer_idx, layer_attn in enumerate(found_attention):
        attn_tensor = layer_attn[0]  # shape: (num_heads, 1, seq_len)
        num_heads = attn_tensor.shape[0]

        for head_idx in range(num_heads):
            head_attn = attn_tensor[head_idx, 0].clone()
            head_attn[0] = 0.0  # 去掉 BOS
            head_attn = torch.nan_to_num(head_attn.float(), nan=0.0)
            # 不再 softmax，因为 flash attention 已是 normalized 分布

            selected_sum = head_attn[selected_token_indices].sum().item()
            total_sum = head_attn.sum().item() + 1e-8
            ratio = selected_sum / total_sum

            if ratio >= attention_ratio_threshold:
                print(f"🔥 Layer {layer_idx:02d} Head {head_idx:02d}: {ratio:.4f}")

    # ✅ 打印每层 attention 的 shape（确认结构）
    shape_info = [a.shape for a in found_attention]
    print(f"\n📐 提取到的 attention shape（每层）:")
    for i, shape in enumerate(shape_info):
        print(f"  Layer {i:02d}: {shape}")
else:
    print(f"\n❌ Needle '{needle}' 未匹配到，跳过 Attention 分析")