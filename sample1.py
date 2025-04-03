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
import math
import argparse

torch.manual_seed(0)
model_path = "lmms-lab/LongVA-7B-DPO"

images_and_bboxes = [
    ["image1.JPG", None],
    ["target.JPG", (1000, 2270, 2357, 2802)],
]
question = "What is the main object in the lower part of the second picture?"
needle = "car"

# 模型加载
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="auto")
vision_tower = model.get_vision_tower()

images = [Image.open(img_path).convert("RGB") for img_path, _ in images_and_bboxes]
images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

pre_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
post_prompt = "<|im_end|>\n<|im_start|>assistant\n"

def vision_embedding(args):
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

def safe_tokenize(tokenizer, text):
    """
    使用 tokenizer.encode() 得到 token id，并去除 BOS token（如果存在）
    """
    tokenized = tokenizer.encode(text, return_tensors="pt")
    if tokenizer.bos_token is not None and tokenized.size(1) > 0 and tokenized[0, 0] == tokenizer.bos_token_id:
        tokenized = tokenized[:, 1:]
    return tokenized

def replace_double_newline_func(token_ids):
    """
    将 token id 271 替换为两个 198
    例如：输入 tensor([[... 271, ...]]) 替换后长度会增加
    """
    # 找到所有等于 271 的位置（注意这里假设 token_ids 的 shape 为 [1, seq_len]）
    double_newline_loc = (token_ids == 271).nonzero()[:, 1]
    # 调整位置索引，确保插入不会出错
    double_newline_loc += torch.arange(len(double_newline_loc))
    if len(double_newline_loc) > 0:
        for loc in double_newline_loc:
            # 将当前位置 token 用两个 198 替换
            token_ids = torch.cat([
                token_ids[:, :loc],
                torch.tensor([[198, 198]], device=token_ids.device),
                token_ids[:, loc+1:]
            ], dim=1)
    return token_ids

def get_text_embedding(text, tokenizer, model, replace_double_newline=False, device=None):
    """
    接收字符串 text，利用 tokenizer 和模型的 embedding 层获取文本 embedding。
    
    参数：
      - text: 待转换的字符串
      - tokenizer: Hugging Face tokenizer 对象
      - model: 包含 .model.embed_tokens 方法的模型（如 Qwen2、LLaVA 等）
      - replace_double_newline: 如果为 True，则将 token id 271 替换成两个 198
      - device: 指定 device，不传时默认使用 model.device

    返回：
      - 一个 tensor，形状为 [1, seq_len, hidden_dim]，数据类型为 bfloat16
    """
    if device is None:
        device = model.device
    # 转为 token id
    token_ids = safe_tokenize(tokenizer, text)
    if replace_double_newline:
        token_ids = replace_double_newline_func(token_ids)
    token_ids = token_ids.to(device)
    with torch.inference_mode():
        embeddings = model.model.embed_tokens(token_ids)
    return embeddings.to(torch.bfloat16)

parser = argparse.ArgumentParser()
parser.add_argument("--pooling_size", type=int, default=0, help="设置池化窗口大小，0 表示不池化")
args = parser.parse_args()
vision_embeddings = vision_embedding(args)
pre_prompt_embeddings = get_text_embedding(pre_prompt, tokenizer, model, replace_double_newline=False)
post_prompt_embeddings = get_text_embedding(post_prompt, tokenizer, model, replace_double_newline=False)
question_embeddings = get_text_embedding(question, tokenizer, model, replace_double_newline=False)

input_emebds = torch.cat([pre_prompt_embeddings, vision_embeddings, question_embeddings, question_embeddings], dim=1)
total_seq_len = input_emebds.shape[1]
print(f"🟢 拼接后的总 embedding 序列长度: {total_seq_len}")
print(f"    pre_prompt 长度: {pre_prompt_embeddings.shape[1]}")
print(f"    视觉 embedding 长度: {vision_embeddings.shape[1]}")
print(f"    question 长度: {question_embeddings.shape[1]}")
print(f"    post_prompt 长度: {post_prompt_embeddings.shape[1]}")

# 构造 position_ids（shape: [1, total_seq_len]）
position_ids = torch.arange(total_seq_len).unsqueeze(0).to(model.device)

# prompt = (
#     "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
#     f"<|im_start|>user\n" + f"{question}<|im_end|>\n"
#     "<|im_start|>assistant\n"
# )
#input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

# === 推理阶段 ===
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
import math
import argparse

torch.manual_seed(0)
model_path = "lmms-lab/LongVA-7B-DPO"

images_and_bboxes = [
    ["image1.JPG", None],
    ["target.JPG", (1000, 2270, 2357, 2802)],
]
question = "What is the main object in the lower part of the second picture?"
needle = "car"

# 模型加载
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="auto")

images = [Image.open(img_path).convert("RGB") for img_path, _ in images_and_bboxes]
images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

pre_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
post_prompt = "<|im_end|>\n<|im_start|>assistant\n"

def vision_embedding(args):
    # 模型加载：加载 tokenizer、model、image_processor 等
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, "llava_qwen", device_map="cuda:0"
    )
    # 删除不需要的层（只保留视觉编码部分）
    del model.model.layers
    model.config.image_aspect_ratio = "pad"
    model.config.mm_patch_merge_type = "flat"

    images = [Image.open(img_path).convert("RGB") for img_path, _ in images_and_bboxes]
    print(f"加载了 {len(images)} 张图片")

    processed_images = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    with torch.inference_mode():
        image_embeddings = model.encode_images(processed_images)

    if args.pooling_size != 0:
        B, L, F = image_embeddings.shape
        n = int(math.sqrt(L))
        image_features_spatial = image_embeddings.view(B, n, n, F).permute(0, 3, 1, 2)
        image_features_spatial_pool = torch.nn.functional.avg_pool2d(
            image_features_spatial, args.pooling_size, args.pooling_size
        )
        image_embeddings = image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous()

    image_embeddings = image_embeddings.view(1, -1, image_embeddings.shape[-1])
    print(f"最终拼接的视觉 embedding shape: {image_embeddings.shape}")
    return image_embeddings

def safe_tokenize(tokenizer, text):
    tokenized = tokenizer.encode(text, return_tensors="pt")
    if tokenizer.bos_token is not None and tokenized.size(1) > 0 and tokenized[0, 0] == tokenizer.bos_token_id:
        tokenized = tokenized[:, 1:]
    return tokenized

def replace_double_newline_func(token_ids):
    double_newline_loc = (token_ids == 271).nonzero()[:, 1]
    double_newline_loc += torch.arange(len(double_newline_loc))
    if len(double_newline_loc) > 0:
        for loc in double_newline_loc:
            token_ids = torch.cat([
                token_ids[:, :loc],
                torch.tensor([[198, 198]], device=token_ids.device),
                token_ids[:, loc+1:]
            ], dim=1)
    return token_ids

def get_text_embedding(text, tokenizer, model, replace_double_newline=False, device=None):
    if device is None:
        device = model.device
    token_ids = safe_tokenize(tokenizer, text)
    if replace_double_newline:
        token_ids = replace_double_newline_func(token_ids)
    token_ids = token_ids.to(device)
    with torch.inference_mode():
        embeddings = model.model.embed_tokens(token_ids)
    return embeddings.to(torch.bfloat16)

parser = argparse.ArgumentParser()
parser.add_argument("--pooling_size", type=int, default=0, help="设置池化窗口大小，0 表示不池化")
args = parser.parse_args()

##################################
# 1) DEBUG: 打印一些关键 DTYPE
##################################
print("==> model.model.embed_tokens.weight.dtype:", model.model.embed_tokens.weight.dtype)
if hasattr(model.model, "layers") and len(model.model.layers) > 0:
    print("==> model.model.layers[0].self_attn.q_proj.weight.dtype:",
          model.model.layers[0].self_attn.q_proj.weight.dtype)

##################################
# 2) 构建embedding
##################################
vision_embeddings = vision_embedding(args)
pre_prompt_embeddings = get_text_embedding(pre_prompt, tokenizer, model, replace_double_newline=False)
post_prompt_embeddings = get_text_embedding(post_prompt, tokenizer, model, replace_double_newline=False)
question_embeddings = get_text_embedding(question, tokenizer, model, replace_double_newline=False)

input_emebds = torch.cat([pre_prompt_embeddings, vision_embeddings, question_embeddings, question_embeddings], dim=1)
print(f"🟢 原始 input_emebds.dtype = {input_emebds.dtype}")
print(f"🟢 拼接后的总长度: {input_emebds.shape[1]}")

##################################
# 3) 可选：将embedding转到模型权重相同的 dtype
##################################
weight_dtype = model.model.embed_tokens.weight.dtype
if input_emebds.dtype != weight_dtype:
    print(f"==> converting input_emebds from {input_emebds.dtype} to {weight_dtype} ...")
    input_emebds = input_emebds.to(weight_dtype)
print(f"🟢 转换后 input_emebds.dtype = {input_emebds.dtype}")

# 构造 position_ids（shape: [1, total_seq_len]）
position_ids = torch.arange(input_emebds.shape[1]).unsqueeze(0).to(model.device)

def eval_forward(model, input_embeds, tokenizer, max_new_tokens=50, temperature=1.0, device=None):
    """
    接受拼接好的 embedding (prompt的embedding)，并生成文本输出。

    参数：
        model: Huggingface transformers 模型
        input_embeds: torch.Tensor, [1, seq_len, hidden_dim]
        tokenizer: 对应的 tokenizer
        max_new_tokens: 生成最大长度
        temperature: 控制随机性程度的超参
        device: torch.device

    返回：
        generated_text: 模型生成的文本（字符串）
    """
    if device is None:
        device = input_embeds.device

    generated_embeds = input_embeds.clone()
    past = None
    generated_ids = []

    with torch.inference_mode():
        # 首先forward一次，得到初始past_key_values
        outputs = model(
            inputs_embeds=generated_embeds,
            use_cache=True,
            output_attentions=False,
            return_dict=True
        )
        past = outputs.past_key_values

        for _ in range(max_new_tokens):
            logits = outputs.logits[:, -1, :] / temperature
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids.append(next_token_id.item())

            # 遇到EOS结束
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            next_token_embed = model.model.embed_tokens(next_token_id).to(input_embeds.dtype)

            outputs = model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                use_cache=True,
                return_dict=True
            )
            past = outputs.past_key_values

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

generated_text = eval_forward(
    model=model,
    input_embeds=input_emebds,
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=0.9
)

print("模型生成的文本是：", generated_text)
##################################
# 4) 进入推理阶段
##################################
# max_new_tokens = 1000
# generated_embeds = input_emebds.clone()  # [1, prompt_len, hidden_dim]
# generated_ids = torch.tensor([], dtype=torch.long, device=model.device)
# eos_token_id = tokenizer.eos_token_id
# past = None
# temperature = 1
# found_attention = None

# print("\n==> Starting forward with inputs_embeds:")
# print("    generated_embeds dtype:", generated_embeds.dtype)
# print("    images_tensor dtype:", images_tensor.dtype)

# with torch.inference_mode():
#     outputs = model(
#         inputs_embeds=generated_embeds,
#         images=images_tensor,
#         image_sizes=[img.size for img in images],
#         modalities=["image"] * len(images),
#         use_cache=True,
#         output_attentions=False,
#         return_dict=True,
#         attn_mode="flash",
#         output_hidden_states=True
#     )
#     past = outputs.past_key_values

# for step in range(max_new_tokens):
#     with torch.inference_mode():
#         next_token_logits = outputs.logits[:, -1, :]
#         probs = torch.softmax(next_token_logits / temperature, dim=-1)
#         next_token_id = torch.multinomial(probs, num_samples=1)

#         if generated_ids.numel() > 0:
#             generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
#         else:
#             generated_ids = next_token_id

#         # 生成 new token embed
#         next_token_embed = model.model.embed_tokens(next_token_id)
#         # debug print
#         if step < 3:  # 前3步看一下
#             print(f"==> step {step}: next_token_embed.dtype = {next_token_embed.dtype}")
#         # 转成和权重匹配
#         next_token_embed = next_token_embed.to(weight_dtype)

#         generated_embeds = torch.cat([generated_embeds, next_token_embed], dim=1)

#         outputs = model(
#             inputs_embeds=next_token_embed,
#             images=images_tensor,
#             image_sizes=[img.size for img in images],
#             modalities=["image"] * len(images),
#             use_cache=True,
#             past_key_values=past,
#             output_attentions=True,
#             return_dict=True,
#             attn_mode="flash"
#         )
#         past = outputs.past_key_values

#         if next_token_id[0, 0] == eos_token_id:
#             print(f"[Step {step}] Encountered EOS, stopping generation.")
#             break

# final_output = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)
# print("\n最终生成文本：")
# print(final_output)


# # === 解析 attention 并打印关注度 ===
# if found_attention is not None:
#     print(f"✅ Needle '{needle}' 被发现，开始 Attention 分析")
    
#     # 视觉 token 区间（基于之前计算）
#     vision_start = text_token_len
#     vision_end = total_seq_len - 1
#     print(f"🎯 视觉 token 范围: [{vision_start}, {vision_end}]")

#     for layer_idx, layer_attn in enumerate(found_attention):
#         print(f"\n🧠 Layer {layer_idx} attention shape: {layer_attn.shape}")
#         batch_size, num_heads, query_len, seq_len = layer_attn.shape
#         assert batch_size == 1 and query_len == 1, "batch=1，query=1才符合生成场景"

#         for head_idx in range(num_heads):
#             attn_scores = layer_attn[0, head_idx, 0, :]  # shape: (seq_len,)
#             attn_scores_no_bos = attn_scores.clone()
#             attn_scores_no_bos[0] = 0.0  # 去除BOS影响

#             total_attn = attn_scores_no_bos.sum().item() + 1e-8
#             max_score, max_idx = torch.max(attn_scores_no_bos, dim=0)
#             ratio = max_score.item() / total_attn

#             # ✅ 只关心视觉区域 token
#             if vision_start <= max_idx.item() <= vision_end:
#                 print(f"   🔎 Head {head_idx}: most attended token idx = {max_idx.item()} "
#                       f"(✅ 视觉区域), score = {max_score.item():.6f}, "
#                       f"占比 = {ratio * 100:.2f}%")

    # num_layers = len(found_attention)
    # num_heads = found_attention[0][0].shape[0]

#     for layer_idx, layer_attn in enumerate(found_attention):
#         attn_tensor = layer_attn[0]  # (num_heads, 1, total_tokens)
#         for head_idx in range(num_heads):
#             head_attn = attn_tensor[head_idx, 0]
#             head_attn[0] = 0.0  # 去掉 BOS
#             head_attn = torch.nan_to_num(head_attn.float(), nan=0.0)
#             head_attn = torch.softmax(head_attn, dim=0)

#             # 计算 BBox 视觉 token 的 attention 总和
#             b_score = sum([head_attn[start:end].sum().item() for _, start, end in selected_token_ranges])

#             # 非 BBox 区域最大 token attention
#             mask = torch.ones_like(head_attn, dtype=torch.bool)
#             for _, start, end in selected_token_ranges:
#                 mask[start:end] = False
#             non_bbox_max = head_attn[mask].max().item()

#             b_ratio = b_score / (head_attn.sum().item() + 1e-8)
#             if b_score > non_bbox_max and b_ratio >= 0.1:
#                 print(f"✅ [L{layer_idx} H{head_idx}] BBox_sum: {b_score:.4f} > max_other: {non_bbox_max:.4f} | 占比: {b_ratio:.2f}")

#             # 🔥 打印 top-k 关注 token
#             topk_values, topk_indices = torch.topk(head_attn[1:], k=5)  # 排除BOS
#             print(f"Layer {layer_idx} Head {head_idx} top-5 attn tokens (attention idx): {topk_indices + 1} values: {topk_values.tolist()}")

# else:
#     print(f"❌ Needle '{needle}' 未匹配到，跳过 Attention 分析")
