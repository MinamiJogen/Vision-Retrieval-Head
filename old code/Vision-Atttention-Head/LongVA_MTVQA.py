import os
import json
import ast
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from functools import partial

from datasets import load_dataset

# LongVA
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX

# --------------------- 注意力 hook 及辅助函数 ---------------------
attention_maps = {}

def hook_attention(module, input, output, layer_idx):
    """Hook 用于保存注意力权重。"""
    if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], torch.Tensor):
        attention_maps[layer_idx] = output[1].detach().cpu()

def print_decoded_tokens(input_ids, tokenizer):
    input_ids_list = input_ids.squeeze(0).tolist()
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
    print("\n **Decoded Tokens**:")
    for i, token in enumerate(decoded_tokens):
        print(f"Token[{i}]: {token}")
    
    if IMAGE_TOKEN_INDEX in input_ids_list:
        image_token_pos = input_ids_list.index(IMAGE_TOKEN_INDEX)
        print(f"\n **Image token found at position: {image_token_pos}**\n")
    else:
        print("\n **No <image> token found in input sequence!**\n")

def resize_image(image, max_size=512):
    """
    按比例缩放图像，使得最长边不超过 max_size。
    """
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h))
    else:
        return image

def call_LongVA_with_attention(question, image_input, tokenizer, model, image_processor):
    """
    根据输入的问题和图像调用 LongVA 模型，并返回生成的文本、注意力权重、
    图像 token 的分割位置、输入 token 的长度以及生成时用到的 input_ids。
    参数 image_input 可以是图像文件路径（str）或 PIL Image 对象。
    """
    global attention_maps
    attention_maps = {}  # 重置 attention_maps

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}\n"
        "<image>\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )

    # 获取输入 token，并保存 input_ids 用于后续分析
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    print_decoded_tokens(input_ids, tokenizer)
    input_token_length = input_ids.size(1)

    try:
        split_index = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][0].item()
    except IndexError:
        split_index = None

    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")
    
    image = resize_image(image, max_size=512)
    images_tensor = process_images([image], image_processor, model.config)
    images_tensor = images_tensor.to(model.device, dtype=torch.float16)

    hook_handles = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            handle = layer.self_attn.register_forward_hook(partial(hook_attention, layer_idx=i))
            hook_handles.append(handle)

    gen_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": 256,
        "output_attentions": True
    }

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=[image.size],
            modalities=["image"],
            **gen_kwargs
        )

    for handle in hook_handles:
        handle.remove()

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # 返回 input_ids 用于后续计算（注意：生成过程中未逐步保存 attentions，这里只得到最终一次 forward 的 attentions）
    return output_text, attention_maps, split_index, input_token_length, input_ids

def force_fp16(module):
    """递归将模块中所有浮点型参数和缓冲区转换为 FP16"""
    for param in module.parameters():
        if param.is_floating_point():
            param.data = param.data.to(torch.float16)
    for name, buf in module.named_buffers():
        if buf.is_floating_point():
            buf.data = buf.data.to(torch.float16)

def find_needle_idx(prompt_ids, needle_ids):
    """
    在 prompt_ids 中查找 needle_ids 出现的位置，要求连续匹配。
    返回 (start_index, end_index)，若未找到则返回 (-1, -1)。
    """
    span_len = len(needle_ids)
    prompt_list = prompt_ids.tolist()
    for i in range(len(prompt_list) - span_len + 1):
        if prompt_list[i:i+span_len] == needle_ids:
            return i, i+span_len
    return -1, -1

# --------------------- 模型加载及预处理 ---------------------
print("Loading LongVA model, please wait...")

model_path = "lmms-lab/LongVA-7B-DPO"

tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name="longva_qwen",
    device_map="auto",
    attn_implementation="eager",
    torch_dtype=torch.float16,
    load_8bit=False,
    load_4bit=False
)

if hasattr(model.config, "use_flash_attention"):
    model.config.use_flash_attention = False
model.config.output_attentions = True

model.eval()

# 统一转换模型及其子模块为 FP16
force_fp16(model)
if hasattr(model, "get_vision_tower"):
    vt = model.get_vision_tower()
    if vt is not None:
        force_fp16(vt)
if hasattr(model.get_model(), "mm_projector"):
    mp = model.get_model().mm_projector
    force_fp16(mp)

print("Model loaded successfully, all floating buffers/params forced to float16, ints kept as int!")

# --------------------- 加载 MTVQA 数据集 ---------------------
print("Loading MTVQA dataset...")
ds = load_dataset("ByteDance/MTVQA")
dataset = ds["train"]

num_samples = min(100, len(dataset))
sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
output_dir = "LongVA_attention_analysis"
os.makedirs(output_dir, exist_ok=True)

# 新增：用于累计每个注意力头的 retrieval score（采用原 retrieval head 代码的逻辑）
aggregated_head_scores = {}

attention_results = {}
for i, idx in enumerate(tqdm(sample_indices, desc="Processing MTVQA dataset", unit="sample")):
    idx = int(idx)
    sample = dataset[idx]
    sample_id = sample["id"]
    
    qa_pairs_str = sample["qa_pairs"]
    try:
        qa_pairs = json.loads(qa_pairs_str)
    except Exception as e:
        try:
            qa_pairs = ast.literal_eval(qa_pairs_str)
        except Exception as e2:
            qa_pairs_str_fixed = (qa_pairs_str.replace("'", "\"")
                                              .replace("None", "null")
                                              .replace("True", "true")
                                              .replace("False", "false"))
            try:
                qa_pairs = json.loads(qa_pairs_str_fixed)
            except Exception as e3:
                print(f"Error parsing qa_pairs for sample {sample_id}: {e3}")
                continue

    if not qa_pairs:
        print(f"No qa_pairs found for sample {sample_id}")
        continue

    qa_pair = qa_pairs[0]
    question = qa_pair["question"]
    answer = qa_pair["answer"]

    image_field = sample["image"]
    if isinstance(image_field, dict):
        if "path" in image_field:
            image_input = image_field["path"]
        else:
            raise ValueError(f"Image field for sample {sample_id} does not contain a 'path' key.")
    elif hasattr(image_field, "convert"):
        image_input = image_field
    else:
        raise ValueError(f"Image field for sample {sample_id} is neither a dict nor a PIL Image object.")
    
    try:
        generated_text, attn_weights, split_index, input_token_length, input_ids = call_LongVA_with_attention(
            question, image_input, tokenizer, model, image_processor
        )
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        continue

    # 使用答案作为 needle，计算其 token id 列表
    needle_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    # 在 prompt（即 input_ids[0]）中查找 needle 出现的位置
    needle_start, needle_end = find_needle_idx(input_ids[0], needle_ids)
    if needle_start == -1:
        print(f"Warning: needle not found for sample {sample_id}")
        # 若未找到，则跳过本次计算 retrieval score（或可设为全 0）
        retrieval_head_scores = {}
    else:
        # 根据原 retrieval head 代码，针对每一层每个 head，取最后一步生成 token的注意力向量，计算 top-1 位置
        retrieval_head_scores = {}
        for layer_idx, attn in attn_weights.items():
            layer_scores = []
            # attn shape: (batch, num_heads, query_length, key_length)，取最后 query token（索引 -1）
            for h in range(attn.shape[1]):
                last_query_attn = attn[0, h, -1, :]  # shape (key_length,)
                # topk(1)
                value, index = last_query_attn.topk(1)
                candidate_index = index.item()
                # 判断 candidate_index 是否落在 needle 范围内，并简单忽略生成 token与 prompt token 比较
                if needle_start <= candidate_index < needle_end:
                    score = 1.0 / (needle_end - needle_start)
                else:
                    score = 0.0
                layer_scores.append(score)
            retrieval_head_scores[layer_idx] = layer_scores

    # 累计每个 head 的 retrieval score 到 aggregated_head_scores（格式 "layer-head" -> list of scores）
    for layer_idx, head_scores in retrieval_head_scores.items():
        for head_idx, score in enumerate(head_scores):
            key = f"{layer_idx}-{head_idx}"
            if key not in aggregated_head_scores:
                aggregated_head_scores[key] = []
            aggregated_head_scores[key].append(score)

    # 保存其他数据，与原来保持一致
    attention_results[sample_id] = {
        "question": question,
        "answer": answer,
        "generated_text": generated_text,
        "attention": {str(k): v.numpy() for k, v in attn_weights.items()},
        "retrieval_head_scores": retrieval_head_scores,  # 新增保存 retrieval head 分数
        "split_index": split_index,
        "input_token_length": input_token_length
    }

    print(f"Processed {i+1}/{num_samples} - ID: {sample_id}, needle range: ({needle_start}, {needle_end})")
    torch.cuda.empty_cache()

# 保存 npz 文件（保持原有输出格式不变）
np.savez_compressed(os.path.join(output_dir, "MTVQA.npz"), **attention_results)
print("Processing complete! Data saved.")

# 额外生成包含注意力头 retrieval score 的 JSON 文件，格式为 {"layer-head": [score1, score2, ...], ...}
json_file = os.path.join(output_dir, "MTVQA_head_scores.json")
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(aggregated_head_scores, f, ensure_ascii=False, indent=2)
print(f"Attention head scores saved to {json_file}")
