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
    # 返回 input_ids 用于后续计算视觉 token 位置
    return output_text, attention_maps, split_index, input_token_length, input_ids

def force_fp16(module):
    """递归将模块中所有浮点型参数和缓冲区转换为 FP16"""
    for param in module.parameters():
        if param.is_floating_point():
            param.data = param.data.to(torch.float16)
    for name, buf in module.named_buffers():
        if buf.is_floating_point():
            buf.data = buf.data.to(torch.float16)

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

# 用于保存其他信息（每个 sample 的其他数据）
combined_other_info = {}
# 用于累加所有样本的格式化注意力（最后仅保存一个 aggregated 注意力文件）
aggregated_attention = {}
attention_sample_count = 0

# 新增函数：格式化注意力，将每个注意力头的最后一个 query 的注意力分布转换为列表，键为 "层号-头号"
def format_attention_maps(attn_maps):
    formatted = {}
    for layer_idx, attn in attn_maps.items():
        # attn shape: (batch, num_heads, query_length, key_length)
        for head_idx in range(attn.shape[1]):
            attn_vec = attn[0, head_idx, -1, :].tolist()
            formatted[f"{layer_idx}-{head_idx}"] = attn_vec
    return formatted

for i, idx in enumerate(tqdm(sample_indices, desc="Processing MTVQA dataset", unit="sample")):
    idx = int(idx)
    sample = dataset[idx]
    sample_id = sample["id"]
    
    qa_pairs_str = sample["qa_pairs"]
    # 尝试用 json.loads，如果失败则使用 ast.literal_eval，再失败则尝试替换修正
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

    # 计算视觉 token 的索引（假设所有等于 IMAGE_TOKEN_INDEX 的位置都视为视觉 token）
    video_indices = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].tolist() if split_index is not None else []
    
    video_attention_scores = {}
    for layer_idx, attn in attn_weights.items():
        layer_scores = []
        for h in range(attn.shape[1]):
            numerator = attn[0, h, :, video_indices].sum().item() if video_indices else 0.0
            denominator = attn[0, h, :, :].sum().item()  # 理论上为1.0（softmax 后）
            score = numerator / denominator if denominator > 0 else 0.0
            layer_scores.append(score)
        video_attention_scores[layer_idx] = layer_scores

    # 获取当前 sample 的格式化注意力
    formatted_attentions = format_attention_maps(attn_weights)
    # 将当前 sample 的格式化注意力累加到 aggregated_attention 中
    for key, vec in formatted_attentions.items():
        vec = np.array(vec)
        if key in aggregated_attention:
            aggregated_attention[key] += vec
        else:
            aggregated_attention[key] = vec
    attention_sample_count += 1

    # 保存其他信息（不含注意力数据）
    combined_other_info[sample_id] = {
        "question": question,
        "answer": answer,
        "generated_text": generated_text,
        "video_attention_scores": video_attention_scores,
        "split_index": split_index,
        "input_token_length": input_token_length
    }

    print(f"Processed {i+1}/{num_samples} - ID: {sample_id}, Split Index: {split_index}")
    torch.cuda.empty_cache()

# 对累加的注意力数据取平均，并转换为列表
aggregated_attention_avg = {}
for key, vec in aggregated_attention.items():
    aggregated_attention_avg[key] = (vec / attention_sample_count).tolist()

# 保存其他信息到一个 JSON 文件中
other_info_save_path = os.path.join(output_dir, "MTVQA_other.json")
with open(other_info_save_path, "w") as f:
    json.dump(combined_other_info, f)

# 保存聚合后的注意力数据到一个 JSON 文件中（文件中仅包含类似 {"0-0": [...], "0-1": [...]} 的内容）
final_attn_save_path = os.path.join(output_dir, "all_formatted_attentions.json")
with open(final_attn_save_path, "w") as f:
    json.dump(aggregated_attention_avg, f)

print("Processing complete! Data saved.")
