import os

# --------------------- GPU 自动选择，并设置环境变量 ---------------------
def select_gpu_with_free_memory():
    try:
        import pynvml
    except ImportError:
        raise ImportError("请先安装 pynvml: pip install pynvml")
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    best_gpu = None
    max_free = 0
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        proc_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if len(proc_info) > 0:
            continue  # 跳过被其他进程占用的 GPU
        if mem_info.free > max_free:
            max_free = mem_info.free
            best_gpu = i
    pynvml.nvmlShutdown()
    return best_gpu

selected_gpu = select_gpu_with_free_memory()
if selected_gpu is None:
    selected_gpu = 0
# 限制程序只看到选定的 GPU（之后在 torch 中，该 GPU 编号将从 0 开始）
os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
print(f"[INFO] 设置 CUDA_VISIBLE_DEVICES={selected_gpu}")

# --------------------- 开始导入其他模块 ---------------------
import json
import math
import torch
import numpy as np
import gc  # 用于垃圾回收
from PIL import Image
import argparse
from datetime import datetime, timezone
import inspect

from transformers import AutoTokenizer, AutoConfig
# 使用 longva 的 builder 加载模型（基座模型为 Qwen2）
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX

def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len,
                                                   device=l.self_attn.rotary_emb.inv_freq.device,
                                                   dtype=torch.float32)
    return

scorer = None  # 如果需要 scorer，请自行加载

# --------------------- 主程序开始 ---------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用设备: {device}")

def print_decoded_tokens_brief(input_ids, tokenizer, max_show=10):
    ids_list = input_ids.squeeze(0).tolist()
    # 如果 ids_list 为嵌套列表，则提取内层列表
    if isinstance(ids_list[0], list):
        ids_list = ids_list[0]
    tokens = tokenizer.convert_ids_to_tokens(ids_list)
    
    front_part = tokens[:max_show]
    back_part = tokens[-max_show:] if len(tokens) > max_show else tokens[max_show:]
    print("[DEBUG] Decoded tokens (front):", front_part)
    if back_part:
        print("[DEBUG] Decoded tokens (back):", back_part)
    
    if IMAGE_TOKEN_INDEX in ids_list:
        pos = ids_list.index(IMAGE_TOKEN_INDEX)
        print(f"[DEBUG] <image> token found at position: {pos}")
    else:
        print("[DEBUG] No <image> token found.")

def resize_image(image, max_size=512):
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        return image.resize((int(w * scale), int(h * scale)))
    return image

# --------------------- 主函数：直接使用 forward() 提取注意力 ---------------------
def evaluate_attention_heads(image_list, question, answer, needle):
    """
    输入：
      image_list: 列表，每个元素为 (PIL.Image, bbox) 元组，其中 bbox 为 (x_min, y_min, x_max, y_max) 或 None。
      question: 问题字符串。
      answer: 参考答案字符串。
      needle: 用于大海捞针测试的“needle”字符串。

    输出：
      retrieval_scores: numpy 数组，形状为 (num_layers, num_heads)；
      generated_text: 模型前向计算得到的文本（贪心解码）；
      attention_maps: dict，键为层号，每个值为对应层 attention 张量（形状：batch, num_heads, seq_len, seq_len）；
      split_index: 输入中 <image> token 的位置；
      input_token_length: 输入 token 总数；
      input_ids: 输入 token id 张量。
    """
    torch.manual_seed(0)
    model_path = "lmms-lab/LongVA-7B-DPO"
    # 使用 builder 加载模型；基座模型为 Qwen2时，model_base 传 None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_path,
        attn_implementation="flash_attention_2",
        use_flash_attention_2=False,  # 关闭 Flash Attention，确保返回 attentions
    )
    # 将模型移动到当前 device
    model = model.to(device)
    
    # 强制启用注意力输出
    model.config.output_attentions = True
    # 使用 builder 中设置的默认注意力模式（如 "eager"），若不存在则默认使用 "flash"
    attn_mode = getattr(model, "default_attn_mode", "flash")
    print(f"[DEBUG] torch.backends.cuda.sdp_kernel = {torch.backends.cuda.sdp_kernel}")
    print(f"[DEBUG] model.config.output_attentions = {model.config.output_attentions}")
    
    # 关闭 CUDA SDP 优化（可选）
    torch.backends.cuda.sdp_kernel = "math"
    model.eval()
    print("[INFO] 模型 device:", next(model.parameters()).device)

    # 构造 prompt
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    for _ in image_list:
        prompt += "<image>\n"
    prompt += f"Question: {question}\nAnswer: {answer}\nNeedle: {needle}\n"
    prompt += "<|im_end|>\n<|im_start|>assistant\n"

    # 将 prompt 转为 token
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    print_decoded_tokens_brief(input_ids, tokenizer, max_show=10)
    input_token_length = input_ids.size(1)
    print(f"[DEBUG] input_ids shape: {input_ids.shape}, total length = {input_token_length}")

    try:
        split_index = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][0].item()
        print(f"[DEBUG] split_index (first <image> token) = {split_index}")
    except IndexError:
        split_index = None

    attention_mask = torch.ones_like(input_ids)

    # 对图像列表使用 builder 返回的 image_processor 进行预处理
    images = [img for (img, bbox) in image_list]
    images_tensor = process_images(images, image_processor, model.config).to(device, dtype=torch.float16)

    # 构造 forward 的参数；检查 forward 接口是否支持 attn_mode 参数
    forward_kwargs = {
        "output_attentions": True,
        "use_cache": False,
        "return_dict": True,
    }
    if "attn_mode" in inspect.signature(model.forward).parameters:
        forward_kwargs["attn_mode"] = attn_mode

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images_tensor,
            image_sizes=[img.size for img in images],
            modalities=["image"],
            **forward_kwargs
        )

    if outputs.attentions is None:
        raise ValueError("[ERROR] 未提取到注意力信息，请检查模型配置。outputs.attentions is None!")
    if not isinstance(outputs.attentions, tuple):
        raise ValueError(f"[ERROR] outputs.attentions 类型异常: {type(outputs.attentions)}")

    # 过滤掉返回 None 的注意力层，确保第一层注意力张量非 None
    attention_maps = {i: attn for i, attn in enumerate(outputs.attentions) if attn is not None}
    if len(attention_maps) == 0:
        raise ValueError("[ERROR] 所有层的 attention 均为 None！")
    # 选取第一个非 None 的注意力张量
    first_layer_attn = next(iter(attention_maps.values()))
    print(f"[INFO] 成功提取 {len(attention_maps)} 层注意力矩阵.")
    print(f"[DEBUG] 第一层注意力张量形状: {first_layer_attn.shape} (batch, num_heads, seq_len, seq_len)")

    logits = outputs.logits  # (batch, seq_len, vocab_size)
    generated_ids = logits.argmax(dim=-1)  # (batch, seq_len)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print("[INFO] 生成结果:")
    print(generated_text)
    print("-" * 50)

    # --------------------- 计算检索得分 ---------------------
    prompt_token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    text_len = len(prompt_token_ids)
    patch_size = getattr(model.config, "patch_size", 16)
    image_bbox_token_indices = []
    for idx, (img, bbox) in enumerate(image_list):
        img_width, img_height = img.size
        grid_w = math.ceil(img_width / patch_size)
        grid_h = math.ceil(img_height / patch_size)
        num_img_tokens = grid_w * grid_h
        img_start = text_len + idx * num_img_tokens
        if bbox is None:
            bbox_token_set = set()
        else:
            x_min, y_min, x_max, y_max = bbox
            grid_x_min = int(x_min // patch_size)
            grid_y_min = int(y_min // patch_size)
            grid_x_max = int(x_max // patch_size)
            grid_y_max = int(y_max // patch_size)
            bbox_token_set = set()
            for i in range(grid_y_min, min(grid_y_max+1, grid_h)):
                for j in range(grid_x_min, min(grid_x_max+1, grid_w)):
                    token_index = img_start + i * grid_w + j
                    bbox_token_set.add(token_index)
        image_bbox_token_indices.append(bbox_token_set)
    all_bbox_indices = set()
    for s in image_bbox_token_indices:
        all_bbox_indices.update(s)

    valid_layers = sorted([layer for layer, attn in attention_maps.items() if attn is not None])
    if not valid_layers:
        raise ValueError("[ERROR] 未提取到注意力信息，请检查模型配置。valid_layers 为空!")

    sample_attn = attention_maps[valid_layers[0]]
    num_heads = sample_attn.shape[1]
    tgt_len = sample_attn.shape[2]
    generated_ids = generated_ids[0]  # 取 batch 0
    needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]
    num_needle_tokens = len(needle_ids)
    retrieval_counts = np.zeros((model.config.num_hidden_layers, num_heads))

    for layer in range(model.config.num_hidden_layers):
        if layer not in attention_maps:
            continue
        attn_tensor = attention_maps[layer][0]  # 取 batch 0
        for pos in range(tgt_len):
            token_id = generated_ids[pos].item()
            if token_id in needle_ids:  # 针对 needle token
                for head in range(num_heads):
                    attn_vector = attn_tensor[head, pos, :]
                    max_idx = torch.argmax(attn_vector).item()
                    if max_idx in all_bbox_indices:
                        retrieval_counts[layer, head] += 1

    retrieval_scores = retrieval_counts / num_needle_tokens
    mean_score = retrieval_scores.mean()
    max_score = retrieval_scores.max()
    print(f"[INFO] Retrieval Score: mean={mean_score:.4f}, max={max_score:.4f}")

    ret_scores = retrieval_scores
    ret_text = generated_text
    del input_ids, outputs, images_tensor
    gc.collect()
    torch.cuda.empty_cache()

    return ret_scores, ret_text, attention_maps, split_index, input_token_length, input_ids

# --------------------- 主程序 ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='模型路径')
    parser.add_argument('--model_name', type=str, default="LongVA", help='模型名称')
    args = parser.parse_args()

    # 测试用例：两张图片，第一张无标注，第二张有标注坐标
    img1 = Image.open("image1.JPG").convert("RGB")
    bbox1 = None
    img2 = Image.open("target.JPG").convert("RGB")
    bbox2 = (1000, 2270, 2357, 2802)
    images = [(img1, bbox1), (img2, bbox2)]

    question = "What is the main object in the lower part of the second picture?"
    answer = "It's a black car."
    needle = "car"

    retrieval_scores, generated_text, attn_maps, split_index, input_token_length, input_ids = evaluate_attention_heads(
        images, question, answer, needle
    )

    results = {
        "generated_text": generated_text,
        "retrieval_scores": retrieval_scores.tolist(),
        "model": "LongVA-7B-DPO",
        "split_index": split_index,
        "input_token_length": input_token_length
    }
    output_dir = "LongVA_attention_analysis"
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Results saved to {result_file}")
