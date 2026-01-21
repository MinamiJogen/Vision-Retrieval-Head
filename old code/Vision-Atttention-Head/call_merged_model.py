import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from functools import partial

# A-OKVQA
from aokvqa.load_aokvqa import load_aokvqa, get_coco_path

# LongVA（注意这里调用的是合并后的模型）
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX

attention_maps = {}

def hook_attention(module, input, output, layer_idx):
    """Hook to store attention weights."""
    if (
        isinstance(output, tuple)
        and len(output) > 1
        and isinstance(output[1], torch.Tensor)
    ):
        attention_maps[layer_idx] = output[1].detach().cpu()

def print_input_token_length(input_ids):
    """
    打印模型输入的 token 长度，不进行 token 解码。
    """
    # input_ids 的 shape 为 [batch, token_length]，这里 batch size 为1
    token_length = input_ids.size(1)
    print(f"\n🔹 **Input token length: {token_length}**\n")
    
    # 如果需要，也可以打印 <image> token 出现的位置（可选）
    input_ids_list = input_ids.squeeze(0).tolist()
    if IMAGE_TOKEN_INDEX in input_ids_list:
        image_token_pos = input_ids_list.index(IMAGE_TOKEN_INDEX)
        print(f"**<image> token found at position: {image_token_pos}**\n")
    else:
        print("**No <image> token found in input sequence!**\n")

def call_LongVA_with_attention(question, image_path, tokenizer, model, image_processor):
    global attention_maps
    attention_maps = {}  # 每次调用前重置 attention_maps

    # 构造 Prompt
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}\n"
        "<image>\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )

    # (1) 构造文本输入 input_ids（int64 格式）
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    
    # 记录并打印输入 token 的长度
    print_input_token_length(input_ids)

    # 注意这里计算 split_index 是基于 input_ids 的，
    # 保证后续分析时用的 split 与 attention 的维度对得上
    try:
        split_index = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][0].item()
    except IndexError:
        split_index = None

    # (2) 处理图像，并转为 float16
    image = Image.open(image_path).convert("RGB")
    images_tensor = process_images([image], image_processor, model.config)
    images_tensor = images_tensor.to(model.device, dtype=torch.float16)

    # 注册 attention hook
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.register_forward_hook(partial(hook_attention, layer_idx=i))

    # 生成参数配置
    gen_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": 512,
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

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # 返回 output_text、attention_maps、split_index 以及输入 token 长度
    return output_text, attention_maps, split_index, input_ids.size(1)

# --------------------- 主流程 ---------------------
# 修改这里的 model_path 为合并后的模型目录
merged_model_path = "huggingface/hub/merged_longva"
print("🔄 Loading merged LongVA model, please wait...")

# 1) 以 FP16 加载合并后的模型
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path=merged_model_path,
    model_base=None,
    model_name="longva_qwen",
    device_map="cuda:0",
    attn_implementation="eager",
    torch_dtype=torch.float16,   # 以 float16 加载权重
    load_8bit=False,
    load_4bit=False
)

# 2) 关闭闪存注意力，并开启输出注意力
if hasattr(model.config, "use_flash_attention"):
    model.config.use_flash_attention = False
model.config.output_attentions = True

model.eval()

# 3) 强制将所有浮点型参数转换为 float16，整数类型不变
for param in model.parameters():
    if param.is_floating_point():
        param.data = param.data.to(torch.float16)

for name, buf in model.named_buffers():
    # 仅转换浮点型缓冲区
    if buf.is_floating_point():
        buf.data = buf.data.to(torch.float16)

# vision tower
if hasattr(model, "get_vision_tower"):
    vt = model.get_vision_tower()
    if vt is not None:
        # 递归转换所有参数
        for p in vt.parameters():
            if p.is_floating_point():
                p.data = p.data.to(torch.float16)
        for bn, bbuf in vt.named_buffers():
            if bbuf.is_floating_point():
                bbuf.data = bbuf.data.to(torch.float16)

# mm_projector
if hasattr(model.get_model(), "mm_projector"):
    mp = model.get_model().mm_projector
    for p in mp.parameters():
        if p.is_floating_point():
            p.data = p.data.to(torch.float16)
    for bn, bbuf in mp.named_buffers():
        if bbuf.is_floating_point():
            bbuf.data = bbuf.data.to(torch.float16)

print("Model loaded successfully, all floating buffers/params forced to float16, ints kept as int!")

# 4) 处理 A-OKVQA 数据集
aokvqa_dir = "./aokvqa/datasets/aokvqa/"
coco_dir   = "./aokvqa/datasets/coco/"
train_dataset = load_aokvqa(aokvqa_dir, 'train')

# 随机抽取 5 个样本（而非前 5 个任务）
num_samples = min(5, len(train_dataset))
sample_indices = np.random.choice(len(train_dataset), num_samples, replace=False)
output_dir = "Merged_attention_analysis"
os.makedirs(output_dir, exist_ok=True)

attention_results = {}
for i, idx in enumerate(tqdm(sample_indices, desc="Processing dataset", unit="sample")):
    dataset_example = train_dataset[idx]

    question_id = dataset_example['question_id']
    question = dataset_example['question']
    choices = dataset_example['choices']
    correct_choice = choices[dataset_example['correct_choice_idx']]
    image_path = get_coco_path('train', dataset_example['image_id'], coco_dir)

    # 获取生成结果，同时返回 attention_maps、split_index 和输入 token 长度
    generated_text, attn_weights, split_index, input_token_length = call_LongVA_with_attention(
        question, image_path, tokenizer, model, image_processor
    )

    attention_results[question_id] = {
        "question": question,
        "choices": choices,
        "correct_choice": correct_choice,
        "generated_text": generated_text,
        "attention": attn_weights,
        "split_index": split_index,
        "input_token_length": input_token_length  # 保存输入 token 数量
    }

    print(f"Processed {i+1}/{num_samples} - QID: {question_id}, Split Index: {split_index}")

np.savez_compressed(os.path.join(output_dir, "aokvqa.npz"), **attention_results)
print("Processing complete! Data saved.")
