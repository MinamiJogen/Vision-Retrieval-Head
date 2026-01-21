from call_LLaVA import call_LLaVA_with_attention
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from aokvqa.load_aokvqa import load_aokvqa, get_coco_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path

# **1. 预加载 LLaVA 模型**
disable_torch_init()
model_path = "liuhaotian/llava-v1.5-7b" # aria, long va
model_name = get_model_name_from_path(model_path)

print("🔄 Loading LLaVA model, please wait...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name, attn_implementation="eager"
)
model.config.output_attentions = True
model.eval()  # 进入推理模式，减少计算
print("Model loaded successfully!")

# 设置数据集路径
aokvqa_dir = "./aokvqa/datasets/aokvqa/"
coco_dir = "./aokvqa/datasets/coco/"
train_dataset = load_aokvqa(aokvqa_dir, 'train')

# 仅处理前 50 个样本
num_samples = min(50, len(train_dataset))

# 创建保存目录
output_dir = "attention_analysis"
os.makedirs(output_dir, exist_ok=True)

# 处理数据并添加进度条
attention_results = {}
for i in tqdm(range(num_samples), desc="Processing dataset", unit="sample"):
    dataset_example = train_dataset[i]

    # 获取问题和图像路径
    question_id = dataset_example['question_id']
    question = dataset_example['question']
    choices = dataset_example['choices']
    correct_choice = choices[dataset_example['correct_choice_idx']]
    image_path = get_coco_path('train', dataset_example['image_id'], coco_dir)

    # 调用 LLaVA，但不重复加载模型
    result, attn_weights, split_index = call_LLaVA_with_attention(
        question, image_path, tokenizer, model, image_processor
    )
    
    # 保存注意力数据，同时记录 split_index
    attention_results[question_id] = {
        "question": question,
        "choices": choices,
        "correct_choice": correct_choice,
        "generated_text": result,
        "attention": attn_weights,
        "split_index": split_index  # 记录 split_index
    }

    # 打印调试信息
    print(f"Processed {i+1}/{num_samples} - QID: {question_id}, Correct: {correct_choice}, Split Index: {split_index}")

# 保存所有注意力权重（包括 split_index）
np.savez_compressed(os.path.join(output_dir, "attention_data.npz"), **attention_results)

print("Processing complete! Data saved.")
