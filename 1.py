import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# --------------------- 导入 CUDA 相关库前请先设置环境变量 ---------------------
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np

# 固定随机种子
torch.manual_seed(0)

model_path = "lmms-lab/LongVA-7B-DPO"
image_path = "target.JPG"
max_frames_num = 16  # 根据 GPU 内存调节
gen_kwargs = {
    "do_sample": True, 
    "temperature": 0.5, 
    "top_p": None, 
    "num_beams": 1, 
    "use_cache": True, 
    "max_new_tokens": 1024
}

# 使用 grid 加载模型（不使用 8bit 加载方式，统一使用 float16）
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, None, "llava_qwen", device_map="auto", offload_folder="offload"
)

# 将模型转换为 float16
model = model.half()

prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<image>\nDescribe the image in details.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(model.device)

# 处理图像，并转换为 float16
image = Image.open(image_path).convert("RGB")
images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

max_new_tokens = 100  # 根据需要设定生成长度
generated_ids = input_ids.clone()
eos_token_id = tokenizer.eos_token_id
all_attentions = []

for step in range(max_new_tokens):
    with torch.inference_mode():
        outputs = model(
            input_ids=generated_ids,   # 一次性输入整个生成序列
            images=images_tensor,
            image_sizes=[image.size],
            modalities=["image"],
            use_cache=False,           # 禁用缓存，确保每一步都重新计算完整 attention
            output_attentions=True,    # 请求返回 attention 权重
            return_dict=True,
        )
    
    # 将每步的 attention 转移到 CPU，并 detach 防止梯度追踪
    step_attentions = [att.detach().cpu() for att in outputs.attentions]
    all_attentions.append(step_attentions)
    
    # 采样下一个 token
    next_token_logits = outputs.logits[:, -1, :]
    probs = torch.softmax(next_token_logits / 0.5, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)
    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    
    # 清理内存
    del outputs
    torch.cuda.empty_cache()
    
    if next_token_id[0] == eos_token_id:
        print("生成到达EOS token，停止")
        break

# 清洗生成的 token，排除特殊 token（例如 IMAGE_TOKEN_INDEX）
clean_generated_ids = [
    token_id.item() for token_id in generated_ids[0]
    if token_id.item() >= 0 and token_id.item() != IMAGE_TOKEN_INDEX
]
final_output = tokenizer.decode(clean_generated_ids, skip_special_tokens=True)
print(final_output)
print(all_attentions)
