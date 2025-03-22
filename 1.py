import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from longva.model.builder import load_pretrained_model
from transformers import AutoTokenizer
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
import torch
import numpy as np

# 固定随机种子
torch.manual_seed(0)

model_path = "lmms-lab/LongVA-7B-DPO"
image_path = "target.JPG"

tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="auto")

prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<image>\nDescribe the image in details.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

image = Image.open(image_path).convert("RGB")

raw_image_tensor = image_processor(image, return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)

images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

vision_tower = model.get_vision_tower()

with torch.no_grad():
    print("Raw image tensor shape:", raw_image_tensor.shape)
    vision_feats = vision_tower(raw_image_tensor)
    print("Vision features shape:", vision_feats.shape)
    print("Vision feature contains NaN?", torch.isnan(vision_feats).any())

max_new_tokens = 1000
generated_ids = input_ids.clone()
eos_token_id = tokenizer.eos_token_id
past = None
temperature = 1
final_attentions = None

# === 初始 prompt 阶段走 flash attention ===
with torch.inference_mode():
    print("[Init] 使用 FlashAttention 处理初始 input")
    outputs = model(
        input_ids=input_ids,
        images=images_tensor,
        image_sizes=[image.size],
        modalities=["image"],
        use_cache=True,
        past_key_values=None,
        output_attentions=False,
        return_dict=True,
        attn_mode="flash"
    )
    past = outputs.past_key_values

# === 逐 token 生成 ===
for step in range(max_new_tokens):
    with torch.inference_mode():
        current_input = generated_ids[:, -1:]

        outputs = model(
            input_ids=current_input,
            images=images_tensor,
            image_sizes=[image.size],
            modalities=["image"],
            use_cache=True,
            past_key_values=past,
            output_attentions=True,
            return_dict=True,
            attn_mode="flash"
        )

        past = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        print(f"[Step {step}] logits min: {next_token_logits.min().item()}, max: {next_token_logits.max().item()}")
        if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
            print(f"[Step {step}] logits 出现异常")
            print(next_token_logits)
            break

        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # 记录 attention
        final_attentions = [att.detach().cpu() for att in outputs.attentions]

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id[0] == eos_token_id:
            print(f"[Step {step}] 生成到达 EOS，结束")
            break

    del outputs
    torch.cuda.empty_cache()

# 后处理
clean_generated_ids = [
    token_id.item() for token_id in generated_ids[0]
    if token_id.item() >= 0 and token_id.item() != IMAGE_TOKEN_INDEX
]
final_output = tokenizer.decode(clean_generated_ids, skip_special_tokens=True)

print("\n最终生成文本：")
print(final_output)
# print("\n最后一步的 attention 分布：")
# print(final_attentions)
