import os, random, torch, math
from PIL import Image
from longva.model.builder import load_pretrained_model
from transformers import AutoTokenizer
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX

torch.manual_seed(0)

# ---------------- 基本配置 ----------------
model_path = "lmms-lab/LongVA-7B-DPO"
question = ("Find the frame of a couple in a wedding. In side the frame, there is a balloon "
            "on the bridegroom's head. What is the color of that balloon? Answer using a single "
            "word or phrase.")
needle_answer = "Yellow"
needle = needle_answer  # 用于匹配生成文本中的关键字
max_new_tokens = 1000
temperature = 1.0

# ---------- 0. cover‑crop 使尺寸完全一致 ----------
def cover_crop(img, target_size):
    tgt_w, tgt_h = target_size
    w, h = img.size
    scale = max(tgt_w / w, tgt_h / h)
    img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    left = (img.width  - tgt_w) // 2
    top  = (img.height - tgt_h) // 2
    return img.crop((left, top, left + tgt_w, top + tgt_h))

needle_img   = Image.open("image/needle.JPG").convert("RGB")
target_size  = needle_img.size
print(f"🖼️ 统一到 needle 尺寸: {target_size}")

other_paths  = ["image/image1.JPG", "image/image3.JPG", "image/image2.JPG"]
other_images = [cover_crop(Image.open(p).convert("RGB"), target_size) for p in other_paths]
needle_img   = cover_crop(needle_img, target_size)

needle_idx = random.randint(0, len(other_images))
images     = other_images.copy()
images.insert(needle_idx, needle_img)

# ---------- 1. 模型 & 一次性 process_images ----------
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, None, "llava_qwen", device_map="cuda")
vision_tower = model.get_vision_tower()

processed_imgs = process_images(images, image_processor, model.config)
if isinstance(processed_imgs, list):
    images_tensor  = torch.stack(processed_imgs).to(model.device, dtype=torch.float16)
else:
    images_tensor  = processed_imgs.to(model.device, dtype=torch.float16)

# ---------- 2. prompt ----------
prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
          "<|im_start|>user\n" + "<image>\n"*len(images) + f"{question}<|im_end|>\n"
          "<|im_start|>assistant\n")
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                  return_tensors="pt").unsqueeze(0).to(model.device)

# ---------- 3. 前向推理 ----------
with torch.inference_mode():
    out = model(input_ids=input_ids,
                images=images_tensor,
                image_sizes=[img.size for img in images],
                modalities=["image"]*len(images),
                output_hidden_states=True,
                use_cache=True,
                attn_mode="flash")

seq_len = out.hidden_states[0].shape[1]
text_len = (input_ids[0] != IMAGE_TOKEN_INDEX).sum().item()
vis_len  = seq_len - text_len
assert vis_len % len(images) == 0
tokens_per_image = vis_len // len(images)
tokens_each = [tokens_per_image] * len(images)

print(f"\n🟢 输入序列长度: {seq_len}")
print(f"🟢 文本 token: {text_len} | 总视觉 token: {vis_len}")
print(f"🟢 每张图视觉 token: {tokens_per_image}")

# ---------- 4. needle token 区间 ----------
cursor = text_len + sum(tokens_each[:needle_idx])
token_start = cursor
token_end   = cursor + tokens_each[needle_idx]
selected_token_ranges = [(needle_idx, token_start, token_end)]
print(f"✅ needle token 区间: {selected_token_ranges}")

# ---------- 5. 生成准备变量补全 ----------
generated_ids = input_ids.clone()
past = out.past_key_values
eos_token_id = tokenizer.eos_token_id
found_attention = None

# ---------- 6. 循环生成新 token ----------
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

# ---------- 7. 输出生成文本 ----------
clean_generated_ids = [
    token_id.item() for token_id in generated_ids[0]
    if token_id.item() >= 0 and token_id.item() != IMAGE_TOKEN_INDEX
]
final_output = tokenizer.decode(clean_generated_ids, skip_special_tokens=True)
print("\n最终生成文本：")
print(final_output)

# ---------- 8. 分析 attention ----------
if found_attention is not None:
    print(f"\n✅ Needle '{needle}' 被发现，开始 Attention 分析")

    selected_token_indices = []
    for _, start, end in selected_token_ranges:
        selected_token_indices.extend(range(start, end))
    selected_token_indices = torch.tensor(selected_token_indices, device="cpu")

    print(f"🎯 所选图片视觉 token 区间: {selected_token_ranges}")
    print(f"🎯 所选 token 总数: {len(selected_token_indices)}")
    print(f"🎯 found_attention 层数: {len(found_attention)}\n")

    attention_ratio_threshold = 0.08

    for layer_idx, layer_attn in enumerate(found_attention):
        attn_tensor = layer_attn[0]
        num_heads = attn_tensor.shape[0]

        for head_idx in range(num_heads):
            head_attn = attn_tensor[head_idx, 0].clone()
            head_attn[0] = 0.0
            head_attn = torch.nan_to_num(head_attn.float(), nan=0.0)

            selected_sum = head_attn[selected_token_indices].sum().item()
            total_sum = head_attn.sum().item() + 1e-8
            ratio = selected_sum / total_sum

            if ratio >= attention_ratio_threshold:
                print(f"🔥 Layer {layer_idx:02d} Head {head_idx:02d}: {ratio:.4f}")

    shape_info = [a.shape for a in found_attention]
    print(f"\n📐 Attention shape（每层）:")
    for i, shape in enumerate(shape_info):
        print(f"  Layer {i:02d}: {shape}")
else:
    print(f"\n❌ Needle '{needle}' 未匹配到，跳过 Attention 分析")
