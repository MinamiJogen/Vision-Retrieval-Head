import torch, numpy as np, re
from PIL import Image
from decord import VideoReader, cpu
from longva.model.builder import load_pretrained_model
from longva.mm_utils       import tokenizer_image_token
from longva.constants      import IMAGE_TOKEN_INDEX

torch.manual_seed(0)
np.random.seed(0)

# ---------------- 配置 ----------------
model_path  = "lmms-lab/LongVA-7B-DPO"
video_path  = "movie.mp4"
needle_path = "image/needle.JPG"
question = ("Find the frame of a couple in a wedding. In side the frame, there is a balloon "
            "on the bridegroom's head. What is the color of that balloon? "
            "Answer the question using a single word or phrase.")
needle = "Yellow"

# ---------------- 模型加载 -----------------
tokenizer, model, image_proc, _ = load_pretrained_model(
    model_path, None, "llava_qwen", device_map="cuda"
)
vision_tower = model.get_vision_tower()
patch_per_image = vision_tower.num_patches_per_side ** 2 // 4  # e.g. 576 // 4 = 144

print("📌 num_patches_per_side:", vision_tower.num_patches_per_side)

# ---------------- needle 图像处理 ---------------
needle_img = Image.open(needle_path).convert("RGB")

# ---------------- 视频帧处理 --------------------
vr          = VideoReader(video_path, ctx=cpu(0))
max_frames  = 16
frame_idx   = np.linspace(0, len(vr) - 1, max_frames, dtype=int).tolist()
frames_np   = vr.get_batch(frame_idx).asnumpy()
video_size  = Image.fromarray(frames_np[0]).size

# needle resize 成视频帧大小
needle_img = needle_img.resize(video_size, Image.BICUBIC)

# pixel_values tensor
video_tensor = image_proc.preprocess(frames_np, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
needle_tensor = image_proc.preprocess(np.array(needle_img), return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

# 拼接视觉输入
insert_idx = max_frames // 2
combined_tensor = torch.cat([
    video_tensor[:insert_idx], needle_tensor, video_tensor[insert_idx:]
], dim=0)
combined_sizes = [video_size] * insert_idx + [needle_img.size] + [video_size] * (max_frames - insert_idx)
modalities = ["video"]

# prompt 构造
prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<image>\n" + question + "<|im_end|>\n<|im_start|>assistant\n"
)
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
prompt_len = input_ids.shape[1]

# ---------------- forward 获取 hidden ----------------
with torch.inference_mode():
    outputs = model(
        input_ids=input_ids,
        images=[combined_tensor],
        image_sizes=[combined_sizes],
        modalities=modalities,
        use_cache=True,
        output_hidden_states=True
    )
past = outputs.past_key_values
total_seq_len = outputs.hidden_states[0].shape[1]
text_len = prompt_len
vision_len = total_seq_len - text_len

print(f"\n🟢 total_seq_len : {total_seq_len}")
print(f"🟢 prompt_len    : {text_len}")
print(f"🟢 vision_len    : {vision_len}")

# ---------------- 视觉 token 动态统计 ----------------
# UniRes 前缀 token 数
vis_prefix = 1
# 假设前缀在最前，实际 token 为 prefix + 每张图 patch
tokens_each = [patch_per_image] * max_frames
actual_token_sum = vision_len - vis_prefix
needle_token_len = actual_token_sum - sum(tokens_each)
tokens_each.insert(insert_idx, needle_token_len)

print(f"🟢 vis_prefix    : {vis_prefix}")
print(f"🟢 needle tokens : {needle_token_len}")
print(f"📊 所有图片视觉 token 分布:")
for i, tok in enumerate(tokens_each):
    tag = "← needle" if i == insert_idx else ""
    print(f"  Img {i:02d}: {tok:3d} {tag}")

# needle token 范围
token_start = text_len + vis_prefix + sum(tokens_each[:insert_idx])
token_end   = token_start + tokens_each[insert_idx]
print(f"✅ needle token 区间: [{token_start}, {token_end})")

# ---------------- 生成 ----------------
max_new = 1000
eos_id = tokenizer.eos_token_id
generated_ids = input_ids.clone()
found_attention = None

for step in range(max_new):
    with torch.inference_mode():
        outg = model(
            input_ids=generated_ids[:, -1:],
            images=[combined_tensor],
            image_sizes=[combined_sizes],
            modalities=modalities,
            use_cache=True,
            past_key_values=past,
            output_attentions=True,
            attn_mode="flash",
        )
    past = outg.past_key_values
    logits = outg.logits[:, -1, :]
    next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
    generated_ids = torch.cat([generated_ids, next_token], dim=1)

    decoded = tokenizer.decode(
        [tid for tid in generated_ids[0, prompt_len:] if tid != IMAGE_TOKEN_INDEX],
        skip_special_tokens=True
    )
    if needle.lower() in decoded.lower() and found_attention is None:
        print(f"🎯 FOUND needle at step {step}")
        found_attention = [att.cpu() for att in outg.attentions]

    if next_token.item() == eos_id:
        break

# ---------------- 输出文本 ----------------
final_output = tokenizer.decode(
    [tid for tid in generated_ids[0].tolist() if tid != IMAGE_TOKEN_INDEX],
    skip_special_tokens=True
)
print("\n📝 最终生成：")
print(final_output)

# ---------------- Attention 分析 ----------------
if found_attention:
    token_range = torch.arange(token_start, token_end)
    print("\n🔍 Attention (ratio ≥ 0.08)")
    for layer_idx, layer in enumerate(found_attention):
        att = layer[0].squeeze(1)  # (heads, seq_len)
        for head_idx in range(att.shape[0]):
            scores = att[head_idx].clone()
            scores[0] = 0.0
            ratio = scores[token_range].sum().item() / (scores.sum().item() + 1e-8)
            if ratio >= 0.08:
                print(f"🔥 Layer {layer_idx:02d} Head {head_idx:02d}: {ratio:.4f}")
    print("\n📐 Attention shape per layer:")
    for i, layer in enumerate(found_attention):
        print(f"  Layer {i:02d}: {layer.shape}")
else:
    print("\n❌ 未捕获到 needle 的 Attention")
