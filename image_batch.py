import os
import random
import re
import json
import cv2
import torch
from PIL import Image
from datasets import load_dataset
from longva.model.builder import load_pretrained_model
from transformers import AutoTokenizer
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX

# 固定随机种子
torch.manual_seed(0)
random.seed(0)

# 配置参数
MODEL_PATH        = "lmms-lab/LongVA-7B-DPO"
VIDEO_PATH        = "movie.mp4"
NUM_FRAMES        = 5
NUM_TRIALS        = 5
MAX_NEW_TOKENS    = 100
TEMP              = 1.0
RESULT_FILE       = "image_results.txt"
ATTN_JSON_IMAGE   = "attention_records_image.jsonl"

# Cover crop 调整尺寸
def cover_crop(img: Image.Image, target_size: tuple[int,int]) -> Image.Image:
    tgt_w, tgt_h = target_size
    w, h = img.size
    scale = max(tgt_w / w, tgt_h / h)
    img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    left = (img.width  - tgt_w) // 2
    top  = (img.height - tgt_h) // 2
    return img.crop((left, top, left + tgt_w, top + tgt_h))

# 提取 assistant 段
def extract_assistant_response(text: str) -> str:
    marker = "<|im_start|>assistant\n"
    parts = text.split(marker)
    return marker.join(parts[1:]).strip() if len(parts) >= 2 else text.strip()

# 检查 needle 是否在文本中出现（独立词匹配）
def strict_needle_detect(text: str, needle: str) -> bool:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return needle.lower() in tokens

# 保存 attention 记录
def save_attention_record(sample_idx, trial_idx, needle_idx, t_start, t_end, question, answer, found_attn):
    record = {
        "sample_index": sample_idx,
        "trial_index": trial_idx,
        "question": question,
        "answer": answer,
        "needle_index": needle_idx,
        "t_start": t_start,
        "t_end": t_end,
        "attention": {}
    }
    for l_idx, layer in enumerate(found_attn):
        att = layer[0].squeeze(1)  # (heads, seq_len)
        record["attention"][f"layer_{l_idx:02d}"] = {
            f"head_{h_idx:02d}": att[h_idx].tolist()
            for h_idx in range(att.shape[0])
        }
    with open(ATTN_JSON_IMAGE, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(record, ensure_ascii=False) + "\n")

# 写入日志头
with open(RESULT_FILE, "w", encoding="utf-8") as out_f:
    out_f.write("Image‑Needle 实验日志\n")

# 加载数据 & 模型
ds = load_dataset("lmms-lab/v_niah_needles")["test"]
tokenizer, model, image_processor, _ = load_pretrained_model(
    MODEL_PATH, None, "llava_qwen", device_map="cuda"
)
vision_tower = model.get_vision_tower()
eos_token_id = tokenizer.eos_token_id

# 主循环
for sample_idx, sample in enumerate(ds):
    question = sample["question"]
    answer   = sample["answer"].strip()
    img_data = sample["image"]
    needle_img = Image.open(img_data["path"]).convert("RGB") if not isinstance(img_data, Image.Image) else img_data.convert("RGB")
    target_size = needle_img.size
    needle_img = cover_crop(needle_img, target_size)

    with open(RESULT_FILE, "a", encoding="utf-8") as out_f:
        out_f.write(f"\n===== Sample {sample_idx} =====\nQ: {question}\nGT: {answer}\n")

    for trial in range(NUM_TRIALS):
        with open(RESULT_FILE, "a", encoding="utf-8") as out_f:
            out_f.write(f"\n--- Trial {trial} ---\n")

        cap = cv2.VideoCapture(VIDEO_PATH)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sorted(random.sample(range(total), NUM_FRAMES))
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: continue
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(cover_crop(pil, target_size))
        cap.release()

        needle_idx = random.randint(0, len(frames))
        images = frames.copy()
        images.insert(needle_idx, needle_img)

        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + "<image>\n" * len(images) + question + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(model.device)
        prompt_len = input_ids.shape[1]

        processed = process_images(images, image_processor, model.config)
        images_tensor = torch.stack(processed).to(model.device, dtype=torch.float16) \
                         if isinstance(processed, list) else processed.to(model.device, dtype=torch.float16)

        with torch.inference_mode():
            outs = model(
                input_ids=input_ids,
                images=images_tensor,
                image_sizes=[im.size for im in images],
                modalities=["image"]*len(images),
                use_cache=True,
                output_attentions=False,
                return_dict=True,
                attn_mode="flash",
                output_hidden_states=True
            )
        past = outs.past_key_values
        seq_len = outs.hidden_states[0].shape[1]
        text_len = (input_ids[0] != IMAGE_TOKEN_INDEX).sum().item()
        vis_len = seq_len - text_len
        assert vis_len % len(images) == 0, "视觉 token 不整除"
        tok_per_img = vis_len // len(images)
        t0 = text_len + needle_idx * tok_per_img
        t1 = t0 + tok_per_img

        with open(RESULT_FILE, "a", encoding="utf-8") as out_f:
            out_f.write(f"seq_len={seq_len}, text_len={text_len}, vis_len={vis_len}\n")
            out_f.write(f"每张图 tok={tok_per_img}, needle range=[{t0},{t1})\n")

        gen_ids = input_ids.clone()
        found_attn = None
        for step in range(MAX_NEW_TOKENS):
            with torch.inference_mode():
                o = model(
                    input_ids=gen_ids[:, -1:],
                    images=images_tensor,
                    image_sizes=[im.size for im in images],
                    modalities=["image"]*len(images),
                    use_cache=True,
                    past_key_values=past,
                    output_attentions=True,
                    attn_mode="flash"
                )
            past = o.past_key_values
            nxt = torch.multinomial(torch.softmax(o.logits[:, -1] / TEMP, dim=-1), 1)
            gen_ids = torch.cat([gen_ids, nxt], dim=-1)

            toks = [i.item() for i in gen_ids[0, prompt_len:] if i.item() != IMAGE_TOKEN_INDEX]
            txt  = tokenizer.decode(toks, skip_special_tokens=True)
            asm  = extract_assistant_response(txt)
            if strict_needle_detect(asm, answer) and found_attn is None:
                found_attn = [a.cpu() for a in o.attentions]
            if nxt.item() == eos_token_id:
                break
            del o
            torch.cuda.empty_cache()

        final = tokenizer.decode(
            [i for i in gen_ids[0].tolist() if i >= 0 and i != IMAGE_TOKEN_INDEX],
            skip_special_tokens=True
        )
        with open(RESULT_FILE, "a", encoding="utf-8") as out_f:
            out_f.write("Generated:\n" + final + "\n")

        if found_attn is not None:
            save_attention_record(sample_idx, trial, needle_idx, t0, t1, question, answer, found_attn)
        else:
            with open(RESULT_FILE, "a", encoding="utf-8") as out_f:
                out_f.write("❌ 未检测到 needle Attention，跳过保存\n")
