import os, random, re, json
from datasets import load_dataset
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch, numpy as np

# ---------------- 固定随机种子 ----------------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ---------------- 常量配置 ----------------
MODEL_PATH    = "lmms-lab/LongVA-7B-DPO"
VIDEO_PATH    = "movie.mp4"
MAX_FRAMES    = 64
NUM_EXPS      = 5
MAX_NEW_TOK   = 1000
TEMP          = 1.0
RESULT_FILE   = "result.txt"
ATTN_JSON_FILE= "attention_records.jsonl"

# ---------------- 载入测试集 ----------------
ds = load_dataset("lmms-lab/v_niah_needles")["test"]

# ---------------- 载入模型 ----------------
tokenizer, model, image_proc, _ = load_pretrained_model(
    MODEL_PATH, None, "llava_qwen", device_map="cuda"
)
vision_tower = model.get_vision_tower()
patch_tok = vision_tower.num_patches_per_side ** 2 // 4  # 144
eos_id = tokenizer.eos_token_id

# ---------------- 预处理视频帧 ----------------
vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
frame_idx = np.linspace(0, len(vr)-1, MAX_FRAMES, dtype=int).tolist()
frames = vr.get_batch(frame_idx).asnumpy()
video_size = Image.fromarray(frames[0]).size
video_t = image_proc.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

# ---------------- 初始化输出文件 ----------------
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    f.write("Vision‑Needle 实验日志\n")

# ====================== 主循环 ======================
for sample_idx, sample in enumerate(ds):
    q_text   = sample["question"]
    ans_text = sample["answer"].strip()
    img_data = sample["image"]

    needle_img = img_data.convert("RGB") if isinstance(img_data, Image.Image) else Image.open(img_data["path"]).convert("RGB")
    needle_img = needle_img.resize(video_size, Image.BICUBIC)

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<image>\n" + q_text + "<|im_end|>\n<|im_start|>assistant\n"
    )
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") \
        .unsqueeze(0).to(model.device)
    prompt_len = input_ids.shape[1]

    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n===== Sample {sample_idx+1} =====\nQ: {q_text}\nGT: {ans_text}\n")
    print(f"\n===== Sample {sample_idx+1} =====\nQ: {q_text}\nGT: {ans_text}")

    for exp in range(NUM_EXPS):
        print(f"\n--- Experiment {exp+1} ---")
        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n--- Experiment {exp+1} ---\n")

        ins = random.randint(0, MAX_FRAMES)
        needle_t = image_proc.preprocess(np.array(needle_img), return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
        combined = torch.cat([video_t[:ins], needle_t, video_t[ins:]], dim=0)
        sizes = [video_size]*ins + [needle_img.size] + [video_size]*(MAX_FRAMES-ins)
        modalities = ["video"]

        with torch.inference_mode():
            outs = model(
                input_ids=input_ids,
                images=[combined],
                image_sizes=[sizes],
                modalities=modalities,
                use_cache=True,
                output_hidden_states=True,
            )
        past = outs.past_key_values
        total_seq_len = outs.hidden_states[0].shape[1]
        text_len = prompt_len
        vision_len = total_seq_len - text_len

        vis_prefix = 1
        video_tok = patch_tok * MAX_FRAMES
        needle_tok = vision_len - vis_prefix - video_tok
        token_list = [patch_tok] * MAX_FRAMES
        token_list.insert(ins, needle_tok)

        t_start = text_len + vis_prefix + sum(token_list[:ins])
        t_end = t_start + needle_tok

        print(f"🟢 total_seq_len : {total_seq_len}", file=open(RESULT_FILE, "a"))
        print(f"🟢 text_len      : {text_len}", file=open(RESULT_FILE, "a"))
        print(f"🟢 vision_len    : {vision_len}", file=open(RESULT_FILE, "a"))
        print(f"🟢 vis_prefix    : {vis_prefix}", file=open(RESULT_FILE, "a"))
        print(f"🟢 needle_tokens : {needle_tok}", file=open(RESULT_FILE, "a"))
        print(f"✅ needle token [{t_start},{t_end})", file=open(RESULT_FILE, "a"))

        gen = input_ids.clone()
        found_attn = None
        for step in range(MAX_NEW_TOK):
            with torch.inference_mode():
                outg = model(
                    input_ids=gen[:, -1:],
                    images=[combined],
                    image_sizes=[sizes],
                    modalities=modalities,
                    use_cache=True,
                    past_key_values=past,
                    output_attentions=True,
                    attn_mode="flash",
                )
            past = outg.past_key_values
            nxt = torch.multinomial(torch.softmax(outg.logits[:, -1]/TEMP, dim=-1), 1)
            gen = torch.cat([gen, nxt], dim=-1)

            toks = [i.item() for i in gen[0, text_len:] if i.item() != IMAGE_TOKEN_INDEX]
            txt = tokenizer.decode(toks, skip_special_tokens=True)
            if ans_text.lower() in txt.lower() and found_attn is None:
                found_attn = [a.cpu() for a in outg.attentions]
            if nxt.item() == eos_id:
                break

        full = tokenizer.decode(
            [i for i in gen[0].tolist() if i >= 0 and i != IMAGE_TOKEN_INDEX],
            skip_special_tokens=True
        )
        print("Generated:", full)
        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write("\nGenerated:\n" + full + "\n")

        if found_attn is not None:
            record = {
                "sample": sample_idx,
                "exp": exp,
                "insert": ins,
                "t_start": t_start,
                "t_end": t_end,
                "layers": {}
            }
            for li, layer in enumerate(found_attn):
                att = layer[0].squeeze(1)
                record["layers"][f"l{li:02d}"] = {
                    f"h{hi:02d}": att[hi].tolist()
                    for hi in range(att.shape[0])
                }
            with open(ATTN_JSON_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        else:
            print("❌ No attention captured, skip save")
