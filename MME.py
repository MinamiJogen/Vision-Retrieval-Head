#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video-MME 推理（LongVA-7B-DPO + fused delta）
→ 2700 条结果写 mme_merge_longva_output.json
"""

import os, json, re, random, gc, collections
from typing import List, Dict

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from decord import VideoReader, cpu
from safetensors.torch import load_file

# reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Paths
PARQUET_PATH = (
    "/disk3/minami/huggingface/hub/datasets--lmms-lab--Video-MME/"
    "snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/"
    "test-00000-of-00001.parquet"
)
VIDEO_DIR  = "/disk3/minami/Vision-Retrieval-Head/videos"
MODEL_PATH = "lmms-lab/LongVA-7B-DPO"
DELTA_PATH = "qwen2_longva_delta/delta.safetensors"

# LongVA
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token
from longva.constants import IMAGE_TOKEN_INDEX

print("⏳ loading LongVA-7B-DPO ...")
tokenizer, model, image_processor, _ = load_pretrained_model(
    MODEL_PATH, None, "llava_qwen", device_map="cuda:0"
)
print("⏳ merging delta ...")
model.load_state_dict(load_file(DELTA_PATH), strict=False)
model.to(torch.float32).eval(); gc.collect()
print("✅ model ready")

GEN_KWARGS = dict(
    do_sample=False,
    temperature=0,
    top_p=None,
    num_beams=1,
    max_new_tokens=128,
    use_cache=False,
)

LETTERS   = list("ABCD")
CHOICE_RE = re.compile(r"\b([ABCD])\b")

def extract_choice(txt: str) -> str:
    m = CHOICE_RE.search(txt.upper())
    if m: return m.group(1)
    tail = txt.strip()[-1:].upper() if txt else ""
    return tail if tail in LETTERS else ""

def build_prompt(question: str, options: List[str]) -> str:
    sys = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    opt_lines = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
    return (
        sys + f"<|im_start|>user\n<image>\nQuestion: {question}\n"
        + "\n".join(opt_lines)
        + "\nAnswer with ONLY the letter (A/B/C/D).\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

# dataset
dataset = load_dataset("parquet", data_files=PARQUET_PATH, split="train")

# frame cache  (Tensor shape = F×3×224×224)
CACHE_MAX = 24
frame_cache: "collections.OrderedDict[str, torch.Tensor]" = collections.OrderedDict()

def get_video_tensor(vid: str) -> torch.Tensor:
    if vid in frame_cache:
        frame_cache.move_to_end(vid)
        return frame_cache[vid]

    path = os.path.join(VIDEO_DIR, "data", f"{vid}.mp4")
    vr = VideoReader(path, ctx=cpu(0))
    idx = np.linspace(0, len(vr) - 1, num=min(16, len(vr)), dtype=int)
    frames = vr.get_batch(idx).asnumpy()                       # (F,H,W,3)  uint8
    tensor = image_processor.preprocess(
        frames, return_tensors="pt"         # ← 官方写法
    )["pixel_values"].to(model.device, dtype=torch.float32)    # (F,3,224,224)

    frame_cache[vid] = tensor
    if len(frame_cache) > CACHE_MAX:
        frame_cache.popitem(last=False)
    return tensor

# main loop
results: List[Dict] = []
for item in tqdm(dataset, total=len(dataset), desc="Processing"):
    vid_id, question_text = item["videoID"], item["question"]
    options, answer_letter = item["options"], item["answer"]
    task_type = item.get("task_type", "Unknown")
    if not options or len(options) < 2: continue

    try:
        video_tensor = get_video_tensor(vid_id)
    except Exception as e:
        print(f"⚠️ {vid_id} frame error: {e}"); continue

    prompt = build_prompt(question_text, options)
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            input_ids, images=[video_tensor], modalities=["video"], **GEN_KWARGS
        )

    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    pred = extract_choice(decoded)
    if not pred: print(f"❓ no letter for {vid_id}: {decoded[:80]}")

    results.append(dict(
        video_id   = vid_id,
        duration   = item.get("duration"),
        domain     = item.get("domain"),
        sub_category = item.get("sub_category"),
        task_type  = task_type,
        question   = question_text,
        options    = options,
        answer     = answer_letter,
        response   = pred,
    ))

    del input_ids
    torch.cuda.empty_cache()

# save
OUT_FILE = "mme_merge_longva_output.json"
with open(OUT_FILE, "w", encoding="utf-8") as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2)

print(f"🎉 done → {OUT_FILE} (total {len(results)} records)")