#!/usr/bin/env python
# merge_longva_full_fp32.py
import json, torch
from pathlib import Path
from longva import LlavaLlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from tqdm import tqdm

BASE_MODEL = Path(
    "/disk3/minami/huggingface/hub/models--lmms-lab--LongVA-7B-DPO/"
    "snapshots/57cadd841419ecdeee050ca2bbdf7c1f96584fcd"
)
DELTA_FILE = Path(
    "/disk3/minami/Vision-Retrieval-Head/qwen2_longva_delta/delta.safetensors"
)
OUTPUT_DIR = Path("/disk3/minami/huggingface/hub/models--LongVA-Merge")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ 1. 载入 LongVA (fp32)
print("◆ load LongVA‑7B (fp32)")
model = LlavaLlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,    # 目标是全 fp32
    low_cpu_mem_usage=False,
    trust_remote_code=True,       # 重要：用到 llava_qwen 自定义类
)

# materialize vision
vision = model.get_vision_tower()            # 新接口
_ = vision.vision_tower.state_dict()

# ----- 2. 合并 delta -----
delta = load_file(DELTA_FILE)
with torch.no_grad():
    for k, v in delta.items():
        if k in model.state_dict():
            model.state_dict()[k].copy_(v.to(dtype=model.state_dict()[k].dtype))

# ----- 3. 保存 -----
model.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size="8GB")
AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True).save_pretrained(OUTPUT_DIR)

vision_dir = OUTPUT_DIR / "vision_tower"
vision.vision_tower.save_pretrained(vision_dir)

cfg = json.load(open(BASE_MODEL / "config.json"))
cfg["vision_tower"] = cfg["mm_vision_tower"] = vision_dir.name   # 相对路径
json.dump(cfg, open(OUTPUT_DIR / "config.json", "w"), indent=2)

print("✅ merged model saved →", OUTPUT_DIR)