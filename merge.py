# save_delta.py
import re, torch
import os
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file   # pip install safetensors

TXT  = "Qwen/Qwen2-7B-Instruct"
LVA  = "lmms-lab/LongVA-7B-DPO"
OUT  = "qwen2_longva_delta"          # 只生成这个目录

# 1. 读两个模型（fp32 方便算平均）
m_txt = AutoModelForCausalLM.from_pretrained(TXT,  torch_dtype=torch.float32, trust_remote_code=True)
m_lva = AutoModelForCausalLM.from_pretrained(LVA,  torch_dtype=torch.float32, trust_remote_code=True)

# 2. 检索头列表（示例）
attn = """
Layer 21: H24
Layer 22: H13,H27
Layer 25: H08
Layer 26: H17
Layer 27: H17,H24
"""
lyr_re, hd_re = re.compile(r"Layer\s+(\d+)"), re.compile(r"H(\d+)")
keep = {(int(lyr_re.search(l).group(1)), int(h))
        for l in attn.strip().splitlines()
        for h in hd_re.findall(l)}

cfg      = m_txt.config
n_layer  = cfg.num_hidden_layers
n_head   = cfg.num_attention_heads
h_dim    = cfg.hidden_size // n_head

# 3. 把 transformer block 找出来
def blocks(m):
    for p in ("model.decoder.layers", "model.layers", "model.transformer.h"):
        cur = m
        try:
            for x in p.split('.'):
                cur = getattr(cur, x)
            return cur
        except AttributeError:
            continue
    raise RuntimeError

b_txt, b_lva = blocks(m_txt), blocks(m_lva)

# 4. 融合：非检索头取平均，检索头保持 LongVA
def row_mix(o, l, ly):
    for h in range(n_head):
        if (ly, h) in keep: continue
        s, e = h*h_dim, (h+1)*h_dim
        l.weight.data[s:e] = 0.5*(o.weight.data[s:e] + l.weight.data[s:e])
        l.bias.data[s:e]   = 0.5*(o.bias.data[s:e]   + l.bias.data[s:e])

def col_mix(o, l, ly):
    for h in range(n_head):
        if (ly, h) in keep: continue
        s, e = h*h_dim, (h+1)*h_dim
        l.weight.data[:,s:e] = 0.5*(o.weight.data[:,s:e] + l.weight.data[:,s:e])
    if o.bias is not None:
        l.bias.data = 0.5*(o.bias.data + l.bias.data)

for i,(bo,bl) in enumerate(zip(b_txt,b_lva),1):
    at, al = (getattr(bo,"self_attn",None) or getattr(bo,"attn")), \
             (getattr(bl,"self_attn",None) or getattr(bl,"attn"))
    if hasattr(at,"q_proj"):
        row_mix(at.q_proj, al.q_proj,i); row_mix(at.k_proj, al.k_proj,i); row_mix(at.v_proj, al.v_proj,i)
    else:
        row_mix(at.qkv_proj, al.qkv_proj, i)
    op = "o_proj" if hasattr(at,"o_proj") else "out_proj"
    col_mix(getattr(at,op), getattr(al,op), i)

print("✅ 融合完成，只导出 transformer 权重")

# 5. 只挑 transformer 权重保存
delta = {k: v.half() for k,v in m_lva.state_dict().items()
         if k.startswith(("model.layers","model.transformer.h","model.decoder.layers"))}
os.makedirs(OUT, exist_ok=True)
save_file(delta, f"{OUT}/delta.safetensors")
print(f"🎉 delta saved to {OUT}/delta.safetensors  ({sum(t.numel() for t in delta.values())/1e9:.2f}B params)")
