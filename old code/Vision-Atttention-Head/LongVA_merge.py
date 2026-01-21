import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX

# ---------------------------
# 1. 加载 LongVA 和 Qwen2 模型
# ---------------------------
longva_tokenizer, longva_model, image_processor, _ = load_pretrained_model(
    model_path="lmms-lab/LongVA-7B-DPO",
    model_base=None,
    model_name="longva_qwen",
    device_map="cuda:0",
    attn_implementation="eager",
    torch_dtype=torch.float16,   # 以 float16 加载权重
    load_8bit=False,
    load_4bit=False
)
qwen2_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
qwen2_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")

# 获取各自的 state_dict
longva_state_dict = longva_model.state_dict()
qwen2_state_dict = qwen2_model.state_dict()

# ---------------------------
# 2. 设置合并参数
# ---------------------------
global_alpha = 0.5  # 非注意力层基座参数的全局合并比例

# 自动获取 Qwen2 的注意力头数量（使用 Qwen2 模型的配置）
num_heads = qwen2_model.config.num_attention_heads

# 接口：可选自定义每个注意力头的合并系数（形状 (1, num_heads, 1)），若为 None，则默认所有头均为 0.5
custom_head_merge_factors = None

# ---------------------------
# 3. 合并 LongVA 的基座参数与 Qwen2 模型权重
# ---------------------------
merged_state_dict = {}

for key, param_longva in longva_state_dict.items():
    # 判断是否属于 Qwen2 基座部分（假设键名以 "transformer." 开头）
    if key.startswith("transformer."):
        if key in qwen2_state_dict:
            param_qwen2 = qwen2_state_dict[key]
        else:
            print(f"Warning: Key {key} not found in Qwen2 model; using LongVA weight only.")
            merged_state_dict[key] = param_longva
            continue

        # 判断是否为注意力层的参数（例如 Q、K、V 投影矩阵）
        if "attn" in key and param_longva.ndim == 2 and (param_longva.shape[1] % num_heads == 0):
            in_features, total_dim = param_longva.shape
            head_dim = total_dim // num_heads

            # 重塑为 (in_features, num_heads, head_dim)
            param_longva_heads = param_longva.view(in_features, num_heads, head_dim)
            param_qwen2_heads = param_qwen2.view(in_features, num_heads, head_dim)

            # 获取每个注意力头的合并系数
            if custom_head_merge_factors is None:
                head_merge_factors = torch.full((1, num_heads, 1), 0.5, dtype=torch.float32, device=param_longva.device)
            else:
                head_merge_factors = custom_head_merge_factors.view(1, num_heads, 1)

            # 对每个注意力头进行合并：
            # merged = head_merge_factor * (LongVA 权重) + (1 - head_merge_factor) * (Qwen2 权重)
            merged_heads = head_merge_factors * param_longva_heads + (1 - head_merge_factors) * param_qwen2_heads
            # 恢复原始形状
            merged_param = merged_heads.view(in_features, total_dim)
        else:
            # 对于其他基座参数，使用全局合并比例
            merged_param = global_alpha * param_longva + (1 - global_alpha) * param_qwen2
        merged_state_dict[key] = merged_param
    else:
        # 对于 LongVA 中非基座部分（例如视觉模块）的参数，直接保留 LongVA 权重
        merged_state_dict[key] = param_longva

# ---------------------------
# 4. 加载合并后的权重并保存模型
# ---------------------------
# 使用 assign=True 参数确保 meta 参数正确赋值，避免警告
longva_model.load_state_dict(merged_state_dict, assign=True)
merged_model_path = "huggingface/hub/merged_longva"  # 请替换为你希望保存的实际路径
longva_model.save_pretrained(merged_model_path)

print(f"模型合并完成，保存到：{merged_model_path}")
