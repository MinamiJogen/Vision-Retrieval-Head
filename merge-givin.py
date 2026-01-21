#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoModelForCausalLM
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX


def get_transformer_layers(model):
    """
    获取 Transformer 模型中的 encoder 或 decoder 层列表（即 .layers）。
    自动适配不同模型的层级结构，如 Qwen、LLaMA、LongVA 等。
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "llama") and hasattr(model.llama, "layers"):
        return model.llama.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        return model.transformer.layers
    raise AttributeError(f"无法在模型 {model.__class__.__name__} 中找到 .layers 属性")


def main():
    # 1. 解析命令行参数：fusion α
    parser = argparse.ArgumentParser(description="Fuse LongVA & Qwen2 attention heads")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Fusion ratio: 0 = pure LongVA, 1 = pure Qwen2 (0 <= α <= 1)"
    )
    parser.add_argument("--output_dir", type=str, default="/disk3/minami/huggingface/hub/models--LongVA-Merge")
    args = parser.parse_args()
    output_dir = args.output_dir

    # Clamp α 到 [0,1]
    ALPHA = max(0.0, min(1.0, args.alpha))
    # ALPHA = 1.0
    BETA = 1.0 - ALPHA
    print(f"[INFO] fusion α = {ALPHA:.2f}  (β = {BETA:.2f})")

    # 2. 加载 LongVA-7B 模型，部署到 GPU
    tokenizer, longva_model, image_processor, _ = load_pretrained_model(
        "lmms-lab/LongVA-7B",
        None,
        "llava_qwen",
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    longva_model = longva_model.eval()
    longva_layers = get_transformer_layers(longva_model)

    # 3. 加载 Qwen2-7B-Instruct 模型，仅加载到 CPU（作为参数源）
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()
    qwen_layers = get_transformer_layers(qwen_model)

    # 4. 提取模型结构参数
    cfg = longva_model.config
    H = cfg.num_attention_heads              # 总注意力头数
    KV = cfg.num_key_value_heads              # KV 共享组数
    D = cfg.hidden_size                       # 隐藏维度
    head_dim = D // H                         # 每个头的维度
    group_size = H // KV                      # 每组包含的头数

    # 5. 指定需融合的注意力头（每层的 head 下标列表）
    heads_to_merge = {
        14: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        18: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        19: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        20: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    }

    # 6. 遍历所有指定层和头，逐个复制合并参数（按 α/β 比例线性插值）
    with torch.no_grad():
        for layer_idx, head_indices in heads_to_merge.items():
            attn_long = longva_layers[layer_idx].self_attn
            attn_qwen = qwen_layers[layer_idx].self_attn.to(attn_long.q_proj.weight.device)

            merged_groups = set()
            for h in head_indices:
                g = h // group_size                  # 当前头所在的 KV 组
                q0, q1 = h * head_dim, (h + 1) * head_dim
                k0, k1 = g * head_dim, (g + 1) * head_dim

                # ---------------- Q ----------------
                attn_long.q_proj.weight.data[q0:q1] \
                    .mul_(BETA).add_(ALPHA * attn_qwen.q_proj.weight.data[q0:q1])
                if attn_long.q_proj.bias is not None:
                    attn_long.q_proj.bias.data[q0:q1] \
                        .mul_(BETA).add_(ALPHA * attn_qwen.q_proj.bias.data[q0:q1])

                # ----------- K / V（去重处理） ----------
                if g not in merged_groups:
                    for proj_long, proj_qwen in (
                        (attn_long.k_proj, attn_qwen.k_proj),
                        (attn_long.v_proj, attn_qwen.v_proj),
                    ):
                        proj_long.weight.data[k0:k1] \
                            .mul_(BETA).add_(ALPHA * proj_qwen.weight.data[k0:k1])
                        if proj_long.bias is not None:
                            proj_long.bias.data[k0:k1] \
                                .mul_(BETA).add_(ALPHA * proj_qwen.bias.data[k0:k1])
                    merged_groups.add(g)

                # ---------------- O ----------------
                attn_long.o_proj.weight.data[:, q0:q1] \
                    .mul_(BETA).add_(ALPHA * attn_qwen.o_proj.weight.data[:, q0:q1])
                if attn_long.o_proj.bias is not None:
                    attn_long.o_proj.bias.data[q0:q1] \
                        .mul_(BETA).add_(ALPHA * attn_qwen.o_proj.bias.data[q0:q1])

                print(f"  ✅ Layer {layer_idx:2d}  Head {h:2d} merged (α={ALPHA:.2f})")

            attn_qwen.to("cpu")
            torch.cuda.empty_cache()


    # 7. 保存合并后的 LongVA 模型
    # output_dir = "/disk3/minami/huggingface/hub/models--LongVA-Merge"
    os.makedirs(output_dir, exist_ok=True)
    longva_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"🎉 模型合并完成，已保存至：{output_dir}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()