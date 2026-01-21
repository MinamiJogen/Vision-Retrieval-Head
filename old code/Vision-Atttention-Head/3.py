import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 设置数据文件路径
data_path = os.path.join("LongVA_attention_analysis", "MTVQA.npz")
data = np.load(data_path, allow_pickle=True)

# 我们将为每一层每个 head 累计视觉注意力的分子和总注意力的分母
agg_numerators = {}    # 格式： { layer_idx: np.array([sum_visual_head0, sum_visual_head1, ...]) }
agg_denominators = {}  # 格式： { layer_idx: np.array([sum_total_head0, sum_total_head1, ...]) }
num_samples = 0

# 遍历 NPZ 文件中的每个 sample
for sample_id in data.files:
    sample = data[sample_id].item()
    # 优先使用新格式： head_token_attention_sums + split_index
    if "head_token_attention_sums" in sample and "split_index" in sample:
        num_samples += 1
        htas = sample["head_token_attention_sums"]  # shape: (num_layers, num_heads, total_tokens)
        split_index = sample["split_index"]
        # 如果 split_index <= 0，则无法正确划分视觉 token，直接跳过该 sample
        if split_index <= 0:
            continue
        num_layers_sample = htas.shape[0]
        for layer_idx in range(num_layers_sample):
            if layer_idx not in agg_numerators:
                agg_numerators[layer_idx] = np.zeros(htas.shape[1])
                agg_denominators[layer_idx] = np.zeros(htas.shape[1])
            for h in range(htas.shape[1]):
                # 排除第一个 token：只考虑 key tokens 从索引 1 开始
                token_scores = htas[layer_idx, h, 1:]
                total = np.sum(token_scores)
                # 原始 split_index 包含 token0，排除 token0 后，新序列中视觉 token 起始索引为 (split_index - 1)
                visual = np.sum(token_scores[split_index - 1:]) if total > 0 else 0.0
                agg_numerators[layer_idx][h] += visual
                agg_denominators[layer_idx][h] += total
    # 回退使用原始格式：video_attention_scores 已保存比值
    elif "video_attention_scores" in sample:
        num_samples += 1
        v_scores = sample["video_attention_scores"]
        for layer_idx, head_scores in v_scores.items():
            try:
                layer_idx_int = int(layer_idx)
            except:
                continue
            if layer_idx_int not in agg_numerators:
                # 直接认为 v_scores 中存储的是比值，分母视为1
                agg_numerators[layer_idx_int] = np.array(head_scores, dtype=float)
                agg_denominators[layer_idx_int] = np.ones_like(np.array(head_scores, dtype=float))
            else:
                agg_numerators[layer_idx_int] += np.array(head_scores, dtype=float)
                agg_denominators[layer_idx_int] += 1
    else:
        # 如果两个字段都不存在，则跳过该 sample
        continue

print(f"Aggregated new video attention ratios from {num_samples} samples.")

if len(agg_numerators) == 0:
    print("No valid samples found with required fields. Please check the saved data.")
    sys.exit(1)

# 计算每个层中各 head 的最终注意力占比
agg_ratio = {}
for layer_idx in agg_numerators.keys():
    ratio = np.divide(agg_numerators[layer_idx],
                      agg_denominators[layer_idx],
                      out=np.zeros_like(agg_numerators[layer_idx]),
                      where=agg_denominators[layer_idx] != 0)
    agg_ratio[layer_idx] = ratio

# 按 layer 排序，并构造二维矩阵（层数 x head 数）
sorted_layers = sorted(agg_ratio.keys())
num_heads = len(agg_ratio[sorted_layers[0]])
ratio_matrix = np.zeros((len(sorted_layers), num_heads))
for i, layer_idx in enumerate(sorted_layers):
    ratio_matrix[i, :] = agg_ratio[layer_idx]

# ---------------- 可视化 ----------------

plt.figure(figsize=(12, 6))
im = plt.imshow(ratio_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, label="Aggregated Video Attention Ratio")
plt.xlabel("Head Index")
plt.ylabel("Layer Index")
plt.title("Aggregated Video Attention Ratios Across Layers and Heads\n(excluding the first token)")
plt.xticks(np.arange(num_heads), [str(i + 1) for i in range(num_heads)])
plt.yticks(np.arange(len(sorted_layers)), labels=[str(l + 1) for l in sorted_layers])
plt.tight_layout()
plt.savefig("video_attention_ratio_heatmap.png")
plt.show()
