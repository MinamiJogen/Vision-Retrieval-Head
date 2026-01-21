import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 设置数据文件路径
data_path = os.path.join("LongVA_attention_analysis", "MTVQA.npz")

# 加载数据，需要允许 pickle（因为保存的是字典对象）
data = np.load(data_path, allow_pickle=True)

# 聚合每个样本的 video attention scores
# 目标结构： agg_scores[layer_idx] = [ [head0, head1, ...], [head0, head1, ...], ... ]
agg_scores = {}
num_samples = 0

# 遍历 NPZ 文件中的每个 sample
for sample_id in data.files:
    sample = data[sample_id].item()  # 获取保存的字典数据
    # 优先使用新保存格式（head_token_attention_sums）进行计算
    if "head_token_attention_sums" in sample and "split_index" in sample:
        num_samples += 1
        htas = sample["head_token_attention_sums"]  # shape: (num_layers, num_heads, total_tokens)
        split_index = sample["split_index"]
        # 如果 split_index 小于等于 0，仍然采用原始格式
        if split_index <= 0:
            # 回退到原始保存格式
            if "video_attention_scores" not in sample:
                continue
            v_scores = sample["video_attention_scores"]
            for layer_idx, head_scores in v_scores.items():
                try:
                    layer_idx_int = int(layer_idx)
                except:
                    layer_idx_int = layer_idx
                if layer_idx_int not in agg_scores:
                    agg_scores[layer_idx_int] = []
                agg_scores[layer_idx_int].append(head_scores)
        else:
            # 新方法：排除第一个 token，视觉 token 在新序列中的起始索引为 split_index - 1
            for layer_idx in range(htas.shape[0]):
                if layer_idx not in agg_scores:
                    agg_scores[layer_idx] = []
                head_scores = []
                for h in range(htas.shape[1]):
                    # 取出当前 head 对所有 token 的注意力分布，排除第一个 token（索引 0）
                    token_scores = htas[layer_idx, h, 1:]
                    total = np.sum(token_scores)
                    # 在原始 token 序号中视觉 token 从 split_index 开始，
                    # 排除第一个 token 后，新序列中视觉 token 的起始索引为 (split_index - 1)
                    visual = np.sum(token_scores[split_index - 1:]) if total > 0 else 0.0
                    ratio = visual / total if total > 0 else 0.0
                    head_scores.append(ratio)
                agg_scores[layer_idx].append(head_scores)
    elif "video_attention_scores" in sample:
        num_samples += 1
        v_scores = sample["video_attention_scores"]
        for layer_idx, head_scores in v_scores.items():
            try:
                layer_idx_int = int(layer_idx)
            except:
                layer_idx_int = layer_idx
            if layer_idx_int not in agg_scores:
                agg_scores[layer_idx_int] = []
            agg_scores[layer_idx_int].append(head_scores)
    else:
        # 如果两个字段都不存在，则仍处理下一个 sample
        continue

print(f"Aggregated new video attention scores (excluding token 0 when available) from {num_samples} samples.")

# 如果聚合结果为空，则打印提示（但不直接退出）
if len(agg_scores) == 0:
    print("No valid samples found with required fields. Please check the saved data.")
    sys.exit(1)

# 计算每个 layer 中各 head 得分的平均值和标准差
agg_mean = {}  # {layer_idx: np.array(平均得分)}
agg_std = {}   # {layer_idx: np.array(标准差)}
for layer_idx, score_lists in agg_scores.items():
    # score_lists: list of lists, shape (num_samples, num_heads)
    score_arr = np.array(score_lists)
    mean_scores = score_arr.mean(axis=0)
    std_scores = score_arr.std(axis=0)
    agg_mean[layer_idx] = mean_scores
    agg_std[layer_idx] = std_scores

# 按 layer 排序
sorted_layers = sorted(agg_mean.keys())
# 假设所有 layer 的 head 数量一致，取第一层的 head 数量
num_heads = len(agg_mean[sorted_layers[0]])

# 构造二维矩阵，用于绘制热力图
mean_matrix = np.zeros((len(sorted_layers), num_heads))
std_matrix = np.zeros((len(sorted_layers), num_heads))
for i, layer_idx in enumerate(sorted_layers):
    mean_matrix[i, :] = agg_mean[layer_idx]
    std_matrix[i, :] = agg_std[layer_idx]

# ---------------- 可视化 ----------------

# 1. 绘制平均得分的热力图
plt.figure(figsize=(12, 6))
im = plt.imshow(mean_matrix, cmap='viridis', aspect='auto')
plt.colorbar(im, label="Average Video Attention Score (Excl. token 0)")
plt.xlabel("Head Index")
plt.ylabel("Layer Index")
plt.title("Mean Video Attention Scores Across Layers and Heads\n(excluding the first token when available)")
plt.xticks(np.arange(num_heads))
plt.yticks(np.arange(len(sorted_layers)), labels=sorted_layers)
plt.tight_layout()
plt.savefig("video_attention_mean_heatmap.png")
plt.show()

# 2. 绘制标准差的热力图
plt.figure(figsize=(12, 6))
im2 = plt.imshow(std_matrix, cmap='magma', aspect='auto')
plt.colorbar(im2, label="Std of Video Attention Score (Excl. token 0)")
plt.xlabel("Head Index")
plt.ylabel("Layer Index")
plt.title("Std Video Attention Scores Across Layers and Heads\n(excluding the first token when available)")
plt.xticks(np.arange(num_heads))
plt.yticks(np.arange(len(sorted_layers)), labels=sorted_layers)
plt.tight_layout()
plt.savefig("video_attention_std_heatmap.png")
plt.show()

# 3. 绘制每一层的折线图（带误差条），展示各 head 的平均得分和波动情况
plt.figure(figsize=(12, 8))
for layer_idx in sorted_layers:
    x = np.arange(num_heads)
    y = agg_mean[layer_idx]
    yerr = agg_std[layer_idx]
    plt.errorbar(x, y, yerr=yerr, capsize=4, label=f"Layer {layer_idx}")
plt.xlabel("Head Index")
plt.ylabel("Video Attention Score (Excl. token 0)")
plt.title("Video Attention Scores per Head Across Layers\n(excluding the first token when available)")
plt.legend()
plt.tight_layout()
plt.savefig("video_attention_lineplot.png")
plt.show()
