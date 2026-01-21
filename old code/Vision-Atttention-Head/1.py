import os
import numpy as np
import matplotlib.pyplot as plt

# 设置数据文件路径
data_path = os.path.join("LongVA_attention_analysis", "MTVQA.npz")

# 加载数据，需要允许 pickle（因为保存的是字典对象）
data = np.load(data_path, allow_pickle=True)

# 聚合每个样本重新计算的视觉注意力得分（排除第一个 token）
# 目标结构： new_agg_scores[layer_idx] = [ [head0, head1, ...], [head0, head1, ...], ... ]
new_agg_scores = {}
num_samples = 0

# 遍历 NPZ 文件中的每个 sample
for sample_id in data.files:
    sample = data[sample_id].item()  # 获取保存的字典数据
    # 这里必须存在 "attention" 和 "split_index"
    if "attention" not in sample or "split_index" not in sample:
        continue
    num_samples += 1
    # 获取 attention 数据，注意保存时 key 为字符串，形状假设为 (batch, num_heads, query_length, key_length)
    attn_dict = sample["attention"]
    split_index = sample["split_index"]
    # 如果 split_index 小于等于0，直接跳过
    if split_index <= 0:
        continue

    new_scores = {}  # 用于存储本样本每一层各 head 的新得分
    for layer_key in attn_dict.keys():
        # 将层号统一转换为 int
        try:
            layer_idx = int(layer_key)
        except:
            continue
        attn = attn_dict[layer_key]
        # 转为 numpy 数组，如果有 batch 维度且 batch=1，则取第一个
        if hasattr(attn, 'numpy'):
            attn = attn.numpy()
        if attn.ndim == 4:
            attn = attn[0]  # shape: (num_heads, query_length, key_length)
        elif attn.ndim == 3:
            pass
        else:
            attn = np.expand_dims(attn, axis=0)
        # 这里排除第一个 token：只使用 key tokens 从索引1开始
        attn_sub = attn[:, :, 1:]
        # 对于视觉部分，由于原始 split_index 表示原始 token 序号（包含第0个），
        # 去掉第0个后，新序列中视觉 token 起始索引为 (split_index - 1)
        visual_start = split_index - 1
        # 计算每个 head 的得分：视觉部分的 attention 和除以全部（排除第0个 token）的 attention 和
        layer_scores = []
        for h in range(attn_sub.shape[0]):
            total = np.sum(attn_sub[h])
            visual = np.sum(attn_sub[h][:, visual_start:]) if total > 0 else 0.0
            ratio = visual / total if total > 0 else 0.0
            layer_scores.append(ratio)
        new_scores[layer_idx] = layer_scores
    # 将 new_scores 聚合到 new_agg_scores 中
    for layer_idx, head_scores in new_scores.items():
        if layer_idx not in new_agg_scores:
            new_agg_scores[layer_idx] = []
        new_agg_scores[layer_idx].append(head_scores)

print(f"Aggregated new video attention scores (excluding token 0) from {num_samples} samples.")

# 如果没有样本满足条件，则退出
if num_samples == 0 or len(new_agg_scores) == 0:
    raise ValueError("No valid samples found with required attention data.")

# 计算每个 layer 中各 head 得分的平均值和标准差
agg_mean = {}  # {layer_idx: np.array(平均得分)}
agg_std = {}   # {layer_idx: np.array(标准差)}
for layer_idx, score_lists in new_agg_scores.items():
    score_arr = np.array(score_lists)  # shape: (num_samples, num_heads)
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
plt.title("Mean Video Attention Scores Across Layers and Heads\n(excluding the first token)")
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
plt.title("Std Video Attention Scores Across Layers and Heads\n(excluding the first token)")
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
plt.title("Video Attention Scores per Head Across Layers\n(excluding the first token)")
plt.legend()
plt.tight_layout()
plt.savefig("video_attention_lineplot.png")
plt.show()
