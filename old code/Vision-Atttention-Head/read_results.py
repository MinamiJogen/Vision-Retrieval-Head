import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于显示进度条

def analyze_input_attention_heads(attn_maps, input_token_length, split_index):
    """
    分析每层中针对输入 token 的注意力分布，并计算：
      1. 每个 attention head 对 image token 的注意力比例（head_image_ratios）。
      2. 各层所有 head 比例的均值（visual_attention_ratios）。
      3. 每层中各 token 被关注的累计比例（归一化后，再乘以 1e3），对所有 head 求均值（token_attention_sums）。
      4. 每个 attention head 对各 token 的注意力分布（归一化后，再乘以 1e3），（head_token_attention_sums）。
      
    参数：
      attn_maps: dict，键为层号（新格式下 key 为字符串），值为对应层的 attention 矩阵
                 （可能的形状包括 [batch, num_heads, query_length, key_length]，
                  或 [num_heads, query_length, key_length]，或 [query_length, key_length]）
      input_token_length: 输入 prompt 的 token 总数（包含所有 token）
      split_index: 分界点索引（基于原始 token 序号），表示从该索引开始为视觉 token
      
    返回：
      head_image_ratios: 每层中每个 attention head 对 image token 的注意力比例，形状为 (num_layers, num_heads)
      visual_attention_ratios: 每层所有 head 比例的均值，形状为 (num_layers, 1)
      token_attention_sums: 每层中各 token 被关注的累计比例（归一化后，再乘以 1e3），形状为 (num_layers, input_token_length)
      head_token_attention_sums: 每层中每个 attention head 对各 token 的注意力分布，形状为 (num_layers, num_heads, input_token_length)
    """
    # 注意新保存格式中，attn_maps 的 key 为字符串，这里转换为整数排序
    sorted_layers = sorted([int(k) for k in attn_maps.keys()])
    if not sorted_layers:
        return None, None, None, None

    num_layers = max(sorted_layers) + 1

    # 检查第一层数据以确定 head 数量
    sample_attn = attn_maps[str(sorted_layers[0])]
    if hasattr(sample_attn, 'numpy'):
        sample_attn = sample_attn.numpy()
    if sample_attn.ndim == 4:
        sample_attn = sample_attn[0]  # 形状变为 (num_heads, query_length, key_length)
    elif sample_attn.ndim == 3:
        pass
    else:
        sample_attn = sample_attn.squeeze()
    if sample_attn.ndim == 2:
        num_heads = 1
    else:
        num_heads = sample_attn.shape[0]

    head_image_ratios = np.zeros((num_layers, num_heads))
    token_attention_sums = np.zeros((num_layers, input_token_length))
    head_token_attention_sums = np.zeros((num_layers, num_heads, input_token_length))

    for layer in sorted_layers:
        attn = attn_maps[str(layer)]
        if hasattr(attn, 'numpy'):
            attn = attn.numpy()
        if attn.ndim == 4:
            attn = attn[0]  # (num_heads, query_length, key_length)
        elif attn.ndim == 3:
            pass
        else:
            attn = attn.squeeze()
        if attn.ndim == 2:
            attn = attn[None, :, :]

        # 截取 input token 部分
        attn_sub = attn[:, :input_token_length, :input_token_length]

        if np.isnan(attn_sub).any():
            print(f"Warning: Found NaN in attention data at layer {layer}, applying mean filling.")
            mean_val = np.nanmean(attn_sub)
            attn_sub = np.where(np.isnan(attn_sub), mean_val, attn_sub)

        # 根据 split_index 计算每个 head 对视觉 token 的注意力比例
        for h in range(num_heads):
            total_attn = np.sum(attn_sub[h])
            numerator = np.sum(attn_sub[h][:, split_index:])
            ratio = numerator / total_attn if total_attn != 0 else 0
            head_image_ratios[layer, h] = ratio

        # 计算所有 head 平均的 token 级注意力
        mean_attn = np.mean(attn_sub, axis=0)  # (query_tokens, key_tokens)
        token_sum = mean_attn.sum(axis=0)
        total = token_sum.sum()
        if total != 0:
            token_ratio = token_sum / total
        else:
            token_ratio = np.zeros_like(token_sum)
        token_attention_sums[layer, :] = token_ratio * 1e3

        # 计算每个 head 对各 token 的注意力分布
        for h in range(num_heads):
            token_sum_h = attn_sub[h].sum(axis=0)
            total_h = token_sum_h.sum()
            if total_h != 0:
                token_ratio_h = token_sum_h / total_h
            else:
                token_ratio_h = np.zeros_like(token_sum_h)
            head_token_attention_sums[layer, h, :] = token_ratio_h * 1e3

    visual_attention_ratios = np.mean(head_image_ratios, axis=1, keepdims=True)
    return head_image_ratios, visual_attention_ratios, token_attention_sums, head_token_attention_sums

def plot_token_attention_bar(visual_attention_ratios, token_attention_sums, output_dir, question_id, split_index):
    """
    绘制 token 级注意力分布条形图，显示每个 token 的累计注意力比例。
    为避免因第一个 token 数值过大影响显示，忽略第一个 token（索引 0）。
    同时根据 split_index 将文本 token 和视觉 token 用不同颜色绘制：
      - 文本 token（索引 < split_index）用蓝色
      - 视觉 token（索引 >= split_index）用橙色
    """
    os.makedirs(output_dir, exist_ok=True)
    num_layers, total_tokens = token_attention_sums.shape
    # 忽略第一个 token
    tokens = np.arange(1, total_tokens)
    token_sums = token_attention_sums.mean(axis=0)[1:]
    print(f"Token attention sums for QID {question_id} (excluding token 0): {token_sums}")
    if np.all(token_sums < 1e-12):
        print(f"Token summed attention for QID {question_id} is too small (<1e-12), skipping bar plot.")
        return

    # 根据 split_index 分组（注意：split_index 以原始 token 编号为准，忽略 token0 后，文本 token 为索引 1 ~ split_index-1）
    text_mask = tokens < split_index
    visual_mask = tokens >= split_index

    plt.figure(figsize=(10, 5))
    plt.bar(tokens[text_mask], token_sums[text_mask], color='blue', alpha=0.6, label='Text Tokens')
    plt.bar(tokens[visual_mask], token_sums[visual_mask], color='orange', alpha=0.6, label='Visual Tokens')
    plt.xlabel("Token Index (excluding token 0)")
    plt.ylabel("Normalized Summed Attention (x1e3)")
    plt.title(f"Token-Level Summed Attention (QID: {question_id})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{question_id}_token_attention_distribution.png"))
    plt.close()

def plot_attention_heatmap(head_image_ratios, output_dir, question_id):
    """
    绘制每层各 attention head 对 image token 的注意力比例热力图。
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    num_layers, num_heads = head_image_ratios.shape

    im = plt.imshow(head_image_ratios, aspect='auto', cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, label='Image Attention Ratio')
    plt.xticks(np.arange(num_heads), [str(i + 1) for i in range(num_heads)])
    plt.yticks(np.arange(num_layers), [str(i + 1) for i in range(num_layers)])
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")
    plt.title(f"Image Attention Ratio per Head (QID: {question_id})")
    plt.savefig(os.path.join(output_dir, f"{question_id}_image_attention_heatmap.png"))
    plt.close()

def plot_visual_attention_ratio(visual_attention_ratios, output_dir, question_id):
    """
    绘制各层平均 image attention 比例折线图。
    """
    os.makedirs(output_dir, exist_ok=True)
    layers = np.arange(visual_attention_ratios.shape[0])
    plt.figure(figsize=(10, 5))
    plt.plot(layers, visual_attention_ratios.squeeze(), marker='o', color='red')
    plt.xlabel("Layer")
    plt.ylabel("Average Image Attention Ratio")
    plt.title(f"Average Image Attention Ratio per Layer (QID: {question_id})")
    plt.savefig(os.path.join(output_dir, f"{question_id}_avg_image_attention_ratio.png"))
    plt.close()

def plot_attention_heads_token_distributions(head_token_attention_sums, output_dir, question_id, split_index):
    """
    绘制一个网格图，横坐标为 attention head，纵坐标为层号。
    每个子图显示对应 attention head 对各 token 的注意力分布（归一化后乘以 1e3）。
    为避免第一个 token 数值过大影响显示，忽略第一个 token（索引 0）。
    同时根据 split_index 使用不同颜色：
      - 文本 token（索引 < split_index）用蓝色
      - 视觉 token（索引 >= split_index）用橙色
    绘图过程中增加了进度条以提示绘制进度。
    """
    os.makedirs(output_dir, exist_ok=True)
    num_layers, num_heads, total_tokens = head_token_attention_sums.shape
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 3, num_layers * 2), squeeze=False)
    
    # 忽略第一个 token：令 tokens 从1开始
    tokens = np.arange(1, total_tokens)
    total_iterations = num_layers * num_heads
    pbar = tqdm(total=total_iterations, desc="Drawing Head Token Attention Grid", leave=False)
    for i in range(num_layers):
        for j in range(num_heads):
            ax = axes[i][j]
            # 仅绘制从 token 1 开始的数据
            attn_dist = head_token_attention_sums[i, j, 1:]
            # 根据 split_index 分组
            text_mask = tokens < split_index
            visual_mask = tokens >= split_index
            ax.bar(tokens[text_mask], attn_dist[text_mask], color='blue', alpha=0.6)
            ax.bar(tokens[visual_mask], attn_dist[visual_mask], color='orange', alpha=0.6)
            ax.set_title(f"L{i+1}, H{j+1}", fontsize=8)
            if i < num_layers - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Token Index (excluding token 0)", fontsize=8)
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("Attention (x1e3)", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=6)
            # 在左上角的第一个子图添加图例
            if i == 0 and j == 0:
                ax.legend(["Text Tokens", "Visual Tokens"], fontsize=6)
            pbar.update(1)
    pbar.close()
    plt.tight_layout()
    plt.suptitle(f"Token Attention Distribution per Head (QID: {question_id})", fontsize=10, y=1.02)
    plt.savefig(os.path.join(output_dir, f"{question_id}_head_token_attention_grid.png"), bbox_inches='tight')
    plt.close()

def pad_token_attention(token_attn_sums, target_length):
    """
    如果 token_attn_sums 的第二维长度小于 target_length，则在右侧填充 0 以匹配 target_length，否则返回原数组。
    """
    current_length = token_attn_sums.shape[1]
    if current_length < target_length:
        pad_width = target_length - current_length
        token_attn_sums = np.pad(token_attn_sums, ((0, 0), (0, pad_width)), mode='constant')
    return token_attn_sums

def main():
    # 指定 NPZ 文件路径
    npz_file = "LongVA_attention_analysis/MTVQA.npz"
    data = np.load(npz_file, allow_pickle=True)
    print("Loaded NPZ file with", len(data.keys()), "entries.\n")

    base_name = os.path.splitext(os.path.basename(npz_file))[0]
    global_output_dir = os.path.join("analysis_results", base_name)
    os.makedirs(global_output_dir, exist_ok=True)

    global_visual_ratios = None  # 累计所有问题的平均 image attention 比率，形状为 (num_layers, 1)
    global_token_sums = None     # 累计所有问题的 token 注意力和，形状为 (num_layers, max_tokens)
    count = 0
    all_questions_dir = os.path.join(global_output_dir, "ALL_QUESTIONS")
    os.makedirs(all_questions_dir, exist_ok=True)
    max_tokens = 0

    # 如果测试用例过多，只分析前 10 个
    keys = list(data.keys())
    if len(keys) > 10:
        keys = keys[:10]

    for i, question_id in enumerate(keys):
        sample_data = data[question_id].item()
        print(f"Processing Question ID: {question_id}")

        input_token_length = sample_data["input_token_length"]
        attn_layers = sample_data["attention"]
        split_index = sample_data.get("split_index", None)
        if split_index is None:
            print(f"No valid split_index for QID {question_id}. Skipping...")
            continue

        if isinstance(attn_layers, dict) and len(attn_layers) > 0:
            head_image_ratios, visual_attention_ratios, token_attn_sums, head_token_attention_sums = analyze_input_attention_heads(
                attn_layers, input_token_length, split_index)
            if visual_attention_ratios is not None and token_attn_sums is not None:
                count += 1
                current_tokens = token_attn_sums.shape[1]
                if current_tokens > max_tokens:
                    max_tokens = current_tokens

                question_output_dir = os.path.join(global_output_dir, f"QID_{question_id}")
                os.makedirs(question_output_dir, exist_ok=True)
                # 传入 split_index 用于区分文本和视觉 token
                plot_token_attention_bar(visual_attention_ratios, token_attn_sums, question_output_dir, question_id, split_index)
                plot_attention_heatmap(head_image_ratios, question_output_dir, question_id)
                plot_visual_attention_ratio(visual_attention_ratios, question_output_dir, question_id)
                plot_attention_heads_token_distributions(head_token_attention_sums, question_output_dir, question_id, split_index)

                if global_visual_ratios is None:
                    global_visual_ratios = visual_attention_ratios.copy()
                else:
                    global_visual_ratios += visual_attention_ratios

                token_attn_sums_padded = pad_token_attention(token_attn_sums, max_tokens)
                if global_token_sums is None:
                    global_token_sums = token_attn_sums_padded.copy()
                else:
                    if global_token_sums.shape[1] < max_tokens:
                        global_token_sums = np.pad(global_token_sums,
                                                   ((0, 0), (0, max_tokens - global_token_sums.shape[1])),
                                                   mode='constant')
                    global_token_sums += token_attn_sums_padded
        else:
            print(f"No valid attention layers for QID {question_id}.")
        print("-" * 80)

    if count > 0 and global_visual_ratios is not None and global_token_sums is not None:
        global_visual_ratios /= count
        global_token_sums /= count
        plot_token_attention_bar(global_visual_ratios, global_token_sums, all_questions_dir, "ALL_QUESTIONS", split_index)
        plot_visual_attention_ratio(global_visual_ratios, all_questions_dir, "ALL_QUESTIONS")
        print(f"Final figures saved to: {global_output_dir}")
    else:
        print("No valid visual attention data to plot globally.")

if __name__ == "__main__":
    main()
