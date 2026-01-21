import os
import json
import numpy as np
import argparse

def aggregate_retrieval_scores(npz_path, activation_threshold=0.0):
    """
    读取 npz 文件，对所有样本中每一层各注意力头的 video_attention_scores 进行统计：
      - average_scores: 每个注意力头在所有样本上的平均 Retrieval Score
      - activation_frequencies: 每个注意力头的激活频率（定义为该 head 在样本中得分大于 activation_threshold 的比例）
    参数：
      npz_path: npz 文件路径
      activation_threshold: 判断一个头“激活”的阈值（由于这里的分数一般大于0，默认阈值设为0）
    返回：
      average_scores, activation_frequencies 两个字典，key 为层号（字符串），value 为各 head 的列表
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"未找到文件：{npz_path}")

    # 加载 npz 文件，allow_pickle=True 以便加载保存的字典
    data = np.load(npz_path, allow_pickle=True)
    
    aggregated_scores = {}      # 按层累加 Retrieval Score
    activation_counts = {}      # 每个 head 激活的样本数（得分 > threshold）
    sample_counts = {}          # 每层参与统计的样本数

    # 遍历每个样本，npz 文件中 key 为样本 id
    for sample_id in data.keys():
        # 每个 sample 是一个字典（通过 .item() 转换为 Python dict）
        sample = data[sample_id].item()
        # 使用之前保存的 "video_attention_scores"（这是基于视觉 token 注意力比例的计算）
        vas = sample.get("video_attention_scores", {})
        for layer, head_scores in vas.items():
            # 统一层号格式为字符串
            layer = str(layer)
            head_scores = np.array(head_scores, dtype=float)
            if layer not in aggregated_scores:
                aggregated_scores[layer] = head_scores.copy()
                activation_counts[layer] = (head_scores > activation_threshold).astype(int)
                sample_counts[layer] = 1
            else:
                aggregated_scores[layer] += head_scores
                activation_counts[layer] += (head_scores > activation_threshold).astype(int)
                sample_counts[layer] += 1

    average_scores = {}
    activation_frequencies = {}
    for layer in aggregated_scores:
        # 计算平均 Retrieval Score
        average_scores[layer] = (aggregated_scores[layer] / sample_counts[layer]).tolist()
        # 计算激活频率（每个 head 在多少比例的样本中被激活）
        activation_frequencies[layer] = (activation_counts[layer] / sample_counts[layer]).tolist()

    return average_scores, activation_frequencies

def main():
    parser = argparse.ArgumentParser(description="Aggregate retrieval head scores from an npz file.")
    parser.add_argument('--file', type=str, default="LongVA_attention_analysis/MTVQA.npz",
                        help="Path to the npz file.")
    parser.add_argument('--activation_threshold', type=float, default=0.0,
                        help="Threshold to consider a head activated (default: 0.0).")
    args = parser.parse_args()

    try:
        avg_scores, act_freqs = aggregate_retrieval_scores(args.file, args.activation_threshold)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # 构建每个注意力头平均 Retrieval Score 的字典，格式为 "层号-头号" -> 平均得分，
    # 同时统计每个头的激活频率（这里激活定义为得分 > threshold）
    avg_score_by_head = {}
    always_activated_heads = []  # 每一次都激活的注意力头（激活频率 == 1.0）
    for layer in act_freqs:
        for head_idx, freq in enumerate(act_freqs[layer]):
            key = f"{layer}-{head_idx}"
            avg_score = avg_scores[layer][head_idx]
            avg_score_by_head[key] = avg_score
            if abs(freq - 1.0) < 1e-6:
                always_activated_heads.append(key)
    
    # 对所有注意力头根据平均 Retrieval Score 从高到低排序
    sorted_avg_scores = sorted(avg_score_by_head.items(), key=lambda x: x[1], reverse=True)
    # 对每一次都激活的注意力头也按平均分数排序
    sorted_always_activated = sorted(always_activated_heads, key=lambda x: avg_score_by_head[x], reverse=True)

    print("【每一次都激活的注意力头】")
    if sorted_always_activated:
        for head in sorted_always_activated:
            print(f"{head}: {avg_score_by_head[head]:.4f}")
    else:
        print("没有注意力头在所有样本中均被激活。")

    print("\n【所有注意力头在所有样本上平均的 Retrieval Score】")
    for key, score in sorted_avg_scores:
        print(f"{key}: {score:.4f}")

    # 保存结果到 JSON 文件
    out_file = "retrieval_score_summary.json"
    output_data = {
        "always_activated_heads": {head: avg_score_by_head[head] for head in sorted_always_activated},
        "average_scores": {head: score for head, score in sorted_avg_scores},
        "activation_frequencies": {f"{layer}-{i}": freq 
                                     for layer in act_freqs 
                                     for i, freq in enumerate(act_freqs[layer])}
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n统计结果已保存到 {out_file}")

if __name__ == "__main__":
    main()
