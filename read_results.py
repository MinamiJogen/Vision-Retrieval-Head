#!/usr/bin/env python3
"""
A script to display the top retrieval heads based on the results saved in a JSON file.
JSON 文件中 key "head_scores" 为字典，键为 "layer-head"（例如 "16-19"），对应值为一组检索得分。
该脚本计算各 head 的平均得分，并按得分降序显示前 N 个检索 head。

Usage:
    python3 display_results.py --file results/LongVA_head_scores.json --top 10
"""

import argparse
import json
import numpy as np

def load_head_scores(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["head_scores"]

def process_head_scores(head_scores):
    """
    处理 head_scores 字典，计算每个 head 的平均得分，返回列表格式：([layer, head], avg_score)
    """
    head_score_list = []
    for head_id, scores in head_scores.items():
        try:
            layer, head = [int(x) for x in head_id.split("-")]
        except ValueError:
            print(f"Skipping invalid head identifier: {head_id}")
            continue
        avg_score = np.mean(scores) if scores else 0.0
        head_score_list.append(([layer, head], avg_score))
    return head_score_list

def display_top_heads(head_score_list, top_n):
    # 按平均检索得分降序排序
    sorted_heads = sorted(head_score_list, key=lambda x: x[1], reverse=True)
    top_heads = sorted_heads[:top_n]
    
    print(f"Top {top_n} Retrieval Heads:")
    for head, score in top_heads:
        print(f"Head: {head}, Retrieval Score: {score:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Display top retrieval heads from a head score JSON file.")
    parser.add_argument('--file', type=str, default='results/LongVA_head_scores.json',
                        help='Path to the head score JSON file.')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of top heads to display.')
    args = parser.parse_args()

    head_scores = load_head_scores(args.file)
    head_score_list = process_head_scores(head_scores)
    display_top_heads(head_score_list, args.top)

if __name__ == "__main__":
    main()
