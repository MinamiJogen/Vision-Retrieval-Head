#!/usr/bin/env python3
"""
A script to display the top retrieval heads based on the results saved in a JSON file.
The JSON file should contain a dictionary where each key is a head identifier in the
format "layer-head_id" (e.g., "16-19") and its value is a list of retrieval scores.
This script computes the average retrieval score for each head, sorts them, and prints
the top N retrieval heads.

Usage:
    python3 read_results.py --file ./head_score/llama-2-7b-80k.json --top 20
"""

import argparse
import json
import numpy as np

def load_head_scores(file_path):
    with open(file_path, 'r') as f:
        # Assumes that the head score file contains a single JSON line
        head_scores = json.loads(f.readline().strip())
    return head_scores

def process_head_scores(head_scores):
    """
    Processes the head_scores dictionary to compute average scores for each head.
    
    Returns:
        A list of tuples: ([layer, head_id], avg_score)
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
    # Sort by average retrieval score in descending order
    sorted_heads = sorted(head_score_list, key=lambda x: x[1], reverse=True)
    top_heads = sorted_heads[:top_n]
    
    print(f"Top {top_n} Retrieval Heads:")
    for head, score in top_heads:
        print(f"Head: {head}, Retrieval Score: {score:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Display top retrieval heads from a head score JSON file.")
    parser.add_argument('--file', type=str, default='./head_score/llama-2-7b-80k.json',
                        help='Path to the head score JSON file.')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of top heads to display.')
    args = parser.parse_args()

    head_scores = load_head_scores(args.file)
    head_score_list = process_head_scores(head_scores)
    display_top_heads(head_score_list, args.top)

if __name__ == "__main__":
    main()
