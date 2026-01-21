import os
import json
import argparse
import re
from typing import List, Dict, Optional, Union

CATEGORIES = [
    "Knowledge", "Film & Television", "Sports Competition",
    "Artistic Performance", "Life Record", "Multilingual"
]

SUB_CATEGORIES = [
    "Humanity & History", "Literature & Art", "Biology & Medicine",
    "Finance & Commerce", "Astronomy", "Geography", "Law", "Life Tip",
    "Technology", "Animation", "Movie & TV Show", "Documentary", "News Report",
    "Esports", "Basketball", "Football", "Athletics", "Other Sports", "Stage Play",
    "Magic Show", "Variety Show", "Acrobatics", "Handicraft", "Food", "Fashion",
    "Daily Life", "Travel", "Pet & Animal", "Exercise", "Multilingual"
]

TASK_CATEGORIES = [
    "Temporal Perception", "Spatial Perception", "Attribute Perception",
    "Action Recognition", "Object Recognition", "OCR Problems", "Counting Problem",
    "Temporal Reasoning", "Spatial Reasoning", "Action Reasoning",
    "Object Reasoning", "Information Synopsis"
]

def extract_characters_regex(s):
    if not s:
        return ""
    s = s.strip()
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is", "The answer",
        "The best option is", "The correct option is", "Best answer:", "Best option:",
        "Answer:", "Option:", "The correct answer", "The correct option"
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    match = re.search(r'[ABCD]', s)
    return match[0] if match else ""

def eval_your_results(
        your_results_path: str,
        skip_missing: bool = True,
        return_categories_accuracy: bool = True,
        return_sub_categories_accuracy: bool = True,
        return_task_types_accuracy: bool = True,
        gt_answer_key: str = "answer",
        your_answer_key: str = "response"
):
    with open(your_results_path, 'r') as f:
        results = json.load(f)

    q_type_dict = {q: {"correct": 0, "answered": 0} for q in TASK_CATEGORIES}
    v_type_dict = {v: {"correct": 0, "answered": 0} for v in CATEGORIES}
    v_sub_type_dict = {s: {"correct": 0, "answered": 0} for s in SUB_CATEGORIES}

    for item in results:
        if skip_missing and item.get("missing", False):
            continue

        domain = item.get("domain")
        sub_category = item.get("sub_category")
        q_type = item.get("task_type")  # now required to be present
        gt_answer = item.get(gt_answer_key)
        response = item.get(your_answer_key)

        pred = extract_characters_regex(response)

        if pred:
            if q_type in q_type_dict:
                q_type_dict[q_type]["answered"] += 1
                q_type_dict[q_type]["correct"] += (pred == gt_answer)
            if domain in v_type_dict:
                v_type_dict[domain]["answered"] += 1
                v_type_dict[domain]["correct"] += (pred == gt_answer)
            if sub_category in v_sub_type_dict:
                v_sub_type_dict[sub_category]["answered"] += 1
                v_sub_type_dict[sub_category]["correct"] += (pred == gt_answer)

    print("================ Evaluation Report ================")

    if return_categories_accuracy:
        print("\n📊 Video Domains Accuracy:")
        for v in CATEGORIES:
            correct = v_type_dict[v]["correct"]
            answered = v_type_dict[v]["answered"]
            acc = 100 * correct / answered if answered > 0 else 0
            print(f"{v:25s}: {acc:.1f}% ({correct}/{answered})")

    if return_sub_categories_accuracy:
        print("\n📊 Video Sub-Categories Accuracy:")
        for s in SUB_CATEGORIES:
            correct = v_sub_type_dict[s]["correct"]
            answered = v_sub_type_dict[s]["answered"]
            acc = 100 * correct / answered if answered > 0 else 0
            print(f"{s:25s}: {acc:.1f}% ({correct}/{answered})")

    if return_task_types_accuracy:
        print("\n📊 Task Categories Accuracy:")
        for q in TASK_CATEGORIES:
            correct = q_type_dict[q]["correct"]
            answered = q_type_dict[q]["answered"]
            acc = 100 * correct / answered if answered > 0 else 0
            print(f"{q:25s}: {acc:.1f}% ({correct}/{answered})")

    total_correct = sum(q_type_dict[q]["correct"] for q in TASK_CATEGORIES)
    total_answered = sum(q_type_dict[q]["answered"] for q in TASK_CATEGORIES)
    overall = 100 * total_correct / total_answered if total_answered > 0 else 0
    print("\n✅ Overall Accuracy:")
    print(f"TOTAL: {overall:.1f}% ({total_correct}/{total_answered})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--return_categories_accuracy", action="store_true")
    parser.add_argument("--return_sub_categories_accuracy", action="store_true")
    parser.add_argument("--return_task_types_accuracy", action="store_true")
    args = parser.parse_args()

    eval_your_results(
        your_results_path=args.results_file,
        skip_missing=args.skip_missing,
        return_categories_accuracy=args.return_categories_accuracy,
        return_sub_categories_accuracy=args.return_sub_categories_accuracy,
        return_task_types_accuracy=args.return_task_types_accuracy
    )
