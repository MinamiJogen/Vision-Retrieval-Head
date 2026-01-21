import json
from collections import defaultdict

# 文件路径
file_path = "mergelogs/hub__models--LongVA-Merge/20250522_212154_samples_videomme.jsonl"

# 初始化统计
total = 0
correct = 0
type_stats = defaultdict(lambda: {"total": 0, "correct": 0})

# 逐行处理
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        score = item.get("videomme_perception_score", {})
        task_type = score.get("task_category") or item["doc"].get("task_type")
        pred = score.get("pred_answer", "").strip()
        answer = score.get("answer", "").strip()

        # 忽略空预测
        if pred == "":
            continue

        total += 1
        type_stats[task_type]["total"] += 1
        if pred == answer:
            correct += 1
            type_stats[task_type]["correct"] += 1

# 打印结果
print(f"整体正确率：{correct}/{total} = {correct / total:.2%}\n")

for t, stat in type_stats.items():
    t_correct = stat["correct"]
    t_total = stat["total"]
    print(f"{t:<25} 正确率：{t_correct}/{t_total} = {t_correct / t_total:.2%}")
