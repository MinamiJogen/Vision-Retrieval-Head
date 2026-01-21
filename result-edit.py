import re
import json

# 将下面的字符串替换成你的实际输出
text = """
Layer 21: H24 (attn: 0.3193, CV: 0.78; ratio: 0.3193, CV: 0.78)
Layer 22: H13 (attn: 0.3787, CV: 0.78; ratio: 0.3787, CV: 0.78), H27 (attn: 0.3611, CV: 0.80; ratio: 0.3611, CV: 0.80)
Layer 25: H08 (attn: 0.4443, CV: 0.71; ratio: 0.4443, CV: 0.71)
Layer 26: H17 (attn: 0.3128, CV: 0.74; ratio: 0.3128, CV: 0.74)
Layer 27: H17 (attn: 0.2323, CV: 0.79; ratio: 0.2323, CV: 0.79), H24 (attn: 0.3026, CV: 0.78; ratio: 0.3026, CV: 0.78)
"""

pattern_layer = re.compile(r'^Layer\s*(\d+):')
pattern_heads  = re.compile(r'H(\d+)')

result = []
for line in text.splitlines():
    m_layer = pattern_layer.search(line)
    if not m_layer:
        continue
    layer_idx = int(m_layer.group(1))
    # 提取该行所有 Hxx
    for h in pattern_heads.findall(line):
        result.append([layer_idx, int(h)])

# 输出为 JSON 格式
print(json.dumps(result, ensure_ascii=False, indent=2))
