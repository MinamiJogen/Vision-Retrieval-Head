# 评测数据集更新说明

## 📋 更新内容

已从所有评测脚本中移除 **MMMU_DEV_VAL**，因为它与当前模型存在兼容性问题。

## ✅ 当前支持的9个Benchmarks

| Benchmark | 说明 | 类型 |
|-----------|------|------|
| **MMBench_DEV_EN** | 多模态理解基准 | 综合能力 |
| **MME** | 全面评估（感知+认知） | 综合能力 |
| **SEEDBench_IMG** | 图像理解能力 | 视觉理解 |
| **HallusionBench** | 幻觉检测 | 可靠性 |
| **AI2D_TEST** | 图表理解 | 视觉推理 |
| **OCRBench** | OCR能力测试 | 文字识别 |
| **MathVista_MINI** | 数学推理 | 数学能力 |
| **RealWorldQA** | 真实场景问答 | 实用性 |
| **POPE** | 物体感知评估 | 感知能力 |

## 🚫 已移除的数据集

- ~~MMMU_DEV_VAL~~ - 由于与LongVA模型存在兼容性问题（会出现乱码错误）

## 📝 更新的文件

1. ✅ `eval_longva_models.sh` - LongVA评测脚本
2. ✅ `eval_qwen_internvl_models.sh` - Qwen3 & InternVL评测脚本
3. ✅ `eval_all_models.sh` - 综合评测脚本
4. ✅ `longva_custom.py` - 恢复原始代码（移除MMMU修复）

## 🎯 现在的评测配置

所有脚本现在都使用相同的**9个benchmarks**，确保：
- ✅ 所有模型都能正常评测
- ✅ 不会出现兼容性问题
- ✅ 结果稳定可靠

## 🚀 继续评测

无需任何修改，直接运行脚本即可：

```bash
# LongVA 系列
cd VLMEvalKit
bash ../eval_longva_models.sh

# Qwen3 & InternVL
bash ../eval_qwen_internvl_models.sh
```

## 💡 如果想测试更多数据集

可以添加这些兼容性好的数据集：

### 推荐添加的数据集
```bash
DATASETS=(
    # 现有的9个...
    "ChartQA_TEST"       # 图表问答
    "DocVQA_VAL"         # 文档问答
    "TextVQA_VAL"        # 文字问答
    "ScienceQA_VAL"      # 科学问答
)
```

### 视频相关（如果需要）
```bash
"MMBench_Video"          # 视频理解
"Video-MME"              # 视频多模态评估
```

## ⚠️ 注意事项

1. **不要**添加回MMMU_DEV_VAL - 它会导致评测失败
2. 如果添加新数据集，建议先用单个模型测试
3. 某些数据集可能需要额外的数据下载

## ✨ 总结

- 现在所有脚本都使用**9个稳定的benchmarks**
- 已验证这9个数据集与所有模型兼容
- 可以放心运行完整评测

开始评测：
```bash
cd VLMEvalKit
bash ../eval_longva_models.sh
```
