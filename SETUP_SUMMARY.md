# LongVA VLMEvalKit 集成完成总结

## ✅ 已完成的工作

### 1. 模型注册
已在VLMEvalKit中成功注册以下三个LongVA模型：

| 模型名称 | 模型路径 | 配置名称 |
|---------|---------|---------|
| LongVA Temporal v1 | `Eculid/Temporal-v1` | `LongVA-Temporal-v1` |
| LongVA Temporal v2 | `Eculid/Temporal-v2` | `LongVA-Temporal-v2` |
| LongVA 7B | `lmms-lab/LongVA-7B-DPO` | `LongVA-7B` |

### 2. 修改的文件

#### 新建文件：
- **VLMEvalKit/vlmeval/vlm/longva_custom.py**
  - LongVA模型的包装类
  - 继承自BaseModel
  - 支持图像输入（视频支持可扩展）

#### 修改文件：
- **VLMEvalKit/vlmeval/vlm/__init__.py**
  - 添加了LongVA类的导入

- **VLMEvalKit/vlmeval/config.py**
  - 在ungrouped字典中注册了三个模型

### 3. 创建的辅助文件

#### 文档：
- **LONGVA_EVAL_GUIDE.md** - 详细的评测使用指南
- **SETUP_SUMMARY.md** - 本文件，集成总结

#### 测试脚本：
- **test_longva_vlmeval.py** - 验证模型注册的Python测试脚本
- **quick_test.sh** - 快速测试单个模型
- **batch_eval.sh** - 批量评测所有三个模型
- **run_vlmeval_example.sh** - 评测命令示例

## 🚀 如何开始评测

### 方法1：快速测试单个模型
```bash
bash quick_test.sh LongVA-Temporal-v1
```

### 方法2：批量评测所有模型
```bash
bash batch_eval.sh
```

### 方法3：使用Python命令（推荐用于生产）
```bash
cd VLMEvalKit

# 单个模型，单个数据集
python run.py --data MMBench_DEV_EN --model LongVA-Temporal-v1 --verbose

# 多个模型，多个数据集
python run.py \
    --data MMBench_DEV_EN SEEDBench_IMG MME \
    --model LongVA-Temporal-v1 LongVA-Temporal-v2 LongVA-7B \
    --verbose
```

## 📋 推荐的测试流程

### 第一步：验证单个模型可以正常加载和推理
```bash
bash quick_test.sh LongVA-Temporal-v1
```
预期结果：成功生成推理结果文件

### 第二步：在小数据集上测试所有三个模型
```bash
cd VLMEvalKit
python run.py \
    --data SEEDBench_IMG \
    --model LongVA-Temporal-v1 LongVA-Temporal-v2 LongVA-7B \
    --mode infer \
    --verbose
```

### 第三步：完整评测
```bash
bash batch_eval.sh
```
或者自定义评测：
```bash
cd VLMEvalKit
python run.py \
    --data MMBench_DEV_EN MME SEEDBench_IMG MMMU_DEV_VAL \
    --model LongVA-Temporal-v1 LongVA-Temporal-v2 LongVA-7B \
    --verbose \
    --work-dir ./longva_results
```

## 📊 结果文件位置

评测完成后，结果会保存在：
```
VLMEvalKit/{work-dir}/
├── LongVA-Temporal-v1/
│   ├── LongVA-Temporal-v1_MMBench_DEV_EN.xlsx    # 详细推理结果
│   ├── LongVA-Temporal-v1_MMBench_DEV_EN.csv     # 评估指标
│   └── ...
├── LongVA-Temporal-v2/
│   └── ...
└── LongVA-7B/
    └── ...
```

## 🔧 GPU使用建议

### 单GPU（推荐用于大模型）
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --data MMBench_DEV_EN --model LongVA-7B --verbose
```

### 多GPU并行（适用于加速评测）
```bash
# 使用2个GPU，每个GPU运行一个模型实例
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 run.py \
    --data MMBench_DEV_EN \
    --model LongVA-Temporal-v1 \
    --verbose
```

### 顺序评测多个模型（内存不足时推荐）
```bash
for model in LongVA-Temporal-v1 LongVA-Temporal-v2 LongVA-7B; do
    CUDA_VISIBLE_DEVICES=0 python run.py \
        --data MMBench_DEV_EN \
        --model $model \
        --verbose
done
```

## 📝 常用数据集

| 数据集 | 说明 | 推荐用途 |
|-------|------|---------|
| `SEEDBench_IMG` | 图像理解基准 | 快速测试 |
| `MMBench_DEV_EN` | 多模态理解 | 全面评估 |
| `MME` | 综合评估 | 性能对比 |
| `MMMU_DEV_VAL` | 多学科理解 | 专业能力 |
| `HallusionBench` | 幻觉检测 | 可靠性测试 |
| `AI2D_TEST` | 图表理解 | 视觉推理 |
| `OCRBench` | OCR能力 | 文字识别 |

## ⚙️ 自定义配置

### 修改生成参数
编辑 `VLMEvalKit/vlmeval/vlm/longva_custom.py` 中的 `kwargs_default`：
```python
kwargs_default = dict(
    do_sample=True,
    temperature=0.5,      # 采样温度
    top_p=None,           # nucleus sampling
    num_beams=1,          # beam search大小
    use_cache=True,       # 使用KV cache
    max_new_tokens=1024   # 最大生成长度
)
```

### 使用本地模型路径
编辑 `VLMEvalKit/vlmeval/config.py`：
```python
"LongVA-Temporal-v1": partial(LongVA, model_path="/path/to/local/model"),
```

### 修改LongVA代码路径
编辑 `VLMEvalKit/vlmeval/vlm/longva_custom.py` 中的路径：
```python
longva_path = "/disk3/minami/Vision-Retrieval-Head/LongVA/longva"
```

## 🐛 常见问题

### Q: 模型加载失败
**A:** 检查：
1. LongVA代码路径是否正确
2. 模型文件是否已下载
3. GPU显存是否充足

### Q: 评测速度慢
**A:** 建议：
1. 使用 `--mode infer` 先只做推理
2. 使用多GPU并行：`torchrun --nproc-per-node=N`
3. 减少数据集数量进行测试

### Q: 内存不足
**A:** 解决方案：
1. 一次只评测一个模型
2. 减少batch size（在代码中修改）
3. 使用较小的模型先测试

### Q: 如何查看详细错误日志
**A:** 添加 `--verbose` 参数会输出详细日志

## 📚 更多信息

- 详细使用指南：查看 `LONGVA_EVAL_GUIDE.md`
- VLMEvalKit官方文档：https://github.com/open-compass/VLMEvalKit
- LongVA模型：https://github.com/EvolvingLMMs-Lab/LongVA

## ✨ 下一步

1. 运行 `bash quick_test.sh LongVA-Temporal-v1` 验证配置
2. 如果测试通过，运行完整评测
3. 查看结果文件并分析性能

祝评测顺利！🎉
