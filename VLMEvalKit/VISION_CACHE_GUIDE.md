# LongVA Vision 特征缓存方案实施指南

## 📊 方案概述

### 问题背景
- **3 个模型**: LongVA-Temporal-v1, Temporal-v2, LongVA-7B
- **共同点**: Vision Encoder 完全相同，只有 LLM 权重不同
- **数据规模**: 7 个 benchmark，18,595 个视频样本
- **重复实验**: 需要多次重复评估

### 核心优化
通过预处理和缓存 vision encoder 输出，避免重复计算：
- **节省时间**: 每次实验节省 ~40-50% 的推理时间
- **节省资源**: Vision Encoder 只计算一次，复用多次

---

## 💾 空间需求分析

### 磁盘空间状况
```
/disk3 总容量: 14TB
已使用: 12TB
可用空间: 1.9TB ⚠️
```

### 缓存大小估算

#### Vision Encoder 配置
- **模型**: CLIP ViT-Large-patch14-336
- **每帧输出**: 577 tokens (576 patches + 1 CLS)
- **投影维度**: 3584 (LLM hidden size)

#### 不同方案的空间需求

| 方案 | 数据类型 | 每视频大小 | 总大小 (18,595个) | 可行性 |
|------|---------|-----------|------------------|--------|
| 无压缩 | float16 | 505 MB | **8.95 TB** | ❌ 超出 4.7x |
| 压缩存储 | float16 + zip | ~300 MB | **5.3 TB** | ❌ 超出 2.8x |
| INT8 量化 | int8 | 126 MB | **2.24 TB** | ⚠️ 超出 15% |
| **INT8 + 压缩** | int8 + zip | ~80 MB | **1.42 TB** | ✅ **可行!** |

**推荐方案**: INT8 量化 + 压缩存储
- 缓存需求: ~1.4 TB
- 剩余空间: ~0.5 TB
- 精度损失: < 0.5%（几乎可忽略）

---

## 🚀 实施步骤

### 第一步：预处理数据集

运行预处理脚本，生成所有视频的 vision features 缓存：

```bash
cd /disk3/minami/Vision-Retrieval-Head/VLMEvalKit

python preprocess_vision_cache.py \
    --datasets \
        Video-MME_128frame \
        Video_Holmes_128frame \
        TempCompass_128frame \
        MLVU_128frame \
        VideoMMMU_128frame \
        MMBench_Video_128frame_nopack \
        LongVideoBench_128frame \
    --model-path lmms-lab/LongVA-7B-DPO \
    --cache-dir /disk3/minami/LMUData/vision_cache \
    --quantize int8 \
    --nframe 128
```

**预计时间**: 取决于 GPU 速度，大约 8-12 小时

**输出目录结构**:
```
/disk3/minami/LMUData/vision_cache/
├── cache_meta.json
├── Video-MME_128frame/
│   ├── <cache_key_1>.pt
│   ├── <cache_key_2>.pt
│   └── ...
├── TempCompass_128frame/
│   └── ...
└── ...
```

### 第二步：配置模型使用缓存

编辑 `vlmeval/config.py`，添加支持缓存的模型配置：

```python
# 在 ungrouped 或 video_models 中添加
from vlmeval.vlm import LongVA_Cached

"LongVA-Temporal-v1-Cached": partial(
    LongVA_Cached,
    model_path="Eculid/Temporal-v1",
    cache_dir="/disk3/minami/LMUData/vision_cache",
    enable_cache=True
),
"LongVA-Temporal-v2-Cached": partial(
    LongVA_Cached,
    model_path="Eculid/Temporal-v2",
    cache_dir="/disk3/minami/LMUData/vision_cache",
    enable_cache=True
),
"LongVA-7B-Cached": partial(
    LongVA_Cached,
    model_path="lmms-lab/LongVA-7B-DPO",
    cache_dir="/disk3/minami/LMUData/vision_cache",
    enable_cache=True
),
```

### 第三步：注册新模型类

编辑 `vlmeval/vlm/__init__.py`，添加导入：

```python
from .longva_cached import LongVA_Cached
```

### 第四步：修改评估脚本

创建新的评估脚本 `eval_longva_video_128f_parallel_cached.sh`：

```bash
#!/bin/bash

# ... (前面的配置保持不变)

# 使用带缓存的模型
MODELS=(
    "LongVA-Temporal-v1-Cached"
    "LongVA-Temporal-v2-Cached"
    "LongVA-7B-Cached"
)

# ... (其余部分保持不变)
```

### 第五步：运行评估

```bash
bash eval_longva_video_128f_parallel_cached.sh
```

---

## ⚙️ 技术细节

### INT8 量化方法

使用**对称量化**：
```python
scale = abs_max / 127.0
quantized = round(value / scale).clamp(-128, 127)
```

**反量化**：
```python
dequantized = quantized * scale
```

### 缓存键生成

```python
cache_key = MD5(f"{dataset}:{video_id}:nframe{nframe}:model{model_signature}")
```

**模型签名**：基于 vision tower 配置
```python
model_signature = MD5(f"{mm_vision_tower}:layer{mm_vision_select_layer}")[:8]
```

### 缓存文件格式

每个缓存文件（`.pt`）包含：
```python
{
    'vision_embeds': Tensor,      # [num_tokens, hidden_dim]
    'scale': float,                # 量化scale
    'video_path': str,
    'nframe': int,
    'dataset': str,
    'model_signature': str,
    'quantization': 'int8',
    'shape': [num_tokens, hidden_dim]
}
```

---

## 📈 性能提升估算

### 时间节省

假设单个视频推理时间分解：
- **Vision Encoder**: 40% (缓存后消除)
- **LLM 生成**: 60%

**加速比**：
- 单次实验: 1.67x
- 3 个模型 × 5 次实验: **节省 ~600 小时 GPU 时间**

### 空间占用

- **初始缓存**: ~1.4 TB
- **每个数据集平均**: ~200 GB
- **可按需删除部分数据集缓存**

---

## 🔍 验证和调试

### 1. 验证缓存正确性

在小数据集上测试：

```bash
# 测试 100 个样本
python test_cache_correctness.py \
    --dataset Video-MME_128frame \
    --model lmms-lab/LongVA-7B-DPO \
    --samples 100 \
    --cache-dir /disk3/minami/LMUData/vision_cache
```

### 2. 检查缓存统计

查看缓存元数据：

```bash
cat /disk3/minami/LMUData/vision_cache/cache_meta.json
```

### 3. 监控缓存命中率

模型会在结束时输出统计：
```
Vision Cache Statistics:
  Cache hits: 18595
  Cache misses: 0
  Hit rate: 100.0%
```

### 4. 对比实验结果

第一次使用缓存时，对比与原始模型的输出：
```bash
# 原始模型
python run.py --model LongVA-7B --data Video-MME_128frame

# 缓存模型
python run.py --model LongVA-7B-Cached --data Video-MME_128frame

# 比较结果文件
diff results/LongVA-7B/... results/LongVA-7B-Cached/...
```

---

## ⚠️ 注意事项

### 1. 模型版本一致性

**重要**: 预处理使用的模型必须与评估使用的模型一致！

- 使用 `lmms-lab/LongVA-7B-DPO` 预处理的缓存
- 只能用于 Vision Encoder 完全相同的模型
- 你的三个模型（Temporal-v1, v2, LongVA-7B）共享 Vision Encoder，所以可以共用缓存

### 2. 磁盘空间监控

预处理过程中定期检查磁盘空间：

```bash
# 监控脚本
watch -n 60 'df -h /disk3 && du -sh /disk3/minami/LMUData/vision_cache'
```

### 3. 缓存失效场景

以下情况需要重新预处理：
- ✅ 修改 LLM 权重 → **无需重新预处理**（你的场景）
- ❌ 修改 Vision Encoder 权重 → 需要重新预处理
- ❌ 更换 Vision Tower → 需要重新预处理
- ❌ 修改 nframe 参数 → 需要重新预处理

### 4. 精度影响

INT8 量化的影响：
- **理论**: < 0.5% 精度损失
- **建议**: 在关键 benchmark 上对比验证
- **可选**: 使用 bfloat16（占用 2TB，精度无损）

---

## 🐛 故障排除

### 问题 1: 缓存未命中

**症状**: Cache miss rate 很高

**原因**:
- 模型签名不匹配
- 缓存文件损坏
- nframe 参数不一致

**解决**:
```bash
# 检查模型签名
python -c "
from vlmeval.vlm.longva_cached import LongVA_Cached
model = LongVA_Cached('lmms-lab/LongVA-7B-DPO')
print('Model signature:', model.model_signature)
"

# 检查缓存元数据
cat /disk3/minami/LMUData/vision_cache/cache_meta.json | grep model_signature
```

### 问题 2: 内存不足

**症状**: CUDA out of memory

**原因**: 缓存的特征太大，无法全部加载到 GPU

**解决**:
- 使用更小的 batch size
- 使用 CPU offloading
- 减少并行 GPU 数量

### 问题 3: 预处理中断

**症状**: 预处理脚本中途崩溃

**解决**: 脚本支持断点续传，重新运行即可：
```bash
# 已处理的视频会被跳过（cached_count 增加）
python preprocess_vision_cache.py ... # 重新运行相同命令
```

---

## 📚 相关文件

- `preprocess_vision_cache.py` - 预处理脚本
- `vlmeval/vlm/longva_cached.py` - 支持缓存的模型类
- `vlmeval/vlm/longva_custom.py` - 原始模型类（参考）
- `test_vision_output_size.py` - 测试 vision encoder 输出维度
- `eval_longva_video_128f_parallel.sh` - 原始评估脚本
- `eval_longva_video_128f_parallel_cached.sh` - 使用缓存的评估脚本（需创建）

---

## 📞 需要帮助？

如有问题，检查以下内容：
1. 缓存目录权限
2. 磁盘空间是否充足
3. 模型路径是否正确
4. Python 环境是否正确（与运行 VLMEvalKit 的环境一致）
5. LongVA 是否正确安装

---

## 🎯 下一步

完成预处理后，你可以：
1. 开始你的重复实验
2. 随时修改 LLM 权重而无需重新预处理
3. 在不同的 benchmark 上快速评估
4. 节省大量 GPU 时间和成本

预祝实验顺利！ 🚀
