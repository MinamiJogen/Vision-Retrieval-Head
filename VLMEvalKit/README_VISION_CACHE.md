# LongVA Vision Cache 使用指南

## 📋 概述

这个方案通过预处理和缓存 vision encoder 的输出来加速 LongVA 模型的评估。

**关键优势**：
- ✅ 节省 40-50% 的推理时间
- ✅ Vision encoder 只计算一次，可以复用多次
- ✅ 三个模型（Temporal-v1, v2, LongVA-7B）共享同一份缓存
- ✅ Float16 无压缩，精度完全无损
- ✅ 与原始流程完全一致（只跳过 vision_tower，mm_projector 和 LLM 正常运行）

**空间需求**：
- Video-MME (2,999 个视频): **1.44 TB**
- 当前可用空间: **1.9 TB**
- 剩余空间: **0.46 TB**

---

## 🚀 使用步骤

### 步骤 1: 预处理 Video-MME

运行预处理脚本，生成 vision features 缓存：

```bash
cd /disk3/minami/Vision-Retrieval-Head/VLMEvalKit

python preprocess_video_mme.py \
    --model-path lmms-lab/LongVA-7B-DPO \
    --cache-dir /disk3/minami/LMUData/vision_cache \
    --nframe 128 \
    --dataset Video-MME_128frame
```

**预计时间**: 8-12 小时（取决于 GPU）

**输出**：
- 缓存目录: `/disk3/minami/LMUData/vision_cache/Video-MME_128frame/`
- 元数据文件: `/disk3/minami/LMUData/vision_cache/cache_meta.json`
- 每个视频一个 `.pt` 文件（约 505 MB/文件）

### 步骤 2: 使用缓存进行评估

运行评估脚本，测试三个模型：

```bash
bash eval_longva_video_cached.sh
```

**测试的模型**：
- LongVA-Temporal-v1-Cached
- LongVA-Temporal-v2-Cached
- LongVA-7B-Cached

**输出**：
- 结果目录: `results/longva_cached_YYYYMMDD_HHMMSS/`
- 日志文件: `logs/longva_cached_YYYYMMDD_HHMMSS.log`

---

## 📊 验证缓存效果

### 检查缓存统计

```bash
# 查看元数据
cat /disk3/minami/LMUData/vision_cache/cache_meta.json | python3 -m json.tool

# 查看缓存文件数量
find /disk3/minami/LMUData/vision_cache/Video-MME_128frame -name "*.pt" | wc -l

# 查看缓存大小
du -sh /disk3/minami/LMUData/vision_cache/Video-MME_128frame
```

### 检查缓存命中率

评估结束后，会输出缓存统计：

```
============================================================
Vision Cache Statistics
============================================================
Cache hits: 2999
Cache misses: 0
Hit rate: 100.0%
============================================================
```

### 对比原始模型和缓存模型

```bash
# 测试原始模型（不使用缓存）
python run.py \
    --data Video-MME_128frame \
    --model LongVA-7B \
    --mode all

# 测试缓存模型
python run.py \
    --data Video-MME_128frame \
    --model LongVA-7B-Cached \
    --mode all

# 比较结果文件（应该完全一致）
# 结果文件位于: results/<model_name>/<eval_id>/<model>_<dataset>.xlsx
```

---

## 🔧 技术细节

### 缓存内容

预处理脚本缓存的是 **vision_tower 的输出**（不包括 mm_projector）：

```python
# 预处理时
vision_features = vision_tower(video_tensor)  # 只到这里
# 保存 vision_features

# 推理时
vision_features = load_cache()  # 从缓存加载
vision_embeds = mm_projector(vision_features)  # 正常运行
output = language_model.generate(...)  # 正常运行
```

**为什么这样设计**：
- 确保 mm_projector 和 LLM 的流程与原始模型完全一致
- mm_projector 的计算量很小（< 5% 总时间）
- 三个模型的 vision_tower 完全相同，但理论上可能有不同的 mm_projector 权重

### 缓存文件格式

每个 `.pt` 文件包含：

```python
{
    'vision_features': Tensor,  # [num_tokens, hidden_dim], float16
    'video_id': str,
    'video_path': str,
    'nframe': int,
    'dataset': str,
    'model_signature': str,     # 基于 vision_tower 配置的哈希
    'dtype': 'float16',
    'shape': [num_tokens, hidden_dim]
}
```

### 缓存键生成

```python
cache_key = MD5(f"{dataset}:{video_id}:nframe{nframe}:model{model_signature}")
```

**模型签名**：
```python
model_signature = MD5(f"{mm_vision_tower}:layer{mm_vision_select_layer}")[:8]
# 例如: "a3b4c5d6"
```

这确保了不同配置的模型不会混用缓存。

### Hook 机制

在推理时，我们使用 hook 替换 vision_tower 的 forward 方法：

```python
# 保存原始 forward
original_forward = vision_tower.forward

# 替换为返回缓存的版本
vision_tower.forward = lambda x: cached_features.unsqueeze(0)

# 正常调用 model.generate（vision_tower 会返回缓存，其他正常）
output = model.generate(...)

# 恢复原始 forward
vision_tower.forward = original_forward
```

---

## ⚠️ 注意事项

### 1. 模型兼容性

**可以共享缓存的条件**：
- ✅ Vision tower 完全相同（包括权重和配置）
- ✅ 只修改 LLM 权重
- ✅ MM projector 可以不同（会重新运行）

**你的三个模型满足这个条件**，所以可以共享缓存。

### 2. 磁盘空间监控

```bash
# 实时监控磁盘空间
watch -n 60 'df -h /disk3'

# 监控缓存大小
watch -n 60 'du -sh /disk3/minami/LMUData/vision_cache'
```

### 3. 预处理中断

脚本支持断点续传，如果中断：

```bash
# 重新运行相同命令，已处理的视频会被跳过
python preprocess_video_mme.py  # 相同参数
```

### 4. 清理缓存

如果需要重新预处理或释放空间：

```bash
# 删除特定数据集的缓存
rm -rf /disk3/minami/LMUData/vision_cache/Video-MME_128frame/

# 删除所有缓存
rm -rf /disk3/minami/LMUData/vision_cache/
```

---

## 🐛 故障排除

### 问题 1: 缓存未命中

**症状**: `Cache misses` 很高，`Hit rate` 很低

**可能原因**:
1. 模型签名不匹配（使用了不同的 vision_tower）
2. nframe 参数不一致
3. 数据集名称不匹配

**解决**:
```bash
# 检查模型签名
python -c "
from vlmeval.vlm.longva_cached import LongVA_Cached
model = LongVA_Cached('lmms-lab/LongVA-7B-DPO')
print('Model signature:', model.model_signature)
"

# 检查缓存元数据
cat /disk3/minami/LMUData/vision_cache/cache_meta.json
```

### 问题 2: 缓存文件损坏

**症状**: 加载缓存时报错

**解决**:
```bash
# 删除损坏的缓存文件，重新预处理
rm -f /disk3/minami/LMUData/vision_cache/Video-MME_128frame/<damaged_file>.pt

# 重新运行预处理（只会处理缺失的文件）
python preprocess_video_mme.py
```

### 问题 3: 内存不足

**症状**: CUDA OOM 或 CPU OOM

**解决**:
```bash
# 减少并行 GPU 数量
export CUDA_VISIBLE_DEVICES=0  # 只使用一个 GPU

# 或者使用原始模型（不使用缓存）
python run.py --model LongVA-7B  # 不带 -Cached 后缀
```

---

## 📈 性能对比

### 预期加速比

| 阶段 | 原始模型 | 缓存模型 | 节省 |
|------|---------|---------|------|
| 视频加载 | 5% | 5% | 0% |
| Vision Encoder | 40% | **0%** | **40%** |
| MM Projector | 5% | 5% | 0% |
| LLM 生成 | 50% | 50% | 0% |
| **总计** | 100% | **60%** | **40%** |

**实际加速比**: 约 **1.67x**

### 多次实验的收益

假设需要测试 3 个模型 × 5 次实验：

| 方案 | 总时间 | GPU 时间节省 |
|------|--------|------------|
| 原始（每次都重新计算） | 15x | - |
| 缓存（预处理 1x + 推理 0.6x × 15） | 10x | **5x = 33%** |

---

## 📚 相关文件

```
VLMEvalKit/
├── preprocess_video_mme.py              # 预处理脚本
├── eval_longva_video_cached.sh          # 评估脚本
├── vlmeval/
│   ├── vlm/
│   │   ├── longva_custom.py             # 原始 LongVA 模型
│   │   └── longva_cached.py             # 支持缓存的 LongVA 模型
│   └── config.py                        # 模型配置（已添加缓存模型）
├── README_VISION_CACHE.md               # 本文件
└── VISION_CACHE_GUIDE.md                # 完整技术指南
```

---

## ✅ 验证清单

在运行评估前，确保：

- [ ] 已运行 `preprocess_video_mme.py` 完成预处理
- [ ] 缓存目录存在: `/disk3/minami/LMUData/vision_cache/Video-MME_128frame/`
- [ ] 缓存文件数量正确: 2,999 个 `.pt` 文件
- [ ] 磁盘空间充足: > 1.5 TB 可用
- [ ] 已修改 `vlmeval/config.py` 添加缓存模型配置
- [ ] 已修改 `vlmeval/vlm/__init__.py` 添加 `LongVA_Cached` 导入

---

## 🎯 下一步

完成 Video-MME 的测试后，如果效果满意，可以：

1. **扩展到其他数据集**（如果有足够空间）：
   ```bash
   python preprocess_video_mme.py --dataset VideoMMMU_128frame
   python preprocess_video_mme.py --dataset LongVideoBench_128frame
   ```

2. **使用压缩或量化**（如果空间不足）：
   - 修改预处理脚本，添加 INT8 量化支持
   - 预期可节省 50% 空间

3. **自动化批量评估**：
   - 修改 `eval_longva_video_cached.sh` 支持多个数据集
   - 实现自动结果对比

---

祝测试顺利！ 🚀

如有问题，请检查日志文件: `logs/longva_cached_*.log`
