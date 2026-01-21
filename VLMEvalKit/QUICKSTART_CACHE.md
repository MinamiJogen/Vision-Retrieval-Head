# 🚀 Vision Cache 快速启动指南

## 一、准备工作

确认环境：
```bash
cd /disk3/minami/Vision-Retrieval-Head/VLMEvalKit

# 检查 LongVA 是否已安装
python3 -c "import longva; print('✓ LongVA installed')" 2>&1

# 检查磁盘空间（需要 > 1.5 TB）
df -h /disk3 | tail -1
```

---

## 二、预处理 Video-MME（一次性，8-12小时）

```bash
# 运行预处理脚本
python preprocess_video_mme.py \
    --model-path lmms-lab/LongVA-7B-DPO \
    --cache-dir /disk3/minami/LMUData/vision_cache \
    --nframe 128 \
    --dataset Video-MME_128frame

# 预期输出:
# - 处理 2,999 个视频
# - 生成约 1.44 TB 缓存
# - 保存到 /disk3/minami/LMUData/vision_cache/Video-MME_128frame/
```

**进度监控**（另开一个终端）：
```bash
# 监控缓存大小
watch -n 60 'du -sh /disk3/minami/LMUData/vision_cache && df -h /disk3'
```

---

## 三、验证缓存

```bash
# 检查缓存文件数量（应该是 2,999）
find /disk3/minami/LMUData/vision_cache/Video-MME_128frame -name "*.pt" | wc -l

# 检查缓存大小（应该约 1.4 TB）
du -sh /disk3/minami/LMUData/vision_cache/Video-MME_128frame

# 查看元数据
cat /disk3/minami/LMUData/vision_cache/cache_meta.json | python3 -m json.tool
```

**预期输出**：
```json
{
  "version": "1.0",
  "dtype": "float16",
  "compression": "none",
  "total_samples": 2999,
  "total_size_bytes": 1548120000000,
  "datasets": {
    "Video-MME_128frame": {
      "count": 2999,
      "size_bytes": 1548120000000
    }
  }
}
```

---

## 四、运行评估（使用缓存）

```bash
# 单 GPU 评估（如果只有一张卡）
export CUDA_VISIBLE_DEVICES=0
bash eval_longva_video_cached.sh

# 或者多 GPU 评估
export CUDA_VISIBLE_DEVICES=0,1
bash eval_longva_video_cached.sh
```

**测试的模型**：
1. LongVA-Temporal-v1-Cached
2. LongVA-Temporal-v2-Cached
3. LongVA-7B-Cached

**预期输出**：
- 结果目录: `results/longva_cached_YYYYMMDD_HHMMSS/`
- 日志文件: `logs/longva_cached_YYYYMMDD_HHMMSS.log`
- 缓存命中率应该是 100%

---

## 五、检查结果

```bash
# 查看最新的结果目录
ls -lht results/ | head -5

# 查看日志（检查缓存统计）
tail -100 logs/longva_cached_*.log

# 应该看到:
# ============================================================
# Vision Cache Statistics
# ============================================================
# Cache hits: 2999
# Cache misses: 0
# Hit rate: 100.0%
# ============================================================
```

---

## 六、对比测试（可选）

验证缓存模型和原始模型的输出是否一致：

```bash
# 1. 测试原始模型（不使用缓存）
export CUDA_VISIBLE_DEVICES=0
python run.py \
    --data Video-MME_128frame \
    --model LongVA-7B \
    --mode all \
    --work-dir results/comparison

# 2. 测试缓存模型
python run.py \
    --data Video-MME_128frame \
    --model LongVA-7B-Cached \
    --mode all \
    --work-dir results/comparison

# 3. 比较结果（应该完全一致或非常接近）
# 结果文件位于:
# - results/comparison/LongVA-7B/<eval_id>/LongVA-7B_Video-MME_128frame.xlsx
# - results/comparison/LongVA-7B-Cached/<eval_id>/LongVA-7B-Cached_Video-MME_128frame.xlsx
```

---

## 七、常见问题

### Q1: 预处理太慢？

**A**: 这是一次性的。完成后可以永久复用。预计时间：
- 单 GPU (A100): ~8-10 小时
- 单 GPU (V100): ~10-12 小时
- 多 GPU 不会加速预处理（因为是顺序处理）

### Q2: 缓存命中率不是 100%？

**A**: 检查以下几点：
```bash
# 1. 检查模型签名是否匹配
python3 -c "
from vlmeval.vlm.longva_cached import LongVA_Cached
model = LongVA_Cached('lmms-lab/LongVA-7B-DPO', cache_dir='/disk3/minami/LMUData/vision_cache')
print('Model signature:', model.model_signature)
"

# 2. 检查数据集名称是否一致
ls /disk3/minami/LMUData/vision_cache/

# 3. 检查缓存文件是否完整
find /disk3/minami/LMUData/vision_cache/Video-MME_128frame -name "*.pt" | wc -l
```

### Q3: 内存不足？

**A**: 减少并行数或使用更少的 GPU：
```bash
# 只使用一张卡
export CUDA_VISIBLE_DEVICES=0
bash eval_longva_video_cached.sh
```

### Q4: 想清理缓存重新开始？

**A**:
```bash
# 删除 Video-MME 缓存
rm -rf /disk3/minami/LMUData/vision_cache/Video-MME_128frame/

# 重新运行预处理
python preprocess_video_mme.py
```

---

## 八、文件清单

确认以下文件都已创建：

```bash
# 检查所有相关文件
ls -lh preprocess_video_mme.py
ls -lh eval_longva_video_cached.sh
ls -lh vlmeval/vlm/longva_cached.py
ls -lh README_VISION_CACHE.md
ls -lh QUICKSTART_CACHE.md

# 检查配置是否正确
grep "LongVA-.*-Cached" vlmeval/config.py
grep "LongVA_Cached" vlmeval/vlm/__init__.py
```

**应该看到**：
- ✅ `preprocess_video_mme.py` (预处理脚本)
- ✅ `eval_longva_video_cached.sh` (评估脚本)
- ✅ `vlmeval/vlm/longva_cached.py` (缓存模型类)
- ✅ `vlmeval/config.py` 包含 "LongVA-*-Cached" 配置
- ✅ `vlmeval/vlm/__init__.py` 导入 `LongVA_Cached`

---

## 九、完整流程示例

```bash
# === 第一步：预处理（一次性） ===
cd /disk3/minami/Vision-Retrieval-Head/VLMEvalKit

python preprocess_video_mme.py \
    --model-path lmms-lab/LongVA-7B-DPO \
    --cache-dir /disk3/minami/LMUData/vision_cache \
    --nframe 128 \
    --dataset Video-MME_128frame

# 等待完成... (8-12 小时)

# === 第二步：验证缓存 ===
find /disk3/minami/LMUData/vision_cache/Video-MME_128frame -name "*.pt" | wc -l
# 应该输出: 2999

du -sh /disk3/minami/LMUData/vision_cache/Video-MME_128frame
# 应该约: 1.4T

# === 第三步：运行评估 ===
export CUDA_VISIBLE_DEVICES=0,1
bash eval_longva_video_cached.sh

# === 第四步：查看结果 ===
tail -100 logs/longva_cached_*.log | grep -A 10 "Cache Statistics"

# 应该看到:
# Cache hits: 2999
# Hit rate: 100.0%
```

---

## 🎯 成功标准

评估完成后，确认：

- ✅ 缓存命中率 = 100%
- ✅ 三个模型都成功完成评估
- ✅ 生成了结果文件 (`*.xlsx` 和 `*.csv`)
- ✅ 结果与原始模型一致（如果做了对比测试）

---

## 📞 需要帮助？

如果遇到问题：

1. **查看日志**: `cat logs/longva_cached_*.log`
2. **检查缓存**: `cat /disk3/minami/LMUData/vision_cache/cache_meta.json`
3. **验证配置**: `grep -r "LongVA_Cached" vlmeval/`

---

祝测试顺利！ 🚀
