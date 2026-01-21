# 🎬 LongVA 视频Benchmark测试说明

## 📝 新脚本信息

**脚本名称**: `eval_longva_video_benchmarks.sh`

**测试内容**: 8个视频benchmark

**预计时间**: 根据视频数量，每个benchmark可能需要30分钟到几小时不等

## 🎯 测试的8个视频Benchmark

| # | Benchmark | 说明 |
|---|-----------|------|
| 1 | VideoMME | 视频多模态评估 - 综合视频理解能力 |
| 2 | Video_Holmes | 视频福尔摩斯 - 视频推理和侦探能力 |
| 3 | LongVideoBench | 长视频理解 - 测试长视频内容理解 |
| 4 | VideoMMMU | 视频MMMU - 多学科视频问答 |
| 5 | MMBenchVideo | MMBench视频版 - 多模态视频理解 |
| 6 | MLVU | 多模态长视频理解 - 长视频综合评估 |
| 7 | TempCompass | 时序理解 - 时间顺序和因果关系 |
| 8 | TempCompass_MCQ | 时序选择题 - TempCompass的选择题版本 |

## 🚀 使用方法

### 快速启动

```bash
cd /disk3/minami/Vision-Retrieval-Head/VLMEvalKit
bash ../eval_longva_video_benchmarks.sh
```

### 修改测试的模型

编辑脚本第21-25行：

```bash
MODELS=(
    "LongVA-Temporal-v1"
    "LongVA-Temporal-v2"
    "LongVA-7B"
)
```

### 添加/删除benchmark

编辑脚本第30-39行：

```bash
VIDEO_BENCHMARKS=(
    "VideoMME"
    "Video_Holmes"
    "LongVideoBench"
    "VideoMMMU"
    "MMBenchVideo"
    "MLVU"
    "TempCompass"
    "TempCompass_MCQ"
    # "MVBench"              # 可以添加其他视频benchmark
    # "QBench_Video"         # 可以添加其他视频benchmark
)
```

## 📊 其他可用的视频Benchmark

如果需要测试更多视频benchmark，可以添加以下选项：

### 视频理解类
- `MVBench` - 多视图视频理解
- `MVBench_MP4` - 多视图视频理解（MP4格式）
- `WorldSense` - 世界感知视频理解

### 视频问答类
- `QBench_Video` - 视频质量问答
- `QBench_Video_MCQ` - 视频质量选择题
- `QBench_Video_VQA` - 视频质量VQA

### 时序和内容类
- `TempCompass_Captioning` - 时序描述
- `TempCompass_YorN` - 时序是非题
- `MLVU_MCQ` - MLVU选择题版本
- `MLVU_OpenEnded` - MLVU开放式问答

### 特定场景
- `MovieChat1k` - 电影对话1k
- `VDC` - 视频描述对话
- `VideoTT` - 视频文本追踪
- `MEGABench` - MEGA视频评测
- `EgoExoBench_MCQ` - 第一/第三人称视频
- `VCRBench` - 视频常识推理
- `CGAVCounting` - 音视频计数
- `MVTamperBench` - 视频篡改检测
- `Video_MMLU_CAP` - 视频MMLU描述
- `Video_MMLU_QA` - 视频MMLU问答

### 视频生成和内容基准
- `CGBench_MCQ_Grounding_Mini` - CG定位选择题（迷你版）
- `CGBench_OpenEnded_Mini` - CG开放式问答（迷你版）
- `CGBench_MCQ_Grounding` - CG定位选择题
- `CGBench_OpenEnded` - CG开放式问答

## 🔍 监控和调试

### 实时查看日志
```bash
tail -f eval_longva_video_*.log
```

### 查看进度
```bash
ls -lh longva_video_results_*/
```

### 查看某个benchmark的状态
```bash
ls -lh longva_video_results_*/LongVA-7B/VideoMME*
```

## ⚠️ 注意事项

1. **视频文件大小**: 视频benchmark通常需要更多存储空间，确保有足够的磁盘空间
2. **处理时间**: 视频处理比图像慢得多，单个benchmark可能需要几小时
3. **GPU内存**: 某些长视频benchmark可能需要大量GPU内存
4. **网络带宽**: 首次运行需要下载视频数据集，可能需要较长时间

## 📈 结果文件

结果将保存在 `longva_video_results_YYYYMMDD_HHMMSS/` 目录下：

```
longva_video_results_20251210_120000/
├── LongVA-Temporal-v1/
│   ├── VideoMME.xlsx
│   ├── Video_Holmes.xlsx
│   ├── LongVideoBench.xlsx
│   └── ...
├── LongVA-Temporal-v2/
└── LongVA-7B/
```

## 💡 快速测试建议

如果想先快速测试，可以只选择几个较小的benchmark：

```bash
VIDEO_BENCHMARKS=(
    "MMBenchVideo"      # 相对较小
    "TempCompass_MCQ"   # 选择题格式，评估快
)
```

然后再逐步添加其他更大的benchmark：

```bash
VIDEO_BENCHMARKS=(
    "MMBenchVideo"
    "TempCompass_MCQ"
    "VideoMME"          # 添加中等规模
    "MLVU"              # 添加中等规模
)
```

最后测试长视频：

```bash
VIDEO_BENCHMARKS=(
    # ... 前面的所有benchmark
    "LongVideoBench"    # 长视频，处理时间较长
    "VideoMMMU"         # 综合测试
)
```

## 📞 问题排查

### 如果某个benchmark失败

1. 查看具体错误信息：
```bash
grep "VideoMME" eval_longva_video_*.log -A 10 -B 5
```

2. 单独测试该benchmark：
```bash
cd VLMEvalKit
python run.py --data VideoMME --model LongVA-7B --mode all --verbose
```

3. 检查数据集是否正确下载：
```bash
ls -lh ~/.cache/vlmeval/  # 或你的数据缓存目录
```

### 常见问题

- **内存不足**: 减少batch size或使用更小的模型
- **下载超时**: 检查网络连接，可能需要使用代理
- **API限制**: 某些评估需要OpenAI API，检查配额和密钥

---

## 🎯 总结

**原始脚本**: 9个图像benchmark
**新脚本**: 8个视频benchmark

新脚本专注于视频理解能力测试，适合评估LongVA模型的视频处理性能。

运行命令：
```bash
cd /disk3/minami/Vision-Retrieval-Head/VLMEvalKit
bash ../eval_longva_video_benchmarks.sh
```
