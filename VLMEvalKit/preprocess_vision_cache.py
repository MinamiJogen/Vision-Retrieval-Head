#!/usr/bin/env python3
"""
LongVA 视频数据集预处理脚本
提取并缓存 vision encoder 输出，支持 int8 量化和压缩存储

用法:
    python preprocess_vision_cache.py \
        --datasets Video-MME_128frame MLVU_128frame \
        --model-path lmms-lab/LongVA-7B-DPO \
        --cache-dir /disk3/minami/LMUData/vision_cache \
        --quantize int8 \
        --num-workers 8
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")

# 环境变量
os.environ['TMPDIR'] = '/disk3/minami/tmp'
os.environ['TORCH_HOME'] = '/disk3/minami/tmp/torch'
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'


@dataclass
class VisionFeatureCache:
    """Vision 特征缓存数据结构"""
    vision_embeds: torch.Tensor  # [num_tokens, hidden_dim]
    video_path: str
    nframe: int
    dataset: str
    model_signature: str  # 模型版本标识
    quantization: Optional[str] = None  # None, 'int8', 'bfloat16'

    # 用于量化的 scale 和 zero_point
    scale: Optional[torch.Tensor] = None
    zero_point: Optional[torch.Tensor] = None


class VisionCacheManager:
    """Vision 特征缓存管理器"""

    def __init__(self, cache_dir: str, quantization: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.quantization = quantization

        # 创建元数据文件
        self.meta_file = self.cache_dir / "cache_meta.json"
        self.load_metadata()

    def load_metadata(self):
        """加载缓存元数据"""
        if self.meta_file.exists():
            with open(self.meta_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'total_samples': 0,
                'total_size_bytes': 0,
                'datasets': {}
            }

    def save_metadata(self):
        """保存缓存元数据"""
        with open(self.meta_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def get_cache_key(self, video_id: str, dataset: str, nframe: int, model_sig: str) -> str:
        """生成缓存键"""
        key_str = f"{dataset}:{video_id}:nframe{nframe}:model{model_sig}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_cache_path(self, cache_key: str, dataset: str) -> Path:
        """获取缓存文件路径"""
        dataset_dir = self.cache_dir / dataset
        dataset_dir.mkdir(exist_ok=True)
        return dataset_dir / f"{cache_key}.pt"

    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将 float16 tensor 量化到 int8"""
        if self.quantization == 'int8':
            # 对称量化
            abs_max = torch.abs(tensor).max()
            scale = abs_max / 127.0
            quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
            return quantized, scale, None
        elif self.quantization == 'bfloat16':
            return tensor.to(torch.bfloat16), None, None
        else:
            return tensor, None, None

    def dequantize_tensor(self, quantized: torch.Tensor, scale: Optional[torch.Tensor],
                         zero_point: Optional[torch.Tensor]) -> torch.Tensor:
        """反量化"""
        if quantized.dtype == torch.int8:
            return quantized.float() * scale
        else:
            return quantized.float()

    def save_cache(self, cache: VisionFeatureCache) -> int:
        """保存缓存"""
        # 量化
        quantized_embeds, scale, zero_point = self.quantize_tensor(cache.vision_embeds)

        # 生成cache key
        video_id = Path(cache.video_path).stem
        cache_key = self.get_cache_key(video_id, cache.dataset, cache.nframe, cache.model_signature)
        cache_path = self.get_cache_path(cache_key, cache.dataset)

        # 保存数据
        save_data = {
            'vision_embeds': quantized_embeds.cpu(),
            'scale': scale.cpu() if scale is not None else None,
            'zero_point': zero_point,
            'video_path': cache.video_path,
            'nframe': cache.nframe,
            'dataset': cache.dataset,
            'model_signature': cache.model_signature,
            'quantization': self.quantization,
            'shape': list(cache.vision_embeds.shape),
        }

        torch.save(save_data, cache_path, _use_new_zipfile_serialization=True)

        # 更新元数据
        file_size = cache_path.stat().st_size
        if cache.dataset not in self.metadata['datasets']:
            self.metadata['datasets'][cache.dataset] = {
                'count': 0,
                'size_bytes': 0
            }
        self.metadata['datasets'][cache.dataset]['count'] += 1
        self.metadata['datasets'][cache.dataset]['size_bytes'] += file_size
        self.metadata['total_samples'] += 1
        self.metadata['total_size_bytes'] += file_size

        return file_size

    def load_cache(self, video_path: str, dataset: str, nframe: int,
                  model_sig: str) -> Optional[torch.Tensor]:
        """加载缓存"""
        video_id = Path(video_path).stem
        cache_key = self.get_cache_key(video_id, dataset, nframe, model_sig)
        cache_path = self.get_cache_path(cache_key, dataset)

        if not cache_path.exists():
            return None

        data = torch.load(cache_path, map_location='cpu')
        vision_embeds = self.dequantize_tensor(
            data['vision_embeds'],
            data.get('scale'),
            data.get('zero_point')
        )

        return vision_embeds


class VisionPreprocessor:
    """Vision 特征预处理器"""

    def __init__(self, model_path: str, cache_manager: VisionCacheManager):
        self.model_path = model_path
        self.cache_manager = cache_manager

        # 加载模型
        print(f"正在加载模型: {model_path}")
        from longva.model.builder import load_pretrained_model

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, "llava_qwen", device_map="auto"
        )
        self.model.eval()

        # 生成模型签名
        self.model_signature = self._generate_model_signature()
        print(f"模型签名: {self.model_signature}")

    def _generate_model_signature(self) -> str:
        """生成模型签名（基于 vision tower 配置）"""
        config = self.model.config
        sig_str = f"{config.mm_vision_tower}:layer{config.mm_vision_select_layer}"
        return hashlib.md5(sig_str.encode()).hexdigest()[:8]

    def process_video(self, video_path: str, dataset: str, nframe: int) -> Optional[int]:
        """处理单个视频"""
        try:
            # 检查缓存
            cached = self.cache_manager.load_cache(
                video_path, dataset, nframe, self.model_signature
            )
            if cached is not None:
                return 0  # 已缓存

            # 加载视频帧
            from decord import VideoReader, cpu

            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)

            # 均匀采样
            if total_frame_num <= nframe:
                frame_idx = list(range(total_frame_num))
            else:
                frame_idx = np.linspace(0, total_frame_num - 1, nframe, dtype=int).tolist()

            frames = vr.get_batch(frame_idx).asnumpy()

            # 预处理
            video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            video_tensor = video_tensor.to(self.model.device, dtype=torch.float16)

            # 提取特征
            with torch.no_grad():
                # 访问 vision tower
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
                    vision_tower = self.model.model.vision_tower
                    mm_projector = self.model.model.mm_projector
                else:
                    raise RuntimeError("无法访问 vision_tower 和 mm_projector")

                # Vision encoder
                vision_features = vision_tower(video_tensor)

                # MM Projector
                vision_embeds = mm_projector(vision_features)

            # 保存缓存
            cache = VisionFeatureCache(
                vision_embeds=vision_embeds.squeeze(0),  # 移除batch维度
                video_path=video_path,
                nframe=nframe,
                dataset=dataset,
                model_signature=self.model_signature,
                quantization=self.cache_manager.quantization
            )

            file_size = self.cache_manager.save_cache(cache)
            return file_size

        except Exception as e:
            print(f"✗ 处理失败 {video_path}: {e}")
            return None

    def process_dataset(self, dataset_name: str, nframe: int):
        """处理整个数据集"""
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*60}")

        # 导入 VLMEvalKit 的数据集配置
        sys.path.insert(0, '/disk3/minami/Vision-Retrieval-Head/VLMEvalKit')
        from vlmeval.dataset.video_dataset_config import supported_video_datasets

        # 构建数据集
        if dataset_name not in supported_video_datasets:
            print(f"✗ 未知数据集: {dataset_name}")
            return

        dataset = supported_video_datasets[dataset_name]()
        print(f"数据集大小: {len(dataset)} 个样本")

        # 处理每个样本
        total_size = 0
        cached_count = 0
        processed_count = 0

        with tqdm(total=len(dataset), desc=f"处理 {dataset_name}") as pbar:
            for idx in range(len(dataset)):
                row = dataset.data.iloc[idx]

                # 获取视频路径
                if 'video' in row:
                    video_path = row['video']
                elif 'video_path' in row:
                    video_path = row['video_path']
                else:
                    print(f"✗ 无法找到视频路径: {row}")
                    continue

                # 处理
                size = self.process_video(video_path, dataset_name, nframe)

                if size is not None:
                    if size == 0:
                        cached_count += 1
                    else:
                        processed_count += 1
                        total_size += size

                pbar.update(1)
                pbar.set_postfix({
                    '已处理': processed_count,
                    '已缓存': cached_count,
                    '累计大小': f'{total_size / (1024**3):.2f} GB'
                })

        # 保存元数据
        self.cache_manager.save_metadata()

        print(f"\n完成! 已处理: {processed_count}, 已缓存: {cached_count}")
        print(f"总大小: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description='LongVA 视频预处理')
    parser.add_argument('--datasets', nargs='+', required=True,
                       help='数据集名称列表')
    parser.add_argument('--model-path', default='lmms-lab/LongVA-7B-DPO',
                       help='模型路径')
    parser.add_argument('--cache-dir', default='/disk3/minami/LMUData/vision_cache',
                       help='缓存目录')
    parser.add_argument('--quantize', choices=['int8', 'bfloat16', 'none'],
                       default='int8', help='量化方法')
    parser.add_argument('--nframe', type=int, default=128,
                       help='采样帧数')

    args = parser.parse_args()

    # 初始化
    quantization = None if args.quantize == 'none' else args.quantize
    cache_manager = VisionCacheManager(args.cache_dir, quantization=quantization)
    preprocessor = VisionPreprocessor(args.model_path, cache_manager)

    # 处理数据集
    for dataset in args.datasets:
        preprocessor.process_dataset(dataset, args.nframe)

    # 最终统计
    print(f"\n{'='*60}")
    print("预处理完成!")
    print(f"{'='*60}")
    print(f"总样本数: {cache_manager.metadata['total_samples']}")
    print(f"总大小: {cache_manager.metadata['total_size_bytes'] / (1024**3):.2f} GB")
    print(f"平均每样本: {cache_manager.metadata['total_size_bytes'] / cache_manager.metadata['total_samples'] / (1024**2):.1f} MB")

    for dataset, stats in cache_manager.metadata['datasets'].items():
        print(f"\n{dataset}:")
        print(f"  样本数: {stats['count']}")
        print(f"  大小: {stats['size_bytes'] / (1024**3):.2f} GB")


if __name__ == '__main__':
    main()
