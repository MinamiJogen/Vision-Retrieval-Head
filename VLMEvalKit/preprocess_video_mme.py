#!/usr/bin/env python3
"""
Video-MME 预处理脚本 (Float16 无压缩)
提取并缓存 vision tower 输出，确保与正常推理流程完全一致

用法:
    python preprocess_video_mme.py
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
import warnings
from typing import Optional
import pandas as pd

warnings.filterwarnings("ignore")

# 环境变量
os.environ['TMPDIR'] = '/disk3/minami/tmp'
os.environ['TORCH_HOME'] = '/disk3/minami/tmp/torch'
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'


class VisionCacheManager:
    """Vision 特征缓存管理器 (Float16 无压缩)"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 元数据文件
        self.meta_file = self.cache_dir / "cache_meta.json"
        self.load_metadata()

    def load_metadata(self):
        """加载缓存元数据"""
        if self.meta_file.exists():
            with open(self.meta_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'version': '1.0',
                'dtype': 'float16',
                'compression': 'none',
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

    def cache_exists(self, video_id: str, dataset: str, nframe: int, model_sig: str) -> bool:
        """检查缓存是否存在"""
        cache_key = self.get_cache_key(video_id, dataset, nframe, model_sig)
        cache_path = self.get_cache_path(cache_key, dataset)
        return cache_path.exists()

    def save_cache(self, video_id: str, video_path: str, dataset: str, nframe: int,
                  model_sig: str, vision_features: torch.Tensor) -> int:
        """保存缓存 (Float16 无压缩)"""
        cache_key = self.get_cache_key(video_id, dataset, nframe, model_sig)
        cache_path = self.get_cache_path(cache_key, dataset)

        # 确保是 float16
        vision_features = vision_features.to(torch.float16).cpu()

        # 保存数据
        save_data = {
            'vision_features': vision_features,
            'video_id': video_id,
            'video_path': video_path,
            'nframe': nframe,
            'dataset': dataset,
            'model_signature': model_sig,
            'dtype': 'float16',
            'shape': list(vision_features.shape),
        }

        torch.save(save_data, cache_path)

        # 更新元数据
        file_size = cache_path.stat().st_size
        if dataset not in self.metadata['datasets']:
            self.metadata['datasets'][dataset] = {
                'count': 0,
                'size_bytes': 0
            }
        self.metadata['datasets'][dataset]['count'] += 1
        self.metadata['datasets'][dataset]['size_bytes'] += file_size
        self.metadata['total_samples'] += 1
        self.metadata['total_size_bytes'] += file_size

        return file_size

    def load_cache(self, video_id: str, dataset: str, nframe: int,
                  model_sig: str) -> Optional[torch.Tensor]:
        """加载缓存"""
        cache_key = self.get_cache_key(video_id, dataset, nframe, model_sig)
        cache_path = self.get_cache_path(cache_key, dataset)

        if not cache_path.exists():
            return None

        data = torch.load(cache_path, map_location='cpu')
        return data['vision_features']


class VideoMMEPreprocessor:
    """Video-MME 预处理器"""

    def __init__(self, model_path: str, cache_dir: str, nframe: int = 128):
        self.model_path = model_path
        self.nframe = nframe
        self.cache_manager = VisionCacheManager(cache_dir)

        print("=" * 60)
        print("Video-MME 预处理脚本")
        print("=" * 60)
        print(f"模型路径: {model_path}")
        print(f"缓存目录: {cache_dir}")
        print(f"采样帧数: {nframe}")
        print(f"数据类型: float16 (无压缩)")
        print()

        # 加载模型
        print("正在加载模型...")
        from longva.model.builder import load_pretrained_model

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, "llava_qwen", device_map="auto"
        )
        self.model.eval()

        # 生成模型签名
        self.model_signature = self._generate_model_signature()
        print(f"✓ 模型加载完成")
        print(f"✓ 模型签名: {self.model_signature}")
        print()

    def _generate_model_signature(self) -> str:
        """生成模型签名（基于 vision tower 配置）"""
        config = self.model.config
        sig_str = f"{config.mm_vision_tower}:layer{config.mm_vision_select_layer}"
        return hashlib.md5(sig_str.encode()).hexdigest()[:8]

    def _load_video_frames(self, video_path: str) -> np.ndarray:
        """加载视频帧（与 longva_custom.py 完全一致）"""
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)

        # 均匀采样（与 longva_custom.py:90-108 一致）
        if total_frame_num <= self.nframe:
            frame_idx = list(range(total_frame_num))
        else:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.nframe, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()

        frames = vr.get_batch(frame_idx).asnumpy()
        return frames

    def process_video(self, video_id: str, video_path: str, dataset_name: str, data_root: str) -> Optional[int]:
        """处理单个视频"""
        try:
            # 检查缓存
            if self.cache_manager.cache_exists(video_id, dataset_name, self.nframe, self.model_signature):
                return 0  # 已缓存

            # 构建完整的视频路径
            # Video-MME 的视频路径格式: {data_root}/video/{video_id}.mp4
            full_video_path = os.path.join(data_root, 'video', f'{video_id}.mp4')

            if not os.path.exists(full_video_path):
                print(f"✗ 视频文件不存在: {full_video_path}")
                return None

            # 加载视频帧（与 longva_custom.py 一致）
            frames = self._load_video_frames(full_video_path)

            # 预处理（与 longva_custom.py:145 一致）
            video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            video_tensor = video_tensor.to(self.model.device, dtype=torch.float16)

            # 提取 vision features（只到 vision_tower，不经过 mm_projector）
            with torch.no_grad():
                # 访问 vision tower
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
                    vision_tower = self.model.model.vision_tower
                elif hasattr(self.model, 'vision_tower'):
                    vision_tower = self.model.vision_tower
                else:
                    raise RuntimeError("无法访问 vision_tower")

                # 只提取 vision_tower 的输出，不经过 mm_projector
                # 这样在推理时可以让 mm_projector 正常运行，确保流程完全一致
                vision_features = vision_tower(video_tensor)

            # 保存缓存
            file_size = self.cache_manager.save_cache(
                video_id=video_id,
                video_path=full_video_path,
                dataset=dataset_name,
                nframe=self.nframe,
                model_sig=self.model_signature,
                vision_features=vision_features.squeeze(0)  # 移除batch维度
            )

            return file_size

        except Exception as e:
            print(f"✗ 处理失败 {video_id}: {e}")
            return None

    def process_dataset(self, dataset_name: str):
        """处理 Video-MME 数据集"""
        print("=" * 60)
        print(f"处理数据集: {dataset_name}")
        print("=" * 60)

        # 导入 VLMEvalKit 的数据集
        sys.path.insert(0, '/disk3/minami/Vision-Retrieval-Head/VLMEvalKit')
        from vlmeval.dataset.video_dataset_config import supported_video_datasets

        # 构建数据集
        if dataset_name not in supported_video_datasets:
            print(f"✗ 未知数据集: {dataset_name}")
            print(f"可用数据集: {list(supported_video_datasets.keys())}")
            return

        dataset = supported_video_datasets[dataset_name]()
        print(f"✓ 数据集加载完成")
        print(f"✓ 样本数量: {len(dataset)}")

        # 获取数据集根目录
        data_root = dataset.data_root
        print(f"✓ 数据集根目录: {data_root}")
        print()

        # 处理每个样本
        total_size = 0
        cached_count = 0
        processed_count = 0
        failed_count = 0

        print("开始处理视频...")
        print()

        with tqdm(total=len(dataset), desc=f"处理 {dataset_name}") as pbar:
            for idx in range(len(dataset)):
                row = dataset.data.iloc[idx]

                # 获取视频 ID (不是完整路径)
                if 'video' in row:
                    video_id = row['video']
                elif 'videoID' in row:
                    video_id = row['videoID']
                elif 'video_id' in row:
                    video_id = row['video_id']
                else:
                    print(f"✗ 样本 {idx} 无法找到视频 ID")
                    failed_count += 1
                    pbar.update(1)
                    continue

                # 处理视频（传入 data_root）
                size = self.process_video(video_id, row.get('video_path', ''), dataset_name, data_root)

                if size is not None:
                    if size == 0:
                        cached_count += 1
                    else:
                        processed_count += 1
                        total_size += size
                else:
                    failed_count += 1

                pbar.update(1)
                pbar.set_postfix({
                    '新处理': processed_count,
                    '已缓存': cached_count,
                    '失败': failed_count,
                    '累计大小': f'{total_size / (1024**3):.2f} GB'
                })

        # 保存元数据
        self.cache_manager.save_metadata()

        # 最终统计
        print()
        print("=" * 60)
        print("处理完成!")
        print("=" * 60)
        print(f"总样本数: {len(dataset)}")
        print(f"新处理: {processed_count}")
        print(f"已缓存: {cached_count}")
        print(f"失败: {failed_count}")
        print(f"本次大小: {total_size / (1024**3):.2f} GB ({total_size / (1024**4):.3f} TB)")
        print()

        # 总体统计
        meta = self.cache_manager.metadata
        if dataset_name in meta['datasets']:
            stats = meta['datasets'][dataset_name]
            print(f"{dataset_name} 总计:")
            print(f"  缓存样本数: {stats['count']}")
            print(f"  总大小: {stats['size_bytes'] / (1024**3):.2f} GB ({stats['size_bytes'] / (1024**4):.3f} TB)")


def main():
    parser = argparse.ArgumentParser(description='Video-MME 预处理脚本')
    parser.add_argument('--model-path', default='lmms-lab/LongVA-7B',
                       help='模型路径 (默认: lmms-lab/LongVA-7B)')
    parser.add_argument('--cache-dir', default='/disk3/minami/LMUData/vision_cache',
                       help='缓存目录 (默认: /disk3/minami/LMUData/vision_cache)')
    parser.add_argument('--nframe', type=int, default=128,
                       help='采样帧数 (默认: 128)')
    parser.add_argument('--dataset', default='Video-MME_128frame',
                       help='数据集名称 (默认: Video-MME_128frame)')

    args = parser.parse_args()

    # 初始化预处理器
    preprocessor = VideoMMEPreprocessor(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        nframe=args.nframe
    )

    # 处理数据集
    preprocessor.process_dataset(args.dataset)

    print()
    print("=" * 60)
    print("全部完成!")
    print("=" * 60)
    print(f"缓存位置: {args.cache_dir}")
    print(f"元数据文件: {args.cache_dir}/cache_meta.json")
    print()
    print("下一步: 使用 eval_longva_video_cached.sh 进行测试")
    print("=" * 60)


if __name__ == '__main__':
    main()
