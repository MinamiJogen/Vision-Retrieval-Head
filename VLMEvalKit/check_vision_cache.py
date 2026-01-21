#!/usr/bin/env python3
"""
检查 Vision Cache 参数和完整性

用法:
    python check_vision_cache.py [--cache-dir PATH] [--dataset NAME] [--detailed]
"""

import argparse
import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import torch
from tqdm import tqdm


def format_bytes(bytes_size):
    """格式化字节大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def check_cache_file(cache_path):
    """检查单个缓存文件的参数"""
    try:
        data = torch.load(cache_path, map_location='cpu')

        info = {
            'video_id': data.get('video_id', 'N/A'),
            'video_path': data.get('video_path', 'N/A'),
            'nframe': data.get('nframe', 'N/A'),
            'dataset': data.get('dataset', 'N/A'),
            'model_signature': data.get('model_signature', 'N/A'),
            'dtype': data.get('dtype', 'N/A'),
            'shape': data.get('shape', 'N/A'),
            'file_size': os.path.getsize(cache_path),
        }

        # 检查 vision_features 的实际 dtype 和 shape
        if 'vision_features' in data:
            features = data['vision_features']
            info['actual_dtype'] = str(features.dtype)
            info['actual_shape'] = list(features.shape)

        return info, None

    except Exception as e:
        return None, str(e)


def check_cache_directory(cache_dir, dataset_name=None, detailed=False):
    """检查缓存目录"""
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        print(f"❌ 缓存目录不存在: {cache_dir}")
        return

    print("=" * 80)
    print("Vision Cache 参数检查")
    print("=" * 80)
    print(f"缓存目录: {cache_dir}")
    print()

    # 检查元数据文件
    meta_file = cache_dir / "cache_meta.json"
    if meta_file.exists():
        print("📄 元数据文件 (cache_meta.json):")
        print("-" * 80)
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print()
    else:
        print("⚠️  元数据文件不存在")
        metadata = None

    # 扫描数据集目录
    dataset_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]

    if dataset_name:
        dataset_dirs = [d for d in dataset_dirs if d.name == dataset_name]
        if not dataset_dirs:
            print(f"❌ 数据集不存在: {dataset_name}")
            return

    if not dataset_dirs:
        print("⚠️  没有找到数据集缓存目录")
        return

    print("=" * 80)
    print("数据集缓存检查")
    print("=" * 80)
    print()

    for dataset_dir in sorted(dataset_dirs):
        print(f"📁 数据集: {dataset_dir.name}")
        print("-" * 80)

        # 查找所有缓存文件
        cache_files = list(dataset_dir.glob("*.pt"))

        if not cache_files:
            print("  ⚠️  没有找到缓存文件")
            print()
            continue

        print(f"  缓存文件数: {len(cache_files)}")

        # 统计信息
        stats = {
            'total_files': len(cache_files),
            'total_size': 0,
            'nframes': defaultdict(int),
            'model_signatures': defaultdict(int),
            'dtypes': defaultdict(int),
            'shapes': defaultdict(int),
            'errors': [],
        }

        # 检查每个文件
        print(f"  正在检查缓存文件...")

        sample_info = None

        for cache_file in tqdm(cache_files, desc="  扫描", disable=not detailed):
            info, error = check_cache_file(cache_file)

            if error:
                stats['errors'].append((cache_file.name, error))
                continue

            if info:
                # 保存第一个样本作为示例
                if sample_info is None:
                    sample_info = info

                # 统计
                stats['total_size'] += info['file_size']
                stats['nframes'][info['nframe']] += 1
                stats['model_signatures'][info['model_signature']] += 1
                stats['dtypes'][info.get('actual_dtype', info['dtype'])] += 1

                shape_str = str(info.get('actual_shape', info['shape']))
                stats['shapes'][shape_str] += 1

        # 输出统计信息
        print()
        print(f"  ✓ 成功读取: {stats['total_files'] - len(stats['errors'])} 个文件")
        print(f"  ✗ 读取失败: {len(stats['errors'])} 个文件")
        print(f"  📊 总大小: {format_bytes(stats['total_size'])}")
        print()

        # 平均文件大小
        if stats['total_files'] > 0:
            avg_size = stats['total_size'] / stats['total_files']
            print(f"  平均文件大小: {format_bytes(avg_size)}")
            print()

        # nframe 分布
        print(f"  📈 nframe 分布:")
        for nframe, count in sorted(stats['nframes'].items()):
            pct = count / stats['total_files'] * 100
            print(f"    {nframe} 帧: {count} 个文件 ({pct:.1f}%)")
        print()

        # 模型签名分布
        print(f"  🔑 模型签名分布:")
        for sig, count in sorted(stats['model_signatures'].items()):
            pct = count / stats['total_files'] * 100
            print(f"    {sig}: {count} 个文件 ({pct:.1f}%)")
        print()

        # dtype 分布
        print(f"  🔢 数据类型分布:")
        for dtype, count in sorted(stats['dtypes'].items()):
            pct = count / stats['total_files'] * 100
            print(f"    {dtype}: {count} 个文件 ({pct:.1f}%)")
        print()

        # shape 分布
        print(f"  📐 特征形状分布:")
        for shape, count in sorted(stats['shapes'].items()):
            pct = count / stats['total_files'] * 100
            print(f"    {shape}: {count} 个文件 ({pct:.1f}%)")
        print()

        # 显示示例
        if sample_info and detailed:
            print(f"  📋 缓存文件示例:")
            print(f"    Video ID: {sample_info['video_id']}")
            print(f"    Video Path: {sample_info['video_path']}")
            print(f"    nframe: {sample_info['nframe']}")
            print(f"    Dataset: {sample_info['dataset']}")
            print(f"    Model Signature: {sample_info['model_signature']}")
            print(f"    Dtype: {sample_info.get('actual_dtype', sample_info['dtype'])}")
            print(f"    Shape: {sample_info.get('actual_shape', sample_info['shape'])}")
            print(f"    File Size: {format_bytes(sample_info['file_size'])}")
            print()

        # 显示错误
        if stats['errors'] and detailed:
            print(f"  ❌ 错误文件:")
            for filename, error in stats['errors'][:10]:  # 只显示前10个
                print(f"    {filename}: {error}")
            if len(stats['errors']) > 10:
                print(f"    ... 还有 {len(stats['errors']) - 10} 个错误")
            print()

        # 一致性检查
        print(f"  ✅ 一致性检查:")

        # 检查 nframe 是否一致
        if len(stats['nframes']) == 1:
            print(f"    ✓ nframe 一致: {list(stats['nframes'].keys())[0]} 帧")
        else:
            print(f"    ⚠️  nframe 不一致! 发现 {len(stats['nframes'])} 种不同的值:")
            for nframe, count in sorted(stats['nframes'].items()):
                print(f"       {nframe} 帧: {count} 个文件")

        # 检查模型签名是否一致
        if len(stats['model_signatures']) == 1:
            print(f"    ✓ 模型签名一致: {list(stats['model_signatures'].keys())[0]}")
        else:
            print(f"    ⚠️  模型签名不一致! 发现 {len(stats['model_signatures'])} 种不同的值")

        # 检查 dtype 是否一致
        if len(stats['dtypes']) == 1:
            print(f"    ✓ 数据类型一致: {list(stats['dtypes'].keys())[0]}")
        else:
            print(f"    ⚠️  数据类型不一致! 发现 {len(stats['dtypes'])} 种不同的值")

        # 检查 shape 是否一致
        if len(stats['shapes']) <= 2:  # 允许少量差异（视频长度可能不同）
            print(f"    ✓ 特征形状基本一致")
        else:
            print(f"    ⚠️  特征形状差异较大! 发现 {len(stats['shapes'])} 种不同的形状")

        print()
        print()

    # 最终总结
    print("=" * 80)
    print("总结")
    print("=" * 80)

    if metadata:
        total_samples = metadata.get('total_samples', 0)
        total_size_bytes = metadata.get('total_size_bytes', 0)
        print(f"✓ 元数据记录的样本数: {total_samples}")
        print(f"✓ 元数据记录的总大小: {format_bytes(total_size_bytes)}")

    print(f"✓ 扫描的数据集数: {len(dataset_dirs)}")

    print()
    print("检查完成!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='检查 Vision Cache 参数和完整性')
    parser.add_argument('--cache-dir', default='/disk3/minami/LMUData/vision_cache',
                       help='缓存目录 (默认: /disk3/minami/LMUData/vision_cache)')
    parser.add_argument('--dataset', default=None,
                       help='只检查特定数据集 (可选)')
    parser.add_argument('--detailed', action='store_true',
                       help='显示详细信息（包括示例和错误）')

    args = parser.parse_args()

    check_cache_directory(args.cache_dir, args.dataset, args.detailed)


if __name__ == '__main__':
    main()
