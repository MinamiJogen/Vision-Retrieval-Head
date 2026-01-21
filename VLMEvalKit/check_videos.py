#!/usr/bin/env python
"""
Video Dataset Check Script

Checks all video files in datasets and reports missing ones.

Usage:
    python check_videos.py [--datasets DATASET1 DATASET2 ...]
    python check_videos.py --frames 64
    python check_videos.py --frames 128
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")


def check_dataset(dataset_name, verbose=True):
    """Check all video files in a dataset."""
    from vlmeval.dataset import build_dataset
    from vlmeval.dataset.video_concat_dataset import ConcatVideoDataset

    if verbose:
        print(f"\nChecking: {dataset_name}")

    try:
        dataset = build_dataset(dataset_name)
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        return None, None, str(e)

    if dataset is None:
        print(f"  ERROR: Dataset returned None")
        return None, None, "Dataset is None"

    # Handle ConcatVideoDataset (like TempCompass)
    if isinstance(dataset, ConcatVideoDataset):
        all_existing = set()
        all_missing = set()
        for sub_name, sub_ds in dataset.dataset_map.items():
            existing, missing = check_sub_dataset(sub_ds, sub_name, verbose)
            all_existing.update(existing)
            all_missing.update(missing)
        if verbose:
            print(f"  Total existing: {len(all_existing)}")
            print(f"  Total missing: {len(all_missing)}")
        return all_existing, all_missing, None
    else:
        existing, missing = check_sub_dataset(dataset, dataset_name, verbose)
        return existing, missing, None


def check_sub_dataset(dataset, name, verbose=True):
    """Check a single (non-concat) dataset."""
    data_root = getattr(dataset, 'data_root', None)
    video_path_attr = getattr(dataset, 'video_path', None)
    data = getattr(dataset, 'data', None)

    if data is None:
        if verbose:
            print(f"  [{name}] ERROR: No data attribute")
        return set(), set()

    if verbose:
        print(f"  [{name}] root: {data_root}, samples: {len(data)}")

    missing_videos = set()
    existing_videos = set()

    for idx, row in data.iterrows():
        full_path = None

        # Try different ways to construct the path
        video = row.get('video', '')
        video_path = row.get('video_path', '')
        prefix = row.get('prefix', '')
        suffix = row.get('suffix', '')

        # Method 1: video_path_attr + video + suffix
        if video_path_attr and video:
            test_path = os.path.join(video_path_attr, f"{video}{suffix}")
            if os.path.exists(test_path):
                full_path = test_path

        # Method 2: data_root + prefix + video + suffix
        if full_path is None and data_root and video:
            if prefix.startswith('./'):
                prefix = prefix[2:]
            test_path = os.path.join(data_root, prefix, f"{video}{suffix}")
            if os.path.exists(test_path):
                full_path = test_path

        # Method 3: data_root + video_path (relative)
        if full_path is None and data_root and video_path:
            if video_path.startswith('./'):
                video_path_clean = video_path[2:]
            else:
                video_path_clean = video_path
            # Check if data_root already contains part of the path
            if 'video' in data_root and video_path_clean.startswith('video/'):
                # data_root ends with /video, video_path starts with video/
                parent = os.path.dirname(data_root)
                test_path = os.path.join(parent, video_path_clean)
            else:
                test_path = os.path.join(data_root, video_path_clean)
            if os.path.exists(test_path):
                full_path = test_path

        # Method 4: Try common extensions
        if full_path is None and data_root and video:
            for ext in ['.mp4', '.avi', '.mkv', '.webm', '']:
                if prefix.startswith('./'):
                    prefix_clean = prefix[2:]
                else:
                    prefix_clean = prefix
                test_path = os.path.join(data_root, prefix_clean, f"{video}{ext}")
                if os.path.exists(test_path):
                    full_path = test_path
                    break

        if full_path and os.path.exists(full_path):
            existing_videos.add(full_path)
        else:
            # Record the expected path for missing videos
            if data_root and video:
                if prefix.startswith('./'):
                    prefix_clean = prefix[2:]
                else:
                    prefix_clean = prefix
                expected = os.path.join(data_root, prefix_clean, f"{video}{suffix}")
            elif video_path:
                expected = video_path
            else:
                expected = f"unknown:{video}"
            missing_videos.add(expected)

    if verbose:
        print(f"  [{name}] existing: {len(existing_videos)}, missing: {len(missing_videos)}")

    return existing_videos, missing_videos


def main():
    parser = argparse.ArgumentParser(description='Check video datasets')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to check')
    parser.add_argument('--frames', type=int, default=None,
                       help='Check datasets with this frame count (64 or 128)')
    parser.add_argument('--save-missing', type=str, default=None,
                       help='Save missing video list to file')
    parser.add_argument('--quiet', action='store_true',
                       help='Less verbose output')
    args = parser.parse_args()

    # Determine which datasets to check
    if args.datasets:
        datasets = args.datasets
    elif args.frames == 64:
        datasets = [
            "Video-MME_64frame",
            "VideoMMMU_64frame",
            "LongVideoBench_64frame",
            "MLVU_64frame",
            "Video_Holmes_64frame",
            "TempCompass_64frame",
            "MMBench_Video_64frame_nopack",
        ]
    elif args.frames == 128:
        datasets = [
            "Video-MME_128frame",
            "VideoMMMU_128frame",
            "LongVideoBench_128frame",
            "MLVU_128frame",
            "Video_Holmes_128frame",
            "TempCompass_128frame",
            "MMBench_Video_128frame_nopack",
        ]
    else:
        # Default to 128 frame
        datasets = [
            "Video-MME_128frame",
            "VideoMMMU_128frame",
            "LongVideoBench_128frame",
            "MLVU_128frame",
            "Video_Holmes_128frame",
            "TempCompass_128frame",
            "MMBench_Video_128frame_nopack",
        ]

    print("=" * 70)
    print("Video Dataset Check")
    print("=" * 70)
    print(f"Checking {len(datasets)} dataset(s)\n")

    results = {}
    all_missing = set()
    all_existing = set()

    for dataset_name in datasets:
        existing, missing, error = check_dataset(dataset_name, verbose=not args.quiet)
        if error:
            results[dataset_name] = {'error': error}
        elif existing is None:
            results[dataset_name] = {'error': 'Unknown error'}
        else:
            results[dataset_name] = {
                'existing': len(existing),
                'missing': len(missing),
                'missing_files': missing
            }
            all_missing.update(missing)
            all_existing.update(existing)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, res in results.items():
        if 'error' in res:
            status = f"ERROR: {res['error']}"
        elif res['missing'] == 0:
            status = f"OK ({res['existing']} videos)"
        else:
            status = f"MISSING {res['missing']} / {res['existing'] + res['missing']} videos"
        print(f"  {name}: {status}")

    print(f"\nTotal unique videos:")
    print(f"  Existing: {len(all_existing)}")
    print(f"  Missing:  {len(all_missing)}")

    if all_missing:
        print(f"\nMissing videos ({len(all_missing)}):")
        for f in sorted(all_missing)[:10]:
            print(f"  {f}")
        if len(all_missing) > 10:
            print(f"  ... and {len(all_missing) - 10} more")

        if args.save_missing:
            with open(args.save_missing, 'w') as f:
                for path in sorted(all_missing):
                    f.write(path + '\n')
            print(f"\nMissing list saved to: {args.save_missing}")

        return 1
    else:
        print("\nAll video files verified successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
