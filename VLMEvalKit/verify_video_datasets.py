#!/usr/bin/env python
"""
Video Dataset Verification Script

This script checks all video files in the specified datasets
to ensure they exist before running evaluation.

Usage:
    python verify_video_datasets.py [--datasets DATASET1 DATASET2 ...]

If no datasets specified, checks all 128-frame datasets by default.
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

def verify_dataset(dataset_name):
    """Verify all video files exist for a dataset."""
    from vlmeval.dataset import build_dataset

    print(f"\n{'='*60}")
    print(f"Checking: {dataset_name}")
    print('='*60)

    try:
        dataset = build_dataset(dataset_name)
    except Exception as e:
        print(f"ERROR: Failed to load dataset {dataset_name}: {e}")
        return [], [str(e)]

    if dataset is None:
        print(f"ERROR: Dataset {dataset_name} returned None")
        return [], [f"Dataset {dataset_name} is None"]

    # Get dataset root and data
    data_root = getattr(dataset, 'data_root', None)
    data = getattr(dataset, 'data', None)

    if data is None:
        print(f"ERROR: Dataset has no data attribute")
        return [], ["No data attribute"]

    print(f"Dataset root: {data_root}")
    print(f"Total samples: {len(data)}")

    missing_files = []
    existing_files = []

    # Check video files
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Checking videos"):
        video_path = None

        # Try different column names for video path
        if 'video_path' in row:
            video_path = row['video_path']
        elif 'video' in row:
            video_path = row['video']

        if video_path is None:
            continue

        # Build full path
        if data_root and not os.path.isabs(video_path):
            # Handle relative paths
            if video_path.startswith('./'):
                video_path = video_path[2:]
            full_path = os.path.join(data_root, video_path)
        else:
            full_path = video_path

        # Check for suffix if needed
        if not os.path.exists(full_path):
            # Try adding common video suffixes
            for suffix in ['.mp4', '.avi', '.mkv', '.webm', '']:
                if 'suffix' in row:
                    test_path = full_path + row['suffix']
                else:
                    test_path = full_path + suffix
                if os.path.exists(test_path):
                    full_path = test_path
                    break

        if os.path.exists(full_path):
            existing_files.append(full_path)
        else:
            missing_files.append(full_path)

    # Report results
    print(f"\nResults for {dataset_name}:")
    print(f"  Existing: {len(existing_files)}")
    print(f"  Missing:  {len(missing_files)}")

    if missing_files:
        print(f"\nMissing files (first 10):")
        for f in missing_files[:10]:
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    return existing_files, missing_files


def main():
    parser = argparse.ArgumentParser(description='Verify video datasets')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Dataset names to check (default: all 128-frame datasets)')
    parser.add_argument('--frames', type=int, default=128,
                       help='Frame count to check (default: 128)')
    args = parser.parse_args()

    # Default datasets based on frame count
    if args.datasets is None:
        if args.frames == 128:
            datasets = [
                "Video-MME_128frame",
                "VideoMMMU_128frame",
                "LongVideoBench_128frame",
                "MLVU_128frame",
                "Video_Holmes_128frame",
                "TempCompass_128frame",
                "MMBench_Video_128frame_nopack",
            ]
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
        else:
            print(f"No default datasets for {args.frames} frames")
            return 1
    else:
        datasets = args.datasets

    print("="*60)
    print("Video Dataset Verification")
    print("="*60)
    print(f"Checking {len(datasets)} datasets:")
    for d in datasets:
        print(f"  - {d}")

    all_missing = {}
    all_existing = {}

    for dataset_name in datasets:
        try:
            existing, missing = verify_dataset(dataset_name)
            all_existing[dataset_name] = existing
            all_missing[dataset_name] = missing
        except Exception as e:
            print(f"\nERROR processing {dataset_name}: {e}")
            all_missing[dataset_name] = [str(e)]

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_missing = 0
    total_existing = 0
    datasets_with_issues = []

    for dataset_name in datasets:
        existing_count = len(all_existing.get(dataset_name, []))
        missing_count = len(all_missing.get(dataset_name, []))
        total_existing += existing_count
        total_missing += missing_count

        status = "OK" if missing_count == 0 else f"MISSING {missing_count}"
        print(f"  {dataset_name}: {status}")

        if missing_count > 0:
            datasets_with_issues.append(dataset_name)

    print(f"\nTotal: {total_existing} existing, {total_missing} missing")

    if datasets_with_issues:
        print(f"\nDatasets with issues:")
        for d in datasets_with_issues:
            print(f"  - {d}")
        print("\nYou may need to re-download these datasets or check the paths.")
        return 1
    else:
        print("\nAll video files verified successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
