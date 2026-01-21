#!/usr/bin/env python
"""Copy missing Video-MME videos from downloaded source."""

import os
import shutil
from pathlib import Path

# Paths
SOURCE_DIR = "/disk3/minami/Vision-Retrieval-Head/videos/data"
TARGET_DIR = "/disk3/minami/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/video"
MISSING_FILE = "/disk3/minami/Vision-Retrieval-Head/VLMEvalKit/missing_videos.txt"

# Read missing videos
with open(MISSING_FILE, 'r') as f:
    missing = [line.strip() for line in f if 'Video-MME' in line]

print(f"Missing Video-MME videos: {len(missing)}")

# Get source videos
source_videos = {f: os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.endswith('.mp4')}
print(f"Source videos available: {len(source_videos)}")

# Copy missing videos
copied = 0
not_found = 0

for missing_path in missing:
    video_name = os.path.basename(missing_path)
    
    if video_name in source_videos:
        src = source_videos[video_name]
        dst = missing_path
        
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1
            if copied <= 10:
                print(f"  Copied: {video_name}")
        else:
            print(f"  Already exists: {video_name}")
    else:
        not_found += 1

print(f"\nSummary:")
print(f"  Copied: {copied}")
print(f"  Not found in source: {not_found}")
print(f"  Total missing: {len(missing)}")
