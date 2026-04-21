#!/usr/bin/env python3
"""
Test run of ByteTrack on all gate 15-second videos with one model
"""
import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

# Import the run_model function from the main script
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics/tracking')
from run_all_dcnv2_bytetrack import run_model

# Configuration
MODEL_PATH = "/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-FPN.pt"
MODEL_NAME = "DCNv2-FPN"
VIDEO_DIR = "/media/mydrive/GitHub/ultralytics/videos"
OUTPUT_BASE_DIR = "/media/mydrive/GitHub/ultralytics/tracking/test_inference_bytetrack_all_gates"

# Gate videos to process (15-second versions)
gate_videos = [
    ("Gate2_Oct7_15sec.mp4", "Gate2_Oct7"),
    ("Gate2.9_Oct7_15sec.mp4", "Gate2.9_Oct7"),
    ("Gate3_Oct7_15sec.mp4", "Gate3_Oct7"),
    ("Gate3.5_Oct7_15sec.mp4", "Gate3.5_Oct7"),
    ("Gate3.5_Oct7_15sec.mp4", "Gate3.1_Oct7")  # Same video, different counting line
]

print("="*70)
print("Testing ByteTrack on all gate 15-second videos")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Videos to process: {len(gate_videos)}")
for video_file, gate_name in gate_videos:
    print(f"  - {video_file}")
print(f"Output: {OUTPUT_BASE_DIR}")
print("="*70)

results = []

for i, (video_file, gate_name) in enumerate(gate_videos, 1):
    video_path = os.path.join(VIDEO_DIR, video_file)

    print(f"\n{'='*70}")
    print(f"Processing {i}/{len(gate_videos)}: {gate_name}")
    print(f"{'='*70}")

    if not os.path.exists(video_path):
        print(f"✗ Video not found: {video_path}")
        continue

    result = run_model(MODEL_PATH, MODEL_NAME, video_path, gate_name, OUTPUT_BASE_DIR)

    if result:
        results.append(result)
    else:
        print(f"✗ Failed to process {gate_name}")

# Print summary
print("\n" + "="*70)
print("ALL TESTS COMPLETED")
print("="*70)
print(f"Successfully processed: {len(results)}/{len(gate_videos)} videos\n")

for result in results:
    print(f"{result['gate_name']}:")
    print(f"  Total vehicles: {result['total_count']}")
    print(f"  Breakdown: {result['vehicle_counts']}")
    print(f"  Total tracks: {result['total_tracks']}")
    print(f"  Avg FPS: {result['avg_fps']:.2f}")
    print()

print("="*70)
