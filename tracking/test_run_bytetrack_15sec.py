#!/usr/bin/env python3
"""
Test run of ByteTrack on Gate2 15-second video with one model
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
VIDEO_PATH = "/media/mydrive/GitHub/ultralytics/videos/Gate2_Oct7_15sec.mp4"
GATE_NAME = "Gate2_Oct7"
OUTPUT_BASE_DIR = "/media/mydrive/GitHub/ultralytics/tracking/test_inference_bytetrack"

print("="*70)
print("Testing ByteTrack on 15-second video")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Video: {GATE_NAME}_15sec.mp4")
print(f"Output: {OUTPUT_BASE_DIR}")
print("="*70)

# Run the model
result = run_model(MODEL_PATH, MODEL_NAME, VIDEO_PATH, GATE_NAME, OUTPUT_BASE_DIR)

if result:
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Total vehicles counted: {result['total_count']}")
    print(f"Vehicle breakdown: {result['vehicle_counts']}")
    print(f"Total tracks: {result['total_tracks']}")
    print(f"Average FPS: {result['avg_fps']:.2f}")
    print("="*70)
else:
    print("\n✗ Test failed")
