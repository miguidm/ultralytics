#!/usr/bin/env python3
"""
Test run of ByteTrack on Gate3.1 only
"""
import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

# Import the run_model function from the main script
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics/tracking')
from run_all_dcnv2_bytetrack import run_model, GATE_COUNTING_LINES

# Configuration
MODEL_PATH = "/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-FPN.pt"
MODEL_NAME = "DCNv2-FPN"
VIDEO_PATH = "/media/mydrive/GitHub/ultralytics/videos/Gate3.5_Oct7_15sec.mp4"
GATE_NAME = "Gate3.1_Oct7"
OUTPUT_BASE_DIR = "/media/mydrive/GitHub/ultralytics/tracking/test_gate3_1_only"

# Get configuration
gate_config = GATE_COUNTING_LINES.get(GATE_NAME, GATE_COUNTING_LINES['default'])

print("="*70)
print("Testing ByteTrack on Gate3.1 15-second video")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Video: Gate3.5_Oct7_15sec.mp4 (using Gate3.1 counting line)")
print(f"\nCounting Line Configuration:")
print(f"  X positions (fractions): {gate_config['x_positions']}")
print(f"  Y positions (fractions): {gate_config['y_positions']}")
print(f"  Actual coordinates (for 1920x1080):")
print(f"    Start: ({int(1920 * gate_config['x_positions'][0])}, {int(1080 * gate_config['y_positions'][0])})")
print(f"    End:   ({int(1920 * gate_config['x_positions'][1])}, {int(1080 * gate_config['y_positions'][1])})")
print(f"\nOutput: {OUTPUT_BASE_DIR}")
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
