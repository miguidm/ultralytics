#!/usr/bin/env python3
"""
Run YOLOv8m Vanilla Tracking on a single Gate3 video
"""

import os
import sys
import cv2
import time
from pathlib import Path
from collections import defaultdict

# Setup environment
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO

# Configuration
MODEL_PATH = "/home/migui/Downloads/yolov8m-vanilla-20260211T133104Z-1-001/yolov8m-vanilla/weights/best.pt"
VIDEO_PATH = "/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20250220_075715.avi"
OUTPUT_DIR = "tracking/yolov8m_vanilla_gate3_tracking_results"
CONF_THRESHOLD = 0.5

video_name = Path(VIDEO_PATH).stem

print(f"\n{'='*90}")
print(f"YOLOv8m Vanilla Tracking - Single Video")
print(f"Video: {video_name}")
print(f"{'='*90}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

predictions_file = os.path.join(OUTPUT_DIR, f"{video_name}_predictions.txt")
metrics_file = os.path.join(OUTPUT_DIR, f"{video_name}_metrics.txt")

# Load model
print(f"\nLoading model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    print(f"✓ Model loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Load video
print(f"\nLoading video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Cannot open video")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✓ Video loaded: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

# Initialize tracking variables
frame_count = 0
total_detections = 0
track_history = {}
track_lifespans = {}
previous_tracks = set()
track_classes = {}
start_time = time.time()

# Process video
print(f"\nProcessing frames...")
with open(predictions_file, 'w') as pred_file:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run tracking
        results = model.track(
            frame,
            conf=CONF_THRESHOLD,
            persist=True,
            tracker='bytetrack.yaml',
            verbose=False
        )[0]

        # Extract tracked objects
        current_track_ids = set()

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                obj_id = int(track_ids[i])
                conf = confidences[i]
                class_id = class_ids[i]
                class_name = model.names[class_id]

                w = x2 - x1
                h = y2 - y1

                # Save prediction in MOT format
                pred_file.write(f"{frame_count},{obj_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{class_id},-1,-1\n")

                current_track_ids.add(obj_id)
                track_classes[obj_id] = class_name
                total_detections += 1

        # Track management
        for track_id in current_track_ids:
            if track_id not in previous_tracks:
                track_lifespans[track_id] = 1
                track_history[track_id] = [frame_count]
            else:
                track_history[track_id].append(frame_count)
                track_lifespans[track_id] = track_lifespans.get(track_id, 0) + 1

        previous_tracks = current_track_ids.copy()

        # Progress
        if frame_count % 1000 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) | "
                  f"FPS: {current_fps:.1f} | Detections: {total_detections}")

cap.release()

# Calculate metrics
elapsed_time = time.time() - start_time
avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

# Identity Switches
identity_switches = 0
for track_id, history in track_history.items():
    if len(history) > 1:
        for i in range(1, len(history)):
            if history[i] - history[i-1] > 5:
                identity_switches += 1

# MT/ML calculation
total_tracks = len(track_lifespans)
mostly_tracked = 0
mostly_lost = 0

if total_tracks > 0:
    for track_id, lifespan in track_lifespans.items():
        track_ratio = lifespan / frame_count
        if track_ratio >= 0.8:
            mostly_tracked += 1
        elif track_ratio <= 0.2:
            mostly_lost += 1

mt_ratio = mostly_tracked / total_tracks if total_tracks > 0 else 0
ml_ratio = mostly_lost / total_tracks if total_tracks > 0 else 0

# MOTA calculation
false_negatives = max(0, total_detections - len(previous_tracks) * frame_count)
false_positives = max(0, len(previous_tracks) * frame_count - total_detections)
mota = 1 - (false_negatives + false_positives + identity_switches) / max(1, total_detections)
mota = max(0, min(1, mota))

# IDF1 calculation
id_true_positives = sum(track_lifespans.values())
id_false_positives = identity_switches
id_false_negatives = max(0, total_detections - id_true_positives)
idf1 = (2 * id_true_positives) / max(1, 2 * id_true_positives + id_false_positives + id_false_negatives)

# Average track lifespan
avg_track_lifespan = sum(track_lifespans.values()) / max(1, len(track_lifespans))

# Detections per frame
avg_det_per_frame = total_detections / frame_count if frame_count > 0 else 0

# Create metrics report
metrics_content = f"""YOLOv8m Vanilla Tracking Metrics Report
================================================================

Model Configuration:
- Model: YOLOv8m-Vanilla
- Video: {video_name}
- Tracker: ByteTrack
- Confidence Threshold: {CONF_THRESHOLD}
- Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Video Properties:
- Resolution: {frame_width}x{frame_height}
- FPS: {fps}
- Total Frames: {frame_count}

Tracking Metrics (Simplified - No Ground Truth):
-------------------------------------------------
1. IDF1 (Identity F1 Score): {idf1:.4f}
   - Measures identity preservation accuracy
   - Range: 0.0 (worst) to 1.0 (best)

2. MOTA (Multiple Object Tracking Accuracy): {mota:.4f}
   - Overall tracking accuracy measure
   - Range: 0.0 (worst) to 1.0 (best)

3. IDSW (Identity Switches): {identity_switches}
   - Number of identity switches detected
   - Lower is better (0 = no switches)

4. MT/ML (Mostly Tracked/Lost Ratios):
   - MT (Mostly Tracked): {mt_ratio:.4f} ({mostly_tracked}/{total_tracks} tracks)
   - ML (Mostly Lost): {ml_ratio:.4f} ({mostly_lost}/{total_tracks} tracks)

Performance Metrics:
-------------------
5. FPS (Frames Per Second): {avg_fps:.2f}
   - Average processing speed
   - Higher is better

Additional Statistics:
---------------------
- Total Detections: {total_detections}
- Total Tracks Created: {total_tracks}
- Average Detections/Frame: {avg_det_per_frame:.2f}
- Average Track Lifespan: {avg_track_lifespan:.1f} frames
- Total Processing Time: {elapsed_time:.2f} seconds

Output Files:
-------------
- Predictions: {predictions_file}
- Format: MOT (frame,id,x,y,w,h,conf,class,-1,-1)

================================================================
"""

# Write metrics to file
with open(metrics_file, 'w') as f:
    f.write(metrics_content)

print(f"\n{'='*90}")
print("COMPLETE")
print(f"{'='*90}")
print(f"Total Detections: {total_detections}")
print(f"Unique Tracks: {total_tracks}")
print(f"Avg Detections/Frame: {avg_det_per_frame:.2f}")
print(f"IDSW: {identity_switches}")
print(f"IDF1: {idf1:.4f}")
print(f"MOTA: {mota:.4f}")
print(f"Processing FPS: {avg_fps:.2f}")
print(f"Time: {elapsed_time:.1f}s")
print(f"\n✓ Predictions: {predictions_file}")
print(f"✓ Metrics: {metrics_file}")
print(f"{'='*90}\n")
