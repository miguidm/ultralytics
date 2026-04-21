#!/usr/bin/env python3
"""
Tracking Metrics Evaluation Script (No Video Output)
Evaluates tracking performance without saving video - faster processing
Saves tracking predictions in MOT format for later analysis
"""

import sys
import os
import argparse
import warnings
from pathlib import Path
import time

# ============================================================================
# DCNv2 ENVIRONMENT SETUP
# ============================================================================

def setup_dcnv2_environment():
    """Configure environment for DCNv2 operations"""
    cuda_lib_path = "/home/migui/miniconda3/envs/dcn/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    torch_lib_path = "/home/migui/miniconda3/envs/dcn/lib/python3.10/site-packages/torch/lib"

    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}"

    ultralytics_root = "/media/mydrive/GitHub/ultralytics"
    if ultralytics_root not in sys.path:
        sys.path.insert(0, ultralytics_root)

setup_dcnv2_environment()

warnings.filterwarnings('ignore', category=UserWarning)

from ultralytics import YOLO
import cv2
import numpy as np


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Tracking Metrics Evaluation (No Video Output)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLO model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output-dir', type=str, default='tracking_results',
                        help='Directory to save tracking predictions and metrics')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml',
                        choices=['bytetrack.yaml', 'botsort.yaml'],
                        help='Tracker configuration')
    return parser.parse_args()


def main():
    """Main tracking evaluation function"""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate filenames based on model name
    model_name = Path(args.model).stem
    video_name = Path(args.source).stem
    predictions_file = os.path.join(args.output_dir, f'{model_name}_{video_name}_predictions.txt')
    metrics_file = os.path.join(args.output_dir, f'{model_name}_{video_name}_metrics.txt')

    print("="*70)
    print("TRACKING METRICS EVALUATION (No Video Output)")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"Video: {video_name}")
    print(f"Tracker: {args.tracker}")

    # Load model
    print(f"\nLoading model...")
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return

    try:
        model = YOLO(args.model)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Load video
    print(f"\nLoading video...")
    if not os.path.exists(args.source):
        print(f"❌ Error: Video file not found: {args.source}")
        return

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {args.source}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✓ Video loaded: {frame_width}x{frame_height} @ {fps}fps")
    print(f"  Total frames: {total_frames}")

    # Initialize tracking variables
    print(f"\nInitializing tracking system...")
    track_classes = {}
    frame_count = 0
    total_processing_time = 0
    identity_switches = 0
    track_history = {}
    track_lifespans = {}
    previous_tracks = {}
    start_time = time.time()
    total_detections = 0

    # Open predictions file (MOT format)
    pred_file = open(predictions_file, 'w')
    print(f"✓ Saving predictions to: {predictions_file}")

    # Start processing
    print(f"\nProcessing frames...")
    print("="*70 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        frame_count += 1

        # Run tracking
        results = model.track(
            frame,
            conf=args.conf,
            persist=True,
            tracker=args.tracker,
            verbose=False
        )[0]

        # Extract tracked objects
        tracked_objects = []
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

                tracked_objects.append([x1, y1, x2, y2, obj_id, conf, class_name])
                current_track_ids.add(obj_id)
                track_classes[obj_id] = class_name

                # Save prediction in MOT format: frame,id,x,y,w,h,conf,class,-1,-1
                w = x2 - x1
                h = y2 - y1
                pred_file.write(f"{frame_count},{obj_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{class_id},-1,-1\n")

        # Update total detections count
        total_detections += len(tracked_objects)

        # Track management and ID switch detection
        for track_id in current_track_ids:
            if track_id not in previous_tracks:
                # New track started
                track_lifespans[track_id] = 1
                track_history[track_id] = [frame_count]
            else:
                track_history[track_id].append(frame_count)
                track_lifespans[track_id] = track_lifespans.get(track_id, 0) + 1

        # Update previous tracks for next iteration
        previous_tracks = current_track_ids.copy()

        # Update timing
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        total_processing_time += frame_processing_time
        current_fps = 1 / frame_processing_time if frame_processing_time > 0 else 0

        # Progress updates every 30 frames
        if frame_count % 30 == 0:
            progress_pct = (frame_count / total_frames) * 100
            print(f"Progress: {progress_pct:.1f}% | Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}")

    # Cleanup
    cap.release()
    pred_file.close()

    # Calculate final metrics
    end_time = time.time()
    total_runtime = end_time - start_time

    # Calculate metrics
    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0

    # Identity Switches (IDSW) - count track fragmentations
    identity_switches = 0
    for track_id, history in track_history.items():
        if len(history) > 1:
            for i in range(1, len(history)):
                if history[i] - history[i-1] > 5:  # Gap larger than 5 frames
                    identity_switches += 1

    # MT/ML calculation (Mostly Tracked / Mostly Lost)
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

    # MOTA calculation (simplified - without ground truth)
    false_negatives = max(0, total_detections - len(current_track_ids) * frame_count)
    false_positives = max(0, len(current_track_ids) * frame_count - total_detections)
    mota = 1 - (false_negatives + false_positives + identity_switches) / max(1, total_detections)
    mota = max(0, min(1, mota))

    # IDF1 calculation (simplified - without ground truth)
    id_true_positives = sum(track_lifespans.values())
    id_false_positives = identity_switches
    id_false_negatives = max(0, total_detections - id_true_positives)
    idf1 = (2 * id_true_positives) / max(1, 2 * id_true_positives + id_false_positives + id_false_negatives)

    # Create metrics report
    output_content = f"""Tracking Metrics Evaluation Report
================================================================

Model Configuration:
- Model: {model_name}
- Video: {video_name}
- Tracker: {args.tracker}
- Confidence Threshold: {args.conf}
- Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Tracking Metrics (Simplified - No Ground Truth):
-------------------------------------------------
1. IDF1 (Identity F1 Score): {idf1:.4f}
   - Measures identity preservation accuracy
   - Range: 0.0 (worst) to 1.0 (best)

2. MT/ML (Mostly Tracked/Lost Ratios):
   - MT (Mostly Tracked): {mt_ratio:.4f} ({mostly_tracked}/{total_tracks} tracks)
   - ML (Mostly Lost): {ml_ratio:.4f} ({mostly_lost}/{total_tracks} tracks)

3. IDSW (Identity Switches): {identity_switches}
   - Number of identity switches detected
   - Lower is better (0 = no switches)

4. MOTA (Multiple Object Tracking Accuracy): {mota:.4f}
   - Overall tracking accuracy measure
   - Range: 0.0 (worst) to 1.0 (best)

Performance Metrics:
-------------------
5. FPS (Frames Per Second): {avg_fps:.2f}
   - Average processing speed
   - Higher is better

Additional Statistics:
---------------------
- Total Frames Processed: {frame_count}
- Total Runtime: {total_runtime:.2f} seconds
- Total Tracks Created: {total_tracks}
- Total Detections: {total_detections}
- Average Track Lifespan: {sum(track_lifespans.values())/max(1,len(track_lifespans)):.1f} frames

Predictions saved to: {predictions_file}
Format: MOT (frame,id,x,y,w,h,conf,class,-1,-1)

================================================================
"""

    # Write metrics to file
    with open(metrics_file, 'w') as f:
        f.write(output_content)

    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\n✓ Predictions saved: {predictions_file}")
    print(f"✓ Metrics saved: {metrics_file}")
    print(f"\nTracking Metrics Summary:")
    print(f"  - IDF1: {idf1:.4f}")
    print(f"  - MOTA: {mota:.4f}")
    print(f"  - IDSW: {identity_switches}")
    print(f"  - MT: {mt_ratio:.4f}")
    print(f"  - ML: {ml_ratio:.4f}")
    print(f"  - FPS: {avg_fps:.2f}")
    print(f"  - Total Tracks: {total_tracks}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
