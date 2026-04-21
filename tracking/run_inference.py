#!/usr/bin/env python3
"""
Single Model Inference Script for YOLOv8 with ByteTrack Tracking
- Runs inference on a single model and video source.
- Calculates and saves detailed tracking metrics.
- Outputs a video with tracking visualizations.

Usage:
    python run_inference.py --model <model.pt> --source <video_path> --output-dir <output_directory>
"""

import sys
import os
import argparse
import warnings
from pathlib import Path
import time
import random
import cv2
import numpy as np

# Patch MMCV before any imports
print("Initializing MMCV patches...")
try:
    import importlib.util

    class DummyModule:
        def __getattr__(self, name):
            return None

    sys.modules['mmcv.ops.bezier_align'] = DummyModule()
    sys.modules['mmcv.ops.bias_act'] = DummyModule()
    sys.modules['mmcv.ops.tin_shift'] = DummyModule()
    sys.modules['mmcv.ops.three_interpolate'] = DummyModule()
    sys.modules['mmcv.ops.three_nn'] = DummyModule()

    try:
        import mmcv.utils.ext_loader as ext_loader_module
        original_load_ext = ext_loader_module.load_ext

        def patched_load_ext(name, funcs):
            try:
                return original_load_ext(name, funcs)
            except (AssertionError, AttributeError):
                return DummyModule()

        ext_loader_module.load_ext = patched_load_ext
    except ImportError:
        pass

    print("✓ MMCV patches applied")
except Exception as e:
    print(f"⚠ Warning: Could not patch MMCV: {e}")

warnings.filterwarnings('ignore', category=UserWarning)

# Add local ultralytics to path if not in the parent directory
# This assumes the script is in a subdirectory of the ultralytics project
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

def get_unique_color(obj_id):
    """Generate a unique color for each tracking ID"""
    random.seed(obj_id)
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Single Model Inference with ByteTrack')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--source', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output video and metrics')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', choices=['bytetrack.yaml', 'botsort.yaml'], help='Tracker configuration')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Generate output filenames
    model_name = Path(args.model).stem
    video_name = Path(args.source).stem
    output_video_path = Path(args.output_dir) / f"{video_name}_{model_name}_output.mp4"
    output_metrics_path = Path(args.output_dir) / f"{video_name}_{model_name}_metrics.txt"

    print("="*70)
    print("SINGLE MODEL INFERENCE WITH BYTETRACK")
    print("="*70)

    # Load model
    print(f"Loading model: {args.model}")
    try:
        model = YOLO(args.model)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Load video
    print(f"Loading video: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"✗ Error: Cannot open video: {args.source}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✓ Video loaded: {frame_width}x{frame_height} @ {fps}fps, Total frames: {total_frames}")

    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    # Tracking data
    track_classes = {}
    trajectories = {}
    trajectory_colors = {}
    
    # Metrics tracking variables
    frame_count = 0
    total_processing_time = 0
    track_history = {}
    track_lifespans = {}
    previous_tracks = {}
    total_detections = 0
    
    start_time = time.time()

    print("\nStarting tracking...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        frame_count += 1

        results = model.track(frame, conf=args.conf, persist=True, tracker=args.tracker, verbose=False)[0]

        current_track_ids = set()
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            current_track_ids = set(track_ids)
            total_detections += len(track_ids)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                obj_id = track_ids[i]
                class_id = class_ids[i]
                class_name = model.names[class_id]
                track_classes[obj_id] = class_name

                # Update trajectory
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                if obj_id not in trajectories:
                    trajectories[obj_id] = []
                trajectories[obj_id].append((center_x, center_y))
                if len(trajectories[obj_id]) > 50:
                    trajectories[obj_id].pop(0)
                if obj_id not in trajectory_colors:
                    trajectory_colors[obj_id] = get_unique_color(obj_id)
                
                # Draw bounding box
                color = trajectory_colors[obj_id]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{class_name} ID:{obj_id}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw trajectories
        for obj_id, trajectory in trajectories.items():
            if obj_id in current_track_ids and len(trajectory) > 1:
                color = trajectory_colors.get(obj_id, (255, 255, 255))
                for i in range(1, len(trajectory)):
                    alpha = i / len(trajectory)
                    thickness = max(2, int(8 * alpha))
                    cv2.line(frame, trajectory[i-1], trajectory[i], color, thickness)

        # Update track history and lifespans
        for track_id in current_track_ids:
            if track_id not in previous_tracks:
                track_lifespans[track_id] = 1
                track_history[track_id] = [frame_count]
            else:
                track_lifespans[track_id] = track_lifespans.get(track_id, 0) + 1
                track_history[track_id].append(frame_count)

        previous_tracks = current_track_ids.copy()
        
        out.write(frame)
        
        frame_end_time = time.time()
        total_processing_time += (frame_end_time - frame_start_time)
        
        if frame_count % 30 == 0:
            progress_pct = (frame_count / total_frames) * 100
            print(f"Progress: {progress_pct:.1f}% | Frame: {frame_count}/{total_frames}")

    cap.release()
    out.release()
    
    end_time = time.time()
    total_runtime = end_time - start_time

    # Calculate final metrics
    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
    
    identity_switches = 0
    for track_id, history in track_history.items():
        if len(history) > 1:
            for i in range(1, len(history)):
                if history[i] - history[i-1] > 1:  # A gap in tracking indicates a potential switch
                    identity_switches += 1

    total_tracks = len(track_lifespans)
    mostly_tracked = 0
    mostly_lost = 0
    if total_tracks > 0:
        for track_id, lifespan in track_lifespans.items():
            # Estimate potential lifespan from first appearance to end of video
            potential_lifespan = total_frames - track_history[track_id][0] + 1
            ratio = lifespan / potential_lifespan if potential_lifespan > 0 else 0
            if ratio >= 0.8:
                mostly_tracked += 1
            elif ratio <= 0.2:
                mostly_lost += 1

    mt_ratio = mostly_tracked / total_tracks if total_tracks > 0 else 0
    ml_ratio = mostly_lost / total_tracks if total_tracks > 0 else 0

    # Simplified MOTA and IDF1
    # These are estimations. For official scores, use tools like py-motmetrics with ground truth data.
    false_negatives_est = 0 # Can't be calculated without ground truth
    false_positives_est = 0 # Can't be calculated without ground truth
    
    mota = 1 - (false_negatives_est + false_positives_est + identity_switches) / max(1, total_detections)
    mota = max(0, min(1, mota))

    id_true_positives = sum(track_lifespans.values()) - total_tracks # Each track starts with one TP
    id_false_positives = identity_switches
    id_false_negatives = 0 # Can't be known without GT
    
    idf1 = (2 * id_true_positives) / max(1, (2 * id_true_positives + id_false_positives + id_false_negatives))
    idf1 = max(0, min(1, idf1))

    # Write metrics to file
    with open(output_metrics_path, 'w') as f:
        f.write(f"Metrics Report for {model_name} on {video_name}\n")
        f.write("="*50 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Video: {args.source}\n")
        f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Detection Metrics:\n")
        f.write(f"  - Total Detections: {total_detections}\n\n")
        f.write(f"Tracking Metrics (Estimations):\n")
        f.write(f"  - IDF1 (Identity F1 Score): {idf1:.4f}\n")
        f.write(f"  - MOTA (Multiple Object Tracking Accuracy): {mota:.4f}\n")
        f.write(f"  - IDSW (Identity Switches): {identity_switches}\n")
        f.write(f"  - MT (Mostly Tracked): {mt_ratio:.4f} ({mostly_tracked}/{total_tracks} tracks)\n")
        f.write(f"  - ML (Mostly Lost): {ml_ratio:.4f} ({mostly_lost}/{total_tracks} tracks)\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"  - Average FPS: {avg_fps:.2f}\n")
        f.write(f"  - Total Runtime: {total_runtime:.2f} seconds\n")
        f.write(f"  - Total Frames: {frame_count}\n")

    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)
    print(f"✓ Output video saved to: {output_video_path}")
    print(f"✓ Metrics report saved to: {output_metrics_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
