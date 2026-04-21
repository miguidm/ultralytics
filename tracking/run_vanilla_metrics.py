#!/usr/bin/env python3
"""
Run ByteTrack inference on Vanilla YOLOv8m model - METRICS ONLY
All gates except Gate 3 variants
"""

import sys
import os
import warnings
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# CRITICAL: Use local custom ultralytics
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np

# Gate-specific counting line configurations
GATE_COUNTING_LINES = {
    'Gate2_Oct7': {
        'x_positions': [0.625, 0.781],
        'y_positions': [1.0, 0.278]
    },
    'Gate2.9_Oct7': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.42, 0.65]
    },
    'Gate3.1_Oct7': {
        'x_positions': [0.885, 0.885],
        'y_positions': [0.0, 0.194]
    },
    'Gate3.5_Oct7': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.547, 0.578]
    },
    'default': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.45, 0.60]
    }
}


def run_model_metrics_only(model_path, model_name, video_path, gate_name, output_base_dir):
    """Run ByteTrack inference - METRICS ONLY (no video output)"""
    print("\n" + "="*70)
    print(f"Processing: {model_name} | Video: {gate_name} | Tracker: ByteTrack")
    print("="*70)

    # Create output directory
    output_dir = os.path.join(output_base_dir, model_name, gate_name)
    os.makedirs(output_dir, exist_ok=True)

    # Output files
    predictions_file = os.path.join(output_dir, f"{gate_name}_predictions.txt")
    metrics_file = os.path.join(output_dir, f"{gate_name}_metrics.txt")

    # Load model
    print(f"Loading model...")
    try:
        model = YOLO(model_path)
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✓ Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

    # Tracking variables
    track_classes = {}
    vehicle_counts = {"car": 0, "motorcycle": 0, "tricycle": 0, "van": 0, "bus": 0, "truck": 0}
    counted_objects = set()
    frame_count = 0
    total_processing_time = 0
    track_history = {}
    track_lifespans = {}
    previous_tracks = {}
    start_time = time.time()
    total_detections = 0

    # Get gate-specific counting line
    gate_config = GATE_COUNTING_LINES.get(gate_name, GATE_COUNTING_LINES['default'])
    line_x_positions = [
        int(frame_width * gate_config['x_positions'][0]),
        int(frame_width * gate_config['x_positions'][1])
    ]
    line_y_positions = [
        int(frame_height * gate_config['y_positions'][0]),
        int(frame_height * gate_config['y_positions'][1])
    ]
    object_positions = {}

    print(f"Counting line: x={line_x_positions}, y={line_y_positions}")

    # Open predictions file (MOT format)
    pred_file = open(predictions_file, 'w')
    print("Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        frame_count += 1

        # Run tracking
        try:
            results = model.track(
                frame,
                conf=0.5,
                persist=True,
                tracker='bytetrack.yaml',
                verbose=False
            )[0]
        except Exception as e:
            print(f"\r⚠ Frame {frame_count}: Tracker error", end='', flush=True)
            continue

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

                # Save prediction (MOT format)
                w = x2 - x1
                h = y2 - y1
                pred_file.write(f"{frame_count},{obj_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{class_id},-1,-1\n")

        total_detections += len(tracked_objects)

        # Update track history
        for track_id in current_track_ids:
            if track_id not in previous_tracks:
                track_lifespans[track_id] = 1
                track_history[track_id] = [frame_count]
            else:
                track_history[track_id].append(frame_count)
                track_lifespans[track_id] = track_lifespans.get(track_id, 0) + 1

        previous_tracks = current_track_ids.copy()

        # Line crossing detection and counting
        for x1, y1, x2, y2, obj_id, conf, class_name in tracked_objects:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Check line crossing
            line_y_at_x = np.interp(center_x, line_x_positions, line_y_positions)
            current_position = 'above' if center_y < line_y_at_x else 'below'

            if obj_id in object_positions:
                previous_position = object_positions[obj_id]
                if previous_position != current_position and obj_id not in counted_objects:
                    if class_name in vehicle_counts:
                        vehicle_counts[class_name] += 1
                        counted_objects.add(obj_id)

            object_positions[obj_id] = current_position

        # Update timing
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        total_processing_time += frame_processing_time

        # Progress
        if frame_count % 30 == 0:
            progress_pct = (frame_count / total_frames) * 100
            current_fps = 1 / frame_processing_time if frame_processing_time > 0 else 0
            print(f"Progress: {progress_pct:.1f}% | Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}")

    # Cleanup
    cap.release()
    pred_file.close()

    # Calculate metrics
    end_time = time.time()
    total_runtime = end_time - start_time
    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0

    # Identity Switches
    identity_switches = 0
    for track_id, history in track_history.items():
        if len(history) > 1:
            for i in range(1, len(history)):
                if history[i] - history[i-1] > 5:
                    identity_switches += 1

    # MT/ML
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

    # MOTA (simplified)
    false_negatives = max(0, total_detections - len(current_track_ids) * frame_count)
    false_positives = max(0, len(current_track_ids) * frame_count - total_detections)
    mota = 1 - (false_negatives + false_positives + identity_switches) / max(1, total_detections)
    mota = max(0, min(1, mota))

    # IDF1 (simplified)
    id_true_positives = sum(track_lifespans.values())
    id_false_positives = identity_switches
    id_false_negatives = max(0, total_detections - id_true_positives)
    idf1 = (2 * id_true_positives) / max(1, 2 * id_true_positives + id_false_positives + id_false_negatives)

    # Save metrics
    metrics_content = f"""Tracking Metrics Report - {model_name} - {gate_name}
================================================================

Configuration:
- Model: {model_name}
- Video: {gate_name}
- Tracker: ByteTrack
- Confidence: 0.5
- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Tracking Metrics:
-----------------
IDF1:  {idf1:.4f}
MOTA:  {mota:.4f}
IDSW:  {identity_switches}
MT:    {mt_ratio:.4f} ({mostly_tracked}/{total_tracks})
ML:    {ml_ratio:.4f} ({mostly_lost}/{total_tracks})
FPS:   {avg_fps:.2f}

Statistics:
-----------
Total Frames:     {frame_count}
Total Runtime:    {total_runtime:.2f}s
Total Tracks:     {total_tracks}
Total Detections: {total_detections}
Avg Track Life:   {sum(track_lifespans.values())/max(1,len(track_lifespans)):.1f} frames

Vehicle Counts:
---------------
Car:        {vehicle_counts['car']}
Motorcycle: {vehicle_counts['motorcycle']}
Tricycle:   {vehicle_counts['tricycle']}
Van:        {vehicle_counts['van']}
Bus:        {vehicle_counts['bus']}
Truck:      {vehicle_counts['truck']}
TOTAL:      {sum(vehicle_counts.values())}

================================================================
"""

    with open(metrics_file, 'w') as f:
        f.write(metrics_content)

    print(f"\n✓ Complete: IDF1={idf1:.4f}, MOTA={mota:.4f}, IDSW={identity_switches}, FPS={avg_fps:.2f}")
    print(f"✓ Saved: {metrics_file}")

    return {
        'model': model_name,
        'gate': gate_name,
        'idf1': idf1,
        'mota': mota,
        'idsw': identity_switches,
        'mt': mt_ratio,
        'ml': ml_ratio,
        'fps': avg_fps,
        'total_tracks': total_tracks,
        'total_count': sum(vehicle_counts.values())
    }


def main():
    """Main function to run Vanilla model on all gates except Gate 3"""
    print("="*70)
    print("VANILLA YOLOV8M - TRACKING METRICS EVALUATION")
    print("All gates except Gate 3 variants")
    print("="*70)

    # Model to evaluate
    model_name = 'Vanilla-YOLOv8m'
    model_path = '/home/migui/Downloads/yolov8m-vanilla-20260211T133104Z-1-001/yolov8m-vanilla/weights/best.pt'

    # Videos to process (excluding Gate 3)
    video_dir = '/media/mydrive/GitHub/ultralytics/videos'
    videos = {
        'Gate2_Oct7': os.path.join(video_dir, 'Gate2_Oct7.mp4'),
        'Gate2.9_Oct7': os.path.join(video_dir, 'Gate2.9_Oct7.mp4'),
        'Gate3.5_Oct7': os.path.join(video_dir, 'Gate3.5_Oct7.mp4'),
        'Gate3.1_Oct7': os.path.join(video_dir, 'Gate3.1_Oct7.mp4'),
    }

    output_base_dir = 'tracking_metrics_results_vanilla'

    # Check model exists
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return

    # Check which videos exist
    available_videos = {k: v for k, v in videos.items() if os.path.exists(v)}
    print(f"\nFound {len(available_videos)}/{len(videos)} videos")
    print(f"Model: {model_name}\n")

    results_summary = []

    # Run on each video
    for gate_name, video_path in available_videos.items():
        result = run_model_metrics_only(model_path, model_name, video_path, gate_name, output_base_dir)
        if result:
            results_summary.append(result)

    # Summary report
    print("\n" + "="*70)
    print(f"SUMMARY - {model_name} - All Gates (except Gate 3)")
    print("="*70)

    if results_summary:
        results_summary.sort(key=lambda x: x['idf1'], reverse=True)

        print(f"\n{'Gate':<15} {'IDF1':<8} {'MOTA':<8} {'IDSW':<6} {'FPS':<8} {'Count':<6}")
        print("-"*70)
        for r in results_summary:
            print(f"{r['gate']:<15} {r['idf1']:<8.4f} {r['mota']:<8.4f} {r['idsw']:<6} {r['fps']:<8.2f} {r['total_count']:<6}")

        # Save summary
        summary_file = os.path.join(output_base_dir, model_name, 'summary.txt')
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        with open(summary_file, 'w') as f:
            f.write(f"{model_name} - Tracking Metrics Summary\n")
            f.write("="*70 + "\n\n")
            f.write(f"{'Gate':<15} {'IDF1':<8} {'MOTA':<8} {'IDSW':<6} {'FPS':<8} {'Count':<6}\n")
            f.write("-"*70 + "\n")
            for r in results_summary:
                f.write(f"{r['gate']:<15} {r['idf1']:<8.4f} {r['mota']:<8.4f} {r['idsw']:<6} {r['fps']:<8.2f} {r['total_count']:<6}\n")

        print(f"\n✓ Summary saved: {summary_file}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
