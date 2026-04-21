#!/usr/bin/env python3
"""
Vehicle Tracking with Classification Smoothing

Demonstrates the classification smoother to reduce car↔truck flickering.
Compares raw vs smoothed classifications.
"""

import sys
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Add ultralytics to path
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np
import random
import time
from pathlib import Path
from collections import defaultdict

from classification_smoother import create_smoother, ClassificationSmoother


def get_unique_color(obj_id):
    """Generate a unique color for each tracking ID"""
    random.seed(obj_id)
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))


def run_tracking_with_smoothing(
    video_path: str,
    model_path: str,
    output_path: str,
    smoother_strategy: str = 'hysteresis',
    conf_threshold: float = 0.5,
    save_predictions: bool = True
):
    """
    Run tracking with classification smoothing.

    Args:
        video_path: Path to input video
        model_path: Path to YOLO model
        output_path: Path for output video
        smoother_strategy: Smoothing strategy to use
        conf_threshold: Detection confidence threshold
        save_predictions: Whether to save predictions to file
    """

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print("Model loaded successfully")

    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

    # Setup output video
    panel_width = 300
    total_width = frame_width + panel_width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, frame_height))

    # Initialize smoother
    smoother = create_smoother(smoother_strategy)
    print(f"Using smoother strategy: {smoother_strategy}")

    # Tracking state
    track_classes_raw = {}      # Raw classifications
    track_classes_smooth = {}   # Smoothed classifications
    vehicle_counts = {"car": 0, "motorcycle": 0, "tricycle": 0, "van": 0, "bus": 0, "truck": 0}
    counted_ids = set()

    # Trajectory tracking
    trajectories = {}
    max_trail_length = 50
    trajectory_colors = {}

    # Line crossing config
    line_y = int(frame_height * 0.7)

    # Track positions for counting
    object_positions = {}
    counted_objects = set()

    # For saving predictions
    predictions_raw = []
    predictions_smooth = []

    # Per-track switch tracking
    track_switch_frames = defaultdict(list)  # track_id -> list of (frame, from_class, to_class)

    frame_count = 0
    start_time = time.time()

    print("\nStarting tracking with smoothing...")
    print("=" * 70)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run tracking
        results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)[0]

        # Process detections
        current_track_ids = set()

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                obj_id = track_ids[i]
                conf = confidences[i]
                class_id = class_ids[i]
                raw_class = model.names[class_id]

                current_track_ids.add(obj_id)

                # Get smoothed class
                smooth_class = smoother.get_stable_class(
                    track_id=obj_id,
                    raw_class=raw_class,
                    confidence=conf,
                    frame_num=frame_count
                )

                # Track switches
                prev_smooth = track_classes_smooth.get(obj_id)
                if prev_smooth and prev_smooth != smooth_class:
                    track_switch_frames[obj_id].append((frame_count, prev_smooth, smooth_class))

                track_classes_raw[obj_id] = raw_class
                track_classes_smooth[obj_id] = smooth_class

                # Save predictions
                if save_predictions:
                    # MOT format: frame,id,x,y,w,h,conf,class,-1,-1
                    w, h = x2 - x1, y2 - y1
                    class_id_for_raw = list(model.names.keys())[list(model.names.values()).index(raw_class)]
                    class_id_for_smooth = list(model.names.keys())[list(model.names.values()).index(smooth_class)]

                    predictions_raw.append(f"{frame_count},{obj_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{class_id_for_raw},-1,-1")
                    predictions_smooth.append(f"{frame_count},{obj_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{class_id_for_smooth},-1,-1")

                # Trajectory tracking
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center_point = (center_x, center_y)

                if obj_id not in trajectories:
                    trajectories[obj_id] = []
                trajectories[obj_id].append(center_point)
                if len(trajectories[obj_id]) > max_trail_length:
                    trajectories[obj_id].pop(0)

                if obj_id not in trajectory_colors:
                    trajectory_colors[obj_id] = get_unique_color(obj_id)

                # Line crossing counting (using smoothed class)
                current_position = 'above' if center_y < line_y else 'below'

                if obj_id in object_positions:
                    previous_position = object_positions[obj_id]
                    if previous_position != current_position and obj_id not in counted_objects:
                        if smooth_class in vehicle_counts:
                            vehicle_counts[smooth_class] += 1
                            counted_objects.add(obj_id)
                            print(f"  Frame {frame_count}: {smooth_class} ID:{obj_id} crossed line")

                object_positions[obj_id] = current_position

                # Draw bounding box
                color = trajectory_colors[obj_id]

                # Show both raw and smooth class if different
                if raw_class != smooth_class:
                    label = f"ID:{obj_id} {smooth_class} (raw:{raw_class})"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)  # Orange for smoothed
                else:
                    label = f"ID:{obj_id} {smooth_class}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                cv2.putText(frame, label, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Cleanup old tracks from smoother
        smoother.cleanup_old_tracks(current_track_ids)

        # Draw trajectories
        for obj_id, trajectory in trajectories.items():
            if len(trajectory) > 1:
                color = trajectory_colors.get(obj_id, (255, 255, 255))
                for j in range(1, len(trajectory)):
                    alpha = j / len(trajectory)
                    thickness = max(1, int(4 * alpha))
                    cv2.line(frame, trajectory[j-1], trajectory[j], color, thickness)

        # Draw counting line
        cv2.line(frame, (int(frame_width * 0.1), line_y),
                (int(frame_width * 0.9), line_y), (0, 255, 255), 3)

        # Create info panel
        panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray

        y_pos = 30
        cv2.putText(panel, "TRACKING + SMOOTHING", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 35

        cv2.putText(panel, f"Strategy: {smoother_strategy}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 25

        cv2.putText(panel, f"Frame: {frame_count}/{total_frames}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 35

        # Smoother stats
        stats = smoother.get_statistics()
        cv2.putText(panel, "SMOOTHER STATS", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos += 22

        cv2.putText(panel, f"Raw switches: {stats['raw_switches']}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)
        y_pos += 20

        cv2.putText(panel, f"After smooth: {stats['smoothed_switches']}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 150), 1)
        y_pos += 20

        cv2.putText(panel, f"Prevented: {stats['switches_prevented']}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 150), 1)
        y_pos += 20

        cv2.putText(panel, f"Reduction: {stats['reduction_pct']:.1f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
        y_pos += 35

        # Vehicle counts
        cv2.putText(panel, "VEHICLE COUNTS", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos += 22

        for vehicle_type, count in vehicle_counts.items():
            if count > 0:
                cv2.putText(panel, f"{vehicle_type}: {count}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y_pos += 20

        # Combine frame and panel
        combined = np.hstack([frame, panel])
        out.write(combined)

        # Progress update
        if frame_count % 500 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            print(f"  Progress: {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%) - {fps_actual:.1f} fps")

    cap.release()
    out.release()

    # Final statistics
    elapsed = time.time() - start_time
    stats = smoother.get_statistics()

    print("\n" + "=" * 70)
    print("TRACKING COMPLETE")
    print("=" * 70)
    print(f"Total frames: {frame_count}")
    print(f"Processing time: {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    print()
    print("CLASSIFICATION SMOOTHING RESULTS:")
    print(f"  Raw switches:      {stats['raw_switches']}")
    print(f"  Smoothed switches: {stats['smoothed_switches']}")
    print(f"  Switches prevented: {stats['switches_prevented']}")
    print(f"  Reduction: {stats['reduction_pct']:.1f}%")
    print()
    print("VEHICLE COUNTS:")
    for vtype, count in vehicle_counts.items():
        if count > 0:
            print(f"  {vtype}: {count}")

    # Save predictions
    if save_predictions:
        base_path = Path(output_path).parent
        video_name = Path(video_path).stem

        raw_pred_file = base_path / f"{video_name}_predictions_raw.txt"
        smooth_pred_file = base_path / f"{video_name}_predictions_smooth.txt"

        with open(raw_pred_file, 'w') as f:
            f.write('\n'.join(predictions_raw))

        with open(smooth_pred_file, 'w') as f:
            f.write('\n'.join(predictions_smooth))

        print(f"\nPredictions saved:")
        print(f"  Raw: {raw_pred_file}")
        print(f"  Smooth: {smooth_pred_file}")

    # Report tracks with remaining switches
    tracks_with_switches = [(tid, switches) for tid, switches in track_switch_frames.items() if len(switches) > 0]
    if tracks_with_switches:
        print(f"\nTracks with switches after smoothing: {len(tracks_with_switches)}")
        for tid, switches in sorted(tracks_with_switches, key=lambda x: -len(x[1]))[:5]:
            print(f"  Track {tid}: {len(switches)} switches")
            for frame, from_cls, to_cls in switches[:3]:
                print(f"    Frame {frame}: {from_cls} → {to_cls}")

    return stats


if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "/media/mydrive/GitHub/ultralytics/videos/Gate2.9_Oct7.mp4"
    MODEL_PATH = "yolov8m.pt"  # Use standard model, or specify DCNv2 path
    OUTPUT_PATH = "/media/mydrive/GitHub/ultralytics/tracking/Gate2.9_Oct7_smoothed_output.mp4"

    # Try to use a trained model if available
    trained_model = "/home/migui/YOLO_outputs/100_dcnv2_yolov8m_pan/weights/best.pt"
    if Path(trained_model).exists():
        MODEL_PATH = trained_model

    run_tracking_with_smoothing(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
        smoother_strategy='hysteresis',  # Best for reducing flickering
        conf_threshold=0.5,
        save_predictions=True
    )
