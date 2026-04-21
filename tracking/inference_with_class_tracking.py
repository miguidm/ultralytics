#!/usr/bin/env python3
"""
Video inference with classification tracking
- Left panel with vehicle counts
- Track IDs displayed on boxes
- Classification changes shown per second
- Total tracks and count displayed
"""

import sys
import os
import warnings
from collections import defaultdict, Counter
import time

warnings.filterwarnings('ignore')

sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np
import random


def get_unique_color(obj_id):
    """Generate a unique color for each tracking ID"""
    random.seed(obj_id)
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))


def run_inference_with_class_tracking(model_path, video_path, output_path):
    """Run inference with classification change tracking"""

    print("="*70)
    print("INFERENCE WITH CLASSIFICATION TRACKING")
    print("="*70)

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    print("Model loaded")

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

    # Output video with left panel
    panel_width = 350
    total_width = frame_width + panel_width

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, frame_height))

    # Tracking data structures
    track_class_history = defaultdict(list)  # All classifications per track
    track_class_with_conf = defaultdict(lambda: defaultdict(float))  # Confidence-weighted
    track_last_class = {}  # Current class per track
    track_class_changes = defaultdict(list)  # [(frame, old_class, new_class), ...]

    # Counting
    vehicle_counts = {"car": 0, "motorcycle": 0, "tricycle": 0, "van": 0, "bus": 0, "truck": 0}
    counted_objects = set()
    object_positions = {}

    # Gate2.9 counting line
    line_x_positions = [0, frame_width]
    line_y_positions = [int(frame_height * 0.42), int(frame_height * 0.65)]

    # Class colors
    class_colors = {
        "car": (55, 250, 250),
        "motorcycle": (83, 179, 36),
        "tricycle": (83, 50, 250),
        "bus": (245, 61, 184),
        "van": (255, 221, 51),
        "truck": (49, 147, 245)
    }

    frame_count = 0
    last_second = -1
    classification_changes_this_second = []
    classification_changes_this_frame = []  # Per-frame changes
    all_time_changes = []  # Store recent changes for display

    print("Processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_second = frame_count // fps
        classification_changes_this_frame = []  # Reset per frame

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
            extended_frame = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
            extended_frame[:, panel_width:] = frame
            out.write(extended_frame)
            continue

        # Process detections
        active_tracks = set()

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

                active_tracks.add(obj_id)

                # Track classification history
                track_class_history[obj_id].append(class_name)
                track_class_with_conf[obj_id][class_name] += conf

                # Detect classification change
                if obj_id in track_last_class:
                    old_class = track_last_class[obj_id]
                    if old_class != class_name:
                        change_info = {
                            'frame': frame_count,
                            'second': current_second,
                            'track_id': obj_id,
                            'old_class': old_class,
                            'new_class': class_name,
                            'conf': conf
                        }
                        track_class_changes[obj_id].append(change_info)
                        classification_changes_this_second.append(change_info)
                        classification_changes_this_frame.append(change_info)

                track_last_class[obj_id] = class_name

                # Get majority class for this track (for display)
                majority_class = max(
                    track_class_with_conf[obj_id],
                    key=track_class_with_conf[obj_id].get
                )

                # Line crossing detection
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                line_y_at_x = np.interp(center_x, line_x_positions, line_y_positions)
                current_position = 'above' if center_y < line_y_at_x else 'below'

                if obj_id in object_positions:
                    previous_position = object_positions[obj_id]
                    if previous_position != current_position and obj_id not in counted_objects:
                        # Use majority class for counting
                        if majority_class in vehicle_counts:
                            vehicle_counts[majority_class] += 1
                            counted_objects.add(obj_id)

                object_positions[obj_id] = current_position

                # Draw bounding box
                color = class_colors.get(class_name.lower(), (0, 255, 0))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Label with ID and current class
                label = f"ID:{obj_id} {class_name}"
                if class_name != majority_class:
                    label += f" (maj:{majority_class})"

                # Background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (int(x1), int(y1) - th - 8), (int(x1) + tw + 4, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1) + 2, int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw counting line
        cv2.line(frame, (line_x_positions[0], line_y_positions[0]),
                 (line_x_positions[1], line_y_positions[1]), (147, 20, 255), 3)

        # Update changes - add per-frame changes to history
        if classification_changes_this_frame:
            all_time_changes.extend(classification_changes_this_frame)
            # Keep only last 15 changes for display
            all_time_changes = all_time_changes[-15:]

        # Reset per-second tracking
        if current_second != last_second:
            classification_changes_this_second = []
            last_second = current_second

        # Create extended frame with LEFT panel
        extended_frame = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
        extended_frame[:, panel_width:] = frame  # Video on right

        # Panel background
        panel_color = (30, 30, 30)
        extended_frame[:, :panel_width] = panel_color

        # === LEFT PANEL CONTENT ===
        y_pos = 30

        # Header
        cv2.putText(extended_frame, "TRACKING ANALYSIS", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 25
        cv2.line(extended_frame, (10, y_pos), (panel_width - 10, y_pos), (100, 100, 100), 1)
        y_pos += 25

        # Frame/Time info
        cv2.putText(extended_frame, f"Frame: {frame_count}/{total_frames}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20
        cv2.putText(extended_frame, f"Time: {current_second}s / {total_frames//fps}s", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 30

        # Stats section
        cv2.putText(extended_frame, "STATISTICS", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        y_pos += 25

        total_tracks = len(track_class_history)
        cv2.putText(extended_frame, f"Total Tracks: {total_tracks}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(extended_frame, f"Active Tracks: {len(active_tracks)}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y_pos += 20

        # Count tracks with class changes
        tracks_with_changes = sum(1 for changes in track_class_changes.values() if changes)
        total_changes = sum(len(changes) for changes in track_class_changes.values())
        cv2.putText(extended_frame, f"Tracks w/ Changes: {tracks_with_changes}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        y_pos += 20
        cv2.putText(extended_frame, f"Total Class Switches: {total_changes}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 100), 1)
        y_pos += 30

        # Vehicle counts section
        cv2.line(extended_frame, (10, y_pos), (panel_width - 10, y_pos), (100, 100, 100), 1)
        y_pos += 20
        cv2.putText(extended_frame, "VEHICLE COUNTS", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        y_pos += 25

        for vehicle_type, count in vehicle_counts.items():
            color = class_colors.get(vehicle_type, (255, 255, 255))
            cv2.putText(extended_frame, f"{vehicle_type.upper()}: {count}",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 22

        total_count = sum(vehicle_counts.values())
        y_pos += 5
        cv2.putText(extended_frame, f"TOTAL: {total_count}",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 35

        # Per-frame changes section (highlighted)
        cv2.line(extended_frame, (10, y_pos), (panel_width - 10, y_pos), (100, 100, 100), 1)
        y_pos += 20
        cv2.putText(extended_frame, "THIS FRAME CHANGES", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y_pos += 22

        if classification_changes_this_frame:
            for change in classification_changes_this_frame:
                text = f"ID:{change['track_id']} {change['old_class']}->{change['new_class']}"
                # Highlight with bright color
                cv2.putText(extended_frame, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
                y_pos += 20
        else:
            cv2.putText(extended_frame, "No changes", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            y_pos += 18

        y_pos += 10

        # Recent changes history section
        cv2.line(extended_frame, (10, y_pos), (panel_width - 10, y_pos), (100, 100, 100), 1)
        y_pos += 20
        cv2.putText(extended_frame, "RECENT CHANGES", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 100), 2)
        y_pos += 22

        if all_time_changes:
            # Show recent changes
            for change in all_time_changes[-6:]:  # Last 6 changes
                text = f"ID:{change['track_id']} {change['old_class'][:3]}->{change['new_class'][:3]} f:{change['frame']}"
                cv2.putText(extended_frame, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 150), 1)
                y_pos += 16
        else:
            cv2.putText(extended_frame, "No changes yet", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        out.write(extended_frame)

        if frame_count % 30 == 0:
            pct = (frame_count / total_frames) * 100
            print(f"\rProgress: {pct:.1f}% | Tracks: {total_tracks} | Changes: {total_changes}", end='', flush=True)

    cap.release()
    out.release()

    # Final summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"Output: {output_path}")
    print(f"Total Tracks: {len(track_class_history)}")
    print(f"Tracks with Class Changes: {tracks_with_changes}")
    print(f"Total Classification Switches: {total_changes}")
    print(f"\nVehicle Counts (using majority class):")
    for vtype, count in vehicle_counts.items():
        print(f"  {vtype}: {count}")
    print(f"  TOTAL: {sum(vehicle_counts.values())}")

    # Show tracks with most changes
    if track_class_changes:
        print(f"\nTracks with Most Class Changes:")
        sorted_changes = sorted(track_class_changes.items(), key=lambda x: len(x[1]), reverse=True)
        for track_id, changes in sorted_changes[:10]:
            class_counts = Counter(track_class_history[track_id])
            print(f"  Track {track_id}: {len(changes)} changes | Classes: {dict(class_counts)}")

    print("="*70)


if __name__ == "__main__":
    # Use DCNv2-LIU model
    model_path = "/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-LIU.pt"
    video_path = "/media/mydrive/GitHub/ultralytics/videos/Gate2.9_Oct7_5min.mp4"
    output_path = "/media/mydrive/GitHub/ultralytics/tracking/Gate2.9_DCNv2-LIU_perframe_5min.mp4"

    run_inference_with_class_tracking(model_path, video_path, output_path)
