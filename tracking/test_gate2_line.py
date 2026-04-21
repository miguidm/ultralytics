#!/usr/bin/env python3
"""
Quick test script to verify Gate2 counting line on 15-second video
"""
import sys
import os
import warnings
from pathlib import Path
import time

warnings.filterwarnings('ignore')

sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np
import random

# Gate2_Oct7 counting line configuration
GATE2_LINE = {
    'x_positions': [0.625, 0.781],  # (1200, 1500) for 1920 width
    'y_positions': [1.0, 0.278]     # (1080, 300) for 1080 height
}

def get_unique_color(obj_id):
    """Generate a unique color for each tracking ID"""
    random.seed(obj_id)
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

# Configuration
model_path = "/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-FPN.pt"
video_path = "/media/mydrive/GitHub/ultralytics/videos/Gate2_Oct7_15sec.mp4"
output_path = "/media/mydrive/GitHub/ultralytics/tracking/test_gate2_line_output.mp4"

print("="*70)
print("Testing Gate2 Counting Line")
print("="*70)
print(f"Model: {os.path.basename(model_path)}")
print(f"Video: {os.path.basename(video_path)}")
print(f"Output: {output_path}")
print("="*70)

# Load model
print("\nLoading model...")
model = YOLO(model_path)
print("✓ Model loaded")

# Load video
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

# Calculate line positions
line_x_positions = [
    int(frame_width * GATE2_LINE['x_positions'][0]),
    int(frame_width * GATE2_LINE['x_positions'][1])
]
line_y_positions = [
    int(frame_height * GATE2_LINE['y_positions'][0]),
    int(frame_height * GATE2_LINE['y_positions'][1])
]

print(f"Counting line: ({line_x_positions[0]}, {line_y_positions[0]}) → ({line_x_positions[1]}, {line_y_positions[1]})")

# Setup output
panel_width = 250
total_width = frame_width + panel_width
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, frame_height))

# Tracking variables
track_classes = {}
vehicle_counts = {"car": 0, "motorcycle": 0, "tricycle": 0, "van": 0, "bus": 0, "truck": 0}
counted_objects = set()
trajectories = {}
max_trail_length = 50
trajectory_colors = {}
object_positions = {}
frame_count = 0

class_colors = {
    "car": (55, 250, 250),
    "motorcycle": (83, 179, 36),
    "tricycle": (83, 50, 250),
    "bus": (245, 61, 184),
    "van": (255, 221, 51),
    "truck": (49, 147, 245)
}

print("\nProcessing frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run tracking
    results = model.track(
        frame,
        conf=0.5,
        persist=True,
        tracker='bytetrack.yaml',
        verbose=False
    )[0]

    # Extract tracked objects
    tracked_objects = []

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            obj_id = track_ids[i]
            class_id = class_ids[i]
            class_name = model.names[class_id]

            tracked_objects.append([x1, y1, x2, y2, obj_id])
            track_classes[obj_id] = class_name

    tracked_objects = np.array(tracked_objects) if tracked_objects else np.array([])

    # Draw tracked boxes and count
    for x1, y1, x2, y2, obj_id in tracked_objects:
        obj_id = int(obj_id)
        class_name = track_classes.get(obj_id, "unknown")

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        center_point = (center_x, center_y)

        # Update trajectory
        if obj_id not in trajectories:
            trajectories[obj_id] = []
        trajectories[obj_id].append(center_point)

        if len(trajectories[obj_id]) > max_trail_length:
            trajectories[obj_id].pop(0)

        if obj_id not in trajectory_colors:
            trajectory_colors[obj_id] = get_unique_color(obj_id)

        # Line crossing detection
        line_y_at_x = np.interp(center_x, line_x_positions, line_y_positions)
        current_position = 'above' if center_y < line_y_at_x else 'below'

        if obj_id in object_positions:
            previous_position = object_positions[obj_id]
            if previous_position != current_position and obj_id not in counted_objects:
                if class_name in vehicle_counts:
                    vehicle_counts[class_name] += 1
                    counted_objects.add(obj_id)

        object_positions[obj_id] = current_position

        # Draw bbox
        bbox_color = class_colors.get(class_name.lower(), (0, 255, 0))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 2)
        cv2.putText(frame, f"{class_name} ID:{obj_id}", (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

    # Draw trajectories
    current_ids = set([int(obj_id) for _, _, _, _, obj_id in tracked_objects]) if len(tracked_objects) > 0 else set()

    for obj_id, trajectory in trajectories.items():
        if len(trajectory) > 1:
            color = trajectory_colors.get(obj_id, (255, 255, 255))
            for i in range(1, len(trajectory)):
                alpha = i / len(trajectory)
                thickness = max(2, int(8 * alpha))
                cv2.line(frame, trajectory[i-1], trajectory[i], color, thickness)
            if trajectory:
                cv2.circle(frame, trajectory[-1], 6, color, -1)

    # Draw counting line
    line_points = np.array([
        [line_x_positions[0], line_y_positions[0]],
        [line_x_positions[1], line_y_positions[1]]
    ], np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [line_points], isClosed=False, color=(0, 0, 255), thickness=5)

    # Create extended frame with panel
    extended_frame = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
    extended_frame[:, :frame_width] = frame

    panel_color = (40, 40, 40)
    extended_frame[:, frame_width:] = panel_color

    # Panel header
    cv2.putText(extended_frame, "VEHICLE COUNT", (frame_width + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(extended_frame, "[ByteTrack]", (frame_width + 10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.line(extended_frame, (frame_width + 10, 65), (frame_width + 240, 65), (255, 255, 255), 1)

    # Vehicle counts
    y_pos = 100
    for vehicle_type, count in vehicle_counts.items():
        color = class_colors.get(vehicle_type, (255, 255, 255))
        cv2.putText(extended_frame, f"{vehicle_type.upper()}: {count}",
                   (frame_width + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_pos += 40

    # Total count
    cv2.line(extended_frame, (frame_width + 10, y_pos + 10),
             (frame_width + 240, y_pos + 10), (100, 100, 100), 1)
    total_count = sum(vehicle_counts.values())
    cv2.putText(extended_frame, f"TOTAL: {total_count}",
               (frame_width + 15, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Frame counter
    cv2.putText(extended_frame, f"Frame: {frame_count}/{total_frames}",
               (frame_width + 15, frame_height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    out.write(extended_frame)

    print(f"\rProcessed frame {frame_count}/{total_frames}", end='', flush=True)

cap.release()
out.release()

print(f"\n\n✓ Output saved to: {output_path}")
print(f"Total vehicles counted: {sum(vehicle_counts.values())}")
print("\nCounting line visualization:")
print(f"  - RED line from (1200, 1080) to (1500, 300)")
print("="*70)
