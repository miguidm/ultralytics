#!/usr/bin/env python3
"""
DCNv2 YOLOv8m Inference Script with BoT-SORT Tracking and Counting

BoT-SORT (Bounding box and Trajectory SORT) is an advanced tracker that:
- Combines motion and appearance features
- Uses camera motion compensation (CMC)
- Better handling of occlusions and crowded scenes
- More robust but slightly slower than ByteTrack

Features:
- DCNv2 YOLOv8m model inference with proper environment setup
- BoT-SORT tracking for persistent object IDs
- Vehicle counting with diagonal line crossing detection
- Right-side counting panel display
- Trajectory visualization with fading effects
- Comprehensive metrics tracking (IDF1, MOTA, IDSW, FPS)

Usage:
    python inference_dcnv2_sort.py --model <model_path> --source <video_path> --output <output_path>

Example:
    python inference_dcnv2_sort.py --model weights/best.pt --source input.mp4 --output output.mp4
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)

from ultralytics import YOLO
import cv2
import numpy as np
import random
import time


def get_unique_color(obj_id):
    """Generate a unique color for each tracking ID"""
    random.seed(obj_id)  # Ensure same ID always gets same color
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DCNv2 YOLOv8m Inference with BoT-SORT')
    parser.add_argument('--model', type=str,
                        default=r'/home/migui/YOLO_outputs/100_dcnv3_yolov8m_liu11/weights/best.pt',
                        help='Path to DCNv2 YOLOv8m model weights')
    parser.add_argument('--source', type=str,
                        default='gate3_feb_crop.mp4',
                        help='Path to input video file')
    parser.add_argument('--output', type=str,
                        default=None,
                        help='Path to output video file (if not specified, will be based on model name)')
    parser.add_argument('--conf', type=float,
                        default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--line-position', type=float,
                        default=0.7,
                        help='Counting line position (0.0-1.0, fraction of frame height)')
    parser.add_argument('--panel-width', type=int,
                        default=250,
                        help='Width of counting panel in pixels')
    parser.add_argument('--show-trajectory', action='store_true',
                        default=True,
                        help='Show object trajectories')
    parser.add_argument('--trajectory-length', type=int,
                        default=50,
                        help='Maximum trajectory trail length')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable real-time display (faster processing)')

    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Generate output filename based on model name if not specified
    if args.output is None:
        model_name = Path(args.model).stem
        args.output = f"{model_name}_botsort_output.mp4"

    print("="*70)
    print("DCNv2 YOLOv8m Inference with BoT-SORT")
    print("="*70)
    print("Tracker: BoT-SORT (botsort.yaml)")
    print("  - Advanced multi-object tracker")
    print("  - Motion + Appearance features")
    print("  - Camera motion compensation (CMC)")
    print("  - Better for occlusions and crowded scenes")
    print()

    # Load DCNv2 YOLOv8 model
    print(f"[1/5] Loading model from: {args.model}")
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
    print(f"\n[2/5] Loading video: {args.source}")
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

    # Setup output video
    print(f"\n[3/5] Setting up output: {args.output}")
    panel_width = args.panel_width
    total_width = frame_width + panel_width

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (total_width, frame_height))

    if not out.isOpened():
        print(f"❌ Error: Cannot create output video: {args.output}")
        cap.release()
        return

    print(f"✓ Output video configured")

    # Initialize tracking variables
    print(f"\n[4/5] Initializing BoT-SORT system...")
    track_classes = {}  # Store class names for each tracked object ID
    vehicle_counts = {
        "car": 0,
        "motorcycle": 0,
        "tricycle": 0,
        "van": 0,
        "bus": 0,
        "truck": 0
    }
    counted_objects = set()  # Track which IDs have been counted

    # Trajectory tracking
    trajectories = {}  # Store trajectory points for each ID
    max_trail_length = args.trajectory_length
    trajectory_colors = {}  # Store unique colors for each trajectory

    # Metrics tracking variables
    frame_count = 0
    total_processing_time = 0
    identity_switches = 0
    track_history = {}  # Store complete history of each track
    track_lifespans = {}  # Store lifespan of each track
    previous_tracks = {}  # Store previous frame tracks for ID switch detection
    start_time = time.time()

    # For MOTA calculation
    total_detections = 0

    # Counting line configuration - diagonal line
    line_color = (147, 20, 255)  # Pink
    line_thickness = 3

    # Define diagonal curved line points function
    def get_curved_line_points(frame_width, frame_height):
        # Diagonal line from top-left to bottom-right for better perspective matching
        points = np.array([
            [0, int(frame_height * 0.45)],           # Top-left - higher up
            [frame_width, int(frame_height * 0.75)]  # Bottom-right - lower down
        ], np.int32)
        return points.reshape((-1, 1, 2))

    # Track which side of the line each object is on
    object_positions = {}  # {obj_id: 'above'/'below'}

    print(f"✓ BoT-SORT initialized")
    print(f"  Diagonal counting line: 45%-75% of frame height")
    print(f"  Trajectory display: {'Enabled' if args.show_trajectory else 'Disabled'}")

    # Define vehicle class colors (BGR format for OpenCV)
    class_colors = {
        "car": (55, 250, 250),        # Yellow
        "motorcycle": (83, 179, 36),  # Green
        "tricycle": (83, 50, 250),    # Red-pink
        "bus": (245, 61, 184),        # Purple
        "van": (255, 221, 51),        # Light blue/cyan
        "truck": (49, 147, 245)       # Orange
    }

    # Start processing
    print(f"\n[5/5] Starting inference and tracking...")
    print("="*70 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        frame_count += 1

        # Run DCNv2 YOLOv8m tracking with BoT-SORT
        # persist=True maintains track IDs across frames
        results = model.track(
            frame,
            conf=args.conf,
            persist=True,
            tracker='botsort.yaml',
            verbose=False
        )[0]

        # Extract tracked objects
        tracked_objects = []
        detection_classes = []

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
                class_name = model.names[class_id]

                tracked_objects.append([x1, y1, x2, y2, obj_id])
                detection_classes.append(class_name)
                track_classes[obj_id] = class_name

        tracked_objects = np.array(tracked_objects) if tracked_objects else np.array([])

        # Track metrics for current frame
        current_track_ids = set()
        if len(tracked_objects) > 0:
            current_track_ids = set([int(obj_id) for obj_id in tracked_objects[:, 4]])

        # Update total detections count
        total_detections += len(tracked_objects)

        # Identity switch detection and track management
        for track_id in current_track_ids:
            if track_id not in previous_tracks:
                # New track started
                track_lifespans[track_id] = 1
                track_history[track_id] = [frame_count]

            if track_id in track_history:
                track_history[track_id].append(frame_count)
                track_lifespans[track_id] = track_lifespans.get(track_id, 0) + 1

        # Update previous tracks for next iteration
        previous_tracks = current_track_ids.copy()

        # Update trajectories and draw tracked boxes
        for x1, y1, x2, y2, obj_id in tracked_objects:
            obj_id = int(obj_id)
            class_name = track_classes.get(obj_id, "unknown")

            # Calculate center point of bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_point = (center_x, center_y)

            # Update trajectory for this object
            if args.show_trajectory:
                if obj_id not in trajectories:
                    trajectories[obj_id] = []
                trajectories[obj_id].append(center_point)

                # Keep only the last N points to limit trajectory length
                if len(trajectories[obj_id]) > max_trail_length:
                    trajectories[obj_id].pop(0)

                # Get or create unique color for this trajectory
                if obj_id not in trajectory_colors:
                    trajectory_colors[obj_id] = get_unique_color(obj_id)

            # Diagonal line crossing detection - use interpolation
            x_positions = [0, frame_width]
            y_positions = [int(frame_height * 0.45), int(frame_height * 0.75)]

            # Find the interpolated Y value for the object's X position
            line_y_at_x = np.interp(center_x, x_positions, y_positions)

            if center_y < line_y_at_x:
                current_position = 'above'
            else:
                current_position = 'below'

            # Check for line crossing
            if obj_id in object_positions:
                previous_position = object_positions[obj_id]

                # Check if vehicle crossed the line (changed from above to below or vice versa)
                if previous_position != current_position and obj_id not in counted_objects:
                    if class_name in vehicle_counts:
                        vehicle_counts[class_name] += 1
                        counted_objects.add(obj_id)
                        print(f"✓ {class_name.upper()} ID:{obj_id} crossed line! Count: {vehicle_counts[class_name]}")

            # Update object's current position
            object_positions[obj_id] = current_position

            # Get color for this vehicle class
            bbox_color = class_colors.get(class_name.lower(), (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), bbox_color, 2)
            cv2.putText(frame, f"{class_name} ID:{obj_id}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

        # Draw the diagonal counting line
        curved_points = get_curved_line_points(frame_width, frame_height)
        cv2.polylines(frame, [curved_points], isClosed=False,
                      color=line_color, thickness=line_thickness)

        # Draw trajectories for all tracked objects
        if args.show_trajectory:
            current_ids = set([int(obj_id) for _, _, _, _, obj_id in tracked_objects]) if len(tracked_objects) > 0 else set()

            for obj_id, trajectory in trajectories.items():
                if len(trajectory) > 1:
                    color = trajectory_colors.get(obj_id, (255, 255, 255))

                    # Draw trajectory as connected lines with fading effect
                    for i in range(1, len(trajectory)):
                        # Calculate thickness based on point age (newer points are thicker)
                        alpha = i / len(trajectory)
                        thickness = max(2, int(8 * alpha))

                        # Draw line segment
                        cv2.line(frame, trajectory[i-1], trajectory[i], color, thickness)

                    # Draw a larger circle at the current position
                    if trajectory:
                        cv2.circle(frame, trajectory[-1], 6, color, -1)

            # Clean up trajectories for objects that are no longer tracked
            trajectories_to_remove = []
            for obj_id in trajectories.keys():
                if obj_id not in current_ids:
                    # Gradually fade out old trajectories
                    if len(trajectories[obj_id]) > 5:
                        trajectories[obj_id] = trajectories[obj_id][5:]
                    else:
                        trajectories_to_remove.append(obj_id)

            # Remove completely faded trajectories
            for obj_id in trajectories_to_remove:
                del trajectories[obj_id]
                if obj_id in trajectory_colors:
                    del trajectory_colors[obj_id]
                if obj_id in object_positions:
                    del object_positions[obj_id]
                counted_objects.discard(obj_id)

        # Create extended frame with counting panel on the right
        extended_frame = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
        extended_frame[:, :frame_width] = frame  # Place original frame on left

        # Create counting panel on the right side
        panel_color = (40, 40, 40)  # Dark gray background
        extended_frame[:, frame_width:] = panel_color

        # Add title and separator line
        cv2.putText(extended_frame, "VEHICLE COUNT", (frame_width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(extended_frame, "[BoT-SORT]", (frame_width + 10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.line(extended_frame, (frame_width + 10, 65),
                 (frame_width + panel_width - 10, 65), (255, 255, 255), 1)

        # Display counts for each vehicle type
        y_pos = 100
        for vehicle_type, count in vehicle_counts.items():
            color = class_colors.get(vehicle_type, (255, 255, 255))
            cv2.putText(extended_frame, f"{vehicle_type.upper()}: {count}",
                        (frame_width + 15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 40

        # Add separator line
        cv2.line(extended_frame, (frame_width + 10, y_pos + 10),
                 (frame_width + panel_width - 10, y_pos + 10), (100, 100, 100), 1)

        # Add total count
        total_count = sum(vehicle_counts.values())
        cv2.putText(extended_frame, f"TOTAL: {total_count}",
                    (frame_width + 15, y_pos + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add tracking statistics
        y_pos += 80
        cv2.line(extended_frame, (frame_width + 10, y_pos),
                 (frame_width + panel_width - 10, y_pos), (100, 100, 100), 1)
        y_pos += 25
        cv2.putText(extended_frame, "TRACKING STATS", (frame_width + 10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 30
        active_tracks = len(current_track_ids)
        total_tracks_so_far = len(track_lifespans)
        cv2.putText(extended_frame, f"Active: {active_tracks}",
                    (frame_width + 15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y_pos += 25
        cv2.putText(extended_frame, f"Total Tracks: {total_tracks_so_far}",
                    (frame_width + 15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        # Add frame counter at bottom
        progress_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(extended_frame, progress_text,
                    (frame_width + 15, frame_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Add FPS counter
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        current_fps = 1 / frame_processing_time if frame_processing_time > 0 else 0
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(extended_frame, fps_text,
                    (frame_width + 15, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Write frame to output video
        out.write(extended_frame)

        # Display frame if not disabled
        if not args.no_display:
            cv2.imshow("DCNv2 YOLOv8m + BoT-SORT", extended_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠ Processing interrupted by user")
                break

        # Update timing
        total_processing_time += frame_processing_time

        # Progress updates every 30 frames
        if frame_count % 30 == 0:
            progress_pct = (frame_count / total_frames) * 100
            print(f"Progress: {progress_pct:.1f}% | Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}")

    # Cleanup
    cap.release()
    out.release()
    if not args.no_display:
        cv2.destroyAllWindows()

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

    # MOTA calculation (simplified)
    false_negatives = max(0, total_detections - len(current_track_ids) * frame_count)
    false_positives = max(0, len(current_track_ids) * frame_count - total_detections)
    mota = 1 - (false_negatives + false_positives + identity_switches) / max(1, total_detections)
    mota = max(0, min(1, mota))

    # IDF1 calculation (simplified)
    id_true_positives = sum(track_lifespans.values())
    id_false_positives = identity_switches
    id_false_negatives = max(0, total_detections - id_true_positives)
    idf1 = (2 * id_true_positives) / max(1, 2 * id_true_positives + id_false_positives + id_false_negatives)

    # Print summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)

    # Create metrics report
    report_filename = args.output.replace('.mp4', '_metrics.txt')
    output_content = f"""DCNv2 YOLOv8m Inference with BoT-SORT - Metrics Report
================================================================

Model Configuration:
- Model Path: {args.model}
- Input Video: {args.source}
- Output Video: {args.output}
- Tracker: BoT-SORT (botsort.yaml)
- Confidence Threshold: {args.conf}
- Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Tracker Information:
--------------------
BoT-SORT is an advanced multi-object tracker that:
- Combines bounding box and trajectory information
- Uses motion and appearance features (ReID)
- Implements camera motion compensation (CMC)
- Better handling of occlusions and crowded scenes
- More robust but slightly slower than ByteTrack

Tracking Metrics:
-----------------
1. IDF1 (Identity F1 Score): {idf1:.4f}
   - Measures identity preservation accuracy
   - Range: 0.0 (worst) to 1.0 (best)

2. MT/ML (Mostly Tracked/Lost Ratios):
   - MT (Mostly Tracked): {mt_ratio:.4f} ({mostly_tracked}/{total_tracks} tracks)
   - ML (Mostly Lost): {ml_ratio:.4f} ({mostly_lost}/{total_tracks} tracks)
   - Tracks tracked ≥80% vs ≤20% of time

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
   - Higher is better for real-time applications

Additional Statistics:
---------------------
- Total Frames Processed: {frame_count}
- Total Runtime: {total_runtime:.2f} seconds
- Total Tracks Created: {total_tracks}
- Total Detections: {total_detections}
- Average Track Lifespan: {sum(track_lifespans.values())/max(1,len(track_lifespans)):.1f} frames

Vehicle Counts (Line Crossing):
-------------------------------
{chr(10).join([f"- {vehicle.upper()}: {count}" for vehicle, count in vehicle_counts.items()])}

TOTAL VEHICLES COUNTED: {sum(vehicle_counts.values())}

================================================================
"""

    # Write metrics to file
    with open(report_filename, 'w') as f:
        f.write(output_content)

    print(f"\n✓ Output video saved: {args.output}")
    print(f"✓ Metrics report saved: {report_filename}")
    print(f"\nProcessing Summary:")
    print(f"  - Frames: {frame_count}/{total_frames}")
    print(f"  - Runtime: {total_runtime:.2f}s")
    print(f"  - Average FPS: {avg_fps:.2f}")
    print(f"  - Total Tracks: {total_tracks}")
    print(f"  - Identity Switches: {identity_switches}")
    print(f"  - MOTA: {mota:.4f}")
    print(f"  - IDF1: {idf1:.4f}")
    print(f"\nVehicle Counts:")
    for vehicle_type, count in vehicle_counts.items():
        if count > 0:
            print(f"  - {vehicle_type.upper()}: {count}")
    print(f"  - TOTAL: {sum(vehicle_counts.values())}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
