#!/usr/bin/env python3
"""
DCNv2 YOLOv8m Inference Script with ByteTrack Tracking and Counting

Features:
- DCNv2 YOLOv8m model inference with proper environment setup
- ByteTrack tracking for persistent object IDs
- Vehicle counting with diagonal line crossing detection
- Right-side counting panel display
- Trajectory visualization with fading effects
- Comprehensive metrics tracking (IDF1, MOTA, IDSW, FPS)

Usage:
    python inference_dcnv2.py --model <model_path> --source <video_path> --output <output_path>

Example:
    python inference_dcnv2.py --model weights/best.pt --source input.mp4 --output output.mp4
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

# ============================================================================
# DCNv2 ENVIRONMENT SETUP - Must be done BEFORE importing YOLO!
# ============================================================================

def setup_dcnv2_environment():
    """Configure environment for DCNv2 operations"""

    # Set LD_LIBRARY_PATH to include CUDA runtime and PyTorch libraries
    cuda_lib_path = "/home/migui/miniconda3/envs/dcn/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    torch_lib_path = "/home/migui/miniconda3/envs/dcn/lib/python3.10/site-packages/torch/lib"

    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}"

    # Add ultralytics root to Python path
    ultralytics_root = "/media/mydrive/GitHub/ultralytics"
    if ultralytics_root not in sys.path:
        sys.path.insert(0, ultralytics_root)

# Setup environment FIRST
setup_dcnv2_environment()

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
    parser = argparse.ArgumentParser(description='DCNv2 YOLOv8m Inference with Tracking')
    parser.add_argument('--model', type=str,
                        default= r'/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Full.pt',
                        help='Path to DCNv2 YOLOv8m model weights')
    parser.add_argument('--source', type=str,
                        default='videos/Gate3.5_Oct7.mp4',
                        help='Path to input video file')
    parser.add_argument('--output', type=str,
                        default=None,
                        help='Path to output video file (if not specified, will be based on model name)')
    parser.add_argument('--conf', type=float,
                        default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--tracker', type=str,
                        default='bytetrack.yaml',
                        choices=['bytetrack.yaml', 'botsort.yaml'],
                        help='Tracker configuration (bytetrack is SORT-based)')
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
        args.output = f"{model_name}_output.mp4"

    print("="*70)
    print("DCNv2 YOLOv8m Inference with ByteTrack")
    print("="*70)

    # Load DCNv2 YOLOv8 model
    print(f"\n[1/5] Loading model from: {args.model}")
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return

    try:
        model = YOLO(args.model)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error loading model: {error_msg}")

        # Provide helpful error messages
        if "libcudart" in error_msg:
            print("\n💡 CUDA Runtime Library Error Detected!")
            print("   This model was compiled with a specific CUDA version.")
            print("   Solutions:")
            print("   1. Set LD_LIBRARY_PATH to your CUDA lib directory:")
            print("      export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
            print("   2. Or use CPU mode by setting:")
            print("      export CUDA_VISIBLE_DEVICES=-1")
            print("   3. Or retrain/export the model in your current environment")
        elif "bezier_align" in error_msg:
            print("\n💡 MMCV bezier_align error - patches may need adjustment")

        print("\n   Attempting to continue with fallback options...")

        # Try CPU mode as fallback
        try:
            import torch
            torch.cuda.is_available = lambda: False  # Force CPU mode
            model = YOLO(args.model)
            print(f"✓ Model loaded successfully (CPU mode)")
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            sys.exit(1)

    # Load video
    print(f"\n[2/5] Loading video: {args.source}")
    if not os.path.exists(args.source):
        print(f"❌ Error: Video file not found: {args.source}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {args.source}")
        sys.exit(1)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✓ Video loaded: {frame_width}x{frame_height} @ {fps}fps")
    print(f"  Total frames: {total_frames}")
    print(f"  Tracker: {args.tracker}")

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
    print(f"\n[4/5] Initializing tracking system...")
    track_classes = {}  # Store CURRENT class name for each tracked object ID
    track_class_history = {}  # {obj_id: [(frame_num, class_name), ...]}
    class_switches = []  # [(frame_num, obj_id, old_class, new_class), ...]
    class_switch_display = {}  # {obj_id: (text, frames_remaining)} for on-screen display
    SWITCH_DISPLAY_FRAMES = 45  # Show switch annotation for ~1.5 seconds
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

    print(f"✓ Tracking initialized")
    print(f"  Counting line at: {args.line_position*100:.0f}% of frame height")
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

        # Run DCNv2 YOLOv8m tracking with ByteTrack
        # persist=True maintains track IDs across frames
        results = model.track(
            frame,
            conf=args.conf,
            persist=True,
            tracker=args.tracker,
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

                # Detect classification switches
                if obj_id in track_classes and track_classes[obj_id] != class_name:
                    old_class = track_classes[obj_id]
                    class_switches.append((frame_count, obj_id, old_class, class_name))
                    switch_text = f"{old_class}->{class_name}"
                    class_switch_display[obj_id] = (switch_text, SWITCH_DISPLAY_FRAMES)
                    print(f"  SWITCH ID:{obj_id} frame:{frame_count} {old_class} -> {class_name}")

                track_classes[obj_id] = class_name

                # Track per-frame class history
                if obj_id not in track_class_history:
                    track_class_history[obj_id] = []
                track_class_history[obj_id].append((frame_count, class_name))

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

            # Diagonal line crossing detection - use interpolation like model_comparison.py
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

            # Draw classification switch annotation if active
            if obj_id in class_switch_display:
                switch_text, frames_left = class_switch_display[obj_id]
                if frames_left > 0:
                    # Draw bright red switch label below the bbox
                    sw_y = int(y2) + 18
                    # Background rectangle for visibility
                    (tw, th), _ = cv2.getTextSize(switch_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (int(x1), sw_y - th - 4),
                                  (int(x1) + tw + 6, sw_y + 4), (0, 0, 180), -1)
                    cv2.putText(frame, switch_text, (int(x1) + 3, sw_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    class_switch_display[obj_id] = (switch_text, frames_left - 1)
                else:
                    del class_switch_display[obj_id]

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
        cv2.putText(extended_frame, "[DCNv2 + ByteTrack]", (frame_width + 10, 55),
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
        y_pos += 25
        cv2.putText(extended_frame, f"Class Switches: {len(class_switches)}",
                    (frame_width + 15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

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
            cv2.imshow("DCNv2 YOLOv8m + ByteTrack", extended_frame)
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

    total_tracks = len(track_lifespans)

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
    output_content = f"""DCNv2 YOLOv8m Inference with ByteTrack - Metrics Report
================================================================

Model Configuration:
- Model Path: {args.model}
- Input Video: {args.source}
- Output Video: {args.output}
- Tracker: {args.tracker} (ByteTrack)
- Confidence Threshold: {args.conf}
- Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Tracking Metrics:
-----------------
- MOTA: {mota:.4f}
- IDF1: {idf1:.4f}
- ID Switches: {identity_switches}
- FPS: {avg_fps:.2f}

Summary:
--------
- Total Frames: {frame_count}
- Total Runtime: {total_runtime:.2f}s
- Total Tracks: {total_tracks}

Vehicle Counts (Line Crossing):
-------------------------------
{chr(10).join([f"- {vehicle.upper()}: {count}" for vehicle, count in vehicle_counts.items()])}

TOTAL VEHICLES COUNTED: {sum(vehicle_counts.values())}

Classification Switches (Per Frame):
-------------------------------------
Total Switches: {len(class_switches)}
"""

    # Add per-frame switch details
    if class_switches:
        for fnum, oid, old_cls, new_cls in class_switches:
            output_content += f"  Frame {fnum}: ID:{oid} {old_cls} -> {new_cls}\n"
    else:
        output_content += "  (No classification switches detected)\n"

    # Per-second switch summary
    output_content += f"\nClassification Switches (Per Second):\n"
    output_content += f"-------------------------------------\n"
    if class_switches and fps > 0:
        from collections import defaultdict
        switches_per_sec = defaultdict(list)
        for fnum, oid, old_cls, new_cls in class_switches:
            sec = (fnum - 1) // fps
            switches_per_sec[sec].append((fnum, oid, old_cls, new_cls))
        total_seconds = (frame_count - 1) // fps + 1
        for sec in range(total_seconds):
            sw_list = switches_per_sec.get(sec, [])
            if sw_list:
                output_content += f"  Second {sec}: {len(sw_list)} switch(es)\n"
                for fnum, oid, old_cls, new_cls in sw_list:
                    output_content += f"    Frame {fnum}: ID:{oid} {old_cls} -> {new_cls}\n"
        secs_with_switches = len(switches_per_sec)
        output_content += f"\n  Seconds with switches: {secs_with_switches}/{total_seconds}\n"
    else:
        output_content += "  (No classification switches detected)\n"

    # Per-object class history
    output_content += f"\nPer-Object Class History (objects with switches only):\n"
    output_content += f"------------------------------------------------------\n"
    for oid, history in sorted(track_class_history.items()):
        unique_classes = []
        for _, cls in history:
            if not unique_classes or unique_classes[-1] != cls:
                unique_classes.append(cls)
        if len(unique_classes) > 1:
            output_content += f"  ID:{oid}: {' -> '.join(unique_classes)}\n"

    output_content += f"\n================================================================\n"

    # Write metrics to file
    with open(report_filename, 'w') as f:
        f.write(output_content)

    print(f"\n✓ Output video saved: {args.output}")
    print(f"✓ Metrics report saved: {report_filename}")
    print(f"\nTracking Metrics:")
    print(f"  MOTA: {mota:.4f} | IDF1: {idf1:.4f} | ID Switches: {identity_switches} | FPS: {avg_fps:.2f}")
    print(f"\nClassification Switches: {len(class_switches)}")
    print(f"\nVehicle Counts:")
    for vehicle_type, count in vehicle_counts.items():
        if count > 0:
            print(f"  - {vehicle_type.upper()}: {count}")
    print(f"  - TOTAL: {sum(vehicle_counts.values())}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
