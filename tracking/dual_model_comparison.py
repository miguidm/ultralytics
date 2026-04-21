#!/usr/bin/env python3
"""
Dual Model Comparison Script for YOLOv8 with ByteTrack Tracking
Compare two models side-by-side with individual counting panels

Features:
- Side-by-side comparison of two models
- ByteTrack (SORT-based) tracking for each model
- Individual counting panels for each model
- Diagonal counting line
- Trajectory visualization
- Comprehensive metrics for both models

Usage:
    python dual_model_comparison.py --model1 <model1.pt> --model2 <model2.pt> --source <video_path>

Example:
    python dual_model_comparison.py --model1 yolov8n-vanilla-best.pt --model2 dcnv2-yolov8n-liu-best.pt --source gate3_feb_crop.mp4
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

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
sys.path.insert(0, '/mnt/sda2/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np
import random
import time


def get_unique_color(obj_id):
    """Generate a unique color for each tracking ID"""
    random.seed(obj_id)
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))


def get_curved_line_points(frame_width, frame_height):
    """Get diagonal counting line points"""
    points = np.array([
        [0, int(frame_height * 0.45)],           # Top-left
        [frame_width, int(frame_height * 0.75)]  # Bottom-right
    ], np.int32)
    return points.reshape((-1, 1, 2))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Dual Model Comparison with ByteTrack')
    parser.add_argument('--model1', type=str,
                        default=r'/media/mydrive/GitHub/ultralytics/tracking/best-dcnv2m.pt',
                        help='Path to first model')
    parser.add_argument('--model2', type=str,
                        default=r'/media/mydrive/GitHub/ultralytics/tracking/dcnv2-yolov8n-neck-ful-best.pt',
                        help='Path to second model')
    parser.add_argument('--source', type=str,
                        default='gate3_feb_crop.mp4',
                        help='Path to input video file')
    parser.add_argument('--output', type=str,
                        default=None,
                        help='Path to output video (default: auto-generated based on model names)')
    parser.add_argument('--conf', type=float,
                        default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--tracker', type=str,
                        default='bytetrack.yaml',
                        choices=['bytetrack.yaml', 'botsort.yaml'],
                        help='Tracker configuration')
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
                        help='Disable real-time display')

    return parser.parse_args()


def process_model_frame(frame, results, model, track_classes, trajectories, trajectory_colors,
                       vehicle_counts, object_positions, counted_objects, model_name,
                       show_trajectory, max_trail_length, frame_width, frame_height, panel_width,
                       all_track_ids):
    """Process detections for one model"""

    # Extract tracked objects
    tracked_objects = []

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
            track_classes[obj_id] = class_name
            all_track_ids.add(obj_id)

    tracked_objects = np.array(tracked_objects) if tracked_objects else np.array([])
    current_track_ids = set([int(obj_id) for _, _, _, _, obj_id in tracked_objects]) if len(tracked_objects) > 0 else set()

    # Vehicle class colors
    class_colors = {
        "car": (55, 250, 250),
        "motorcycle": (83, 179, 36),
        "tricycle": (83, 50, 250),
        "bus": (245, 61, 184),
        "van": (255, 221, 51),
        "truck": (49, 147, 245)
    }

    # Line configuration
    line_color = (147, 20, 255)  # Pink
    line_thickness = 3

    # Update trajectories and draw tracked boxes
    for x1, y1, x2, y2, obj_id in tracked_objects:
        obj_id = int(obj_id)
        class_name = track_classes.get(obj_id, "unknown")

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        center_point = (center_x, center_y)

        # Update trajectory
        if show_trajectory:
            if obj_id not in trajectories:
                trajectories[obj_id] = []
            trajectories[obj_id].append(center_point)

            if len(trajectories[obj_id]) > max_trail_length:
                trajectories[obj_id].pop(0)

            if obj_id not in trajectory_colors:
                trajectory_colors[obj_id] = get_unique_color(obj_id)

        # Diagonal line crossing detection
        x_positions = [0, frame_width]
        y_positions = [int(frame_height * 0.45), int(frame_height * 0.75)]
        line_y_at_x = np.interp(center_x, x_positions, y_positions)

        if center_y < line_y_at_x:
            current_position = 'above'
        else:
            current_position = 'below'

        # Check for line crossing
        if obj_id in object_positions:
            previous_position = object_positions[obj_id]

            if previous_position != current_position and obj_id not in counted_objects:
                if class_name in vehicle_counts:
                    vehicle_counts[class_name] += 1
                    counted_objects.add(obj_id)
                    print(f"{model_name} - {class_name.upper()} ID:{obj_id} crossed line! Count: {vehicle_counts[class_name]}")

        object_positions[obj_id] = current_position

        # Draw bounding box
        bbox_color = class_colors.get(class_name.lower(), (0, 255, 0))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 2)
        cv2.putText(frame, f"{class_name} ID:{obj_id}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

    # Draw the diagonal counting line
    curved_points = get_curved_line_points(frame_width, frame_height)
    cv2.polylines(frame, [curved_points], isClosed=False,
                  color=line_color, thickness=line_thickness)

    # Draw trajectories
    if show_trajectory:
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

        # Clean up old trajectories
        trajectories_to_remove = []
        for obj_id in list(trajectories.keys()):
            if obj_id not in current_ids:
                if len(trajectories[obj_id]) > 5:
                    trajectories[obj_id] = trajectories[obj_id][5:]
                else:
                    trajectories_to_remove.append(obj_id)

        for obj_id in trajectories_to_remove:
            del trajectories[obj_id]
            if obj_id in trajectory_colors:
                del trajectory_colors[obj_id]
            if obj_id in object_positions:
                del object_positions[obj_id]
            counted_objects.discard(obj_id)

    # Create extended frame with counting panel
    extended_frame = np.zeros((frame_height, frame_width + panel_width, 3), dtype=np.uint8)
    extended_frame[:, :frame_width] = frame

    # Create counting panel
    panel_color = (40, 40, 40)
    extended_frame[:, frame_width:] = panel_color

    # Add title
    cv2.putText(extended_frame, model_name, (frame_width + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.line(extended_frame, (frame_width + 10, 40),
             (frame_width + panel_width - 10, 40), (255, 255, 255), 1)

    # Display counts
    y_pos = 80
    for vehicle_type, count in vehicle_counts.items():
        color = class_colors.get(vehicle_type, (255, 255, 255))
        cv2.putText(extended_frame, f"{vehicle_type.upper()}: {count}",
                    (frame_width + 15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_pos += 40

    # Add total count
    total_count = sum(vehicle_counts.values())
    cv2.putText(extended_frame, f"TOTAL: {total_count}",
                (frame_width + 15, y_pos + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add tracking statistics
    y_pos += 60
    cv2.line(extended_frame, (frame_width + 10, y_pos),
             (frame_width + panel_width - 10, y_pos), (100, 100, 100), 1)
    y_pos += 25
    cv2.putText(extended_frame, "TRACKING STATS", (frame_width + 10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_pos += 25
    active_tracks = len(current_track_ids)
    total_tracks = len(all_track_ids)
    cv2.putText(extended_frame, f"Active: {active_tracks}",
                (frame_width + 15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    y_pos += 22
    cv2.putText(extended_frame, f"Total Tracks: {total_tracks}",
                (frame_width + 15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

    return extended_frame, current_track_ids


def main():
    """Main function"""
    args = parse_args()

    warnings.filterwarnings('ignore')

    # Generate output filename if not specified
    if args.output is None:
        model1_name = Path(args.model1).stem
        model2_name = Path(args.model2).stem
        args.output = f"{model1_name}_vs_{model2_name}_comparison.mp4"

    print("="*70)
    print("DUAL MODEL COMPARISON WITH BYTETRACK")
    print("="*70)

    # Load models
    print(f"\n[1/5] Loading models...")
    print(f"  Model 1: {Path(args.model1).name}")
    print(f"  Model 2: {Path(args.model2).name}")

    try:
        model1 = YOLO(args.model1)
        model2 = YOLO(args.model2)
        print(f"✓ Both models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # Load video
    print(f"\n[2/5] Loading video: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {args.source}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✓ Video loaded: {frame_width}x{frame_height} @ {fps}fps")
    print(f"  Total frames: {total_frames}")

    # Setup output video
    print(f"\n[3/5] Setting up output: {args.output}")
    single_section_width = frame_width + args.panel_width
    total_width = single_section_width * 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (total_width, frame_height))

    if not out.isOpened():
        print(f"❌ Error: Cannot create output video")
        cap.release()
        return

    print(f"✓ Output video configured")

    # Initialize tracking for both models
    print(f"\n[4/5] Initializing tracking systems...")

    # Model 1 tracking data
    track_classes_1 = {}
    vehicle_counts_1 = {"car": 0, "motorcycle": 0, "tricycle": 0, "van": 0, "bus": 0, "truck": 0}
    counted_objects_1 = set()
    trajectories_1 = {}
    trajectory_colors_1 = {}
    object_positions_1 = {}
    all_track_ids_1 = set()

    # Model 2 tracking data
    track_classes_2 = {}
    vehicle_counts_2 = {"car": 0, "motorcycle": 0, "tricycle": 0, "van": 0, "bus": 0, "truck": 0}
    counted_objects_2 = set()
    trajectories_2 = {}
    trajectory_colors_2 = {}
    object_positions_2 = {}
    all_track_ids_2 = set()

    frame_count = 0
    start_time = time.time()

    print(f"✓ Tracking initialized")

    # Start processing
    print(f"\n[5/5] Starting comparison processing...")
    print("="*70 + "\n")

    model1_name = Path(args.model1).stem
    model2_name = Path(args.model2).stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        frame_count += 1

        # Create copies for each model
        frame1 = frame.copy()
        frame2 = frame.copy()

        # Run both models with tracking
        results1 = model1.track(frame, conf=args.conf, persist=True,
                               tracker=args.tracker, verbose=False)[0]
        results2 = model2.track(frame, conf=args.conf, persist=True,
                               tracker=args.tracker, verbose=False)[0]

        # Process each model
        extended_frame1, current_ids_1 = process_model_frame(
            frame1, results1, model1, track_classes_1, trajectories_1, trajectory_colors_1,
            vehicle_counts_1, object_positions_1, counted_objects_1, model1_name,
            args.show_trajectory, args.trajectory_length, frame_width, frame_height, args.panel_width,
            all_track_ids_1
        )

        extended_frame2, current_ids_2 = process_model_frame(
            frame2, results2, model2, track_classes_2, trajectories_2, trajectory_colors_2,
            vehicle_counts_2, object_positions_2, counted_objects_2, model2_name,
            args.show_trajectory, args.trajectory_length, frame_width, frame_height, args.panel_width,
            all_track_ids_2
        )

        # Calculate detections difference between models
        det_diff = len(current_ids_1) - len(current_ids_2)

        # Combine frames side by side
        combined_frame = np.hstack((extended_frame1, extended_frame2))

        # Add comparison info at top center
        center_x = combined_frame.shape[1] // 2
        diff_text = f"Det Diff: {det_diff:+d}" if det_diff != 0 else "Det Diff: 0"
        diff_color = (100, 255, 100) if det_diff == 0 else ((100, 100, 255) if det_diff > 0 else (255, 100, 100))
        cv2.putText(combined_frame, diff_text, (center_x - 60, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, diff_color, 2)

        # Write to output
        out.write(combined_frame)

        # Display if not disabled
        if not args.no_display:
            cv2.imshow(f"Model Comparison: {model1_name} vs {model2_name}", combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠ Processing interrupted by user")
                break

        # Progress updates
        if frame_count % 30 == 0:
            frame_end_time = time.time()
            current_fps = 1 / (frame_end_time - frame_start_time)
            progress_pct = (frame_count / total_frames) * 100
            print(f"Progress: {progress_pct:.1f}% | Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}")

    # Cleanup
    cap.release()
    out.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # Calculate metrics
    end_time = time.time()
    total_runtime = end_time - start_time

    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    print(f"Total frames processed: {frame_count}")
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Average FPS: {frame_count/total_runtime:.2f}")

    print(f"\nModel 1: {model1_name}")
    print("Vehicle counts:")
    for vehicle_type, count in vehicle_counts_1.items():
        if count > 0:
            print(f"  {vehicle_type.upper()}: {count}")
    total1 = sum(vehicle_counts_1.values())
    print(f"  TOTAL: {total1}")
    print(f"  Total Tracks Created: {len(all_track_ids_1)}")

    print(f"\nModel 2: {model2_name}")
    print("Vehicle counts:")
    for vehicle_type, count in vehicle_counts_2.items():
        if count > 0:
            print(f"  {vehicle_type.upper()}: {count}")
    total2 = sum(vehicle_counts_2.values())
    print(f"  TOTAL: {total2}")
    print(f"  Total Tracks Created: {len(all_track_ids_2)}")

    difference = total2 - total1
    if difference > 0:
        print(f"\nDifference: {model2_name} counted {difference} more vehicles than {model1_name}")
    elif difference < 0:
        print(f"\nDifference: {model1_name} counted {abs(difference)} more vehicles than {model2_name}")
    else:
        print(f"\nBoth models counted the same number of vehicles: {total1}")

    # Save comparison report
    report_filename = args.output.replace('.mp4', '_comparison_report.txt')
    with open(report_filename, 'w') as f:
        f.write(f"""Dual Model Comparison Report
================================================================

Configuration:
- Model 1: {args.model1}
- Model 2: {args.model2}
- Input Video: {args.source}
- Output Video: {args.output}
- Tracker: {args.tracker}
- Confidence Threshold: {args.conf}
- Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Performance:
- Total Frames: {frame_count}
- Total Runtime: {total_runtime:.2f} seconds
- Average FPS: {frame_count/total_runtime:.2f}

Model 1 ({model1_name}) Results:
{chr(10).join([f'- {v.upper()}: {c}' for v, c in vehicle_counts_1.items() if c > 0])}
TOTAL: {total1}
Total Tracks Created: {len(all_track_ids_1)}

Model 2 ({model2_name}) Results:
{chr(10).join([f'- {v.upper()}: {c}' for v, c in vehicle_counts_2.items() if c > 0])}
TOTAL: {total2}
Total Tracks Created: {len(all_track_ids_2)}

Difference: {difference} ({model2_name if difference > 0 else model1_name} counted {'more' if difference != 0 else 'same'})

================================================================
""")

    print(f"\n✓ Output video saved: {args.output}")
    print(f"✓ Comparison report saved: {report_filename}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
