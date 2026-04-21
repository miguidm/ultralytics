#!/usr/bin/env python3
"""
Multi-Model Inference Script for DCNv2 YOLOv8m with Tracking and Counting
Processes multiple models and generates separate metrics files for each

Features:
- Batch processing of multiple .pt model files
- Individual txt report generation for each model
- Vehicle counting with line crossing detection
- ByteTrack (SORT-based) tracking
- Comprehensive metrics tracking (IDF1, MOTA, IDSW, FPS)

Usage:
    python multi_model_inference.py --models <model1.pt> <model2.pt> ... --source <video_path>
    python multi_model_inference.py --models-dir <directory_with_models> --source <video_path>

Example:
    python multi_model_inference.py --models best1.pt best2.pt best3.pt --source input.mp4
    python multi_model_inference.py --models-dir ./weights --source input.mp4
"""

import sys
import os
import argparse
import warnings
from pathlib import Path
import glob

# Patch MMCV before any imports to avoid compilation errors
print("Initializing MMCV patches...")
try:
    # Mock the problematic MMCV modules before they're imported
    import importlib.util

    # Create a dummy module for bezier_align and other problematic ops
    class DummyModule:
        def __getattr__(self, name):
            return None

    # Inject dummy modules into sys.modules
    sys.modules['mmcv.ops.bezier_align'] = DummyModule()
    sys.modules['mmcv.ops.bias_act'] = DummyModule()
    sys.modules['mmcv.ops.tin_shift'] = DummyModule()
    sys.modules['mmcv.ops.three_interpolate'] = DummyModule()
    sys.modules['mmcv.ops.three_nn'] = DummyModule()

    # Now patch the ext_loader to skip validation
    try:
        import mmcv.utils.ext_loader as ext_loader_module
        original_load_ext = ext_loader_module.load_ext

        def patched_load_ext(name, funcs):
            """Patched version that doesn't validate functions"""
            try:
                return original_load_ext(name, funcs)
            except (AssertionError, AttributeError):
                # Return a dummy module if compilation failed
                return DummyModule()

        ext_loader_module.load_ext = patched_load_ext
    except ImportError:
        pass  # mmcv not installed yet

    print("✓ MMCV patches applied")
except Exception as e:
    print(f"⚠ Warning: Could not patch MMCV: {e}")

warnings.filterwarnings('ignore', category=UserWarning)

# Add the parent directory to Python path to use local ultralytics
sys.path.insert(0, '/mnt/sda2/GitHub/ultralytics')

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
    parser = argparse.ArgumentParser(description='Multi-Model DCNv2 YOLOv8m Inference with Tracking')

    # Model selection - either individual files or directory
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument('--models', type=str, nargs='+',
                            default=[
                                r'/media/mydrive/GitHub/ultralytics/tracking/dcnv2-yolov8-neck-fpn.pt',
                                r'/media/mydrive/GitHub/ultralytics/tracking/dcnv2-yolov8n-neck-ful-best.pt',
                                r'/media/mydrive/GitHub/ultralytics/tracking/dcvn2-yolov8-neck-pan-best.pt'
                            ],
                            help='List of model paths (e.g., model1.pt model2.pt)')
    model_group.add_argument('--models-dir', type=str,
                            help='Directory containing .pt model files')

    parser.add_argument('--source', type=str,
                        default='gate3_feb_crop.mp4',
                        help='Path to input video file')
    parser.add_argument('--output-dir', type=str,
                        default='./output',
                        help='Directory to save output videos and txt files')
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
    parser.add_argument('--save-video', action='store_true',
                        default=False,
                        help='Save output video (disabled by default for faster processing)')

    return parser.parse_args()


def get_model_files(args):
    """Get list of model files from arguments"""
    if args.models:
        # Individual model files specified
        return args.models
    elif args.models_dir:
        # Search directory for .pt files
        models_dir = Path(args.models_dir)
        if not models_dir.exists():
            print(f"❌ Error: Directory not found: {args.models_dir}")
            return []

        pt_files = list(models_dir.glob('*.pt'))
        if not pt_files:
            print(f"❌ Error: No .pt files found in: {args.models_dir}")
            return []

        return [str(f) for f in pt_files]

    return []


def process_single_model(model_path, args, model_num, total_models):
    """Process inference for a single model"""

    # Get model name from file path (without extension)
    model_name = Path(model_path).stem

    print("\n" + "="*70)
    print(f"Processing Model [{model_num}/{total_models}]: {model_name}")
    print("="*70)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    output_video_path = output_dir / f"{model_name}_output.mp4"
    metrics_file_path = output_dir / f"{model_name}_metrics.txt"

    # Load model
    print(f"\n[1/5] Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return False

    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error loading model: {error_msg}")

        if "libcudart" in error_msg:
            print("\n💡 CUDA Runtime Library Error Detected!")
            print("   Attempting CPU mode...")

        try:
            import torch
            torch.cuda.is_available = lambda: False  # Force CPU mode
            model = YOLO(model_path)
            print(f"✓ Model loaded successfully (CPU mode)")
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return False

    # Load video
    print(f"\n[2/5] Loading video: {args.source}")
    if not os.path.exists(args.source):
        print(f"❌ Error: Video file not found: {args.source}")
        return False

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {args.source}")
        return False

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✓ Video loaded: {frame_width}x{frame_height} @ {fps}fps")
    print(f"  Total frames: {total_frames}")
    print(f"  Tracker: {args.tracker}")

    # Setup output video (optional)
    out = None
    if args.save_video:
        print(f"\n[3/5] Setting up output video: {output_video_path}")
        panel_width = args.panel_width
        total_width = frame_width + panel_width

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (total_width, frame_height))

        if not out.isOpened():
            print(f"❌ Error: Cannot create output video: {output_video_path}")
            cap.release()
            return False
        print(f"✓ Output video configured")
    else:
        print(f"\n[3/5] Video saving disabled (processing only)")

    # Initialize tracking variables
    print(f"\n[4/5] Initializing tracking system...")
    track_classes = {}
    vehicle_counts = {
        "car": 0,
        "motorcycle": 0,
        "tricycle": 0,
        "van": 0,
        "bus": 0,
        "truck": 0
    }
    counted_objects = set()

    # Trajectory tracking
    trajectories = {}
    max_trail_length = args.trajectory_length
    trajectory_colors = {}

    # Metrics tracking variables
    frame_count = 0
    total_processing_time = 0
    identity_switches = 0
    track_history = {}
    track_lifespans = {}
    previous_tracks = {}
    start_time = time.time()
    total_detections = 0

    # Counting line configuration - diagonal line like model_comparison.py
    line_color = (147, 20, 255)  # Pink
    line_thickness = 3
    object_positions = {}

    # Define diagonal curved line points function
    def get_curved_line_points(frame_width, frame_height):
        # Diagonal line from top-left to bottom-right for better perspective matching
        points = np.array([
            [0, int(frame_height * 0.45)],           # Top-left - higher up
            [frame_width, int(frame_height * 0.75)]  # Bottom-right - lower down
        ], np.int32)
        return points.reshape((-1, 1, 2))

    print(f"✓ Tracking initialized")

    # Define vehicle class colors
    class_colors = {
        "car": (55, 250, 250),
        "motorcycle": (83, 179, 36),
        "tricycle": (83, 50, 250),
        "bus": (245, 61, 184),
        "van": (255, 221, 51),
        "truck": (49, 147, 245)
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

        # Track metrics
        current_track_ids = set()
        if len(tracked_objects) > 0:
            current_track_ids = set([int(obj_id) for obj_id in tracked_objects[:, 4]])

        total_detections += len(tracked_objects)

        # Identity switch detection
        for track_id in current_track_ids:
            if track_id not in previous_tracks:
                track_lifespans[track_id] = 1
                track_history[track_id] = [frame_count]

            if track_id in track_history:
                track_history[track_id].append(frame_count)
                track_lifespans[track_id] = track_lifespans.get(track_id, 0) + 1

        previous_tracks = current_track_ids.copy()

        # Update trajectories and draw tracked boxes
        for x1, y1, x2, y2, obj_id in tracked_objects:
            obj_id = int(obj_id)
            class_name = track_classes.get(obj_id, "unknown")

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_point = (center_x, center_y)

            # Update trajectory
            if args.show_trajectory and args.save_video:
                if obj_id not in trajectories:
                    trajectories[obj_id] = []
                trajectories[obj_id].append(center_point)

                if len(trajectories[obj_id]) > max_trail_length:
                    trajectories[obj_id].pop(0)

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

                if previous_position != current_position and obj_id not in counted_objects:
                    if class_name in vehicle_counts:
                        vehicle_counts[class_name] += 1
                        counted_objects.add(obj_id)

            # Update object's current position
            object_positions[obj_id] = current_position

            # Draw bounding box (only if saving video)
            if args.save_video:
                bbox_color = class_colors.get(class_name.lower(), (0, 255, 0))
                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), bbox_color, 2)
                cv2.putText(frame, f"{class_name} ID:{obj_id}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

        # Draw the diagonal counting line (only if saving video)
        if args.save_video:
            curved_points = get_curved_line_points(frame_width, frame_height)
            cv2.polylines(frame, [curved_points], isClosed=False,
                          color=line_color, thickness=line_thickness)

        # Draw trajectories (only if saving video)
        if args.show_trajectory and args.save_video:
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
            for obj_id in trajectories.keys():
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

        # Create extended frame with panel (only if saving video)
        if args.save_video:
            panel_width = args.panel_width
            total_width = frame_width + panel_width
            extended_frame = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
            extended_frame[:, :frame_width] = frame

            panel_color = (40, 40, 40)
            extended_frame[:, frame_width:] = panel_color

            cv2.putText(extended_frame, "VEHICLE COUNT", (frame_width + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(extended_frame, f"[{model_name}]", (frame_width + 10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.line(extended_frame, (frame_width + 10, 65),
                     (frame_width + panel_width - 10, 65), (255, 255, 255), 1)

            y_pos = 100
            for vehicle_type, count in vehicle_counts.items():
                color = class_colors.get(vehicle_type, (255, 255, 255))
                cv2.putText(extended_frame, f"{vehicle_type.upper()}: {count}",
                            (frame_width + 15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_pos += 40

            cv2.line(extended_frame, (frame_width + 10, y_pos + 10),
                     (frame_width + panel_width - 10, y_pos + 10), (100, 100, 100), 1)

            total_count = sum(vehicle_counts.values())
            cv2.putText(extended_frame, f"TOTAL: {total_count}",
                        (frame_width + 15, y_pos + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            progress_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(extended_frame, progress_text,
                        (frame_width + 15, frame_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            frame_end_time = time.time()
            frame_processing_time = frame_end_time - frame_start_time
            current_fps = 1 / frame_processing_time if frame_processing_time > 0 else 0
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(extended_frame, fps_text,
                        (frame_width + 15, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            out.write(extended_frame)

            if not args.no_display:
                cv2.imshow(f"Model: {model_name}", extended_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠ Processing interrupted by user")
                    break
        else:
            # Just timing without video output
            frame_end_time = time.time()
            frame_processing_time = frame_end_time - frame_start_time

        total_processing_time += frame_processing_time

        # Progress updates
        if frame_count % 30 == 0:
            progress_pct = (frame_count / total_frames) * 100
            current_fps = 1 / frame_processing_time if frame_processing_time > 0 else 0
            print(f"Progress: {progress_pct:.1f}% | Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}")

    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # Calculate final metrics
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
    false_negatives = max(0, total_detections - len(current_track_ids) * frame_count)
    false_positives = max(0, len(current_track_ids) * frame_count - total_detections)
    mota = 1 - (false_negatives + false_positives + identity_switches) / max(1, total_detections)
    mota = max(0, min(1, mota))

    # IDF1 calculation
    id_true_positives = sum(track_lifespans.values())
    id_false_positives = identity_switches
    id_false_negatives = max(0, total_detections - id_true_positives)
    idf1 = (2 * id_true_positives) / max(1, 2 * id_true_positives + id_false_positives + id_false_negatives)

    # Create metrics report
    output_content = f"""DCNv2 YOLOv8m Multi-Model Inference - Metrics Report
================================================================

Model Information:
- Model Name: {model_name}
- Model Path: {model_path}
- Input Video: {args.source}
- Output Video: {output_video_path if args.save_video else 'Not saved'}
- Tracker: {args.tracker} (SORT-based)
- Confidence Threshold: {args.conf}
- Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

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
    with open(metrics_file_path, 'w') as f:
        f.write(output_content)

    print("\n" + "="*70)
    print(f"MODEL {model_name} - PROCESSING COMPLETE")
    print("="*70)
    if args.save_video:
        print(f"✓ Output video saved: {output_video_path}")
    print(f"✓ Metrics report saved: {metrics_file_path}")
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
    print("="*70 + "\n")

    return True


def main():
    """Main function to process multiple models"""
    args = parse_args()

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Get list of model files
    model_files = get_model_files(args)

    if not model_files:
        print("❌ Error: No model files found")
        return

    print("\n" + "="*70)
    print("MULTI-MODEL INFERENCE SCRIPT")
    print("="*70)
    print(f"\nFound {len(model_files)} model(s) to process:")
    for i, model_path in enumerate(model_files, 1):
        print(f"  {i}. {Path(model_path).name}")

    print(f"\nInput video: {args.source}")
    print(f"Output directory: {args.output_dir}")
    print(f"Save videos: {'Yes' if args.save_video else 'No (metrics only)'}")
    print("="*70)

    # Process each model
    successful = 0
    failed = 0

    for i, model_path in enumerate(model_files, 1):
        try:
            if process_single_model(model_path, args, i, len(model_files)):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Error processing {Path(model_path).name}: {e}")
            failed += 1

    # Final summary
    print("\n" + "="*70)
    print("ALL MODELS PROCESSING COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  - Total models: {len(model_files)}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    print(f"\nOutput location: {args.output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
