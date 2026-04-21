#!/usr/bin/env python3
"""
Run BoT-SORT inference on all DCNv2 models for all gate videos

Processes all videos from /media/mydrive/GitHub/ultralytics/videos/
Outputs to /media/mydrive/GitHub/ultralytics/tracking/inference_results_botsort/{model_name}/{gate_name}/

Features:
- BoT-SORT tracking (botsort.yaml)
- Gate-specific counting lines
- Right-side counting panel
- Comprehensive metrics tracking
"""

import sys
import os
import warnings
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# CRITICAL: Use local custom ultralytics with DCNv2/DCNv3 modules
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np
import random

# Gate-specific counting line configurations
# Based on reference images in /media/mydrive/GitHub/ultralytics/References
# Format: gate_name: {'x_positions': [x1, x2], 'y_positions': [y1, y2]}
# Coordinates are in fractions (0.0 to 1.0) of frame dimensions
GATE_COUNTING_LINES = {
    'Gate2_Oct7': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.62, 0.38]  # Diagonal - based on /media/mydrive/GitHub/ultralytics/References/Gate2.jpeg
    },
    'Gate2.9_Oct7': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.42, 0.65]  # Diagonal - based on Gate2.9.jpeg reference
    },
    'Gate3_Oct7': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.47, 0.60]  # Diagonal - interpolated
    },
    'Gate3.5_Oct7': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.52, 0.55]  # Nearly horizontal - based on Gate3.5.jpeg reference
    },
    'default': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.45, 0.60]
    }
}


def get_unique_color(obj_id):
    """Generate a unique color for each tracking ID"""
    random.seed(obj_id)
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))


def run_model(model_path, model_name, video_path, gate_name, output_base_dir):
    """Run BoT-SORT inference for a single model on a specific video"""
    print("\n" + "="*70)
    print(f"Processing: {model_name} | Video: {gate_name} | Tracker: BoT-SORT")
    print("="*70)

    # Create output directory structure
    output_dir = os.path.join(output_base_dir, model_name, gate_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    print(f"Loading model from: {model_path}")
    try:
        model = YOLO(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"  1. Make sure CUDA libraries are accessible")
        print(f"  2. Check that model file exists: {os.path.exists(model_path)}")
        print(f"  3. Try running with CPU mode if GPU is unavailable")
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

    print(f"✓ Video loaded: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

    panel_width = 250
    total_width = frame_width + panel_width

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = os.path.join(output_dir, f"{gate_name}_botsort_output.mp4")
    out = cv2.VideoWriter(output_video, fourcc, fps, (total_width, frame_height))

    if not out.isOpened():
        print(f"✗ Cannot create output video")
        cap.release()
        return None

    # Tracking variables
    track_classes = {}
    vehicle_counts = {"car": 0, "motorcycle": 0, "tricycle": 0, "van": 0, "bus": 0, "truck": 0}
    counted_objects = set()
    trajectories = {}
    max_trail_length = 50
    trajectory_colors = {}

    frame_count = 0
    total_processing_time = 0
    track_history = {}
    track_lifespans = {}
    previous_tracks = {}
    start_time = time.time()
    total_detections = 0

    # Get gate-specific counting line configuration
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

    print(f"BoT-SORT: Counting line for {gate_name}: x={line_x_positions}, y={line_y_positions}")

    # Class colors
    class_colors = {
        "car": (55, 250, 250),
        "motorcycle": (83, 179, 36),
        "tricycle": (83, 50, 250),
        "bus": (245, 61, 184),
        "van": (255, 221, 51),
        "truck": (49, 147, 245)
    }

    print("Starting BoT-SORT processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        frame_count += 1

        # Run tracking with BoT-SORT
        try:
            results = model.track(
                frame,
                conf=0.5,
                persist=True,
                tracker='botsort.yaml',
                verbose=False
            )[0]
        except Exception as e:
            print(f"\r⚠ Frame {frame_count}: Tracker error - {type(e).__name__}", end='', flush=True)
            # Create blank extended frame
            extended_frame = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
            extended_frame[:, :frame_width] = frame
            out.write(extended_frame)
            continue

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
                class_id = class_ids[i]
                class_name = model.names[class_id]

                tracked_objects.append([x1, y1, x2, y2, obj_id])
                track_classes[obj_id] = class_name

        tracked_objects = np.array(tracked_objects) if tracked_objects else np.array([])

        # Track metrics
        current_track_ids = set()
        if len(tracked_objects) > 0:
            current_track_ids = set([int(obj_id) for obj_id in tracked_objects[:, 4]])

        total_detections += len(tracked_objects)

        # Update track history
        for track_id in current_track_ids:
            if track_id not in previous_tracks:
                track_lifespans[track_id] = 1
                track_history[track_id] = [frame_count]
            if track_id in track_history:
                track_history[track_id].append(frame_count)
                track_lifespans[track_id] = track_lifespans.get(track_id, 0) + 1

        previous_tracks = current_track_ids.copy()

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

            # Diagonal line crossing detection using interpolation
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

        # Clean up trajectories
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

        # Draw diagonal counting line on frame
        line_points = np.array([
            [line_x_positions[0], line_y_positions[0]],
            [line_x_positions[1], line_y_positions[1]]
        ], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [line_points], isClosed=False, color=(147, 20, 255), thickness=3)

        # Create extended frame with panel
        extended_frame = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
        extended_frame[:, :frame_width] = frame

        panel_color = (40, 40, 40)
        extended_frame[:, frame_width:] = panel_color

        # Panel header
        cv2.putText(extended_frame, "VEHICLE COUNT", (frame_width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(extended_frame, "[BoT-SORT]", (frame_width + 10, 55),
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

        # Tracking stats
        y_pos += 80
        cv2.line(extended_frame, (frame_width + 10, y_pos),
                 (frame_width + 240, y_pos), (100, 100, 100), 1)
        y_pos += 25
        cv2.putText(extended_frame, "TRACKING STATS", (frame_width + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 30
        cv2.putText(extended_frame, f"Active: {len(current_track_ids)}",
                   (frame_width + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y_pos += 25
        cv2.putText(extended_frame, f"Total: {len(track_lifespans)}",
                   (frame_width + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        # Frame counter
        cv2.putText(extended_frame, f"Frame: {frame_count}/{total_frames}",
                   (frame_width + 15, frame_height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # FPS
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        current_fps = 1 / frame_processing_time if frame_processing_time > 0 else 0
        cv2.putText(extended_frame, f"FPS: {current_fps:.1f}",
                   (frame_width + 15, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        out.write(extended_frame)
        total_processing_time += frame_processing_time

        if frame_count % 30 == 0:
            progress_pct = (frame_count / total_frames) * 100
            print(f"\rProgress: {progress_pct:.1f}% | Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}", end='', flush=True)

    cap.release()
    out.release()

    # Calculate metrics
    end_time = time.time()
    total_runtime = end_time - start_time
    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0

    identity_switches = 0
    for track_id, history in track_history.items():
        if len(history) > 1:
            for i in range(1, len(history)):
                if history[i] - history[i-1] > 5:
                    identity_switches += 1

    total_tracks = len(track_lifespans)
    total_count = sum(vehicle_counts.values())

    # Print summary
    print(f"\n\n✓ Output video: {output_video}")
    print(f"  Frames: {frame_count}/{total_frames}")
    print(f"  Runtime: {total_runtime:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Total Tracks: {total_tracks}")
    print(f"  Identity Switches: {identity_switches}")
    print(f"  Vehicle Counts: {total_count}")

    # Calculate additional tracking metrics
    avg_track_length = sum(track_lifespans.values()) / len(track_lifespans) if track_lifespans else 0
    detections_per_frame = total_detections / frame_count if frame_count > 0 else 0

    # Calculate mostly tracked (>80% of their lifespan) and mostly lost (<20% of their lifespan)
    # Based on track persistence relative to video length
    mostly_tracked = sum(1 for lifespan in track_lifespans.values() if lifespan > frame_count * 0.8)
    mostly_lost = sum(1 for lifespan in track_lifespans.values() if lifespan < frame_count * 0.2)

    # Save metrics to file
    metrics_file = os.path.join(output_dir, f"{gate_name}_botsort_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"BoT-SORT Results - {model_name} - {gate_name}\n")
        f.write(f"{'='*70}\n\n")

        f.write(f"Model: {model_name}\n")
        f.write(f"Video: {os.path.basename(video_path)}\n")
        f.write(f"Gate: {gate_name}\n")
        f.write(f"Tracker: BoT-SORT (botsort.yaml)\n\n")

        f.write(f"{'='*70}\n")
        f.write(f"VIDEO INFORMATION\n")
        f.write(f"{'='*70}\n")
        f.write(f"Resolution: {frame_width}x{frame_height}\n")
        f.write(f"FPS (Video): {fps}\n")
        f.write(f"Total Frames: {total_frames}\n")
        f.write(f"Frames Processed: {frame_count}\n\n")

        f.write(f"{'='*70}\n")
        f.write(f"MULTIPLE OBJECT TRACKING (MOT) METRICS\n")
        f.write(f"{'='*70}\n")
        f.write(f"IDF1 (Identity F1 Score): N/A (requires ground truth)\n")
        f.write(f"MOTA (MOT Accuracy): N/A (requires ground truth)\n")
        f.write(f"IDS (Identity Switches): {identity_switches}\n")
        f.write(f"MT (Mostly Tracked): {mostly_tracked} tracks (>{frame_count*0.8:.0f} frames)\n")
        f.write(f"ML (Mostly Lost): {mostly_lost} tracks (<{frame_count*0.2:.0f} frames)\n")
        f.write(f"FPS (Processing): {avg_fps:.2f}\n\n")

        f.write(f"{'='*70}\n")
        f.write(f"TRACKING STATISTICS\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total Runtime: {total_runtime:.2f}s\n")
        f.write(f"Processing Time: {total_processing_time:.2f}s\n")
        f.write(f"Total Unique Tracks: {total_tracks}\n")
        f.write(f"Total Detections: {total_detections}\n")
        f.write(f"Average Track Length: {avg_track_length:.2f} frames\n")
        f.write(f"Detections Per Frame: {detections_per_frame:.2f}\n\n")

        f.write(f"{'='*70}\n")
        f.write(f"VEHICLE COUNTS\n")
        f.write(f"{'='*70}\n")
        for vehicle_type, count in vehicle_counts.items():
            f.write(f"{vehicle_type.upper()}: {count}\n")
        f.write(f"\nTOTAL VEHICLES COUNTED: {total_count}\n\n")

        f.write(f"{'='*70}\n")
        f.write(f"COUNTING LINE CONFIGURATION\n")
        f.write(f"{'='*70}\n")
        f.write(f"X positions: {line_x_positions}\n")
        f.write(f"Y positions: {line_y_positions}\n")
        f.write(f"Fractions: {gate_config['y_positions']}\n\n")

        f.write(f"{'='*70}\n")
        f.write(f"OUTPUT FILES\n")
        f.write(f"{'='*70}\n")
        f.write(f"Video: {output_video}\n")
        f.write(f"Metrics: {metrics_file}\n")

    print(f"✓ Metrics saved: {metrics_file}")

    return {
        'model_name': model_name,
        'gate_name': gate_name,
        'total_count': total_count,
        'vehicle_counts': vehicle_counts,
        'total_tracks': total_tracks,
        'identity_switches': identity_switches,
        'avg_fps': avg_fps,
        'frames_processed': frame_count
    }


def main():
    """Main batch processing function"""

    print("\n" + "="*70)
    print("DCNv2 Models - BoT-SORT Batch Processing")
    print("="*70)
    print("\nTracker: BoT-SORT (botsort.yaml)")
    print("  - Advanced multi-object tracker")
    print("  - Motion + Appearance features (ReID)")
    print("  - Camera motion compensation (CMC)")
    print("  - Better for occlusions and crowded scenes\n")

    # Configuration - matches run_all_dcnv2_batch.py structure
    MODEL_DIR = Path("/media/mydrive/GitHub/ultralytics/modified_model")
    VIDEO_DIR = Path("/media/mydrive/GitHub/ultralytics/videos")
    OUTPUT_BASE_DIR = "/media/mydrive/GitHub/ultralytics/tracking/inference_results_botsort"

    # Get all DCNv2 model files from modified_model directory
    model_files = sorted(MODEL_DIR.glob("DCNv2-*.pt"))
    # Exclude _fixed versions
    model_files = [m for m in model_files if '_fixed' not in m.name]

    if not model_files:
        print(f"✗ No DCNv2 model files found in {MODEL_DIR}")
        print(f"  Looking for: DCNv2-*.pt")
        sys.exit(1)

    # Create models list as (path, name) tuples
    models = [(str(model_path), model_path.stem) for model_path in model_files]

    print(f"Found {len(models)} DCNv2 model(s) in {MODEL_DIR}:")
    for model_path, model_name in models:
        print(f"  - {model_name}")
    print()

    # Get all video files from VIDEO_DIR
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
        video_files.extend(VIDEO_DIR.glob(ext))

    # Filter out directories and sort
    video_files = [f for f in video_files if f.is_file()]
    video_files.sort()

    if not video_files:
        print(f"✗ No video files found in {VIDEO_DIR}")
        sys.exit(1)

    print("="*70)
    print(f"Found {len(video_files)} video(s) to process:")
    for vf in video_files:
        gate_name = vf.stem
        gate_config = GATE_COUNTING_LINES.get(gate_name, GATE_COUNTING_LINES['default'])
        print(f"  - {vf.name} | Line: {gate_config['y_positions']}")
    print("="*70)

    # Track all results
    all_results = []

    # Process each video with each model
    total_runs = len(video_files) * len(models)
    current_run = 0

    for video_path in video_files:
        # Extract gate name from filename (remove extension)
        gate_name = video_path.stem

        for model_path, model_name in models:
            current_run += 1
            print(f"\n{'='*70}")
            print(f"RUN {current_run}/{total_runs} - BoT-SORT")
            print(f"{'='*70}")

            result = run_model(model_path, model_name, str(video_path), gate_name, OUTPUT_BASE_DIR)
            if result:
                all_results.append(result)

    # Print final summary
    print("\n" + "="*70)
    print("BOT-SORT BATCH PROCESSING COMPLETED!")
    print("="*70)
    print(f"\nProcessed {len(video_files)} video(s) with {len(models)} model(s)")
    print(f"Total runs: {len(all_results)}/{total_runs}")
    print(f"\nResults saved to: {OUTPUT_BASE_DIR}")
    print("\nSummary by Model:")
    for model_path, model_name in models:
        model_results = [r for r in all_results if r['model_name'] == model_name]
        if model_results:
            print(f"\n  {model_name}:")
            for result in model_results:
                print(f"    - {result['gate_name']}: {result['total_count']} vehicles, "
                      f"{result['total_tracks']} tracks, {result['avg_fps']:.2f} fps")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
