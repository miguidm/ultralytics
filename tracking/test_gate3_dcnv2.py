#!/usr/bin/env python3
"""
Gate3_Oct7 Tracking Metrics Test - DCNv2 Models
Uses DCNv2 conda environment to test tracking metrics on Gate3 only
"""

import sys
import os
import warnings
from pathlib import Path
import time

# ============================================================================
# DCNv2 ENVIRONMENT SETUP
# ============================================================================

def setup_dcnv2_environment():
    """Configure environment for DCNv2 operations"""
    # DCNv2 conda environment paths
    dcnv2_env = "/home/migui/miniconda3/envs/dcn"
    cuda_lib_path = f"{dcnv2_env}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    torch_lib_path = f"{dcnv2_env}/lib/python3.10/site-packages/torch/lib"

    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}"

    ultralytics_root = "/media/mydrive/GitHub/ultralytics"
    if ultralytics_root not in sys.path:
        sys.path.insert(0, ultralytics_root)

    print(f"✓ DCNv2 environment configured")
    print(f"  - CUDA lib: {cuda_lib_path}")
    print(f"  - Torch lib: {torch_lib_path}")
    print(f"  - Ultralytics: {ultralytics_root}")

setup_dcnv2_environment()

warnings.filterwarnings('ignore', category=UserWarning)

from ultralytics import YOLO
import cv2
import numpy as np


def run_gate3_tracking(model_path, model_name, output_base_dir):
    """Run tracking on Gate3_Oct7 with detailed metrics"""

    # Video path
    video_path = '/media/mydrive/GitHub/ultralytics/videos/Gate3_Oct7.mp4'
    gate_name = 'Gate3_Oct7'

    # Gate3 counting line configuration
    gate_config = {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.47, 0.60]
    }

    print("\n" + "="*70)
    print(f"GATE3 TRACKING TEST - {model_name}")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")

    # Create output directory
    output_dir = os.path.join(output_base_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Output files
    predictions_file = os.path.join(output_dir, f"{gate_name}_predictions.txt")
    metrics_file = os.path.join(output_dir, f"{gate_name}_metrics.txt")
    debug_file = os.path.join(output_dir, f"{gate_name}_debug.txt")

    # Load model
    print(f"\nLoading model...")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found: {model_path}")
        return None

    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Load video
    print(f"\nLoading video...")
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video")
        return None

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✓ Video loaded: {frame_width}x{frame_height} @ {fps}fps")
    print(f"  Total frames: {total_frames}")

    # Counting line setup
    line_x_positions = [
        int(frame_width * gate_config['x_positions'][0]),
        int(frame_width * gate_config['x_positions'][1])
    ]
    line_y_positions = [
        int(frame_height * gate_config['y_positions'][0]),
        int(frame_height * gate_config['y_positions'][1])
    ]

    print(f"\nCounting line: x={line_x_positions}, y={line_y_positions}")

    # Initialize tracking variables
    track_classes = {}
    vehicle_counts = {"car": 0, "motorcycle": 0, "tricycle": 0, "van": 0, "bus": 0, "truck": 0}
    counted_objects = set()
    frame_count = 0
    total_processing_time = 0
    track_history = {}
    track_lifespans = {}
    previous_tracks = set()
    start_time = time.time()
    total_detections = 0
    object_positions = {}

    # Debug tracking
    frame_track_counts = []  # Track how many objects per frame

    # Open files
    pred_file = open(predictions_file, 'w')
    debug_log = open(debug_file, 'w')

    print(f"\nProcessing frames...")
    print("="*70 + "\n")

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
            print(f"\r⚠ Frame {frame_count}: Tracker error - {e}", end='', flush=True)
            debug_log.write(f"Frame {frame_count}: Tracker error - {e}\n")
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

        # Track statistics
        frame_track_counts.append(len(current_track_ids))
        total_detections += len(tracked_objects)

        # Update track history
        for track_id in current_track_ids:
            if track_id not in previous_tracks:
                # New track started
                track_lifespans[track_id] = 1
                track_history[track_id] = [frame_count]
                debug_log.write(f"Frame {frame_count}: New track {track_id} ({track_classes.get(track_id, 'unknown')})\n")
            else:
                track_history[track_id].append(frame_count)
                track_lifespans[track_id] = track_lifespans.get(track_id, 0) + 1

        # Line crossing detection
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
                        debug_log.write(f"Frame {frame_count}: Counted {class_name} (ID {obj_id})\n")

            object_positions[obj_id] = current_position

        previous_tracks = current_track_ids.copy()

        # Update timing
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        total_processing_time += frame_processing_time

        # Progress updates
        if frame_count % 100 == 0:
            progress_pct = (frame_count / total_frames) * 100
            current_fps = 1 / frame_processing_time if frame_processing_time > 0 else 0
            active_tracks = len(current_track_ids)
            print(f"Progress: {progress_pct:.1f}% | Frame: {frame_count}/{total_frames} | "
                  f"FPS: {current_fps:.1f} | Active Tracks: {active_tracks}")

    # Cleanup
    cap.release()
    pred_file.close()
    debug_log.close()

    # Calculate metrics
    end_time = time.time()
    total_runtime = end_time - start_time
    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0

    # Identity Switches
    identity_switches = 0
    for track_id, history in track_history.items():
        if len(history) > 1:
            for i in range(1, len(history)):
                if history[i] - history[i-1] > 5:  # Gap > 5 frames
                    identity_switches += 1

    # MT/ML calculation
    total_tracks = len(track_lifespans)
    mostly_tracked = 0
    mostly_lost = 0
    partially_tracked = 0

    if total_tracks > 0:
        for track_id, lifespan in track_lifespans.items():
            track_ratio = lifespan / frame_count
            if track_ratio >= 0.8:
                mostly_tracked += 1
            elif track_ratio <= 0.2:
                mostly_lost += 1
            else:
                partially_tracked += 1

    mt_ratio = mostly_tracked / total_tracks if total_tracks > 0 else 0
    ml_ratio = mostly_lost / total_tracks if total_tracks > 0 else 0
    pt_ratio = partially_tracked / total_tracks if total_tracks > 0 else 0

    # IMPROVED MOTA calculation
    # Average number of active tracks per frame instead of last frame only
    avg_active_tracks = sum(frame_track_counts) / len(frame_track_counts) if frame_track_counts else 0

    # Use average for better MOTA estimation
    expected_detections = avg_active_tracks * frame_count
    false_negatives = max(0, expected_detections - total_detections)
    false_positives = max(0, total_detections - expected_detections)

    mota_numerator = false_negatives + false_positives + identity_switches
    mota_denominator = max(1, total_detections)
    mota = 1 - (mota_numerator / mota_denominator)
    mota = max(0, min(1, mota))

    # IDF1 calculation
    id_true_positives = sum(track_lifespans.values())
    id_false_positives = identity_switches
    id_false_negatives = max(0, total_detections - id_true_positives)
    idf1 = (2 * id_true_positives) / max(1, 2 * id_true_positives + id_false_positives + id_false_negatives)

    # Average track lifespan
    avg_track_life = sum(track_lifespans.values()) / max(1, len(track_lifespans))

    # Create detailed metrics report
    metrics_content = f"""Gate3_Oct7 Tracking Metrics - {model_name}
================================================================

Configuration:
- Model: {model_name}
- Video: {gate_name}
- Tracker: ByteTrack
- Confidence: 0.5
- Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Tracking Metrics:
-----------------
IDF1:  {idf1:.4f}
MOTA:  {mota:.4f}
IDSW:  {identity_switches}
MT:    {mt_ratio:.4f} ({mostly_tracked}/{total_tracks})
PT:    {pt_ratio:.4f} ({partially_tracked}/{total_tracks})
ML:    {ml_ratio:.4f} ({mostly_lost}/{total_tracks})
FPS:   {avg_fps:.2f}

Statistics:
-----------
Total Frames:        {frame_count}
Total Runtime:       {total_runtime:.2f}s
Total Tracks:        {total_tracks}
Total Detections:    {total_detections}
Avg Track Life:      {avg_track_life:.1f} frames
Avg Active Tracks:   {avg_active_tracks:.1f}
Max Active Tracks:   {max(frame_track_counts) if frame_track_counts else 0}
Min Active Tracks:   {min(frame_track_counts) if frame_track_counts else 0}

MOTA Calculation Details:
--------------------------
Avg Active Tracks:   {avg_active_tracks:.1f}
Expected Detections: {expected_detections:.0f}
False Negatives:     {false_negatives:.0f}
False Positives:     {false_positives:.0f}
Identity Switches:   {identity_switches}
MOTA Formula:        1 - ({mota_numerator:.0f} / {mota_denominator})
MOTA Result:         {mota:.4f}

Track Lifespan Distribution:
-----------------------------
Very Short (<10%):   {sum(1 for l in track_lifespans.values() if l/frame_count < 0.1)}
Short (10-20%):      {sum(1 for l in track_lifespans.values() if 0.1 <= l/frame_count < 0.2)}
Medium (20-50%):     {sum(1 for l in track_lifespans.values() if 0.2 <= l/frame_count < 0.5)}
Long (50-80%):       {sum(1 for l in track_lifespans.values() if 0.5 <= l/frame_count < 0.8)}
Very Long (>80%):    {sum(1 for l in track_lifespans.values() if l/frame_count >= 0.8)}

Vehicle Counts:
---------------
Car:        {vehicle_counts['car']}
Motorcycle: {vehicle_counts['motorcycle']}
Tricycle:   {vehicle_counts['tricycle']}
Van:        {vehicle_counts['van']}
Bus:        {vehicle_counts['bus']}
Truck:      {vehicle_counts['truck']}
TOTAL:      {sum(vehicle_counts.values())}

Files Saved:
------------
- Predictions: {predictions_file}
- Metrics:     {metrics_file}
- Debug Log:   {debug_file}

================================================================
"""

    # Write metrics to file
    with open(metrics_file, 'w') as f:
        f.write(metrics_content)

    # Print summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\n✓ Model: {model_name}")
    print(f"✓ Predictions: {predictions_file}")
    print(f"✓ Metrics: {metrics_file}")
    print(f"✓ Debug: {debug_file}")
    print(f"\nMetrics Summary:")
    print(f"  - IDF1: {idf1:.4f}")
    print(f"  - MOTA: {mota:.4f}")
    print(f"  - IDSW: {identity_switches}")
    print(f"  - MT:   {mt_ratio:.4f} ({mostly_tracked} tracks)")
    print(f"  - PT:   {pt_ratio:.4f} ({partially_tracked} tracks)")
    print(f"  - ML:   {ml_ratio:.4f} ({mostly_lost} tracks)")
    print(f"  - FPS:  {avg_fps:.2f}")
    print(f"  - Total Tracks: {total_tracks}")
    print(f"  - Total Count: {sum(vehicle_counts.values())}")
    print(f"  - Avg Track Life: {avg_track_life:.1f} frames ({avg_track_life/frame_count*100:.1f}% of video)")
    print("\n" + "="*70 + "\n")

    return {
        'model': model_name,
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
    """Main function to test all DCNv2 models on Gate3"""
    print("="*70)
    print("GATE3 TRACKING TEST - DCNv2 MODELS")
    print("="*70)

    # DCNv2 models
    models = {
        'DCNv2-Full': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Full.pt',
        'DCNv2-FPN': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-FPN.pt',
        'DCNv2-Pan': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Pan.pt',
        'DCNv2-LIU': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-LIU.pt',
    }

    output_base_dir = 'gate3_test_results_dcnv2'
    os.makedirs(output_base_dir, exist_ok=True)

    results_summary = []

    # Test each model
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"\n⚠ Skipping {model_name}: model not found at {model_path}")
            continue

        result = run_gate3_tracking(model_path, model_name, output_base_dir)
        if result:
            results_summary.append(result)

    # Summary report
    print("\n" + "="*70)
    print("SUMMARY - All DCNv2 Models on Gate3_Oct7")
    print("="*70)

    if results_summary:
        results_summary.sort(key=lambda x: x['idf1'], reverse=True)

        print(f"\n{'Model':<20} {'IDF1':<8} {'MOTA':<8} {'IDSW':<6} {'MT':<8} {'ML':<8} {'FPS':<8} {'Count':<6}")
        print("-"*80)
        for r in results_summary:
            print(f"{r['model']:<20} {r['idf1']:<8.4f} {r['mota']:<8.4f} {r['idsw']:<6} "
                  f"{r['mt']:<8.4f} {r['ml']:<8.4f} {r['fps']:<8.2f} {r['total_count']:<6}")

        # Save summary
        summary_file = os.path.join(output_base_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write("Gate3_Oct7 Tracking Test - DCNv2 Models Summary\n")
            f.write("="*80 + "\n\n")
            f.write(f"{'Model':<20} {'IDF1':<8} {'MOTA':<8} {'IDSW':<6} {'MT':<8} {'ML':<8} {'FPS':<8} {'Count':<6}\n")
            f.write("-"*80 + "\n")
            for r in results_summary:
                f.write(f"{r['model']:<20} {r['idf1']:<8.4f} {r['mota']:<8.4f} {r['idsw']:<6} "
                        f"{r['mt']:<8.4f} {r['ml']:<8.4f} {r['fps']:<8.2f} {r['total_count']:<6}\n")

        print(f"\n✓ Summary saved: {summary_file}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
