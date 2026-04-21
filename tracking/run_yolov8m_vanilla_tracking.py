#!/usr/bin/env python3
"""
Run YOLOv8m Vanilla Tracking on Gate3 Videos
Performs tracking and calculates metrics for the vanilla YOLOv8m model
"""

import os
import sys
import cv2
import time
from pathlib import Path
from collections import defaultdict

# Setup environment
def setup_environment():
    """Configure environment"""
    ultralytics_root = "/media/mydrive/GitHub/ultralytics"
    if ultralytics_root not in sys.path:
        sys.path.insert(0, ultralytics_root)

setup_environment()

from ultralytics import YOLO


def run_tracking(model_path, video_path, output_dir, conf_threshold=0.5):
    """Run tracking on a video and calculate metrics"""

    model_name = "YOLOv8m-Vanilla"
    video_name = Path(video_path).stem

    print(f"\n{'='*90}")
    print(f"Model: {model_name} | Video: {video_name}")
    print(f"Confidence: {conf_threshold}")
    print(f"{'='*90}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    predictions_file = os.path.join(output_dir, f"{video_name}_predictions.txt")
    metrics_file = os.path.join(output_dir, f"{video_name}_metrics.txt")

    # Load model
    print(f"Loading model...")
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Load video
    print(f"Loading video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✓ Video loaded: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

    # Initialize tracking variables
    frame_count = 0
    total_detections = 0
    track_history = {}
    track_lifespans = {}
    previous_tracks = {}
    track_classes = {}
    start_time = time.time()

    # Process video
    with open(predictions_file, 'w') as pred_file:
        print(f"Processing frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run tracking
            results = model.track(
                frame,
                conf=conf_threshold,
                persist=True,
                tracker='bytetrack.yaml',
                verbose=False
            )[0]

            # Extract tracked objects
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

                    w = x2 - x1
                    h = y2 - y1

                    # Save prediction in MOT format: frame,id,x,y,w,h,conf,class,-1,-1
                    pred_file.write(f"{frame_count},{obj_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{class_id},-1,-1\n")

                    current_track_ids.add(obj_id)
                    track_classes[obj_id] = class_name
                    total_detections += 1

            # Track management
            for track_id in current_track_ids:
                if track_id not in previous_tracks:
                    # New track started
                    track_lifespans[track_id] = 1
                    track_history[track_id] = [frame_count]
                else:
                    track_history[track_id].append(frame_count)
                    track_lifespans[track_id] = track_lifespans.get(track_id, 0) + 1

            previous_tracks = current_track_ids.copy()

            # Progress
            if frame_count % 1000 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  Frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) | "
                      f"FPS: {current_fps:.1f} | Detections: {total_detections}")

    cap.release()

    # Calculate metrics
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

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

    # MOTA calculation (simplified - without ground truth)
    false_negatives = max(0, total_detections - len(previous_tracks) * frame_count)
    false_positives = max(0, len(previous_tracks) * frame_count - total_detections)
    mota = 1 - (false_negatives + false_positives + identity_switches) / max(1, total_detections)
    mota = max(0, min(1, mota))

    # IDF1 calculation (simplified - without ground truth)
    id_true_positives = sum(track_lifespans.values())
    id_false_positives = identity_switches
    id_false_negatives = max(0, total_detections - id_true_positives)
    idf1 = (2 * id_true_positives) / max(1, 2 * id_true_positives + id_false_positives + id_false_negatives)

    # Average track lifespan
    avg_track_lifespan = sum(track_lifespans.values()) / max(1, len(track_lifespans))

    # Detections per frame
    avg_det_per_frame = total_detections / frame_count if frame_count > 0 else 0

    # Create metrics report
    metrics_content = f"""YOLOv8m Vanilla Tracking Metrics Report
================================================================

Model Configuration:
- Model: YOLOv8m-Vanilla
- Video: {video_name}
- Tracker: ByteTrack
- Confidence Threshold: {conf_threshold}
- Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Video Properties:
- Resolution: {frame_width}x{frame_height}
- FPS: {fps}
- Total Frames: {frame_count}

Tracking Metrics (Simplified - No Ground Truth):
-------------------------------------------------
1. IDF1 (Identity F1 Score): {idf1:.4f}
   - Measures identity preservation accuracy
   - Range: 0.0 (worst) to 1.0 (best)

2. MOTA (Multiple Object Tracking Accuracy): {mota:.4f}
   - Overall tracking accuracy measure
   - Range: 0.0 (worst) to 1.0 (best)

3. IDSW (Identity Switches): {identity_switches}
   - Number of identity switches detected
   - Lower is better (0 = no switches)

4. MT/ML (Mostly Tracked/Lost Ratios):
   - MT (Mostly Tracked): {mt_ratio:.4f} ({mostly_tracked}/{total_tracks} tracks)
   - ML (Mostly Lost): {ml_ratio:.4f} ({mostly_lost}/{total_tracks} tracks)

Performance Metrics:
-------------------
5. FPS (Frames Per Second): {avg_fps:.2f}
   - Average processing speed
   - Higher is better

Additional Statistics:
---------------------
- Total Detections: {total_detections}
- Total Tracks Created: {total_tracks}
- Average Detections/Frame: {avg_det_per_frame:.2f}
- Average Track Lifespan: {avg_track_lifespan:.1f} frames
- Total Processing Time: {elapsed_time:.2f} seconds

Output Files:
-------------
- Predictions: {predictions_file}
- Format: MOT (frame,id,x,y,w,h,conf,class,-1,-1)

================================================================
"""

    # Write metrics to file
    with open(metrics_file, 'w') as f:
        f.write(metrics_content)

    print(f"\n✓ Complete!")
    print(f"  Total Detections: {total_detections}")
    print(f"  Unique Tracks: {total_tracks}")
    print(f"  Avg Detections/Frame: {avg_det_per_frame:.2f}")
    print(f"  IDSW: {identity_switches}")
    print(f"  IDF1: {idf1:.4f}")
    print(f"  MOTA: {mota:.4f}")
    print(f"  Processing FPS: {avg_fps:.2f}")
    print(f"  Time: {elapsed_time:.1f}s")
    print(f"  Saved: {predictions_file}")
    print(f"  Saved: {metrics_file}")

    return {
        'video': video_name,
        'frames': frame_count,
        'detections': total_detections,
        'unique_tracks': total_tracks,
        'avg_det_per_frame': avg_det_per_frame,
        'identity_switches': identity_switches,
        'idf1': idf1,
        'mota': mota,
        'mt_ratio': mt_ratio,
        'ml_ratio': ml_ratio,
        'fps': avg_fps,
        'elapsed': elapsed_time,
        'predictions_file': predictions_file,
        'metrics_file': metrics_file
    }


def main():
    """Main function"""

    print("="*90)
    print("YOLOv8m VANILLA TRACKING - GATE3 VIDEOS")
    print("="*90)

    # Model path
    model_path = "/home/migui/Downloads/yolov8m-vanilla-20260211T133104Z-1-001/yolov8m-vanilla/weights/best.pt"

    # Videos to process
    videos = [
        "/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20251007_075715.avi",
        "/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20250403_154426.avi",
        "/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20250220_075715.avi"
    ]

    # Output directory
    output_dir = "yolov8m_vanilla_gate3_tracking_results"

    # Confidence threshold
    conf_threshold = 0.5

    # Check model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found: {model_path}")
        return

    print(f"\n✓ Model found: {model_path}")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Confidence threshold: {conf_threshold}")

    all_results = []

    # Process each video
    for video in videos:
        if not os.path.exists(video):
            print(f"\n⚠ Warning: Video not found: {video}")
            continue

        result = run_tracking(
            model_path,
            video,
            output_dir,
            conf_threshold
        )

        if result:
            all_results.append(result)

    # Summary
    if all_results:
        print("\n" + "="*90)
        print("SUMMARY - ALL RESULTS")
        print("="*90)

        summary_file = os.path.join(output_dir, "summary_report.txt")

        # Console output
        print(f"\n{'Video':<50} {'Frames':<8} {'Tracks':<8} {'Det/F':<8} {'IDSW':<6} {'IDF1':<8} {'MOTA':<8}")
        print("-"*90)

        for r in all_results:
            video_short = r['video'][:48]
            print(f"{video_short:<50} {r['frames']:<8} {r['unique_tracks']:<8} "
                  f"{r['avg_det_per_frame']:<8.2f} {r['identity_switches']:<6} "
                  f"{r['idf1']:<8.4f} {r['mota']:<8.4f}")

        # Calculate averages
        avg_idf1 = sum(r['idf1'] for r in all_results) / len(all_results)
        avg_mota = sum(r['mota'] for r in all_results) / len(all_results)
        avg_idsw = sum(r['identity_switches'] for r in all_results) / len(all_results)
        avg_tracks = sum(r['unique_tracks'] for r in all_results) / len(all_results)
        avg_fps = sum(r['fps'] for r in all_results) / len(all_results)

        print("\n" + "-"*90)
        print(f"{'AVERAGES:':<50} {'':<8} {avg_tracks:<8.1f} {'':<8} {avg_idsw:<6.1f} "
              f"{avg_idf1:<8.4f} {avg_mota:<8.4f}")
        print(f"\nAverage Processing FPS: {avg_fps:.2f}")

        # Save detailed summary
        with open(summary_file, 'w') as f:
            f.write("YOLOv8m Vanilla Tracking - Gate3 Videos Summary Report\n")
            f.write("="*90 + "\n\n")
            f.write(f"Model: YOLOv8m-Vanilla\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Tracker: ByteTrack\n")
            f.write(f"Confidence Threshold: {conf_threshold}\n")
            f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Videos Processed: {len(all_results)}\n\n")

            f.write("="*90 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*90 + "\n\n")

            for r in all_results:
                f.write(f"Video: {r['video']}\n")
                f.write("-"*90 + "\n")
                f.write(f"  Frames Processed: {r['frames']}\n")
                f.write(f"  Total Detections: {r['detections']}\n")
                f.write(f"  Unique Tracks: {r['unique_tracks']}\n")
                f.write(f"  Avg Detections/Frame: {r['avg_det_per_frame']:.2f}\n")
                f.write(f"  Identity Switches (IDSW): {r['identity_switches']}\n")
                f.write(f"  IDF1: {r['idf1']:.4f}\n")
                f.write(f"  MOTA: {r['mota']:.4f}\n")
                f.write(f"  MT Ratio: {r['mt_ratio']:.4f}\n")
                f.write(f"  ML Ratio: {r['ml_ratio']:.4f}\n")
                f.write(f"  Processing FPS: {r['fps']:.2f}\n")
                f.write(f"  Processing Time: {r['elapsed']:.2f}s\n")
                f.write(f"  Predictions: {r['predictions_file']}\n")
                f.write(f"  Metrics: {r['metrics_file']}\n")
                f.write("\n")

            f.write("="*90 + "\n")
            f.write("SUMMARY METRICS\n")
            f.write("="*90 + "\n\n")
            f.write(f"Average IDF1: {avg_idf1:.4f}\n")
            f.write(f"Average MOTA: {avg_mota:.4f}\n")
            f.write(f"Average IDSW: {avg_idsw:.1f}\n")
            f.write(f"Average Unique Tracks: {avg_tracks:.1f}\n")
            f.write(f"Average Processing FPS: {avg_fps:.2f}\n\n")

            f.write("="*90 + "\n")
            f.write("INTERPRETATION\n")
            f.write("="*90 + "\n\n")
            f.write("Metric Descriptions:\n")
            f.write("- IDF1: Identity F1 Score (0-1, higher is better)\n")
            f.write("  Measures how well the tracker maintains object identities\n\n")
            f.write("- MOTA: Multiple Object Tracking Accuracy (0-1, higher is better)\n")
            f.write("  Overall tracking accuracy combining detection and identity\n\n")
            f.write("- IDSW: Identity Switches (lower is better)\n")
            f.write("  Number of times tracked objects lost their identity\n\n")
            f.write("- MT: Mostly Tracked ratio (0-1, higher is better)\n")
            f.write("  Proportion of tracks present for 80%+ of their lifetime\n\n")
            f.write("- ML: Mostly Lost ratio (0-1, lower is better)\n")
            f.write("  Proportion of tracks present for only 20% or less\n\n")
            f.write("\nNote: These metrics are simplified estimates without ground truth.\n")
            f.write("For accurate evaluation, compare predictions against annotated ground truth.\n")

        print(f"\n✓ Summary saved: {summary_file}")
        print("="*90 + "\n")


if __name__ == "__main__":
    main()
