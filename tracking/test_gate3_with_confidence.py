#!/usr/bin/env python3
"""
Test Gate3 Videos with Different Confidence Thresholds
Finds optimal confidence to improve MOTA by reducing false positives
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import time

# Setup environment
def setup_dcnv2_environment():
    """Configure environment for DCNv2"""
    cuda_lib_path = "/home/migui/miniconda3/envs/dcn/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    torch_lib_path = "/home/migui/miniconda3/envs/dcn/lib/python3.10/site-packages/torch/lib"

    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}"

    ultralytics_root = "/media/mydrive/GitHub/ultralytics"
    if ultralytics_root not in sys.path:
        sys.path.insert(0, ultralytics_root)

setup_dcnv2_environment()

from ultralytics import YOLO


def run_tracking_with_confidence(model_path, video_path, conf_threshold, output_dir, tracker='bytetrack.yaml'):
    """
    Run tracking with specific confidence threshold
    """

    video_name = Path(video_path).stem
    model_name = Path(model_path).stem

    print(f"\n{'='*80}")
    print(f"Testing: {model_name} on {video_name}")
    print(f"Confidence: {conf_threshold} | Tracker: {tracker}")
    print(f"{'='*80}")

    # Create output directory
    output_subdir = os.path.join(output_dir, f"conf_{conf_threshold}", model_name)
    os.makedirs(output_subdir, exist_ok=True)

    predictions_file = os.path.join(output_subdir, f"{video_name}_predictions.txt")
    summary_file = os.path.join(output_subdir, f"{video_name}_summary.txt")

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
        print(f"❌ Error: Cannot open video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15

    print(f"✓ Video: {total_frames} frames @ {fps}fps")

    # Initialize tracking
    frame_count = 0
    start_time = time.time()
    total_detections = 0

    # Open predictions file
    pred_file = open(predictions_file, 'w')

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
            tracker=tracker,
            verbose=False
        )[0]

        # Extract predictions
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

                w = x2 - x1
                h = y2 - y1

                # MOT format: frame,id,x,y,w,h,conf,class,-1,-1
                pred_file.write(f"{frame_count},{obj_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{class_id},-1,-1\n")
                total_detections += 1

        # Progress
        if frame_count % 1000 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"  {progress:.1f}% | Frame {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Detections: {total_detections}")

    cap.release()
    pred_file.close()

    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # Calculate statistics
    unique_tracks = set()
    with open(predictions_file, 'r') as f:
        for line in f:
            track_id = int(line.split(',')[1])
            unique_tracks.add(track_id)

    stats = {
        'model': model_name,
        'video': video_name,
        'conf': conf_threshold,
        'frames': frame_count,
        'detections': total_detections,
        'unique_tracks': len(unique_tracks),
        'avg_detections_per_frame': total_detections / frame_count if frame_count > 0 else 0,
        'avg_fps': avg_fps,
        'elapsed_time': elapsed_time
    }

    # Save summary
    with open(summary_file, 'w') as f:
        f.write(f"Tracking Summary\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"Confidence Threshold: {conf_threshold}\n")
        f.write(f"Tracker: {tracker}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Total Frames: {frame_count}\n")
        f.write(f"  Total Detections: {total_detections}\n")
        f.write(f"  Unique Tracks: {len(unique_tracks)}\n")
        f.write(f"  Avg Detections/Frame: {stats['avg_detections_per_frame']:.2f}\n")
        f.write(f"  Processing FPS: {avg_fps:.2f}\n")
        f.write(f"  Total Time: {elapsed_time:.1f}s\n\n")
        f.write(f"Predictions saved to: {predictions_file}\n")

    print(f"\n✓ Complete!")
    print(f"  Detections: {total_detections}")
    print(f"  Unique Tracks: {len(unique_tracks)}")
    print(f"  Avg Detections/Frame: {stats['avg_detections_per_frame']:.2f}")
    print(f"  Predictions: {predictions_file}")

    return stats


def main():
    """Main function"""

    print("="*80)
    print("GATE3 CONFIDENCE THRESHOLD TESTING")
    print("="*80)

    # Videos to test
    videos = [
        "/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20251007_075715.avi",
        "/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20250403_154426.avi",
        "/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20250220_075715.avi"
    ]

    # Model to test
    model_path = "/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Full.pt"

    # Confidence thresholds to test
    confidence_levels = [0.5, 0.6, 0.7]

    output_dir = "gate3_confidence_test_results"

    all_results = []

    for video in videos:
        if not os.path.exists(video):
            print(f"\n⚠ Warning: Video not found: {video}")
            continue

        for conf in confidence_levels:
            result = run_tracking_with_confidence(
                model_path,
                video,
                conf,
                output_dir
            )

            if result:
                all_results.append(result)

    # Summary comparison
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY COMPARISON")
        print("="*80)
        print(f"\n{'Video':<40} {'Conf':<6} {'Detections':<12} {'Tracks':<8} {'Det/Frame':<10}")
        print("-"*80)

        for r in all_results:
            video_short = r['video'][:38]
            print(f"{video_short:<40} {r['conf']:<6.2f} {r['detections']:<12} {r['unique_tracks']:<8} {r['avg_detections_per_frame']:<10.2f}")

        # Save comparison
        comparison_file = os.path.join(output_dir, "confidence_comparison.txt")
        with open(comparison_file, 'w') as f:
            f.write("Gate3 Confidence Threshold Comparison\n")
            f.write("="*80 + "\n\n")
            f.write(f"{'Video':<40} {'Conf':<6} {'Detections':<12} {'Tracks':<8} {'Det/Frame':<10}\n")
            f.write("-"*80 + "\n")

            for r in all_results:
                video_short = r['video'][:38]
                f.write(f"{video_short:<40} {r['conf']:<6.2f} {r['detections']:<12} {r['unique_tracks']:<8} {r['avg_detections_per_frame']:<10.2f}\n")

        print(f"\n✓ Comparison saved: {comparison_file}")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
