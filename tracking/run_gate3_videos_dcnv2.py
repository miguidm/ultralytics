#!/usr/bin/env python3
"""
Run DCNv2 Tracking on Gate3 Videos with Different Confidence Thresholds
Tests confidence levels to find optimal setting
"""

import os
import sys
import cv2
import time
from pathlib import Path
from collections import defaultdict

# Setup DCNv2 environment
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


def run_tracking(model_path, video_path, conf_threshold, output_dir, model_name):
    """Run tracking on a video with specific confidence threshold"""

    video_name = Path(video_path).stem

    print(f"\n{'='*90}")
    print(f"Model: {model_name} | Video: {video_name} | Confidence: {conf_threshold}")
    print(f"{'='*90}")

    # Create output directory
    conf_dir = f"conf_{conf_threshold:.1f}"
    output_subdir = os.path.join(output_dir, conf_dir, model_name)
    os.makedirs(output_subdir, exist_ok=True)

    predictions_file = os.path.join(output_subdir, f"{video_name}_predictions.txt")

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

    print(f"✓ Video loaded: {total_frames} frames @ {fps}fps")

    # Process video
    frame_count = 0
    total_detections = 0
    start_time = time.time()

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

            # Save predictions
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
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  Frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) | "
                      f"FPS: {current_fps:.1f} | Detections: {total_detections}")

    cap.release()

    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # Calculate statistics
    unique_tracks = set()
    detections_per_frame = defaultdict(int)

    with open(predictions_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            track_id = int(parts[1])
            unique_tracks.add(track_id)
            detections_per_frame[frame] += 1

    avg_det_per_frame = total_detections / frame_count if frame_count > 0 else 0

    print(f"\n✓ Complete!")
    print(f"  Total Detections: {total_detections}")
    print(f"  Unique Tracks: {len(unique_tracks)}")
    print(f"  Avg Detections/Frame: {avg_det_per_frame:.2f}")
    print(f"  Processing FPS: {avg_fps:.2f}")
    print(f"  Time: {elapsed_time:.1f}s")
    print(f"  Saved: {predictions_file}")

    return {
        'model': model_name,
        'video': video_name,
        'conf': conf_threshold,
        'frames': frame_count,
        'detections': total_detections,
        'unique_tracks': len(unique_tracks),
        'avg_det_per_frame': avg_det_per_frame,
        'fps': avg_fps,
        'elapsed': elapsed_time,
        'predictions_file': predictions_file
    }


def main():
    """Main function"""

    print("="*90)
    print("GATE3 DCNv2 TRACKING - CONFIDENCE THRESHOLD TESTING")
    print("="*90)

    # Videos to process
    videos = [
        "/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20251007_075715.avi",
        "/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20250403_154426.avi",
        "/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20250220_075715.avi"
    ]

    # DCNv2 models
    models = {
        'DCNv2-Full': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Full.pt',
        'DCNv2-FPN': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-FPN.pt',
        'DCNv2-Pan': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Pan.pt',
        'DCNv2-LIU': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-LIU.pt',
    }

    # Confidence thresholds to test
    confidence_levels = [0.5, 0.6, 0.7]

    output_dir = "gate3_new_videos_dcnv2_results"

    all_results = []

    # Process each combination
    for video in videos:
        if not os.path.exists(video):
            print(f"\n⚠ Warning: Video not found: {video}")
            continue

        video_short = Path(video).stem[:40]
        print(f"\n{'='*90}")
        print(f"Processing Video: {video_short}")
        print(f"{'='*90}")

        for model_name, model_path in models.items():
            if not os.path.exists(model_path):
                print(f"\n⚠ Warning: Model not found: {model_path}")
                continue

            for conf in confidence_levels:
                result = run_tracking(
                    model_path,
                    video,
                    conf,
                    output_dir,
                    model_name
                )

                if result:
                    all_results.append(result)

    # Summary
    if all_results:
        print("\n" + "="*90)
        print("SUMMARY - ALL RESULTS")
        print("="*90)

        # Group by confidence
        for conf in confidence_levels:
            conf_results = [r for r in all_results if r['conf'] == conf]

            if conf_results:
                print(f"\n{'='*90}")
                print(f"CONFIDENCE = {conf}")
                print(f"{'='*90}")
                print(f"{'Video':<40} {'Model':<15} {'Detections':<12} {'Tracks':<8} {'Det/Frame':<10}")
                print("-"*90)

                for r in conf_results:
                    video_short = r['video'][:38]
                    print(f"{video_short:<40} {r['model']:<15} {r['detections']:<12} "
                          f"{r['unique_tracks']:<8} {r['avg_det_per_frame']:<10.2f}")

        # Save summary
        summary_file = os.path.join(output_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write("Gate3 DCNv2 Tracking Results Summary\n")
            f.write("="*90 + "\n\n")

            for conf in confidence_levels:
                conf_results = [r for r in all_results if r['conf'] == conf]

                if conf_results:
                    f.write(f"\nCONFIDENCE = {conf}\n")
                    f.write("-"*90 + "\n")
                    f.write(f"{'Video':<40} {'Model':<15} {'Detections':<12} {'Tracks':<8} {'Det/Frame':<10}\n")
                    f.write("-"*90 + "\n")

                    for r in conf_results:
                        video_short = r['video'][:38]
                        f.write(f"{video_short:<40} {r['model']:<15} {r['detections']:<12} "
                               f"{r['unique_tracks']:<8} {r['avg_det_per_frame']:<10.2f}\n")

        print(f"\n✓ Summary saved: {summary_file}")
        print("\nNext step: Run MOTA evaluation on these predictions")
        print("="*90 + "\n")


if __name__ == "__main__":
    main()
