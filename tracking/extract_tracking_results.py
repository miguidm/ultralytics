#!/usr/bin/env python3
"""
Extract tracking results from ByteTrack inference videos and save in MOT format

Converts tracker output to MOT Challenge format:
frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence, class_id, visibility

Output: tracking_results/{model_name}/{gate_name}/tracker_output.txt
"""

import sys
import os
import warnings
from pathlib import Path
import cv2

warnings.filterwarnings('ignore')

sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import numpy as np


def extract_tracking_results(model_path, video_path, output_file, model_type='dcnv2'):
    """
    Extract tracking results from a video using the specified model

    Args:
        model_path: Path to the model file
        video_path: Path to the video file
        output_file: Path to save tracking results in MOT format
        model_type: 'dcnv2' or 'dcnv3' (affects confidence threshold)
    """
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"Model: {os.path.basename(model_path)}")

    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # Set confidence based on model type
    conf_threshold = 0.25 if model_type == 'dcnv3' else 0.5

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Open output file
    with open(output_file, 'w') as f:
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run tracking
            try:
                results = model.track(
                    frame,
                    conf=conf_threshold,
                    persist=True,
                    tracker='bytetrack.yaml',
                    verbose=False
                )[0]
            except Exception as e:
                print(f"Frame {frame_count}: Error - {e}")
                continue

            # Extract tracked objects
            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    track_id = track_ids[i]
                    confidence = confidences[i]
                    class_id = class_ids[i]

                    # Convert to MOT format: x, y, w, h
                    bbox_left = x1
                    bbox_top = y1
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    # Write to file
                    # Format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
                    f.write(f"{frame_count},{track_id},{bbox_left:.2f},{bbox_top:.2f},"
                           f"{bbox_width:.2f},{bbox_height:.2f},{confidence:.4f},{class_id},1\n")

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames", end='\r', flush=True)

    cap.release()
    print(f"\n✓ Saved tracking results to: {output_file}")
    return True


def main():
    """Extract tracking results for all models and videos"""

    print("="*70)
    print("Extract Tracking Results to MOT Format")
    print("="*70)

    # Configuration
    MODEL_DIR = Path("/media/mydrive/GitHub/ultralytics/modified_model")
    VIDEO_DIR = Path("/media/mydrive/GitHub/ultralytics/videos")
    OUTPUT_BASE_DIR = Path("/media/mydrive/GitHub/ultralytics/tracking/tracking_results")

    # Get DCNv2 models
    dcnv2_models = sorted(MODEL_DIR.glob("DCNv2-*.pt"))
    dcnv2_models = [m for m in dcnv2_models if '_fixed' not in m.name]

    # Get DCNv3 models from specific directories
    dcnv3_model_dirs = [
        "/home/migui/YOLO_outputs/100_dcnv3_yolov8m_fpn_second",
        "/home/migui/YOLO_outputs/100_dcnv3_yolov8m_full_second",
        "/home/migui/YOLO_outputs/100_dcnv3_yolov8m_liu_second",
        "/home/migui/YOLO_outputs/100_dcnv3_yolov8m_pan_second"
    ]

    dcnv3_models = []
    for model_dir in dcnv3_model_dirs:
        weights_file = Path(model_dir) / "weights" / "best.pt"
        if weights_file.exists():
            dcnv3_models.append(weights_file)
        else:
            print(f"Warning: DCNv3 model not found: {weights_file}")

    # Get all video files (exclude 15sec clips for now)
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(VIDEO_DIR.glob(ext))
    video_files = [v for v in video_files if v.is_file() and '15sec' not in v.name]
    video_files.sort()

    print(f"\nFound {len(dcnv2_models)} DCNv2 models")
    print(f"Found {len(dcnv3_models)} DCNv3 models")
    print(f"Found {len(video_files)} videos")

    total_runs = len(video_files) * (len(dcnv2_models) + len(dcnv3_models))
    current_run = 0

    # Process DCNv2 models
    for model_path in dcnv2_models:
        model_name = model_path.stem

        for video_path in video_files:
            gate_name = video_path.stem
            current_run += 1

            print(f"\n[{current_run}/{total_runs}] {model_name} - {gate_name}")

            output_file = OUTPUT_BASE_DIR / "dcnv2" / model_name / gate_name / "tracker_output.txt"
            extract_tracking_results(str(model_path), str(video_path), str(output_file), 'dcnv2')

    # Process DCNv3 models
    for model_path in dcnv3_models:
        # Extract model name from directory
        dir_name = model_path.parent.parent.name
        if 'fpn' in dir_name.lower():
            model_name = "DCNv3-FPN"
        elif 'full' in dir_name.lower():
            model_name = "DCNv3-Full"
        elif 'liu' in dir_name.lower():
            model_name = "DCNv3-Liu"
        elif 'pan' in dir_name.lower():
            model_name = "DCNv3-Pan"
        else:
            model_name = dir_name

        for video_path in video_files:
            gate_name = video_path.stem
            current_run += 1

            print(f"\n[{current_run}/{total_runs}] {model_name} - {gate_name}")

            output_file = OUTPUT_BASE_DIR / "dcnv3" / model_name / gate_name / "tracker_output.txt"
            extract_tracking_results(str(model_path), str(video_path), str(output_file), 'dcnv3')

    print("\n" + "="*70)
    print("Extraction Complete!")
    print(f"Results saved to: {OUTPUT_BASE_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
