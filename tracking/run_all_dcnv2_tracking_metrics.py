#!/usr/bin/env python3
"""
Run ByteTrack tracking metrics evaluation on all DCNv2 models
Outputs only core MOT metrics: MOTA, IDF1, IDSW, MT, ML

No video output - metrics only for fast evaluation
"""

import sys
import os
import warnings
from pathlib import Path
import time

warnings.filterwarnings('ignore')

sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def run_tracking_metrics(model_path, model_name, video_path, gate_name, output_base_dir):
    """Run ByteTrack and calculate core tracking metrics"""
    print("\n" + "="*70)
    print(f"Processing: {model_name} | Video: {gate_name}")
    print("="*70)

    output_dir = os.path.join(output_base_dir, model_name, gate_name)
    os.makedirs(output_dir, exist_ok=True)

    metrics_file = os.path.join(output_dir, f"{gate_name}_tracking_metrics.txt")

    # Load model
    print(f"Loading model...")
    try:
        model = YOLO(model_path)
        print("Model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"Video: {total_frames} frames @ {fps}fps")

    # Tracking state
    track_history = {}        # track_id -> list of frame numbers
    track_first_frame = {}    # track_id -> first frame seen
    track_last_frame = {}     # track_id -> last frame seen
    track_last_box = {}       # track_id -> last known bounding box [x1,y1,x2,y2]

    # For IDSW detection
    lost_tracks = {}          # track_id -> (last_frame, last_box) for recently lost tracks
    identity_switches = 0
    lost_track_window = 30    # frames to consider a track as "recently lost"
    iou_threshold = 0.3       # IoU threshold to consider same object

    prev_track_ids = set()
    frame_count = 0
    total_detections = 0
    start_time = time.time()

    print("Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
        except Exception:
            continue

        current_track_ids = set()
        current_boxes = {}

        # Extract track IDs and boxes
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)

            for i, track_id in enumerate(track_ids):
                track_id = int(track_id)
                box = boxes[i]
                current_track_ids.add(track_id)
                current_boxes[track_id] = box

                if track_id not in track_history:
                    track_history[track_id] = []
                    track_first_frame[track_id] = frame_count

                    # Check if this new track matches a recently lost track (IDSW!)
                    for lost_id, (lost_frame, lost_box) in list(lost_tracks.items()):
                        if frame_count - lost_frame <= lost_track_window:
                            iou = compute_iou(box, lost_box)
                            if iou > iou_threshold:
                                identity_switches += 1
                                del lost_tracks[lost_id]
                                break

                track_history[track_id].append(frame_count)
                track_last_frame[track_id] = frame_count
                track_last_box[track_id] = box

            total_detections += len(track_ids)

        # Detect lost tracks (were present last frame, not present now)
        for track_id in prev_track_ids - current_track_ids:
            if track_id in track_last_box:
                lost_tracks[track_id] = (frame_count, track_last_box[track_id])

        # Clean up old lost tracks
        lost_tracks = {k: v for k, v in lost_tracks.items()
                      if frame_count - v[0] <= lost_track_window}

        prev_track_ids = current_track_ids

        # Progress
        if frame_count % 100 == 0:
            pct = (frame_count / total_frames) * 100
            print(f"\rProgress: {pct:.1f}% ({frame_count}/{total_frames})", end='', flush=True)

    cap.release()

    end_time = time.time()
    runtime = end_time - start_time
    processing_fps = frame_count / runtime if runtime > 0 else 0

    # Calculate metrics
    total_tracks = len(track_history)

    # Also count track fragmentations (same ID with gaps - indicates tracker re-ID'd same object)
    fragmentations = 0
    for track_id, frames in track_history.items():
        if len(frames) > 1:
            sorted_frames = sorted(frames)
            for i in range(1, len(sorted_frames)):
                if sorted_frames[i] - sorted_frames[i-1] > 5:
                    fragmentations += 1

    # MT (Mostly Tracked) and ML (Mostly Lost)
    # MT: track present for >80% of its lifespan (first to last frame)
    # ML: track present for <20% of its lifespan
    mostly_tracked = 0
    mostly_lost = 0

    for track_id in track_history:
        lifespan = track_last_frame[track_id] - track_first_frame[track_id] + 1
        frames_present = len(track_history[track_id])

        if lifespan > 0:
            coverage = frames_present / lifespan
            if coverage >= 0.8:
                mostly_tracked += 1
            elif coverage <= 0.2:
                mostly_lost += 1

    mt_ratio = mostly_tracked / total_tracks if total_tracks > 0 else 0
    ml_ratio = mostly_lost / total_tracks if total_tracks > 0 else 0

    # MOTA (approximated)
    # MOTA = 1 - (FN + FP + IDSW) / GT
    # Without ground truth, we use IDSW + fragmentations as error proxy
    total_track_frames = sum(len(frames) for frames in track_history.values())
    total_errors = identity_switches + fragmentations
    mota = max(0, min(1, 1 - (total_errors / max(1, total_track_frames))))

    # IDF1 (approximated)
    # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    # IDFP ≈ identity switches (wrong ID assigned)
    # IDFN ≈ fragmentations (ID lost)
    id_true_positives = total_track_frames - total_errors
    id_false_positives = identity_switches
    id_false_negatives = fragmentations

    idf1 = (2 * id_true_positives) / max(1, 2 * id_true_positives + id_false_positives + id_false_negatives)

    # Print results
    print(f"\n\n{'='*50}")
    print(f"TRACKING METRICS - {model_name} - {gate_name}")
    print(f"{'='*50}")
    print(f"IDF1:  {idf1:.4f}")
    print(f"MOTA:  {mota:.4f}")
    print(f"IDSW:  {identity_switches}")
    print(f"FRAG:  {fragmentations}")
    print(f"MT:    {mt_ratio:.4f} ({mostly_tracked}/{total_tracks})")
    print(f"ML:    {ml_ratio:.4f} ({mostly_lost}/{total_tracks})")
    print(f"FPS:   {processing_fps:.2f}")
    print(f"{'='*50}")

    # Save metrics
    metrics_content = f"""Tracking Metrics Report
{'='*50}
Model: {model_name}
Video: {gate_name}
Tracker: ByteTrack
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

CORE MOT METRICS
{'='*50}
IDF1:  {idf1:.4f}
MOTA:  {mota:.4f}
IDSW:  {identity_switches}
FRAG:  {fragmentations}
MT:    {mt_ratio:.4f} ({mostly_tracked}/{total_tracks})
ML:    {ml_ratio:.4f} ({mostly_lost}/{total_tracks})
FPS:   {processing_fps:.2f}

STATISTICS
{'='*50}
Total Frames:  {frame_count}
Total Tracks:  {total_tracks}
Runtime:       {runtime:.2f}s
"""

    with open(metrics_file, 'w') as f:
        f.write(metrics_content)

    print(f"Saved: {metrics_file}")

    return {
        'model': model_name,
        'gate': gate_name,
        'idf1': idf1,
        'mota': mota,
        'idsw': identity_switches,
        'frag': fragmentations,
        'mt': mt_ratio,
        'ml': ml_ratio,
        'fps': processing_fps,
        'total_tracks': total_tracks
    }


def main():
    print("="*70)
    print("DCNv2 TRACKING METRICS EVALUATION")
    print("="*70)

    models = {
        'DCNv2-Full': r'/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Full.pt',
        'DCNv2-FPN': r'/media/mydrive/GitHub/ultralytics/tracking/dcnv2-yolov8-neck-fpn.pt',
        'DCNv2-Pan': r'/media/mydrive/GitHub/ultralytics/tracking/dcvn2-yolov8-neck-pan-best.pt',
        'DCNv2-Liu': r'/media/mydrive/GitHub/ultralytics/tracking/dcnv2-yolov8n-liu-best.pt',
    }

    videos = {
        'Gate3_Oct7': '/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20251007_075715.avi',
        'Gate3_Apr3': '/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20250403_154426.avi',
        'Gate3_Feb20': '/home/migui/Downloads/GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_20250220_075715.avi',
    }

    output_base_dir = 'tracking_metrics_dcnv2'

    available_videos = {k: v for k, v in videos.items() if os.path.exists(v)}
    print(f"\nVideos: {len(available_videos)}/{len(videos)}")
    print(f"Models: {len(models)}\n")

    results = []

    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: not found")
            continue

        for gate_name, video_path in available_videos.items():
            result = run_tracking_metrics(model_path, model_name, video_path, gate_name, output_base_dir)
            if result:
                results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if results:
        results.sort(key=lambda x: x['idf1'], reverse=True)

        print(f"\n{'Model':<15} {'Gate':<12} {'IDF1':<7} {'MOTA':<7} {'IDSW':<5} {'FRAG':<5} {'MT':<6} {'ML':<6} {'FPS':<5}")
        print("-"*75)
        for r in results:
            print(f"{r['model']:<15} {r['gate']:<12} {r['idf1']:<7.4f} {r['mota']:<7.4f} {r['idsw']:<5} {r['frag']:<5} {r['mt']:<6.3f} {r['ml']:<6.3f} {r['fps']:<5.1f}")

        # Save summary CSV
        summary_file = os.path.join(output_base_dir, 'summary.csv')
        os.makedirs(output_base_dir, exist_ok=True)
        with open(summary_file, 'w') as f:
            f.write("Model,Gate,IDF1,MOTA,IDSW,FRAG,MT,ML,FPS,Tracks\n")
            for r in results:
                f.write(f"{r['model']},{r['gate']},{r['idf1']:.4f},{r['mota']:.4f},{r['idsw']},{r['frag']},{r['mt']:.4f},{r['ml']:.4f},{r['fps']:.2f},{r['total_tracks']}\n")
        print(f"\nSummary: {summary_file}")


if __name__ == "__main__":
    main()
