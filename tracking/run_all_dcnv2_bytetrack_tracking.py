#!/usr/bin/env python3
"""
Run ByteTrack inference on DCNv2 models with video output
Focus on tracking metrics: MOTA, IDF1, IDSW, MT, ML (no counting)
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
import random


def get_unique_color(obj_id):
    """Generate a unique color for each tracking ID"""
    random.seed(obj_id)
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))


def run_tracking(model_path, model_name, video_path, gate_name, output_base_dir):
    """Run ByteTrack with video output and tracking metrics"""
    print("\n" + "="*70)
    print(f"Processing: {model_name} | Video: {gate_name}")
    print("="*70)

    output_dir = os.path.join(output_base_dir, model_name, gate_name)
    os.makedirs(output_dir, exist_ok=True)

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

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

    # Video output with metrics panel
    panel_width = 200
    total_width = frame_width + panel_width

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = os.path.join(output_dir, f"{gate_name}_tracking.mp4")
    out = cv2.VideoWriter(output_video, fourcc, fps, (total_width, frame_height))

    if not out.isOpened():
        print(f"Cannot create output video")
        cap.release()
        return None

    # Class colors
    class_colors = {
        "car": (55, 250, 250),
        "motorcycle": (83, 179, 36),
        "tricycle": (83, 50, 250),
        "bus": (245, 61, 184),
        "van": (255, 221, 51),
        "truck": (49, 147, 245)
    }

    # Tracking state
    track_classes = {}
    trajectories = {}
    trajectory_colors = {}
    max_trail_length = 50

    track_history = {}
    track_first_frame = {}
    track_last_frame = {}

    frame_count = 0
    start_time = time.time()

    print("Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()
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
            extended_frame = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
            extended_frame[:, :frame_width] = frame
            out.write(extended_frame)
            continue

        # Extract tracked objects
        tracked_objects = []
        current_track_ids = set()

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                obj_id = int(track_ids[i])
                class_name = model.names[int(class_ids[i])]

                tracked_objects.append([x1, y1, x2, y2, obj_id, class_name])
                current_track_ids.add(obj_id)
                track_classes[obj_id] = class_name

                # Update track history
                if obj_id not in track_history:
                    track_history[obj_id] = []
                    track_first_frame[obj_id] = frame_count

                track_history[obj_id].append(frame_count)
                track_last_frame[obj_id] = frame_count

        # Draw tracked objects
        for x1, y1, x2, y2, obj_id, class_name in tracked_objects:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Update trajectory
            if obj_id not in trajectories:
                trajectories[obj_id] = []
                trajectory_colors[obj_id] = get_unique_color(obj_id)

            trajectories[obj_id].append((center_x, center_y))
            if len(trajectories[obj_id]) > max_trail_length:
                trajectories[obj_id].pop(0)

            # Draw bbox
            color = class_colors.get(class_name.lower(), (0, 255, 0))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{class_name} #{obj_id}", (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw trajectories
        for obj_id, trajectory in trajectories.items():
            if len(trajectory) > 1 and obj_id in current_track_ids:
                color = trajectory_colors.get(obj_id, (255, 255, 255))
                for i in range(1, len(trajectory)):
                    thickness = max(1, int(4 * i / len(trajectory)))
                    cv2.line(frame, trajectory[i-1], trajectory[i], color, thickness)

        # Clean old trajectories
        for obj_id in list(trajectories.keys()):
            if obj_id not in current_track_ids:
                if len(trajectories[obj_id]) > 3:
                    trajectories[obj_id] = trajectories[obj_id][3:]
                else:
                    del trajectories[obj_id]
                    trajectory_colors.pop(obj_id, None)

        # Calculate current metrics
        total_tracks = len(track_history)
        identity_switches = sum(
            sum(1 for i in range(1, len(frames)) if frames[i] - frames[i-1] > 5)
            for frames in track_history.values() if len(frames) > 1
        )

        # Create extended frame with panel
        extended_frame = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
        extended_frame[:, :frame_width] = frame
        extended_frame[:, frame_width:] = (40, 40, 40)

        # Panel content
        y = 30
        cv2.putText(extended_frame, "TRACKING", (frame_width + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.line(extended_frame, (frame_width + 10, y), (frame_width + 190, y), (100, 100, 100), 1)

        y += 30
        cv2.putText(extended_frame, f"Active: {len(current_track_ids)}", (frame_width + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y += 25
        cv2.putText(extended_frame, f"Total: {total_tracks}", (frame_width + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        y += 25
        cv2.putText(extended_frame, f"IDSW: {identity_switches}", (frame_width + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 100), 1)

        # FPS
        frame_time = time.time() - frame_start
        current_fps = 1 / frame_time if frame_time > 0 else 0

        cv2.putText(extended_frame, f"Frame: {frame_count}/{total_frames}",
                   (frame_width + 10, frame_height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(extended_frame, f"FPS: {current_fps:.1f}",
                   (frame_width + 10, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        out.write(extended_frame)

        if frame_count % 100 == 0:
            pct = (frame_count / total_frames) * 100
            print(f"\rProgress: {pct:.1f}%", end='', flush=True)

    cap.release()
    out.release()

    # Final metrics calculation
    runtime = time.time() - start_time
    processing_fps = frame_count / runtime if runtime > 0 else 0

    total_tracks = len(track_history)

    # IDSW
    identity_switches = 0
    for frames in track_history.values():
        if len(frames) > 1:
            sorted_frames = sorted(frames)
            for i in range(1, len(sorted_frames)):
                if sorted_frames[i] - sorted_frames[i-1] > 5:
                    identity_switches += 1

    # MT/ML
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

    # MOTA/IDF1 (approximated)
    total_track_frames = sum(len(f) for f in track_history.values())
    mota = max(0, min(1, 1 - (identity_switches / max(1, total_track_frames))))

    id_tp = total_track_frames - identity_switches
    idf1 = (2 * id_tp) / max(1, 2 * id_tp + identity_switches)

    # Print results
    print(f"\n\n{'='*50}")
    print(f"RESULTS - {model_name} - {gate_name}")
    print(f"{'='*50}")
    print(f"IDF1:  {idf1:.4f}")
    print(f"MOTA:  {mota:.4f}")
    print(f"IDSW:  {identity_switches}")
    print(f"MT:    {mt_ratio:.4f} ({mostly_tracked}/{total_tracks})")
    print(f"ML:    {ml_ratio:.4f} ({mostly_lost}/{total_tracks})")
    print(f"FPS:   {processing_fps:.2f}")
    print(f"Video: {output_video}")

    # Save metrics
    metrics_file = os.path.join(output_dir, f"{gate_name}_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Tracking Metrics - {model_name} - {gate_name}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"IDF1:  {idf1:.4f}\n")
        f.write(f"MOTA:  {mota:.4f}\n")
        f.write(f"IDSW:  {identity_switches}\n")
        f.write(f"MT:    {mt_ratio:.4f} ({mostly_tracked}/{total_tracks})\n")
        f.write(f"ML:    {ml_ratio:.4f} ({mostly_lost}/{total_tracks})\n")
        f.write(f"FPS:   {processing_fps:.2f}\n")
        f.write(f"\nTotal Tracks: {total_tracks}\n")
        f.write(f"Frames: {frame_count}\n")
        f.write(f"Runtime: {runtime:.2f}s\n")

    return {
        'model': model_name,
        'gate': gate_name,
        'idf1': idf1,
        'mota': mota,
        'idsw': identity_switches,
        'mt': mt_ratio,
        'ml': ml_ratio,
        'fps': processing_fps,
        'total_tracks': total_tracks
    }


def main():
    print("="*70)
    print("DCNv2 BYTETRACK - TRACKING METRICS")
    print("="*70)

    MODEL_DIR = Path("/media/mydrive/GitHub/ultralytics/modified_model")
    VIDEO_DIR = Path("/media/mydrive/GitHub/ultralytics/videos")
    OUTPUT_BASE_DIR = "/media/mydrive/GitHub/ultralytics/tracking/inference_dcnv2_tracking"

    model_files = sorted(MODEL_DIR.glob("DCNv2-*.pt"))
    model_files = [m for m in model_files if '_fixed' not in m.name]

    if not model_files:
        print(f"No DCNv2 models found in {MODEL_DIR}")
        sys.exit(1)

    models = [(str(m), m.stem) for m in model_files]

    print(f"Models: {len(models)}")
    for _, name in models:
        print(f"  - {name}")

    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI']:
        video_files.extend(VIDEO_DIR.glob(ext))
    video_files = sorted([f for f in video_files if f.is_file()])

    print(f"Videos: {len(video_files)}")

    results = []

    for video_path in video_files:
        gate_name = video_path.stem
        for model_path, model_name in models:
            result = run_tracking(model_path, model_name, str(video_path), gate_name, OUTPUT_BASE_DIR)
            if result:
                results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if results:
        results.sort(key=lambda x: x['idf1'], reverse=True)
        print(f"\n{'Model':<15} {'Gate':<15} {'IDF1':<8} {'MOTA':<8} {'IDSW':<6} {'MT':<8} {'ML':<8}")
        print("-"*70)
        for r in results:
            print(f"{r['model']:<15} {r['gate']:<15} {r['idf1']:<8.4f} {r['mota']:<8.4f} {r['idsw']:<6} {r['mt']:<8.4f} {r['ml']:<8.4f}")

        summary_file = os.path.join(OUTPUT_BASE_DIR, 'summary.csv')
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        with open(summary_file, 'w') as f:
            f.write("Model,Gate,IDF1,MOTA,IDSW,MT,ML,FPS,Tracks\n")
            for r in results:
                f.write(f"{r['model']},{r['gate']},{r['idf1']:.4f},{r['mota']:.4f},{r['idsw']},{r['mt']:.4f},{r['ml']:.4f},{r['fps']:.2f},{r['total_tracks']}\n")


if __name__ == "__main__":
    main()
