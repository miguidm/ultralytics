#!/usr/bin/env python3
"""
1-hour annotated video output for 6to6_g3.mp4.
Models: DCNv2-Full and Vanilla-YOLOv8m.
Counting line: Gate g29 [(0.0, 0.42), -> (0.55, 0.72)]

Usage (run in dcnv2 env — has both models):
  /home/migui/miniconda3/envs/dcnv2/bin/python run_video_g3_1hr.py
"""

import sys
import os
import warnings
import time
import random
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np

VIDEO_PATH  = '/media/mydrive/GitHub/ultralytics/videos/6to6_g3.mp4'
OUTPUT_DIR  = '/media/mydrive/GitHub/ultralytics/tracking/video_g3_1hr'
FPS_VIDEO   = 15
MAX_FRAMES  = 3600 * FPS_VIDEO          # 1 hour = 54 000 frames

# g29 counting line fractions
LX1F, LY1F = 0.0,  0.52
LX2F, LY2F = 1.0,  0.62

VEHICLE_CLASSES = ["car", "motorcycle", "tricycle", "van", "bus", "truck"]
CLASS_COLORS = {
    "car":        (55,  250, 250),
    "motorcycle": (83,  179, 36),
    "tricycle":   (83,  50,  250),
    "bus":        (245, 61,  184),
    "van":        (255, 221, 51),
    "truck":      (49,  147, 245),
}

MODELS = {
    'DCNv2-Full':      '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Full.pt',
    'Vanilla-YOLOv8m': '/home/migui/Downloads/yolov8m-vanilla-20260211T133104Z-1-001/yolov8m-vanilla/weights/best.pt',
}


def get_color(obj_id):
    random.seed(obj_id)
    return (random.randint(80, 255), random.randint(80, 255), random.randint(80, 255))


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run(model_path, model_name):
    print(f"\n{'='*65}")
    print(f"  {model_name}  |  6to6_g3.mp4  |  first 1 hour")
    print(f"{'='*65}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        model = YOLO(model_path)
        print("  Model loaded")
    except Exception as e:
        print(f"  Error loading model: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("  Cannot open video")
        return

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = int(cap.get(cv2.CAP_PROP_FPS)) or FPS_VIDEO
    print(f"  Video: {frame_width}x{frame_height} @ {fps}fps  ({total_frames} frames total)")

    panel_w   = 260
    out_w     = frame_width + panel_w
    out_path  = os.path.join(OUTPUT_DIR, f"{model_name}_g3_1hr.mp4")
    fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
    writer    = cv2.VideoWriter(out_path, fourcc, fps, (out_w, frame_height))
    if not writer.isOpened():
        print("  Cannot create output video")
        cap.release()
        return
    print(f"  Output: {out_path}")

    # Pixel coords for counting line
    lx1 = int(LX1F * frame_width);  ly1 = int(LY1F * frame_height)
    lx2 = int(LX2F * frame_width);  ly2 = int(LY2F * frame_height)

    vehicle_counts  = {c: 0 for c in VEHICLE_CLASSES}
    counted_objects = set()
    object_positions = {}
    trajectories    = {}
    traj_colors     = {}
    max_trail       = 50

    frame_count = 0
    start_time  = time.time()

    consecutive_failures = 0

    print("  Processing...")

    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= 10:
                break
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            seek = cur + 100
            if seek >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
            frame_count = seek
            continue
        consecutive_failures = 0
        frame_count += 1

        video_time = (frame_count - 1) / fps

        try:
            results = model.track(
                frame,
                conf=0.5,
                persist=True,
                tracker='bytetrack.yaml',
                verbose=False
            )[0]
        except Exception:
            writer.write(np.zeros((frame_height, out_w, 3), dtype=np.uint8))
            continue

        current_ids = set()

        if results.boxes is not None and results.boxes.id is not None:
            boxes     = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for i, tid in enumerate(track_ids):
                tid        = int(tid)
                box        = boxes[i]
                class_name = model.names[int(class_ids[i])]
                current_ids.add(tid)

                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2

                # Trajectory
                if tid not in trajectories:
                    trajectories[tid] = []
                    traj_colors[tid]  = get_color(tid)
                trajectories[tid].append((int(cx), int(cy)))
                if len(trajectories[tid]) > max_trail:
                    trajectories[tid].pop(0)

                # Counting line crossing
                line_y = np.interp(cx, [lx1, lx2], [ly1, ly2])
                side   = 'above' if cy < line_y else 'below'
                if tid in object_positions and object_positions[tid] != side:
                    if tid not in counted_objects and class_name in vehicle_counts:
                        vehicle_counts[class_name] += 1
                        counted_objects.add(tid)
                object_positions[tid] = side

                # Draw bbox
                color = CLASS_COLORS.get(class_name.lower(), (0, 255, 0))
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame, f"{class_name} #{tid}",
                            (int(box[0]), int(box[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw trajectories
        for tid, trail in trajectories.items():
            if len(trail) > 1 and tid in current_ids:
                col = traj_colors[tid]
                for j in range(1, len(trail)):
                    thick = max(1, int(4 * j / len(trail)))
                    cv2.line(frame, trail[j-1], trail[j], col, thick)

        # Fade out-of-view trajectories
        for tid in list(trajectories.keys()):
            if tid not in current_ids:
                if len(trajectories[tid]) > 4:
                    trajectories[tid] = trajectories[tid][4:]
                else:
                    del trajectories[tid]
                    traj_colors.pop(tid, None)

        # Counting line
        cv2.line(frame, (lx1, ly1), (lx2, ly2), (147, 20, 255), 3)

        # Side panel
        canvas = np.zeros((frame_height, out_w, 3), dtype=np.uint8)
        canvas[:, :frame_width] = frame
        canvas[:, frame_width:] = (35, 35, 35)

        yp = 30
        cv2.putText(canvas, model_name, (frame_width + 10, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        yp += 25
        cv2.putText(canvas, format_time(video_time), (frame_width + 10, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        yp += 20
        cv2.line(canvas, (frame_width + 10, yp),
                 (frame_width + panel_w - 10, yp), (80, 80, 80), 1)

        yp += 20
        cv2.putText(canvas, f"Active: {len(current_ids)}", (frame_width + 10, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        yp += 20
        cv2.putText(canvas, f"Counted: {len(counted_objects)}", (frame_width + 10, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        yp += 25
        cv2.line(canvas, (frame_width + 10, yp),
                 (frame_width + panel_w - 10, yp), (80, 80, 80), 1)
        yp += 20
        cv2.putText(canvas, "COUNTS", (frame_width + 10, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        yp += 22
        for vtype in VEHICLE_CLASSES:
            cnt = vehicle_counts[vtype]
            color = CLASS_COLORS.get(vtype, (255, 255, 255))
            cv2.putText(canvas, f"{vtype.capitalize()}: {cnt}",
                        (frame_width + 15, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            yp += 20
        yp += 5
        cv2.line(canvas, (frame_width + 10, yp),
                 (frame_width + panel_w - 10, yp), (80, 80, 80), 1)
        yp += 22
        total = sum(vehicle_counts.values())
        cv2.putText(canvas, f"TOTAL: {total}", (frame_width + 10, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Frame counter bottom
        cv2.putText(canvas, f"Frame {frame_count}/{MAX_FRAMES}",
                    (frame_width + 10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        writer.write(canvas)

        if frame_count % 500 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / frame_count) * (MAX_FRAMES - frame_count)
            pct = (frame_count / MAX_FRAMES) * 100
            print(f"\r  {pct:.1f}% | Frame {frame_count}/{MAX_FRAMES} | "
                  f"Time: {format_time(video_time)} | "
                  f"Counted: {total} | ETA: {format_time(eta)}",
                  end='', flush=True)

    cap.release()
    writer.release()
    print(f"\n  Done. Total counted: {sum(vehicle_counts.values())}")
    print(f"  Video saved: {out_path}")
    for v in VEHICLE_CLASSES:
        if vehicle_counts[v]:
            print(f"    {v}: {vehicle_counts[v]}")


def main():
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_name}: not found at {model_path}")
            continue
        run(model_path, model_name)


if __name__ == '__main__':
    main()
