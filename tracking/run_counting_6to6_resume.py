#!/usr/bin/env python3
"""
Resume counting from after the corrupted frame in each 6to6 video.
Seeks to the first clean 5-minute boundary past the failure, runs
tracking+counting to end of video, appends new intervals to the
existing *_5min_counts.csv, and updates LineCrossingCount in summary.csv.

Usage:
  # DCNv2 env:
  /home/migui/miniconda3/envs/dcnv2/bin/python run_counting_6to6_resume.py --model-type dcnv2 --video-path /media/mydrive/GitHub/ultralytics/videos/6to6_g2.mp4
  # DCNv3 env:
  /home/migui/miniconda3/envs/dcn/bin/python run_counting_6to6_resume.py --model-type dcnv3 --video-path /media/mydrive/GitHub/ultralytics/videos/6to6_g29.mp4
"""

import sys
import os
import argparse
import warnings
from collections import defaultdict
import time
import csv

warnings.filterwarnings('ignore')
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np

VEHICLE_CLASSES = ["car", "motorcycle", "tricycle", "van", "bus", "truck"]
INTERVAL_SECONDS = 300  # 5 minutes

COUNTING_LINES = {
    '6to6_g2':  [(0.02, 0.55), (0.70, 0.38)],
    '6to6_g3':  [(0.0, 0.52), (1.0, 0.62)],
    '6to6_g29': [(0.0, 0.42), (0.55, 0.72)],
    '6to6_g35': [(0.0, 0.45), (1.0, 0.75)],
}
DEFAULT_COUNTING_LINE = [(0.0, 0.45), (1.0, 0.75)]

# First clean 5-min boundary frame after each video's corrupt frame
# Aligned to the next 5-min interval start (562500 = 10:25:00 at 15fps, 585000 = 10:50:00)
RESUME_FRAMES = {
    '6to6_g2':  562500,   # 10:25:00
    '6to6_g3':  562500,   # 10:25:00
    '6to6_g29': 585000,   # 10:50:00
    '6to6_g35': 562500,   # 10:25:00
}


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def run_resume(model_path, model_name, video_path, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    line_cfg = COUNTING_LINES.get(video_name, DEFAULT_COUNTING_LINE)
    (lx1f, ly1f), (lx2f, ly2f) = line_cfg

    resume_frame = RESUME_FRAMES.get(video_name)
    if resume_frame is None:
        print(f"No resume frame configured for {video_name}, skipping.")
        return None

    resume_time = resume_frame / 15.0  # always 15fps
    start_interval = int(resume_time // INTERVAL_SECONDS)

    print(f"\n{'='*70}")
    print(f"  RESUME: {model_name} | {video_name}.mp4")
    print(f"  Starting from frame {resume_frame} ({format_time(resume_time)}, interval {start_interval})")
    print(f"{'='*70}")

    # Paths for existing data
    interval_csv = os.path.join(output_dir, f"{model_name}_5min_counts.csv")
    summary_csv = os.path.join(os.path.dirname(output_dir), 'summary.csv')

    if not os.path.exists(interval_csv):
        print(f"  ERROR: existing interval CSV not found: {interval_csv}")
        return None

    # Read existing interval data to know which intervals are already covered
    existing_intervals = set()
    with open(interval_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_intervals.add(row['Interval_Start'])

    print(f"  Existing intervals: {len(existing_intervals)} (up to {max(existing_intervals)})")

    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Seek to resume position
    cap.set(cv2.CAP_PROP_POS_FRAMES, resume_frame)
    actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print(f"  Seeked to frame {actual_pos}")

    # Counting state
    vehicle_counts = {c: 0 for c in VEHICLE_CLASSES}
    counted_objects = set()
    object_positions = {}

    interval_counts = defaultdict(lambda: {c: 0 for c in VEHICLE_CLASSES})
    interval_counted_objects = defaultdict(set)
    interval_id_switches = defaultdict(int)
    interval_class_switches = defaultdict(int)
    interval_active_tracks = defaultdict(set)

    track_classes = {}
    track_last_box = {}

    frame_count = resume_frame
    start_time = time.time()
    consecutive_failures = 0
    max_consecutive_failures = 10
    new_intervals_written = 0

    print("  Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            seek_to = current_pos + 100
            if seek_to >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
            frame_count = seek_to
            continue
        consecutive_failures = 0
        frame_count += 1

        video_time = (frame_count - 1) / fps
        current_interval = int(video_time // INTERVAL_SECONDS)

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

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for i, track_id in enumerate(track_ids):
                track_id = int(track_id)
                box = boxes[i]
                class_name = model.names[int(class_ids[i])]

                interval_active_tracks[current_interval].add(track_id)

                if track_id in track_classes and track_classes[track_id] != class_name:
                    interval_class_switches[current_interval] += 1
                track_classes[track_id] = class_name

                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                line_y_at_x = np.interp(cx,
                    [lx1f * frame_width, lx2f * frame_width],
                    [ly1f * frame_height, ly2f * frame_height])
                current_pos_side = 'above' if cy < line_y_at_x else 'below'

                if track_id in object_positions:
                    if object_positions[track_id] != current_pos_side:
                        if track_id not in counted_objects:
                            if class_name in vehicle_counts:
                                vehicle_counts[class_name] += 1
                                counted_objects.add(track_id)
                        if track_id not in interval_counted_objects[current_interval]:
                            if class_name in interval_counts[current_interval]:
                                interval_counts[current_interval][class_name] += 1
                                interval_counted_objects[current_interval].add(track_id)

                object_positions[track_id] = current_pos_side
                track_last_box[track_id] = box

        if frame_count % 1000 == 0:
            elapsed = time.time() - start_time
            pct = (frame_count / total_frames) * 100
            eta = (elapsed / max(1, frame_count - resume_frame)) * (total_frames - frame_count)
            counted = sum(vehicle_counts.values())
            print(f"\r  {pct:.1f}% | Frame {frame_count}/{total_frames} | "
                  f"Time: {format_time(video_time)} | New count: {counted} | "
                  f"ETA: {format_time(eta)}", end='', flush=True)

    cap.release()
    print()

    total_new = sum(vehicle_counts.values())
    video_time_end = (frame_count - 1) / fps
    print(f"\n  New counts from resume: {total_new} vehicles")
    for c in VEHICLE_CLASSES:
        if vehicle_counts[c] > 0:
            print(f"    {c}: {vehicle_counts[c]}")

    # Append new intervals to the existing CSV
    new_interval_rows = []
    for idx in sorted(interval_counts.keys()):
        t_start = format_time(idx * INTERVAL_SECONDS)
        t_end = format_time((idx + 1) * INTERVAL_SECONDS)
        if t_start in existing_intervals:
            print(f"  Skipping already-existing interval {t_start}")
            continue
        row = [t_start, t_end]
        interval_total = 0
        for vtype in VEHICLE_CLASSES:
            c = interval_counts[idx][vtype]
            interval_total += c
            row.append(c)
        n_idsw = interval_id_switches[idx]
        n_clssw = interval_class_switches[idx]
        n_tracks = len(interval_active_tracks[idx])
        pct_id = (n_idsw / max(1, n_tracks)) * 100
        pct_cls = (n_clssw / max(1, n_tracks)) * 100
        row.extend([interval_total, n_idsw, n_clssw, n_tracks,
                    f"{pct_id:.2f}", f"{pct_cls:.2f}"])
        new_interval_rows.append(row)

    with open(interval_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        for row in new_interval_rows:
            writer.writerow(row)
    new_intervals_written = len(new_interval_rows)
    print(f"  Appended {new_intervals_written} new intervals to {interval_csv}")

    # Update summary.csv: add new counts to LineCrossingCount and per-class totals
    if os.path.exists(summary_csv) and total_new > 0:
        rows = []
        with open(summary_csv) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row['Model'] == model_name:
                    row['LineCrossingCount'] = int(row['LineCrossingCount']) + total_new
                    for c in VEHICLE_CLASSES:
                        col = c.capitalize() if c.capitalize() in row else c
                        # summary uses Car, Motorcycle, Tricycle, Van, Bus, Truck
                        cap_c = c.capitalize()
                        if cap_c in row:
                            row[cap_c] = int(row[cap_c]) + vehicle_counts[c]
                rows.append(row)
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Updated summary.csv: +{total_new} to LineCrossingCount for {model_name}")

    return {
        'model': model_name,
        'new_count': total_new,
        'new_intervals': new_intervals_written,
        **{c: vehicle_counts[c] for c in VEHICLE_CLASSES},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['dcnv2', 'dcnv3', 'vanilla'], required=True)
    parser.add_argument('--video-path', type=str, required=True)
    args = parser.parse_args()

    video_path = args.video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if args.model_type == 'dcnv2':
        models = {
            'DCNv2-Full': '/home/migui/Downloads/dcnv2-yolov8-neck-full-20260318T004120Z-1-001/dcnv2-yolov8-neck-full/weights/best.pt',
            'DCNv2-FPN':  '/home/migui/Downloads/dcnv2-yolov8-neck-fpn-20260318T004118Z-1-001/dcnv2-yolov8-neck-fpn/weights/best.pt',
            'DCNv2-Pan':  '/home/migui/Downloads/dcnv2-yolov8-neck-pan-20260318T004653Z-1-001/dcnv2-yolov8-neck-pan/weights/best.pt',
            'DCNv2-Liu':  '/home/migui/Downloads/dcnv2-yolov8-liu-20260318T004538Z-1-001/dcnv2-yolov8-liu/weights/best.pt',
        }
        output_base = f'/media/mydrive/GitHub/ultralytics/tracking/counting_{video_name}/dcnv2'
    elif args.model_type == 'dcnv3':
        models = {
            'DCNv3-Full': '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_full/weights/best.pt',
            'DCNv3-FPN':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n-neck-fpn/weights/best.pt',
            'DCNv3-Pan':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_pan/weights/best.pt',
            'DCNv3-Liu':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_liu/weights/best.pt',
        }
        output_base = f'/media/mydrive/GitHub/ultralytics/tracking/counting_{video_name}/dcnv3'
    else:
        models = {
            'Vanilla-YOLOv8n': '/home/migui/Downloads/100_yolov8n_300epochs_b32-20260318T004620Z-1-001/100_yolov8n_300epochs_b32/weights/best.pt',
        }
        output_base = f'/media/mydrive/GitHub/ultralytics/tracking/counting_{video_name}/vanilla'

    all_results = []
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_name}: not found at {model_path}")
            continue
        output_dir = os.path.join(output_base, model_name)
        result = run_resume(model_path, model_name, video_path, output_dir)
        if result:
            all_results.append(result)

    if all_results:
        print(f"\n{'='*80}")
        print(f"  RESUME SUMMARY — {args.model_type.upper()} on {video_name}.mp4")
        print(f"{'='*80}")
        print(f"  {'Model':<18} {'NewCount':<10} {'Car':<6} {'Moto':<6} {'Tri':<6} {'Van':<6} {'Bus':<6} {'Truck':<6} {'Intervals'}")
        print(f"  {'-'*75}")
        for r in all_results:
            print(f"  {r['model']:<18} {r['new_count']:<10} "
                  f"{r['car']:<6} {r['motorcycle']:<6} {r['tricycle']:<6} "
                  f"{r['van']:<6} {r['bus']:<6} {r['truck']:<6} {r['new_intervals']}")


if __name__ == '__main__':
    main()
