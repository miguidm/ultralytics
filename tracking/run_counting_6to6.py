#!/usr/bin/env python3
"""
DCNv2 vs DCNv3 Tracking + Counting (full 12hr video)
All metrics: MOTA, IDF1, ID Switches, FPS, Classification switches
+ Total vehicle counts + Per-5-minute interval counts

Usage (run separately for each env):
  # DCNv2 in dcnv2 env:
  /home/migui/miniconda3/envs/dcnv2/bin/python run_counting_6to6.py --model-type dcnv2
  # DCNv3 in dcn env:
  /home/migui/miniconda3/envs/dcn/bin/python run_counting_6to6.py --model-type dcnv3
  # Both (if deps allow):
  python run_counting_6to6.py --model-type dcnv2
  python run_counting_6to6.py --model-type dcnv3
"""

import sys
import os
import argparse
import warnings
from collections import defaultdict
import time
import random
import csv

warnings.filterwarnings('ignore')
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np

CLASS_COLORS = {
    "car": (55, 250, 250),
    "motorcycle": (83, 179, 36),
    "tricycle": (83, 50, 250),
    "bus": (245, 61, 184),
    "van": (255, 221, 51),
    "truck": (49, 147, 245),
}

VEHICLE_CLASSES = ["car", "motorcycle", "tricycle", "van", "bus", "truck"]
INTERVAL_SECONDS = 300  # 5 minutes

# Per-video counting line configs: (x1_frac, y1_frac) -> (x2_frac, y2_frac)
# x/y fractions of frame width/height
COUNTING_LINES = {
    '6to6_g2':  [(0.02, 0.55), (0.70, 0.38)],
    '6to6_g3':  [(0.0, 0.52), (1.0, 0.62)],
    '6to6_g29': [(0.0, 0.42), (0.55, 0.72)],
    '6to6_g35': [(0.0, 0.45), (1.0, 0.75)],
}
DEFAULT_COUNTING_LINE = [(0.0, 0.45), (1.0, 0.75)]


def merge_duplicate_tracks(track_first_frame, track_last_frame, track_first_box, track_last_box,
                           max_gap=60, min_iou=0.2):
    """Merge tracks that likely belong to the same vehicle.
    If track B appears near where track A disappeared within max_gap frames, merge them."""
    track_ids = sorted(track_first_frame.keys(), key=lambda t: track_first_frame[t])
    merged = {}  # track_id -> canonical_id

    for tid in track_ids:
        best_match = None
        best_iou = min_iou

        for prev_tid in track_ids:
            if prev_tid == tid:
                break  # only look at earlier tracks
            canonical = merged.get(prev_tid, prev_tid)
            # find the last track in this canonical group
            last_frame_of_prev = track_last_frame.get(prev_tid, 0)
            gap = track_first_frame[tid] - last_frame_of_prev
            if 0 < gap <= max_gap:
                iou = compute_iou(track_last_box[prev_tid], track_first_box[tid])
                if iou > best_iou:
                    best_iou = iou
                    best_match = canonical

        merged[tid] = best_match if best_match is not None else tid

    unique_vehicles = len(set(merged.values()))
    return unique_vehicles, merged


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def format_time(seconds):
    """Format seconds as HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_tracking(model_path, model_name, video_path, output_dir, max_frames=0, save_video=False):
    """Run ByteTrack tracking with counting and per-5-min intervals"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    line_cfg = COUNTING_LINES.get(video_name, DEFAULT_COUNTING_LINE)
    (lx1f, ly1f), (lx2f, ly2f) = line_cfg

    print("\n" + "=" * 70)
    print(f"  {model_name} | Video: {video_name}.mp4")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
        try:
            model_info = model.info(verbose=False)
            model_gflops = model_info[1] if isinstance(model_info, (list, tuple)) and len(model_info) > 1 else float('nan')
        except Exception:
            model_gflops = float('nan')
        print(f"Model loaded | GFLOPs: {model_gflops:.2f}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / fps
    print(f"Video: {total_frames} frames @ {fps}fps ({frame_width}x{frame_height})")
    print(f"Duration: {format_time(video_duration)}")

    # Video writer
    video_writer = None
    panel_width = 280
    if save_video:
        video_file = os.path.join(output_dir, f"{model_name}_6to6_counted.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_file, fourcc, fps, (frame_width + panel_width, frame_height))
        print(f"Video output: {video_file}")

    # Tracking state
    track_history = {}
    track_first_frame = {}
    track_last_frame = {}
    track_first_box = {}
    track_last_box = {}

    # IDSW detection
    lost_tracks = {}
    identity_switches = 0
    lost_track_window = 30
    iou_threshold = 0.3

    # Classification tracking
    track_classes = {}
    track_class_history = {}
    class_switches = []

    # Object tracking timeline
    track_active_frames = {}

    # Vehicle counting - TOTAL
    vehicle_counts = {c: 0 for c in VEHICLE_CLASSES}
    counted_objects = set()
    object_positions = {}

    # Vehicle counting - PER INTERVAL (5 min)
    # Key: interval index (0, 1, 2, ...), Value: {class: count}
    interval_counts = defaultdict(lambda: {c: 0 for c in VEHICLE_CLASSES})
    interval_counted_objects = defaultdict(set)  # interval -> set of track_ids counted in that interval

    # Per-interval switch tracking
    interval_id_switches = defaultdict(int)
    interval_class_switches = defaultdict(int)
    interval_active_tracks = defaultdict(set)  # tracks seen per interval

    # Video annotation state
    class_switch_display = {}
    trajectories = {}
    trajectory_colors = {}
    max_trail_length = 50

    prev_track_ids = set()
    frame_count = 0
    total_detections = 0
    start_time = time.time()

    print("Processing frames...")

    consecutive_failures = 0
    max_consecutive_failures = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break
            # Seek past the corrupted frame and continue
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            seek_to = current_pos + 100
            if max_frames > 0 and seek_to > max_frames:
                break
            if seek_to >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
            frame_count = seek_to
            continue
        consecutive_failures = 0

        frame_count += 1
        if max_frames > 0 and frame_count > max_frames:
            break

        # Current time in video
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

        current_track_ids = set()
        current_boxes = {}

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for i, track_id in enumerate(track_ids):
                track_id = int(track_id)
                box = boxes[i]
                class_name = model.names[int(class_ids[i])]

                current_track_ids.add(track_id)
                current_boxes[track_id] = box

                if track_id not in track_history:
                    track_history[track_id] = []
                    track_first_frame[track_id] = frame_count
                    track_first_box[track_id] = box
                    track_active_frames[track_id] = set()
                    track_class_history[track_id] = []

                    # IDSW detection
                    for lost_id, (lost_frame, lost_box) in list(lost_tracks.items()):
                        if frame_count - lost_frame <= lost_track_window:
                            iou = compute_iou(box, lost_box)
                            if iou > iou_threshold:
                                identity_switches += 1
                                interval_id_switches[current_interval] += 1
                                del lost_tracks[lost_id]
                                break

                track_history[track_id].append(frame_count)
                track_last_frame[track_id] = frame_count
                track_last_box[track_id] = box
                track_active_frames[track_id].add(frame_count)

                # Classification switch detection
                track_class_history[track_id].append((frame_count, class_name))
                if track_id in track_classes and track_classes[track_id] != class_name:
                    old_class = track_classes[track_id]
                    class_switches.append((frame_count, track_id, old_class, class_name))
                    interval_class_switches[current_interval] += 1
                    if save_video:
                        class_switch_display[track_id] = (f"{old_class}->{class_name}", 45)
                track_classes[track_id] = class_name

                # Track active tracks per interval
                interval_active_tracks[current_interval].add(track_id)

                # Counting line detection (diagonal: 0,45% -> width,75%)
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                line_y_at_x = np.interp(cx, [lx1f * frame_width, lx2f * frame_width],
                                            [ly1f * frame_height, ly2f * frame_height])
                current_pos = 'above' if cy < line_y_at_x else 'below'

                if track_id in object_positions:
                    if object_positions[track_id] != current_pos:
                        # TOTAL count (once per object ever)
                        if track_id not in counted_objects:
                            if class_name in vehicle_counts:
                                vehicle_counts[class_name] += 1
                                counted_objects.add(track_id)

                        # PER-INTERVAL count (once per object per interval)
                        if track_id not in interval_counted_objects[current_interval]:
                            if class_name in interval_counts[current_interval]:
                                interval_counts[current_interval][class_name] += 1
                                interval_counted_objects[current_interval].add(track_id)

                object_positions[track_id] = current_pos

            total_detections += len(track_ids)

        # Video annotation
        if save_video and results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids_arr = results.boxes.id.cpu().numpy().astype(int)
            class_ids_arr = results.boxes.cls.cpu().numpy().astype(int)

            active_ids = set()
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                tid = int(track_ids_arr[i])
                cls_name = model.names[int(class_ids_arr[i])]
                active_ids.add(tid)

                cxd, cyd = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if tid not in trajectories:
                    trajectories[tid] = []
                    trajectory_colors[tid] = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                trajectories[tid].append((cxd, cyd))
                if len(trajectories[tid]) > max_trail_length:
                    trajectories[tid].pop(0)

                color = CLASS_COLORS.get(cls_name.lower(), (0, 255, 0))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{cls_name} ID:{tid}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if tid in class_switch_display:
                    sw_text, sw_left = class_switch_display[tid]
                    if sw_left > 0:
                        sw_y = int(y2) + 18
                        (tw, th), _ = cv2.getTextSize(sw_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(frame, (int(x1), sw_y - th - 4),
                                      (int(x1) + tw + 6, sw_y + 4), (0, 0, 180), -1)
                        cv2.putText(frame, sw_text, (int(x1) + 3, sw_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                        class_switch_display[tid] = (sw_text, sw_left - 1)
                    else:
                        del class_switch_display[tid]

            for tid, trail in trajectories.items():
                if len(trail) > 1:
                    tcolor = trajectory_colors.get(tid, (255, 255, 255))
                    for j in range(1, len(trail)):
                        thickness = max(2, int(8 * j / len(trail)))
                        cv2.line(frame, trail[j - 1], trail[j], tcolor, thickness)
                    cv2.circle(frame, trail[-1], 6, tcolor, -1)

            for tid in list(trajectories.keys()):
                if tid not in active_ids:
                    if len(trajectories[tid]) > 5:
                        trajectories[tid] = trajectories[tid][5:]
                    else:
                        del trajectories[tid]
                        trajectory_colors.pop(tid, None)

            # Counting line
            cv2.line(frame, (int(lx1f * frame_width), int(ly1f * frame_height)),
                     (int(lx2f * frame_width), int(ly2f * frame_height)), (147, 20, 255), 3)

            # Side panel
            extended = np.zeros((frame_height, frame_width + panel_width, 3), dtype=np.uint8)
            extended[:, :frame_width] = frame
            extended[:, frame_width:] = (40, 40, 40)

            cv2.putText(extended, model_name, (frame_width + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(extended, f"Time: {format_time(video_time)}", (frame_width + 10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.line(extended, (frame_width + 10, 65), (frame_width + panel_width - 10, 65), (255, 255, 255), 1)

            yp = 95
            cv2.putText(extended, f"Active Tracks: {len(active_ids)}", (frame_width + 15, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            yp += 25
            cv2.putText(extended, f"ID Switches: {identity_switches}", (frame_width + 15, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            yp += 25
            cv2.putText(extended, f"Class Switches: {len(class_switches)}", (frame_width + 15, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

            yp += 35
            cv2.line(extended, (frame_width + 10, yp), (frame_width + panel_width - 10, yp), (100, 100, 100), 1)
            yp += 25
            cv2.putText(extended, "TOTAL COUNTS", (frame_width + 10, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            yp += 20
            for vtype in VEHICLE_CLASSES:
                vcount = vehicle_counts[vtype]
                if vcount > 0:
                    color = CLASS_COLORS.get(vtype, (255, 255, 255))
                    cv2.putText(extended, f"{vtype.upper()}: {vcount}", (frame_width + 15, yp),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    yp += 18
            total_count = sum(vehicle_counts.values())
            cv2.putText(extended, f"TOTAL: {total_count}", (frame_width + 15, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Current interval counts
            yp += 35
            cv2.line(extended, (frame_width + 10, yp), (frame_width + panel_width - 10, yp), (100, 100, 100), 1)
            yp += 25
            int_start = format_time(current_interval * INTERVAL_SECONDS)
            int_end = format_time((current_interval + 1) * INTERVAL_SECONDS)
            cv2.putText(extended, f"INTERVAL {int_start}-{int_end}", (frame_width + 10, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            yp += 20
            int_total = 0
            for vtype in VEHICLE_CLASSES:
                ic = interval_counts[current_interval][vtype]
                if ic > 0:
                    color = CLASS_COLORS.get(vtype, (255, 255, 255))
                    cv2.putText(extended, f"{vtype.upper()}: {ic}", (frame_width + 15, yp),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    yp += 16
                    int_total += ic
            cv2.putText(extended, f"INTERVAL TOTAL: {int_total}", (frame_width + 15, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.putText(extended, f"Frame: {frame_count}/{max_frames if max_frames > 0 else total_frames}",
                        (frame_width + 15, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            video_writer.write(extended)

        # Detect lost tracks
        for track_id in prev_track_ids - current_track_ids:
            if track_id in track_last_box:
                lost_tracks[track_id] = (frame_count, track_last_box[track_id])

        lost_tracks = {k: v for k, v in lost_tracks.items()
                      if frame_count - v[0] <= lost_track_window}

        prev_track_ids = current_track_ids

        if frame_count % 500 == 0:
            elapsed = time.time() - start_time
            pct = (frame_count / total_frames) * 100
            eta = (elapsed / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
            print(f"\r  {pct:.1f}% | Frame {frame_count}/{total_frames} | "
                  f"Time: {format_time(video_time)} | "
                  f"Counted: {sum(vehicle_counts.values())} | "
                  f"ETA: {format_time(eta)}", end='', flush=True)

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved: {video_file}")

    end_time = time.time()
    runtime = end_time - start_time
    processing_fps = frame_count / runtime if runtime > 0 else 0

    # Calculate metrics
    total_tracks = len(track_history)

    # Deduplicate tracks
    unique_vehicles, track_merge_map = merge_duplicate_tracks(
        track_first_frame, track_last_frame, track_first_box, track_last_box,
        max_gap=60, min_iou=0.2
    )
    duplicate_tracks = total_tracks - unique_vehicles

    fragmentations = 0
    mostly_tracked = 0
    mostly_lost = 0
    for track_id, frames in track_history.items():
        span = track_last_frame[track_id] - track_first_frame[track_id] + 1
        coverage = len(track_active_frames[track_id]) / span if span > 0 else 0
        if coverage > 0.8:
            mostly_tracked += 1
        elif coverage < 0.2:
            mostly_lost += 1

        if len(frames) > 1:
            sorted_frames = sorted(frames)
            for i in range(1, len(sorted_frames)):
                if sorted_frames[i] - sorted_frames[i - 1] > 5:
                    fragmentations += 1

    pct_mt = (mostly_tracked / max(1, total_tracks)) * 100
    pct_ml = (mostly_lost / max(1, total_tracks)) * 100

    total_track_frames = sum(len(frames) for frames in track_history.values())
    total_errors = identity_switches + fragmentations
    mota = max(0, min(1, 1 - (total_errors / max(1, total_track_frames))))

    id_true_positives = total_track_frames - total_errors
    id_false_positives = identity_switches
    id_false_negatives = fragmentations
    idf1 = (2 * id_true_positives) / max(1, 2 * id_true_positives + id_false_positives + id_false_negatives)

    # ===================== PRINT RESULTS =====================

    print(f"\n\n{'=' * 70}")
    print(f"  RESULTS: {model_name}")
    print(f"{'=' * 70}")

    print(f"\n  TRACKING METRICS")
    print(f"  {'Metric':<20} {'Value':<10}")
    print(f"  {'-' * 30}")
    print(f"  {'MOTA':<20} {mota:.4f}")
    print(f"  {'IDF1':<20} {idf1:.4f}")
    print(f"  {'ID Switches':<20} {identity_switches}")
    print(f"  {'Fragmentations':<20} {fragmentations}")
    print(f"  {'Mostly Tracked (MT)':<20} {mostly_tracked} ({pct_mt:.1f}%)")
    print(f"  {'Mostly Lost (ML)':<20} {mostly_lost} ({pct_ml:.1f}%)")
    print(f"  {'Track IDs Assigned':<20} {total_tracks}")
    print(f"  {'Unique Vehicles':<20} {unique_vehicles}")
    print(f"  {'Duplicate Tracks':<20} {duplicate_tracks}")
    print(f"  {'GFLOPs':<20} {model_gflops:.2f}")
    print(f"  {'Processing FPS':<20} {processing_fps:.2f}")
    print(f"  {'Runtime':<20} {format_time(runtime)}")
    print(f"  {'Class Switches':<20} {len(class_switches)}")

    # --- Total vehicle counts ---
    total_counted = sum(vehicle_counts.values())
    print(f"\n  TOTAL VEHICLE COUNTS (line crossing)")
    print(f"  {'-' * 30}")
    for vtype in VEHICLE_CLASSES:
        print(f"  {vtype.upper():<15} {vehicle_counts[vtype]}")
    print(f"  {'TOTAL':<15} {total_counted}")

    # --- Per-5-minute interval counts ---
    num_intervals = max(interval_counts.keys()) + 1 if interval_counts else 0
    if num_intervals == 0 and frame_count > 0:
        num_intervals = int(video_time // INTERVAL_SECONDS) + 1

    print(f"\n  VEHICLE COUNTS PER 5-MINUTE INTERVAL")
    header = f"  {'Interval':<18}"
    for vtype in VEHICLE_CLASSES:
        header += f" {vtype.capitalize():<8}"
    header += f" {'Total':<8}"
    print(header)
    print(f"  {'-' * (18 + 8 * (len(VEHICLE_CLASSES) + 1))}")

    running_total = {c: 0 for c in VEHICLE_CLASSES}
    for idx in range(num_intervals):
        t_start = format_time(idx * INTERVAL_SECONDS)
        t_end = format_time((idx + 1) * INTERVAL_SECONDS)
        row = f"  {t_start}-{t_end:<7}"
        interval_total = 0
        for vtype in VEHICLE_CLASSES:
            c = interval_counts[idx][vtype]
            running_total[vtype] += c
            interval_total += c
            row += f" {c:<8}"
        row += f" {interval_total:<8}"
        print(row)

    # Cumulative check
    print(f"\n  Cumulative total from intervals: {sum(sum(interval_counts[i].values()) for i in range(num_intervals))}")
    print(f"  Total from line crossing:        {total_counted}")

    # --- Switch statistics ---
    tracks_with_cls_switch = sum(1 for tid in track_class_history
                                 if len(set(cls for _, cls in track_class_history[tid])) > 1)
    pct_tracks_cls_switch = (tracks_with_cls_switch / max(1, total_tracks)) * 100

    # Counted vehicles (line crossing) that experienced class switches
    counted_with_cls_switch = sum(1 for tid in counted_objects
                                  if tid in track_class_history
                                  and len(set(cls for _, cls in track_class_history[tid])) > 1)
    pct_counted_cls_switch = (counted_with_cls_switch / max(1, total_counted)) * 100

    # Counted vehicles that experienced ID switches (via merge map: >1 track assigned to canonical)
    from collections import defaultdict as _dd
    canonical_to_tracks = _dd(list)
    for tid, canonical in track_merge_map.items():
        canonical_to_tracks[canonical].append(tid)
    counted_with_id_switch = sum(
        1 for tid in counted_objects
        if len(canonical_to_tracks.get(track_merge_map.get(tid, tid), [tid])) > 1
    )
    pct_counted_id_switch = (counted_with_id_switch / max(1, total_counted)) * 100

    # Per-frame switch counts per track
    track_frame_switches = defaultdict(int)  # tid -> number of frame-level switches
    for fnum, tid, old, new in class_switches:
        track_frame_switches[tid] += 1

    # Per-second switch counts per track
    # Group switches by (tid, second), count unique seconds where a switch happened
    track_second_switches = defaultdict(set)  # tid -> set of seconds with a switch
    for fnum, tid, old, new in class_switches:
        sec = (fnum - 1) // fps
        track_second_switches[tid].add(sec)
    track_sec_switch_counts = {tid: len(secs) for tid, secs in track_second_switches.items()}

    # Per-frame stats for counted vehicles
    counted_frame_switches = {tid: track_frame_switches[tid] for tid in counted_objects
                              if track_frame_switches[tid] > 0}
    total_counted_frame_switches = sum(counted_frame_switches.values())
    avg_frame_switches_per_affected = total_counted_frame_switches / max(1, counted_with_cls_switch)

    # Per-second stats for counted vehicles
    counted_sec_switches = {tid: track_sec_switch_counts[tid] for tid in counted_objects
                            if tid in track_sec_switch_counts}
    total_counted_sec_switches = sum(counted_sec_switches.values())
    avg_sec_switches_per_affected = total_counted_sec_switches / max(1, counted_with_cls_switch)

    # Totals for all tracks
    total_frame_switches = len(class_switches)
    total_sec_switches = sum(track_sec_switch_counts.values())
    pct_idsw_per_detection = (identity_switches / max(1, total_detections)) * 100
    pct_clssw_per_detection = (len(class_switches) / max(1, total_detections)) * 100
    video_hours = video_time / 3600 if video_time > 0 else 1
    idsw_per_hour = identity_switches / video_hours
    clssw_per_hour = len(class_switches) / video_hours

    print(f"\n  SWITCH STATISTICS (over {format_time(video_time)} period)")
    print(f"  {'-' * 60}")
    print(f"  {'ID Switches Total':<40} {identity_switches}")
    print(f"  {'ID Switches / Hour':<40} {idsw_per_hour:.2f}")
    print(f"  {'Counted Vehs with ID Switch':<40} {counted_with_id_switch}/{total_counted} ({pct_counted_id_switch:.2f}%)")
    print(f"  {'Tracks with Class Switch':<40} {tracks_with_cls_switch}/{total_tracks} ({pct_tracks_cls_switch:.2f}%)")
    print(f"  {'Counted Vehs with Class Switch':<40} {counted_with_cls_switch}/{total_counted} ({pct_counted_cls_switch:.2f}%)")

    print(f"\n  CLASS SWITCHES - PER FRAME (each frame where class changes)")
    print(f"  {'-' * 60}")
    print(f"  {'Total (all tracks)':<40} {total_frame_switches}")
    print(f"  {'On Counted Vehicles':<40} {total_counted_frame_switches}")
    print(f"  {'Avg per Affected Counted Veh':<40} {avg_frame_switches_per_affected:.2f}")
    print(f"  {'Per Hour':<40} {clssw_per_hour:.2f}")

    print(f"\n  CLASS SWITCHES - PER SECOND (unique seconds with a switch)")
    print(f"  {'-' * 60}")
    print(f"  {'Total (all tracks)':<40} {total_sec_switches}")
    print(f"  {'On Counted Vehicles':<40} {total_counted_sec_switches}")
    print(f"  {'Avg per Affected Counted Veh':<40} {avg_sec_switches_per_affected:.2f}")
    sec_switch_per_hour = total_sec_switches / video_hours
    print(f"  {'Per Hour':<40} {sec_switch_per_hour:.2f}")

    # --- Per-5-min interval switch breakdown ---
    print(f"\n  ID & CLASS SWITCHES PER 5-MINUTE INTERVAL")
    print(f"  {'Interval':<18} {'IDSw':<8} {'ClsSw':<8} {'Tracks':<8} {'IDSw%':<10} {'ClsSw%':<10}")
    print(f"  {'-' * 62}")
    for idx in range(num_intervals):
        t_start = format_time(idx * INTERVAL_SECONDS)
        t_end = format_time((idx + 1) * INTERVAL_SECONDS)
        n_idsw = interval_id_switches[idx]
        n_clssw = interval_class_switches[idx]
        n_tracks = len(interval_active_tracks[idx])
        pct_id = (n_idsw / max(1, n_tracks)) * 100
        pct_cls = (n_clssw / max(1, n_tracks)) * 100
        print(f"  {t_start}-{t_end:<7} {n_idsw:<8} {n_clssw:<8} {n_tracks:<8} {pct_id:<10.2f} {pct_cls:<10.2f}")

    # --- Classification switches summary ---
    print(f"\n  CLASSIFICATION SWITCHES: {len(class_switches)} total")
    if class_switches and fps > 0:
        switches_per_sec = defaultdict(list)
        for fnum, tid, old_cls, new_cls in class_switches:
            sec = (fnum - 1) // fps
            switches_per_sec[sec].append((fnum, tid, old_cls, new_cls))
        total_seconds = (frame_count - 1) // fps + 1
        print(f"  Seconds with switches: {len(switches_per_sec)}/{total_seconds}")

    # --- Per-object class history (switched only) ---
    print(f"\n  PER-OBJECT CLASS HISTORY (switched objects only):")
    any_switched = False
    for tid in sorted(track_class_history.keys()):
        history = track_class_history[tid]
        unique_seq = []
        for _, cls in history:
            if not unique_seq or unique_seq[-1] != cls:
                unique_seq.append(cls)
        if len(unique_seq) > 1:
            any_switched = True
            n_frame = track_frame_switches.get(tid, 0)
            n_sec = track_sec_switch_counts.get(tid, 0)
            counted_tag = " [COUNTED]" if tid in counted_objects else ""
            print(f"  ID:{tid} ({n_frame}f/{n_sec}s): {' -> '.join(unique_seq)}{counted_tag}")
    if not any_switched:
        print(f"  (No objects switched class)")

    print(f"\n{'=' * 70}\n")

    # ===================== SAVE REPORTS =====================

    # Text report
    report_file = os.path.join(output_dir, f"{model_name}_report.txt")
    with open(report_file, 'w') as f:
        f.write(f"{model_name} - {video_name}.mp4 Full Tracking Report\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Frames: {frame_count} | FPS(video): {fps} | Duration: {format_time(video_duration)}\n\n")

        f.write(f"TRACKING METRICS\n{'-' * 30}\n")
        f.write(f"MOTA:               {mota:.4f}\n")
        f.write(f"IDF1:               {idf1:.4f}\n")
        f.write(f"ID Switches:        {identity_switches}\n")
        f.write(f"Fragmentations:     {fragmentations}\n")
        f.write(f"Mostly Tracked (MT):{mostly_tracked} ({pct_mt:.1f}%)\n")
        f.write(f"Mostly Lost (ML):   {mostly_lost} ({pct_ml:.1f}%)\n")
        f.write(f"Track IDs Assigned: {total_tracks}\n")
        f.write(f"Unique Vehicles:    {unique_vehicles}\n")
        f.write(f"Duplicate Tracks:   {duplicate_tracks}\n")
        f.write(f"GFLOPs:             {model_gflops:.2f}\n")
        f.write(f"FPS:                {processing_fps:.2f}\n")
        f.write(f"Runtime:            {format_time(runtime)}\n")
        f.write(f"Class Switches:     {len(class_switches)}\n\n")

        f.write(f"TOTAL VEHICLE COUNTS\n{'-' * 30}\n")
        for vtype in VEHICLE_CLASSES:
            f.write(f"  {vtype.upper():<15} {vehicle_counts[vtype]}\n")
        f.write(f"  {'TOTAL':<15} {total_counted}\n\n")

        f.write(f"VEHICLE COUNTS PER 5-MINUTE INTERVAL\n{'-' * 60}\n")
        hdr = f"{'Interval':<18}"
        for vtype in VEHICLE_CLASSES:
            hdr += f" {vtype.capitalize():<8}"
        hdr += f" {'Total':<8}\n"
        f.write(hdr)
        for idx in range(num_intervals):
            t_start = format_time(idx * INTERVAL_SECONDS)
            t_end = format_time((idx + 1) * INTERVAL_SECONDS)
            row = f"{t_start}-{t_end:<7}"
            interval_total = 0
            for vtype in VEHICLE_CLASSES:
                c = interval_counts[idx][vtype]
                interval_total += c
                row += f" {c:<8}"
            row += f" {interval_total:<8}\n"
            f.write(row)

        f.write(f"\nSWITCH STATISTICS (over {format_time(video_time)} period)\n{'-' * 60}\n")
        f.write(f"ID Switches Total:                      {identity_switches}\n")
        f.write(f"ID Switches / Hour:                     {idsw_per_hour:.2f}\n")
        f.write(f"Counted Vehs with ID Switch:            {counted_with_id_switch}/{total_counted} ({pct_counted_id_switch:.2f}%)\n")
        f.write(f"Tracks with Class Switch:               {tracks_with_cls_switch}/{total_tracks} ({pct_tracks_cls_switch:.2f}%)\n")
        f.write(f"Counted Vehs with Class Switch:         {counted_with_cls_switch}/{total_counted} ({pct_counted_cls_switch:.2f}%)\n")
        f.write(f"\nCLASS SWITCHES - PER FRAME\n{'-' * 60}\n")
        f.write(f"Total (all tracks):                     {total_frame_switches}\n")
        f.write(f"On Counted Vehicles:                    {total_counted_frame_switches}\n")
        f.write(f"Avg per Affected Counted Veh:           {avg_frame_switches_per_affected:.2f}\n")
        f.write(f"Per Hour:                               {clssw_per_hour:.2f}\n")
        sec_switch_per_hour = total_sec_switches / video_hours
        f.write(f"\nCLASS SWITCHES - PER SECOND\n{'-' * 60}\n")
        f.write(f"Total (all tracks):                     {total_sec_switches}\n")
        f.write(f"On Counted Vehicles:                    {total_counted_sec_switches}\n")
        f.write(f"Avg per Affected Counted Veh:           {avg_sec_switches_per_affected:.2f}\n")
        f.write(f"Per Hour:                               {sec_switch_per_hour:.2f}\n")

        f.write(f"\nID & CLASS SWITCHES PER 5-MINUTE INTERVAL\n{'-' * 62}\n")
        f.write(f"{'Interval':<18} {'IDSw':<8} {'ClsSw':<8} {'Tracks':<8} {'IDSw%':<10} {'ClsSw%':<10}\n")
        for idx in range(num_intervals):
            t_start = format_time(idx * INTERVAL_SECONDS)
            t_end = format_time((idx + 1) * INTERVAL_SECONDS)
            n_idsw = interval_id_switches[idx]
            n_clssw = interval_class_switches[idx]
            n_tracks = len(interval_active_tracks[idx])
            pct_id = (n_idsw / max(1, n_tracks)) * 100
            pct_cls = (n_clssw / max(1, n_tracks)) * 100
            f.write(f"{t_start}-{t_end:<7} {n_idsw:<8} {n_clssw:<8} {n_tracks:<8} {pct_id:<10.2f} {pct_cls:<10.2f}\n")

        f.write(f"\nCLASSIFICATION SWITCHES: {len(class_switches)} total\n")
        if class_switches:
            f.write(f"{'Frame':<8} {'ID':<6} {'From':<12} {'To':<12}\n")
            for fnum, tid, old_cls, new_cls in class_switches:
                f.write(f"{fnum:<8} {tid:<6} {old_cls:<12} {new_cls:<12}\n")

            f.write(f"\nPER-OBJECT CLASS HISTORY (switched only):\n")
            f.write(f"  (Nf=frame switches, Ns=second switches)\n")
            for tid in sorted(track_class_history.keys()):
                history = track_class_history[tid]
                unique_seq = []
                for _, cls in history:
                    if not unique_seq or unique_seq[-1] != cls:
                        unique_seq.append(cls)
                if len(unique_seq) > 1:
                    n_frame = track_frame_switches.get(tid, 0)
                    n_sec = track_sec_switch_counts.get(tid, 0)
                    counted_tag = " [COUNTED]" if tid in counted_objects else ""
                    f.write(f"  ID:{tid} ({n_frame}f/{n_sec}s): {' -> '.join(unique_seq)}{counted_tag}\n")

    print(f"Report saved: {report_file}")

    # Per-5-min CSV
    interval_csv = os.path.join(output_dir, f"{model_name}_5min_counts.csv")
    with open(interval_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Interval_Start", "Interval_End"] + [c.capitalize() for c in VEHICLE_CLASSES] +
                        ["Total", "ID_Switches", "Class_Switches", "Active_Tracks", "IDSw_Pct", "ClsSw_Pct"])
        for idx in range(num_intervals):
            t_start = format_time(idx * INTERVAL_SECONDS)
            t_end = format_time((idx + 1) * INTERVAL_SECONDS)
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
            row.extend([interval_total, n_idsw, n_clssw, n_tracks, f"{pct_id:.2f}", f"{pct_cls:.2f}"])
            writer.writerow(row)
    print(f"Interval CSV saved: {interval_csv}")

    return {
        'model': model_name,
        'mota': mota,
        'idf1': idf1,
        'idsw': identity_switches,
        'fragmentations': fragmentations,
        'fps': processing_fps,
        'runtime': format_time(runtime),
        'class_switches': len(class_switches),
        'total_tracks': total_tracks,
        'unique_vehicles': unique_vehicles,
        'duplicate_tracks': duplicate_tracks,
        'vehicle_total': total_counted,
        'car': vehicle_counts.get('car', 0),
        'motorcycle': vehicle_counts.get('motorcycle', 0),
        'tricycle': vehicle_counts.get('tricycle', 0),
        'van': vehicle_counts.get('van', 0),
        'bus': vehicle_counts.get('bus', 0),
        'truck': vehicle_counts.get('truck', 0),
        'idsw_per_hour': idsw_per_hour,
        'clssw_per_hour': clssw_per_hour,
        'pct_idsw': pct_idsw_per_detection,
        'pct_clssw': pct_clssw_per_detection,
        'tracks_with_cls_switch': tracks_with_cls_switch,
        'pct_tracks_cls_switch': pct_tracks_cls_switch,
        'counted_with_cls_switch': counted_with_cls_switch,
        'pct_counted_cls_switch': pct_counted_cls_switch,
        'counted_with_id_switch': counted_with_id_switch,
        'pct_counted_id_switch': pct_counted_id_switch,
        'counted_frame_switches': total_counted_frame_switches,
        'counted_sec_switches': total_counted_sec_switches,
        'avg_frame_switches_per_affected': avg_frame_switches_per_affected,
        'avg_sec_switches_per_affected': avg_sec_switches_per_affected,
        'mostly_tracked': mostly_tracked,
        'pct_mt': pct_mt,
        'mostly_lost': mostly_lost,
        'pct_ml': pct_ml,
        'gflops': model_gflops,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['dcnv2', 'dcnv3', 'vanilla'], required=True)
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Max frames (0=all)')
    parser.add_argument('--video', action='store_true',
                        help='Save annotated output video')
    parser.add_argument('--video-path', type=str,
                        default='/media/mydrive/GitHub/ultralytics/videos/6to6_g35.mp4',
                        help='Path to input video')
    args = parser.parse_args()

    video_path = args.video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if args.model_type == 'dcnv2':
        models = {
            'DCNv2-Full': r'/home/migui/Downloads/dcnv2-yolov8-neck-full-20260318T004120Z-1-001/dcnv2-yolov8-neck-full/weights/best.pt',
            'DCNv2-FPN':  r'/home/migui/Downloads/dcnv2-yolov8-neck-fpn-20260318T004118Z-1-001/dcnv2-yolov8-neck-fpn/weights/best.pt',
            'DCNv2-Pan':  r'/home/migui/Downloads/dcnv2-yolov8-neck-pan-20260318T004653Z-1-001/dcnv2-yolov8-neck-pan/weights/best.pt',
            'DCNv2-Liu':  r'/home/migui/Downloads/dcnv2-yolov8-liu-20260318T004538Z-1-001/dcnv2-yolov8-liu/weights/best.pt',
        }
        output_base = f'/media/mydrive/GitHub/ultralytics/tracking/counting_{video_name}/dcnv2'
    elif args.model_type == 'dcnv3':
        models = {
            'DCNv3-Full': r'/home/migui/YOLO_outputs/100_dcnv3_yolov8n_full/weights/best.pt',
            'DCNv3-FPN':  r'/home/migui/YOLO_outputs/100_dcnv3_yolov8n-neck-fpn/weights/best.pt',
            'DCNv3-Pan':  r'/home/migui/YOLO_outputs/100_dcnv3_yolov8n_pan/weights/best.pt',
            'DCNv3-Liu':  r'/home/migui/YOLO_outputs/100_dcnv3_yolov8n_liu/weights/best.pt',
        }
        output_base = f'/media/mydrive/GitHub/ultralytics/tracking/counting_{video_name}/dcnv3'
    else:
        models = {
            'Vanilla-YOLOv8n': r'/home/migui/Downloads/100_yolov8n_300epochs_b32-20260318T004620Z-1-001/100_yolov8n_300epochs_b32/weights/best.pt',
        }
        output_base = f'/media/mydrive/GitHub/ultralytics/tracking/counting_{video_name}/vanilla'

    all_results = []

    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_name}: not found at {model_path}")
            continue

        output_dir = os.path.join(output_base, model_name)
        result = run_tracking(model_path, model_name, video_path, output_dir,
                              max_frames=args.max_frames, save_video=args.video)
        if result:
            all_results.append(result)

    # Combined summary
    if all_results:
        print(f"\n{'=' * 120}")
        print(f"  COMBINED SUMMARY - {args.model_type.upper()} models on {video_name}.mp4")
        print(f"{'=' * 120}")
        print(f"  {'Model':<15} {'MOTA':<10} {'IDF1':<10} {'IDSW':<8} {'MT%':<7} {'ML%':<7} {'GFLOPs':<9} {'FPS':<8} {'ClsSw':<8} {'TrkIDs':<8} {'Unique':<8} {'Dupes':<8} {'Car':<6} {'Moto':<6} {'Tri':<6} {'Van':<6} {'Bus':<6} {'Truck':<6} {'Count':<6} {'IDSw/hr':<8} {'ClsSw/hr':<9} {'Ctd_IDSw%':<11} {'Ctd_ClsSw%':<12} {'Runtime':<10}")
        print(f"  {'-' * 195}")
        for r in all_results:
            print(f"  {r['model']:<15} {r['mota']:<10.4f} {r['idf1']:<10.4f} {r['idsw']:<8} {r['pct_mt']:<7.1f} {r['pct_ml']:<7.1f} {r['gflops']:<9.2f} {r['fps']:<8.2f} {r['class_switches']:<8} {r['total_tracks']:<8} {r['unique_vehicles']:<8} {r['duplicate_tracks']:<8} {r['car']:<6} {r['motorcycle']:<6} {r['tricycle']:<6} {r['van']:<6} {r['bus']:<6} {r['truck']:<6} {r['vehicle_total']:<6} {r['idsw_per_hour']:<8.2f} {r['clssw_per_hour']:<9.2f} {r['pct_counted_id_switch']:<11.2f} {r['pct_counted_cls_switch']:<12.2f} {r['runtime']:<10}")

        # Summary CSV
        summary_file = os.path.join(output_base, 'summary.csv')
        os.makedirs(output_base, exist_ok=True)
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "MOTA", "IDF1", "IDSW", "Fragmentations",
                            "MostlyTracked", "Pct_MT", "MostlyLost", "Pct_ML",
                            "GFLOPs", "FPS", "Runtime",
                            "ClassSwitches", "TrackIDs", "UniqueVehicles", "DuplicateTracks",
                            "Car", "Motorcycle", "Tricycle", "Van", "Bus", "Truck", "LineCrossingCount",
                            "IDSw_per_Hour", "ClsSw_per_Hour", "IDSw_Pct", "ClsSw_Pct",
                            "Counted_with_IDSwitch", "Pct_Counted_IDSwitch",
                            "Counted_with_ClsSwitch", "Pct_Counted_ClsSwitch",
                            "Tracks_with_ClsSwitch", "Pct_Tracks_ClsSwitch"])
            for r in all_results:
                writer.writerow([r['model'], f"{r['mota']:.4f}", f"{r['idf1']:.4f}", r['idsw'],
                                r['fragmentations'],
                                r['mostly_tracked'], f"{r['pct_mt']:.1f}",
                                r['mostly_lost'], f"{r['pct_ml']:.1f}",
                                f"{r['gflops']:.2f}", f"{r['fps']:.2f}", r['runtime'],
                                r['class_switches'], r['total_tracks'],
                                r['unique_vehicles'], r['duplicate_tracks'],
                                r['car'], r['motorcycle'], r['tricycle'], r['van'], r['bus'], r['truck'],
                                r['vehicle_total'],
                                f"{r['idsw_per_hour']:.2f}", f"{r['clssw_per_hour']:.2f}",
                                f"{r['pct_idsw']:.4f}", f"{r['pct_clssw']:.4f}",
                                r['counted_with_id_switch'], f"{r['pct_counted_id_switch']:.2f}",
                                r['counted_with_cls_switch'], f"{r['pct_counted_cls_switch']:.2f}",
                                r['tracks_with_cls_switch'], f"{r['pct_tracks_cls_switch']:.2f}"])
        print(f"\n  Summary CSV: {summary_file}")


if __name__ == "__main__":
    main()
