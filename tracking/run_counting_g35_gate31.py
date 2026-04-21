#!/usr/bin/env python3
"""
Full 12-hour tracking + counting on 6to6_g35.mp4 using the Gate3.1 counting line.

Gate3.1 counting line: vertical line at x=0.885 (1700px @ 1920w),
from y=0.0 to y=0.194 (0–210px @ 1080h).

Outputs to: tracking/counting_g35_gate31/{dcnv2|dcnv3|vanilla}/{ModelName}/

Usage:
  # DCNv2 env:
  /home/migui/miniconda3/envs/dcnv2/bin/python run_counting_g35_gate31.py --model-type dcnv2
  # DCNv3 env:
  /home/migui/miniconda3/envs/dcn/bin/python run_counting_g35_gate31.py --model-type dcnv3
  # Vanilla env:
  /home/migui/miniconda3/envs/dcnv2/bin/python run_counting_g35_gate31.py --model-type vanilla
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

# Gate3.1 counting line (vertical, x=0.885, y=0.0 to 0.194)
GATE31_LINE = [(0.885, 0.0), (0.885, 0.194)]

VIDEO_PATH = '/media/mydrive/GitHub/ultralytics/videos/6to6_g35.mp4'
OUTPUT_BASE = '/media/mydrive/GitHub/ultralytics/tracking/counting_g35_gate31'


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


def merge_duplicate_tracks(track_first_frame, track_last_frame, track_first_box, track_last_box,
                           max_gap=60, min_iou=0.2):
    track_ids = sorted(track_first_frame.keys(), key=lambda t: track_first_frame[t])
    merged = {}
    for tid in track_ids:
        best_match = None
        best_iou = min_iou
        for prev_tid in track_ids:
            if prev_tid == tid:
                break
            canonical = merged.get(prev_tid, prev_tid)
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


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_tracking(model_path, model_name, output_dir):
    (lx1f, ly1f), (lx2f, ly2f) = GATE31_LINE

    print("\n" + "=" * 70)
    print(f"  {model_name} | 6to6_g35.mp4 | Gate3.1 counting line")
    print(f"  Line: ({lx1f},{ly1f}) -> ({lx2f},{ly2f})")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

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

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / fps
    print(f"Video: {total_frames} frames @ {fps}fps ({frame_width}x{frame_height})")
    print(f"Duration: {format_time(video_duration)}")

    # Tracking state
    track_history = {}
    track_first_frame = {}
    track_last_frame = {}
    track_first_box = {}
    track_last_box = {}

    lost_tracks = {}
    identity_switches = 0
    lost_track_window = 30
    iou_threshold = 0.3

    track_classes = {}
    track_class_history = {}
    class_switches = []

    track_active_frames = {}

    vehicle_counts = {c: 0 for c in VEHICLE_CLASSES}
    counted_objects = set()
    object_positions = {}

    interval_counts = defaultdict(lambda: {c: 0 for c in VEHICLE_CLASSES})
    interval_counted_objects = defaultdict(set)
    interval_id_switches = defaultdict(int)
    interval_class_switches = defaultdict(int)
    interval_active_tracks = defaultdict(set)

    prev_track_ids = set()
    frame_count = 0
    total_detections = 0
    start_time = time.time()

    consecutive_failures = 0
    max_consecutive_failures = 10

    print("Processing frames...")

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

                track_class_history[track_id].append((frame_count, class_name))
                if track_id in track_classes and track_classes[track_id] != class_name:
                    old_class = track_classes[track_id]
                    class_switches.append((frame_count, track_id, old_class, class_name))
                    interval_class_switches[current_interval] += 1
                track_classes[track_id] = class_name

                interval_active_tracks[current_interval].add(track_id)

                # Gate3.1 line crossing: vertical line at x = lx1f * frame_width
                # Use np.interp for consistency (handles vertical line via clamping)
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

            total_detections += len(track_ids)

        # Detect lost tracks for IDSW
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

    end_time = time.time()
    runtime = end_time - start_time
    processing_fps = frame_count / runtime if runtime > 0 else 0

    total_tracks = len(track_history)
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

    total_track_frames = sum(len(f) for f in track_history.values())
    total_errors = identity_switches + fragmentations
    mota = max(0, min(1, 1 - (total_errors / max(1, total_track_frames))))
    id_tp = total_track_frames - total_errors
    idf1 = (2 * id_tp) / max(1, 2 * id_tp + identity_switches + fragmentations)

    total_counted = sum(vehicle_counts.values())
    video_hours = video_time / 3600 if video_time > 0 else 1
    idsw_per_hour = identity_switches / video_hours
    clssw_per_hour = len(class_switches) / video_hours

    tracks_with_cls_switch = sum(1 for tid in track_class_history
                                 if len(set(cls for _, cls in track_class_history[tid])) > 1)
    pct_tracks_cls_switch = (tracks_with_cls_switch / max(1, total_tracks)) * 100

    counted_with_cls_switch = sum(1 for tid in counted_objects
                                  if tid in track_class_history
                                  and len(set(cls for _, cls in track_class_history[tid])) > 1)
    pct_counted_cls_switch = (counted_with_cls_switch / max(1, total_counted)) * 100

    from collections import defaultdict as _dd
    canonical_to_tracks = _dd(list)
    for tid, canonical in track_merge_map.items():
        canonical_to_tracks[canonical].append(tid)
    counted_with_id_switch = sum(
        1 for tid in counted_objects
        if len(canonical_to_tracks.get(track_merge_map.get(tid, tid), [tid])) > 1
    )
    pct_counted_id_switch = (counted_with_id_switch / max(1, total_counted)) * 100

    track_frame_switches = defaultdict(int)
    for fnum, tid, old, new in class_switches:
        track_frame_switches[tid] += 1

    track_second_switches = defaultdict(set)
    for fnum, tid, old, new in class_switches:
        sec = (fnum - 1) // fps
        track_second_switches[tid].add(sec)
    track_sec_switch_counts = {tid: len(secs) for tid, secs in track_second_switches.items()}

    counted_frame_switches = {tid: track_frame_switches[tid] for tid in counted_objects
                              if track_frame_switches[tid] > 0}
    total_counted_frame_switches = sum(counted_frame_switches.values())
    avg_frame_switches_per_affected = total_counted_frame_switches / max(1, counted_with_cls_switch)

    counted_sec_switches = {tid: track_sec_switch_counts[tid] for tid in counted_objects
                            if tid in track_sec_switch_counts}
    total_counted_sec_switches = sum(counted_sec_switches.values())
    avg_sec_switches_per_affected = total_counted_sec_switches / max(1, counted_with_cls_switch)

    total_frame_switches = len(class_switches)
    total_sec_switches = sum(track_sec_switch_counts.values())
    sec_switch_per_hour = total_sec_switches / video_hours

    num_intervals = max(interval_counts.keys()) + 1 if interval_counts else 0
    if num_intervals == 0 and frame_count > 0:
        num_intervals = int(video_time // INTERVAL_SECONDS) + 1

    # Print results
    print(f"\n\n{'=' * 70}")
    print(f"  RESULTS: {model_name}  [Gate3.1 line on 6to6_g35.mp4]")
    print(f"{'=' * 70}")
    print(f"\n  TRACKING METRICS")
    print(f"  {'MOTA':<20} {mota:.4f}")
    print(f"  {'IDF1':<20} {idf1:.4f}")
    print(f"  {'ID Switches':<20} {identity_switches}")
    print(f"  {'Fragmentations':<20} {fragmentations}")
    print(f"  {'Mostly Tracked (MT)':<20} {mostly_tracked} ({pct_mt:.1f}%)")
    print(f"  {'Mostly Lost (ML)':<20} {mostly_lost} ({pct_ml:.1f}%)")
    print(f"  {'Track IDs Assigned':<20} {total_tracks}")
    print(f"  {'Unique Vehicles':<20} {unique_vehicles}")
    print(f"  {'GFLOPs':<20} {model_gflops:.2f}")
    print(f"  {'Processing FPS':<20} {processing_fps:.2f}")
    print(f"  {'Runtime':<20} {format_time(runtime)}")
    print(f"  {'Class Switches':<20} {len(class_switches)}")

    print(f"\n  TOTAL VEHICLE COUNTS (Gate3.1 line crossing)")
    for vtype in VEHICLE_CLASSES:
        print(f"  {vtype.upper():<15} {vehicle_counts[vtype]}")
    print(f"  {'TOTAL':<15} {total_counted}")

    print(f"\n  SWITCH STATISTICS (over {format_time(video_time)})")
    print(f"  {'ID Switches / Hour':<40} {idsw_per_hour:.2f}")
    print(f"  {'Counted Vehs with ID Switch':<40} {counted_with_id_switch}/{total_counted} ({pct_counted_id_switch:.2f}%)")
    print(f"  {'Class Switches / Hour':<40} {clssw_per_hour:.2f}")
    print(f"  {'Tracks with Class Switch':<40} {tracks_with_cls_switch}/{total_tracks} ({pct_tracks_cls_switch:.2f}%)")
    print(f"  {'Counted Vehs with Class Switch':<40} {counted_with_cls_switch}/{total_counted} ({pct_counted_cls_switch:.2f}%)")

    # Save text report
    report_file = os.path.join(output_dir, f"{model_name}_report.txt")
    with open(report_file, 'w') as f:
        f.write(f"{model_name} - 6to6_g35.mp4 [Gate3.1 counting line]\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Video: {VIDEO_PATH}\n")
        f.write(f"Counting line: Gate3.1 ({lx1f},{ly1f}) -> ({lx2f},{ly2f})\n")
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

        f.write(f"\nSWITCH STATISTICS (over {format_time(video_time)})\n{'-' * 60}\n")
        f.write(f"ID Switches Total:                      {identity_switches}\n")
        f.write(f"ID Switches / Hour:                     {idsw_per_hour:.2f}\n")
        f.write(f"Counted Vehs with ID Switch:            {counted_with_id_switch}/{total_counted} ({pct_counted_id_switch:.2f}%)\n")
        f.write(f"Class Switches / Hour:                  {clssw_per_hour:.2f}\n")
        f.write(f"Tracks with Class Switch:               {tracks_with_cls_switch}/{total_tracks} ({pct_tracks_cls_switch:.2f}%)\n")
        f.write(f"Counted Vehs with Class Switch:         {counted_with_cls_switch}/{total_counted} ({pct_counted_cls_switch:.2f}%)\n")
        f.write(f"\nCLASS SWITCHES - PER FRAME\n{'-' * 60}\n")
        f.write(f"Total (all tracks):                     {total_frame_switches}\n")
        f.write(f"On Counted Vehicles:                    {total_counted_frame_switches}\n")
        f.write(f"Avg per Affected Counted Veh:           {avg_frame_switches_per_affected:.2f}\n")
        f.write(f"Per Hour:                               {clssw_per_hour:.2f}\n")
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

    # Per-5-min interval CSV
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
        'vehicle_total': total_counted,
        'car': vehicle_counts.get('car', 0),
        'motorcycle': vehicle_counts.get('motorcycle', 0),
        'tricycle': vehicle_counts.get('tricycle', 0),
        'van': vehicle_counts.get('van', 0),
        'bus': vehicle_counts.get('bus', 0),
        'truck': vehicle_counts.get('truck', 0),
        'idsw_per_hour': idsw_per_hour,
        'clssw_per_hour': clssw_per_hour,
        'tracks_with_cls_switch': tracks_with_cls_switch,
        'pct_tracks_cls_switch': pct_tracks_cls_switch,
        'counted_with_cls_switch': counted_with_cls_switch,
        'pct_counted_cls_switch': pct_counted_cls_switch,
        'counted_with_id_switch': counted_with_id_switch,
        'pct_counted_id_switch': pct_counted_id_switch,
        'mostly_tracked': mostly_tracked,
        'pct_mt': pct_mt,
        'mostly_lost': mostly_lost,
        'pct_ml': pct_ml,
        'gflops': model_gflops,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['dcnv2', 'dcnv3', 'vanilla'], required=True)
    args = parser.parse_args()

    if args.model_type == 'dcnv2':
        models = {
            'DCNv2-Full': '/home/migui/Downloads/dcnv2-yolov8-neck-full-20260318T004120Z-1-001/dcnv2-yolov8-neck-full/weights/best.pt',
            'DCNv2-FPN':  '/home/migui/Downloads/dcnv2-yolov8-neck-fpn-20260318T004118Z-1-001/dcnv2-yolov8-neck-fpn/weights/best.pt',
            'DCNv2-Pan':  '/home/migui/Downloads/dcnv2-yolov8-neck-pan-20260318T004653Z-1-001/dcnv2-yolov8-neck-pan/weights/best.pt',
            'DCNv2-Liu':  '/home/migui/Downloads/dcnv2-yolov8-liu-20260318T004538Z-1-001/dcnv2-yolov8-liu/weights/best.pt',
        }
        output_base = os.path.join(OUTPUT_BASE, 'dcnv2')
    elif args.model_type == 'dcnv3':
        models = {
            'DCNv3-Full': '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_full/weights/best.pt',
            'DCNv3-FPN':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n-neck-fpn/weights/best.pt',
            'DCNv3-Pan':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_pan/weights/best.pt',
            'DCNv3-Liu':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_liu/weights/best.pt',
        }
        output_base = os.path.join(OUTPUT_BASE, 'dcnv3')
    else:
        models = {
            'Vanilla-YOLOv8n': '/home/migui/Downloads/100_yolov8n_300epochs_b32-20260318T004620Z-1-001/100_yolov8n_300epochs_b32/weights/best.pt',
        }
        output_base = os.path.join(OUTPUT_BASE, 'vanilla')

    all_results = []
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_name}: not found at {model_path}")
            continue
        output_dir = os.path.join(output_base, model_name)
        result = run_tracking(model_path, model_name, output_dir)
        if result:
            all_results.append(result)

    if all_results:
        # Write summary CSV
        summary_csv = os.path.join(output_base, 'summary.csv')
        fieldnames = ['Model', 'MOTA', 'IDF1', 'IDSW', 'Fragmentations',
                      'MostlyTracked', 'Pct_MT', 'MostlyLost', 'Pct_ML',
                      'GFLOPs', 'FPS', 'Runtime',
                      'ClassSwitches', 'TotalTracks', 'UniqueVehicles',
                      'LineCrossingCount', 'Car', 'Motorcycle', 'Tricycle', 'Van', 'Bus', 'Truck',
                      'IDSw_per_hour', 'ClsSw_per_hour',
                      'Counted_with_IDSw', 'Pct_Counted_IDSw',
                      'Tracks_with_ClsSw', 'Pct_Tracks_ClsSw',
                      'Counted_with_ClsSw', 'Pct_Counted_ClsSw']
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow({
                    'Model': r['model'],
                    'MOTA': f"{r['mota']:.4f}",
                    'IDF1': f"{r['idf1']:.4f}",
                    'IDSW': r['idsw'],
                    'Fragmentations': r['fragmentations'],
                    'MostlyTracked': r['mostly_tracked'],
                    'Pct_MT': f"{r['pct_mt']:.1f}",
                    'MostlyLost': r['mostly_lost'],
                    'Pct_ML': f"{r['pct_ml']:.1f}",
                    'GFLOPs': f"{r['gflops']:.2f}",
                    'FPS': f"{r['fps']:.2f}",
                    'Runtime': r['runtime'],
                    'ClassSwitches': r['class_switches'],
                    'TotalTracks': r['total_tracks'],
                    'UniqueVehicles': r['unique_vehicles'],
                    'LineCrossingCount': r['vehicle_total'],
                    'Car': r['car'],
                    'Motorcycle': r['motorcycle'],
                    'Tricycle': r['tricycle'],
                    'Van': r['van'],
                    'Bus': r['bus'],
                    'Truck': r['truck'],
                    'IDSw_per_hour': f"{r['idsw_per_hour']:.2f}",
                    'ClsSw_per_hour': f"{r['clssw_per_hour']:.2f}",
                    'Counted_with_IDSw': r['counted_with_id_switch'],
                    'Pct_Counted_IDSw': f"{r['pct_counted_id_switch']:.2f}",
                    'Tracks_with_ClsSw': r['tracks_with_cls_switch'],
                    'Pct_Tracks_ClsSw': f"{r['pct_tracks_cls_switch']:.2f}",
                    'Counted_with_ClsSw': r['counted_with_cls_switch'],
                    'Pct_Counted_ClsSw': f"{r['pct_counted_cls_switch']:.2f}",
                })
        print(f"\nSummary CSV saved: {summary_csv}")

        print(f"\n{'=' * 80}")
        print(f"  SUMMARY — {args.model_type.upper()} | 6to6_g35.mp4 [Gate3.1 line]")
        print(f"{'=' * 80}")
        print(f"  {'Model':<18} {'Count':<8} {'Car':<6} {'Moto':<6} {'Tri':<6} {'Van':<6} {'Bus':<6} {'Truck':<6} {'IDSW':<8} {'FPS'}")
        print(f"  {'-'*75}")
        for r in all_results:
            print(f"  {r['model']:<18} {r['vehicle_total']:<8} "
                  f"{r['car']:<6} {r['motorcycle']:<6} {r['tricycle']:<6} "
                  f"{r['van']:<6} {r['bus']:<6} {r['truck']:<6} "
                  f"{r['idsw']:<8} {r['fps']:.2f}")


if __name__ == '__main__':
    main()
