#!/usr/bin/env python3
"""
DCNv2 vs DCNv3 Tracking Comparison on 6to6_g35.mp4
Metrics: MOTA, IDF1, ID Switches, FPS
+ Classification switching analysis (per frame & per second)
+ Object tracking timeline

Usage (run separately for each env):
  # DCNv2 in dcnv2 env:
  /home/migui/miniconda3/envs/dcnv2/bin/python run_comparison_6to6.py --model-type dcnv2
  # DCNv3 in dcn env:
  /home/migui/miniconda3/envs/dcn/bin/python run_comparison_6to6.py --model-type dcnv3
"""

import sys
import os
import argparse
import warnings
from collections import defaultdict
import time
import random

warnings.filterwarnings('ignore')
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np

# Vehicle class colors (BGR)
CLASS_COLORS = {
    "car": (55, 250, 250),
    "motorcycle": (83, 179, 36),
    "tricycle": (83, 50, 250),
    "bus": (245, 61, 184),
    "van": (255, 221, 51),
    "truck": (49, 147, 245),
}

SWITCH_DISPLAY_FRAMES = 45  # Show switch label for ~1.5s at 30fps

# Per-video counting line configs: (x1_frac, y1_frac) -> (x2_frac, y2_frac)
COUNTING_LINES = {
    '6to6_g2':  [(0.02, 0.55), (0.70, 0.38)],
    '6to6_g3':  [(0.0, 0.52), (1.0, 0.62)],
    '6to6_g29': [(0.0, 0.42), (0.55, 0.72)],
    '6to6_g35': [(0.0, 0.45), (1.0, 0.75)],
}
DEFAULT_COUNTING_LINE = [(0.0, 0.45), (1.0, 0.75)]


def get_unique_color(obj_id):
    random.seed(obj_id)
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))


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


def run_tracking(model_path, model_name, video_path, output_dir, max_frames=0, label='', save_video=False):
    """Run ByteTrack tracking with classification switch detection"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    line_cfg = COUNTING_LINES.get(video_name, DEFAULT_COUNTING_LINE)
    (lx1f, ly1f), (lx2f, ly2f) = line_cfg

    print("\n" + "=" * 70)
    print(f"  {model_name} | Video: {video_name}.mp4" + (" [VIDEO]" if save_video else ""))
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {model_path}")
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
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total_frames} frames @ {fps}fps ({frame_width}x{frame_height})")

    # Setup video writer
    video_writer = None
    panel_width = 280
    if save_video:
        suffix = f"_{label}" if label else ""
        video_file = os.path.join(output_dir, f"{model_name}_{video_name}{suffix}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_file, fourcc, fps, (frame_width + panel_width, frame_height))
        print(f"Video output: {video_file}")

    # Tracking state
    track_history = {}        # track_id -> list of frame numbers
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
    track_classes = {}           # track_id -> current class
    track_class_history = {}     # track_id -> [(frame, class), ...]
    class_switches = []          # [(frame, track_id, old_class, new_class), ...]

    # Object tracking timeline
    track_active_frames = {}     # track_id -> set of frames where active

    # Vehicle counting (line crossing)
    vehicle_counts = {"car": 0, "motorcycle": 0, "tricycle": 0, "van": 0, "bus": 0, "truck": 0}
    counted_objects = set()
    object_positions = {}  # track_id -> 'above'/'below'

    # Video annotation state
    class_switch_display = {}    # track_id -> (text, frames_remaining)
    trajectories = {}            # track_id -> list of (cx, cy)
    trajectory_colors = {}
    max_trail_length = 50

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

        if max_frames > 0 and frame_count > max_frames:
            break

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
                    if save_video:
                        class_switch_display[track_id] = (f"{old_class}->{class_name}", SWITCH_DISPLAY_FRAMES)
                track_classes[track_id] = class_name

                # Counting line detection (diagonal: 0,45% -> width,75%)
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                line_y_at_x = np.interp(cx, [lx1f * frame_width, lx2f * frame_width],
                                            [ly1f * frame_height, ly2f * frame_height])
                current_pos = 'above' if cy < line_y_at_x else 'below'
                if track_id in object_positions:
                    if object_positions[track_id] != current_pos and track_id not in counted_objects:
                        if class_name in vehicle_counts:
                            vehicle_counts[class_name] += 1
                            counted_objects.add(track_id)
                object_positions[track_id] = current_pos

            total_detections += len(track_ids)

        # Draw annotations on frame for video output
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

                # Trajectory
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if tid not in trajectories:
                    trajectories[tid] = []
                    trajectory_colors[tid] = get_unique_color(tid)
                trajectories[tid].append((cx, cy))
                if len(trajectories[tid]) > max_trail_length:
                    trajectories[tid].pop(0)

                # Bounding box
                color = CLASS_COLORS.get(cls_name.lower(), (0, 255, 0))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{cls_name} ID:{tid}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Switch annotation
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

            # Draw trajectories
            for tid, trail in trajectories.items():
                if len(trail) > 1:
                    tcolor = trajectory_colors.get(tid, (255, 255, 255))
                    for j in range(1, len(trail)):
                        thickness = max(2, int(8 * j / len(trail)))
                        cv2.line(frame, trail[j - 1], trail[j], tcolor, thickness)
                    cv2.circle(frame, trail[-1], 6, tcolor, -1)

            # Clean up old trajectories
            for tid in list(trajectories.keys()):
                if tid not in active_ids:
                    if len(trajectories[tid]) > 5:
                        trajectories[tid] = trajectories[tid][5:]
                    else:
                        del trajectories[tid]
                        trajectory_colors.pop(tid, None)

            # Draw counting line on frame
            line_pt1 = (int(lx1f * frame_width), int(ly1f * frame_height))
            line_pt2 = (int(lx2f * frame_width), int(ly2f * frame_height))
            cv2.line(frame, line_pt1, line_pt2, (147, 20, 255), 3)

            # Build side panel
            extended = np.zeros((frame_height, frame_width + panel_width, 3), dtype=np.uint8)
            extended[:, :frame_width] = frame
            extended[:, frame_width:] = (40, 40, 40)

            # Panel content
            cv2.putText(extended, model_name, (frame_width + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(extended, "ByteTrack", (frame_width + 10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.line(extended, (frame_width + 10, 65), (frame_width + panel_width - 10, 65), (255, 255, 255), 1)

            yp = 95
            cv2.putText(extended, f"Active Tracks: {len(active_ids)}", (frame_width + 15, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            yp += 25
            cv2.putText(extended, f"Total Tracks: {len(track_history)}", (frame_width + 15, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
            yp += 25
            cv2.putText(extended, f"ID Switches: {identity_switches}", (frame_width + 15, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            yp += 25
            cv2.putText(extended, f"Class Switches: {len(class_switches)}", (frame_width + 15, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

            yp += 40
            cv2.line(extended, (frame_width + 10, yp), (frame_width + panel_width - 10, yp), (100, 100, 100), 1)
            yp += 25
            cv2.putText(extended, "RECENT SWITCHES", (frame_width + 10, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            yp += 20

            # Show last 8 switches
            recent = class_switches[-8:] if class_switches else []
            for fnum, stid, old_c, new_c in recent:
                txt = f"F{fnum} ID:{stid} {old_c}->{new_c}"
                cv2.putText(extended, txt, (frame_width + 15, yp),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 255), 1)
                yp += 16

            # Vehicle counts section
            yp += 15
            cv2.line(extended, (frame_width + 10, yp), (frame_width + panel_width - 10, yp), (100, 100, 100), 1)
            yp += 25
            cv2.putText(extended, "VEHICLE COUNT", (frame_width + 10, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            yp += 20
            for vtype, vcount in vehicle_counts.items():
                if vcount > 0:
                    color = CLASS_COLORS.get(vtype, (255, 255, 255))
                    cv2.putText(extended, f"{vtype.upper()}: {vcount}", (frame_width + 15, yp),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    yp += 18
            total_count = sum(vehicle_counts.values())
            cv2.putText(extended, f"TOTAL: {total_count}", (frame_width + 15, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Frame counter at bottom
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

        if frame_count % 100 == 0:
            pct = (frame_count / total_frames) * 100
            print(f"\rProgress: {pct:.1f}% ({frame_count}/{total_frames})", end='', flush=True)

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
    for track_id, frames in track_history.items():
        if len(frames) > 1:
            sorted_frames = sorted(frames)
            for i in range(1, len(sorted_frames)):
                if sorted_frames[i] - sorted_frames[i - 1] > 5:
                    fragmentations += 1

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

    # --- Metrics table ---
    print(f"\n  TRACKING METRICS")
    print(f"  {'Metric':<20} {'Value':<10}")
    print(f"  {'-' * 30}")
    print(f"  {'MOTA':<20} {mota:.4f}")
    print(f"  {'IDF1':<20} {idf1:.4f}")
    print(f"  {'ID Switches':<20} {identity_switches}")
    print(f"  {'FPS':<20} {processing_fps:.2f}")

    # --- Object tracking timeline ---
    print(f"  {'Track IDs Assigned':<20} {total_tracks}")
    print(f"  {'Unique Vehicles':<20} {unique_vehicles}")
    print(f"  {'Duplicate Tracks':<20} {duplicate_tracks}")

    print(f"\n  TRACKED OBJECTS ({total_tracks} total)")
    print(f"  {'ID':<6} {'Class':<12} {'First Frame':<14} {'Last Frame':<14} {'Frames Active':<15}")
    print(f"  {'-' * 60}")
    for tid in sorted(track_history.keys()):
        # Get dominant class (most frequent)
        cls_hist = track_class_history.get(tid, [])
        if cls_hist:
            cls_counts = defaultdict(int)
            for _, c in cls_hist:
                cls_counts[c] += 1
            dominant_class = max(cls_counts, key=cls_counts.get)
        else:
            dominant_class = "unknown"
        first_f = track_first_frame[tid]
        last_f = track_last_frame[tid]
        n_frames = len(track_active_frames.get(tid, []))
        print(f"  {tid:<6} {dominant_class:<12} {first_f:<14} {last_f:<14} {n_frames:<15}")

    # --- Classification switches per second (FIRST) ---
    print(f"\n  CLASSIFICATION SWITCHES: {len(class_switches)} total")
    print(f"\n  SWITCHES PER SECOND:")
    if class_switches and fps > 0:
        switches_per_sec = defaultdict(list)
        for fnum, tid, old_cls, new_cls in class_switches:
            sec = (fnum - 1) // fps
            switches_per_sec[sec].append((fnum, tid, old_cls, new_cls))

        total_seconds = (frame_count - 1) // fps + 1
        print(f"  {'Second':<10} {'Count':<8} {'Details':<40}")
        print(f"  {'-' * 55}")
        for sec in range(total_seconds):
            sw_list = switches_per_sec.get(sec, [])
            if sw_list:
                details = "; ".join([f"ID:{tid} {o}->{n}" for _, tid, o, n in sw_list])
                print(f"  {sec:<10} {len(sw_list):<8} {details}")
        print(f"\n  Seconds with switches: {len(switches_per_sec)}/{total_seconds}")
    else:
        print(f"  (No switches)")

    # --- Classification switches per frame (AFTER per-second) ---
    if class_switches:
        print(f"\n  SWITCHES PER FRAME:")
        print(f"  {'Frame':<8} {'ID':<6} {'From':<12} {'To':<12}")
        print(f"  {'-' * 38}")
        for fnum, tid, old_cls, new_cls in class_switches:
            print(f"  {fnum:<8} {tid:<6} {old_cls:<12} {new_cls:<12}")

    # --- Per-object class history (only objects with switches) ---
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
            print(f"  ID:{tid}: {' -> '.join(unique_seq)}")
    if not any_switched:
        print(f"  (No objects switched class)")

    print(f"\n{'=' * 70}\n")

    # --- Save report ---
    suffix = f"_{label}" if label else ""
    report_file = os.path.join(output_dir, f"{model_name}_{video_name}{suffix}_report.txt")
    with open(report_file, 'w') as f:
        f.write(f"{model_name} - {video_name}.mp4 Tracking Report\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Frames: {frame_count} | FPS(video): {fps}\n\n")

        f.write(f"TRACKING METRICS\n")
        f.write(f"{'-' * 30}\n")
        f.write(f"MOTA:         {mota:.4f}\n")
        f.write(f"IDF1:         {idf1:.4f}\n")
        f.write(f"ID Switches:  {identity_switches}\n")
        f.write(f"FPS:          {processing_fps:.2f}\n\n")

        f.write(f"Track IDs Assigned: {total_tracks}\n")
        f.write(f"Unique Vehicles:    {unique_vehicles}\n")
        f.write(f"Duplicate Tracks:   {duplicate_tracks}\n\n")
        f.write(f"TRACKED OBJECTS ({total_tracks} total)\n")
        f.write(f"{'-' * 60}\n")
        f.write(f"{'ID':<6} {'Class':<12} {'First':<10} {'Last':<10} {'Frames':<10}\n")
        for tid in sorted(track_history.keys()):
            cls_hist = track_class_history.get(tid, [])
            if cls_hist:
                cls_counts = defaultdict(int)
                for _, c in cls_hist:
                    cls_counts[c] += 1
                dominant_class = max(cls_counts, key=cls_counts.get)
            else:
                dominant_class = "unknown"
            f.write(f"{tid:<6} {dominant_class:<12} {track_first_frame[tid]:<10} {track_last_frame[tid]:<10} {len(track_active_frames.get(tid, [])):<10}\n")

        f.write(f"\nCLASSIFICATION SWITCHES: {len(class_switches)} total\n")
        f.write(f"{'-' * 40}\n")
        if class_switches:
            f.write(f"{'Frame':<8} {'ID':<6} {'From':<12} {'To':<12}\n")
            for fnum, tid, old_cls, new_cls in class_switches:
                f.write(f"{fnum:<8} {tid:<6} {old_cls:<12} {new_cls:<12}\n")

            f.write(f"\nSWITCHES PER SECOND:\n")
            switches_per_sec = defaultdict(list)
            for fnum, tid, old_cls, new_cls in class_switches:
                sec = (fnum - 1) // fps
                switches_per_sec[sec].append((fnum, tid, old_cls, new_cls))
            total_seconds = (frame_count - 1) // fps + 1
            for sec in range(total_seconds):
                sw_list = switches_per_sec.get(sec, [])
                if sw_list:
                    details = "; ".join([f"ID:{tid} {o}->{n}" for _, tid, o, n in sw_list])
                    f.write(f"  Sec {sec}: {len(sw_list)} - {details}\n")
            f.write(f"\nSeconds with switches: {len(switches_per_sec)}/{total_seconds}\n")

            f.write(f"\nPER-OBJECT CLASS HISTORY (switched only):\n")
            for tid in sorted(track_class_history.keys()):
                history = track_class_history[tid]
                unique_seq = []
                for _, cls in history:
                    if not unique_seq or unique_seq[-1] != cls:
                        unique_seq.append(cls)
                if len(unique_seq) > 1:
                    f.write(f"  ID:{tid}: {' -> '.join(unique_seq)}\n")
        else:
            f.write("(No switches)\n")

        # Vehicle counts in report
        total_counted = sum(vehicle_counts.values())
        f.write(f"\nVEHICLE COUNTS (Line Crossing)\n")
        f.write(f"{'-' * 30}\n")
        for vtype, vcount in vehicle_counts.items():
            f.write(f"  {vtype.upper()}: {vcount}\n")
        f.write(f"  TOTAL: {total_counted}\n")

    total_counted = sum(vehicle_counts.values())
    print(f"Report saved: {report_file}")

    # Print vehicle counts
    if total_counted > 0:
        print(f"\n  VEHICLE COUNTS (line crossing):")
        for vtype, vcount in vehicle_counts.items():
            if vcount > 0:
                print(f"  {vtype.upper()}: {vcount}")
        print(f"  TOTAL: {total_counted}")

    return {
        'model': model_name,
        'mota': mota,
        'idf1': idf1,
        'idsw': identity_switches,
        'fps': processing_fps,
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['dcnv2', 'dcnv3'], required=True)
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Max frames to process (0=all). E.g. 225=15s, 450=30s, 900=1min, 4500=5min at 15fps')
    parser.add_argument('--label', type=str, default='',
                        help='Label for this run (e.g. "15sec", "30sec")')
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
            'DCNv2-Full': r'/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Full.pt',
            'DCNv2-FPN': r'/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-FPN.pt',
            'DCNv2-Pan': r'/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Pan.pt',
            'DCNv2-Liu': r'/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-LIU.pt',
        }
        output_base = f'/media/mydrive/GitHub/ultralytics/tracking/comparison_{video_name}/dcnv2'
    else:
        models = {
            'DCNv3-Full': r'/home/migui/YOLO_outputs/100_dcnv3_yolov8m_full_second/weights/best.pt',
            'DCNv3-FPN': r'/home/migui/YOLO_outputs/100_dcnv3_yolov8m_fpn_second/weights/best.pt',
            'DCNv3-Pan': r'/home/migui/YOLO_outputs/100_dcnv3_yolov8m_pan_second/weights/best.pt',
            'DCNv3-Liu': r'/home/migui/YOLO_outputs/100_dcnv3_yolov8m_liu_second/weights/best.pt',
        }
        output_base = f'/media/mydrive/GitHub/ultralytics/tracking/comparison_{video_name}/dcnv3'

    all_results = []

    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_name}: not found at {model_path}")
            continue

        output_dir = os.path.join(output_base, model_name)
        result = run_tracking(model_path, model_name, video_path, output_dir,
                              max_frames=args.max_frames, label=args.label,
                              save_video=args.video)
        if result:
            all_results.append(result)

    # Print combined summary table
    if all_results:
        print(f"\n{'=' * 70}")
        print(f"  COMBINED SUMMARY - {args.model_type.upper()} models on {video_name}.mp4")
        print(f"{'=' * 70}")
        print(f"  {'Model':<15} {'MOTA':<10} {'IDF1':<10} {'IDSW':<8} {'FPS':<8} {'ClsSw':<8} {'TrkIDs':<8} {'Unique':<8} {'Dupes':<8} {'Car':<6} {'Moto':<6} {'Tri':<6} {'Van':<6} {'Bus':<6} {'Truck':<6} {'Count':<6}")
        print(f"  {'-' * 131}")
        for r in all_results:
            print(f"  {r['model']:<15} {r['mota']:<10.4f} {r['idf1']:<10.4f} {r['idsw']:<8} {r['fps']:<8.2f} {r['class_switches']:<8} {r['total_tracks']:<8} {r['unique_vehicles']:<8} {r['duplicate_tracks']:<8} {r['car']:<6} {r['motorcycle']:<6} {r['tricycle']:<6} {r['van']:<6} {r['bus']:<6} {r['truck']:<6} {r['vehicle_total']:<6}")

        # Save summary CSV
        summary_file = os.path.join(output_base, 'summary.csv')
        os.makedirs(output_base, exist_ok=True)
        with open(summary_file, 'w') as f:
            f.write("Model,MOTA,IDF1,IDSW,FPS,ClassSwitches,TrackIDs,UniqueVehicles,DuplicateTracks,Car,Motorcycle,Tricycle,Van,Bus,Truck,LineCrossingCount\n")
            for r in all_results:
                f.write(f"{r['model']},{r['mota']:.4f},{r['idf1']:.4f},{r['idsw']},{r['fps']:.2f},{r['class_switches']},{r['total_tracks']},{r['unique_vehicles']},{r['duplicate_tracks']},{r['car']},{r['motorcycle']},{r['tricycle']},{r['van']},{r['bus']},{r['truck']},{r['vehicle_total']}\n")
        print(f"\n  Summary CSV: {summary_file}")


if __name__ == "__main__":
    main()
