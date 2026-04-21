#!/usr/bin/env python3
"""
Benchmark detection-only FPS for all models (no tracking, no counting).
Runs model.predict() on frames sampled from the video and measures throughput.

Usage:
  /home/migui/miniconda3/envs/dcnv2/bin/python benchmark_detection_fps.py --model-type dcnv2
  /home/migui/miniconda3/envs/dcn/bin/python benchmark_detection_fps.py --model-type dcnv3
  /home/migui/miniconda3/envs/dcnv2/bin/python benchmark_detection_fps.py --model-type vanilla
"""

import sys
import os
import argparse
import warnings
import time
import csv

warnings.filterwarnings('ignore')
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np
import torch

VIDEO_PATH = '/media/mydrive/GitHub/ultralytics/videos/6to6_g35.mp4'
WARMUP_FRAMES = 50    # frames to discard before timing starts
BENCHMARK_FRAMES = 500  # frames to time


def benchmark_model(model_path, model_name):
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    try:
        model = YOLO(model_path)
        print(f"  Model loaded")
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"  Cannot open video")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {frame_width}x{frame_height}, {total_frames} frames")

    # Collect frames into memory for clean timing (avoids disk I/O skewing results)
    needed = WARMUP_FRAMES + BENCHMARK_FRAMES
    # Seek to middle of video to avoid any corrupt frames near the start
    start_frame = total_frames // 4
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    while len(frames) < needed:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < needed:
        print(f"  Only got {len(frames)} frames, need {needed}")
        needed = len(frames)
        warmup = min(WARMUP_FRAMES, needed // 5)
        bench = needed - warmup
    else:
        warmup = WARMUP_FRAMES
        bench = BENCHMARK_FRAMES

    print(f"  Warming up ({warmup} frames)...")
    for i in range(warmup):
        _ = model.predict(frames[i], conf=0.5, verbose=False)

    # Sync GPU before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"  Benchmarking ({bench} frames)...")
    t_start = time.perf_counter()
    for i in range(warmup, warmup + bench):
        _ = model.predict(frames[i], conf=0.5, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    fps = bench / elapsed
    ms_per_frame = (elapsed / bench) * 1000

    print(f"\n  Detection FPS:      {fps:.2f}")
    print(f"  ms / frame:         {ms_per_frame:.2f}")
    print(f"  Frames benchmarked: {bench}")
    print(f"  Total time:         {elapsed:.2f}s")

    return {
        'model': model_name,
        'fps': fps,
        'ms_per_frame': ms_per_frame,
        'frames': bench,
        'elapsed': elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['dcnv2', 'dcnv3', 'vanilla'], required=True)
    args = parser.parse_args()

    if args.model_type == 'dcnv2':
        models = {
            'DCNv2-Full': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Full.pt',
            'DCNv2-FPN':  '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-FPN.pt',
            'DCNv2-Pan':  '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Pan.pt',
            'DCNv2-Liu':  '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-LIU.pt',
        }
    elif args.model_type == 'dcnv3':
        models = {
            'DCNv3-Full': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_full_second/weights/best.pt',
            'DCNv3-FPN':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_fpn_second/weights/best.pt',
            'DCNv3-Pan':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_pan_second/weights/best.pt',
            'DCNv3-Liu':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_liu_second/weights/best.pt',
        }
    else:
        models = {
            'Vanilla-YOLOv8m': '/home/migui/Downloads/yolov8m-vanilla-20260211T133104Z-1-001/yolov8m-vanilla/weights/best.pt',
        }

    results = []
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_name}: not found at {model_path}")
            continue
        r = benchmark_model(model_path, model_name)
        if r:
            results.append(r)

    if results:
        out_dir = '/media/mydrive/GitHub/ultralytics/tracking/benchmark_fps'
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f'detection_fps_{args.model_type}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'fps', 'ms_per_frame', 'frames', 'elapsed'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved: {csv_path}")

        print(f"\n{'='*60}")
        print(f"  DETECTION FPS SUMMARY — {args.model_type.upper()}")
        print(f"{'='*60}")
        print(f"  {'Model':<20} {'FPS':>8}  {'ms/frame':>10}")
        print(f"  {'-'*42}")
        for r in results:
            print(f"  {r['model']:<20} {r['fps']:>8.2f}  {r['ms_per_frame']:>10.2f}")


if __name__ == '__main__':
    main()
