#!/usr/bin/env python3
"""
Debug MOTA calculation - check a single frame in detail
"""

import os
import sys
import numpy as np
import cv2

def parse_yolo_annotation(label_file, img_width, img_height, class_names):
    """Parse YOLO format annotation file"""
    detections = []

    if not os.path.exists(label_file):
        return detections

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert from YOLO format (normalized) to absolute coordinates
            x1 = (x_center - width/2) * img_width
            y1 = (y_center - height/2) * img_height
            x2 = (x_center + width/2) * img_width
            y2 = (y_center + height/2) * img_height

            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

            detections.append({
                'class': class_name,
                'bbox': [x1, y1, x2, y2],
                'class_id': class_id
            })

    return detections


def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max < x_min or y_max < y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)

    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def debug_frame(frame_num):
    """Debug a single frame"""

    print(f"="*70)
    print(f"Debugging Frame {frame_num}")
    print(f"="*70)

    # Paths
    gt_file = f'/media/mydrive/GitHub/YOLO-20260129T044456Z-3-001/YOLO/G3-Merged-YOLO/obj_train_data/gate3_oct_{frame_num:04d}.txt'
    pred_file = 'gate3_test_results_dcnv2/DCNv2-Full/Gate3_Oct7_predictions.txt'
    class_names = ['Car', 'Motorcycle', 'Tricycle', 'Bus', 'Van', 'Truck']

    # Video dimensions
    img_width = 1920
    img_height = 1080

    # Load GT
    print(f"\nGround Truth:")
    gt_dets = parse_yolo_annotation(gt_file, img_width, img_height, class_names)
    print(f"  Total: {len(gt_dets)} objects")
    for i, gt in enumerate(gt_dets):
        print(f"  GT{i}: {gt['class']} - bbox: [{gt['bbox'][0]:.1f}, {gt['bbox'][1]:.1f}, {gt['bbox'][2]:.1f}, {gt['bbox'][3]:.1f}]")

    # Load predictions
    print(f"\nPredictions:")
    pred_dets = []
    with open(pred_file) as f:
        for line in f:
            if line.startswith(f"{frame_num},"):
                parts = line.strip().split(',')
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6])
                class_id = int(parts[7])

                bbox = [x, y, x+w, y+h]
                pred_dets.append({
                    'track_id': track_id,
                    'class_id': class_id,
                    'bbox': bbox,
                    'conf': conf
                })

    print(f"  Total: {len(pred_dets)} predictions")
    for i, pred in enumerate(pred_dets):
        class_name = class_names[pred['class_id']] if pred['class_id'] < len(class_names) else f"class_{pred['class_id']}"
        print(f"  PRED{i}: {class_name} (ID:{pred['track_id']}) - bbox: [{pred['bbox'][0]:.1f}, {pred['bbox'][1]:.1f}, {pred['bbox'][2]:.1f}, {pred['bbox'][3]:.1f}] - conf: {pred['conf']:.3f}")

    # Calculate IoU matrix
    print(f"\nIoU Matrix:")
    print(f"{'':>10}", end='')
    for i in range(len(pred_dets)):
        print(f"PRED{i:>2}", end='  ')
    print()

    matches = []
    for i, gt in enumerate(gt_dets):
        print(f"GT{i:<8}", end='  ')
        for j, pred in enumerate(pred_dets):
            iou = 0.0
            if gt['class_id'] == pred['class_id']:
                iou = calculate_iou(gt['bbox'], pred['bbox'])
                if iou >= 0.5:
                    matches.append((i, j, iou))
            print(f"{iou:.3f}", end='  ')
        print()

    # Matching results
    print(f"\nMatches (IoU >= 0.5):")
    if matches:
        for gt_idx, pred_idx, iou in matches:
            print(f"  GT{gt_idx} <-> PRED{pred_idx}: IoU = {iou:.3f}")
    else:
        print("  None!")

    # Summary
    tp = len(matches)
    fp = len(pred_dets) - tp
    fn = len(gt_dets) - tp

    print(f"\nFrame Metrics:")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")

    if len(gt_dets) > 0:
        frame_recall = tp / len(gt_dets)
        print(f"  Recall: {frame_recall:.3f}")

    if len(pred_dets) > 0:
        frame_precision = tp / len(pred_dets)
        print(f"  Precision: {frame_precision:.3f}")

    print(f"\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Test several frames
    test_frames = [50, 100, 200, 400]

    for frame_num in test_frames:
        debug_frame(frame_num)
