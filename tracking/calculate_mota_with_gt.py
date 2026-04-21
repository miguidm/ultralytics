#!/usr/bin/env python3
"""
Calculate proper MOTA metrics using YOLO ground truth annotations
Compares tracking predictions against ground truth for Gate3_Oct7
"""

import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import cv2

def parse_yolo_annotation(label_file, img_width, img_height, class_names):
    """
    Parse YOLO format annotation file
    Returns list of [class_name, x1, y1, x2, y2]
    """
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


def parse_mot_predictions(predictions_file):
    """
    Parse MOT format predictions file
    Format: frame,id,x,y,w,h,conf,class,-1,-1
    Returns dict: {frame_num: [(track_id, class_id, bbox, conf), ...]}
    """
    predictions = defaultdict(list)

    if not os.path.exists(predictions_file):
        return predictions

    with open(predictions_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue

            frame_num = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            class_id = int(parts[7])

            bbox = [x, y, x+w, y+h]  # Convert to x1,y1,x2,y2

            predictions[frame_num].append({
                'track_id': track_id,
                'class_id': class_id,
                'bbox': bbox,
                'conf': conf
            })

    return predictions


def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max < x_min or y_max < y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)

    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def match_detections(gt_detections, pred_detections, iou_threshold=0.5):
    """
    Match predictions to ground truth using Hungarian algorithm (greedy approximation)
    Returns: (true_positives, false_positives, false_negatives, id_switches)
    """
    if len(gt_detections) == 0 and len(pred_detections) == 0:
        return 0, 0, 0

    if len(gt_detections) == 0:
        return 0, len(pred_detections), 0  # All predictions are false positives

    if len(pred_detections) == 0:
        return 0, 0, len(gt_detections)  # All ground truths are false negatives

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(gt_detections), len(pred_detections)))

    for i, gt in enumerate(gt_detections):
        for j, pred in enumerate(pred_detections):
            # Only match if same class
            if gt['class_id'] == pred['class_id']:
                iou_matrix[i, j] = calculate_iou(gt['bbox'], pred['bbox'])

    # Greedy matching (simplified Hungarian algorithm)
    matched_gt = set()
    matched_pred = set()
    true_positives = 0

    # Sort by IoU (highest first)
    matches = []
    for i in range(len(gt_detections)):
        for j in range(len(pred_detections)):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((iou_matrix[i, j], i, j))

    matches.sort(reverse=True)

    for iou, gt_idx, pred_idx in matches:
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            true_positives += 1

    false_positives = len(pred_detections) - len(matched_pred)
    false_negatives = len(gt_detections) - len(matched_gt)

    return true_positives, false_positives, false_negatives


def calculate_mota_with_groundtruth(gt_dir, predictions_file, video_path, model_name):
    """Calculate MOTA using ground truth annotations"""

    print("="*70)
    print(f"MOTA Calculation with Ground Truth - {model_name}")
    print("="*70)

    # Load class names
    class_names = ['Car', 'Motorcycle', 'Tricycle', 'Bus', 'Van', 'Truck']

    # Get video dimensions
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return None

    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"\nVideo: {video_path}")
    print(f"  Dimensions: {img_width}x{img_height}")
    print(f"  Total frames: {total_frames}")

    # Load ground truth annotations
    print(f"\nLoading ground truth from: {gt_dir}")
    gt_data_dir = os.path.join(gt_dir, "obj_train_data")

    ground_truth = {}  # {frame_num: [detections]}
    gt_files = [f for f in os.listdir(gt_data_dir) if f.endswith('.txt')]

    for gt_file in gt_files:
        # Extract frame number from filename
        # e.g., gate3_oct_0031.txt -> frame 31
        # or gate3.5_oct_0001.txt -> frame 1
        basename = os.path.splitext(gt_file)[0]

        # Try to extract frame number
        parts = basename.split('_')
        if len(parts) >= 3:
            try:
                frame_num = int(parts[-1])

                # Load annotation
                label_path = os.path.join(gt_data_dir, gt_file)
                detections = parse_yolo_annotation(label_path, img_width, img_height, class_names)

                if detections:
                    ground_truth[frame_num] = detections
            except ValueError:
                continue

    print(f"✓ Loaded ground truth for {len(ground_truth)} frames")

    # Load predictions
    print(f"\nLoading predictions from: {predictions_file}")
    predictions = parse_mot_predictions(predictions_file)
    print(f"✓ Loaded predictions for {len(predictions)} frames")

    # Find common frames
    gt_frames = set(ground_truth.keys())
    pred_frames = set(predictions.keys())
    common_frames = sorted(gt_frames & pred_frames)

    print(f"\nFrame overlap:")
    print(f"  Ground truth frames: {len(gt_frames)}")
    print(f"  Prediction frames: {len(pred_frames)}")
    print(f"  Common frames: {len(common_frames)}")

    if len(common_frames) == 0:
        print("\n❌ No overlapping frames between ground truth and predictions!")
        print(f"   GT frame range: {min(gt_frames) if gt_frames else 'N/A'} - {max(gt_frames) if gt_frames else 'N/A'}")
        print(f"   Pred frame range: {min(pred_frames) if pred_frames else 'N/A'} - {max(pred_frames) if pred_frames else 'N/A'}")
        return None

    # Calculate metrics frame by frame
    print(f"\nCalculating MOTA metrics...")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt_objects = 0

    # Per-class statistics
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0})

    for frame_num in common_frames:
        gt_dets = ground_truth[frame_num]
        pred_dets = predictions[frame_num]

        # Match detections
        tp, fp, fn = match_detections(gt_dets, pred_dets, iou_threshold=0.5)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt_objects += len(gt_dets)

        # Per-class stats
        for gt in gt_dets:
            class_stats[gt['class']]['gt'] += 1

    # Calculate MOTA
    # MOTA = 1 - (FN + FP + IDSW) / GT
    # Note: We're not tracking ID switches in this simplified version
    id_switches = 0  # Would need frame-to-frame tracking to calculate this properly

    if total_gt_objects == 0:
        mota = 0.0
        precision = 0.0
        recall = 0.0
    else:
        mota = 1 - (total_fn + total_fp + id_switches) / total_gt_objects
        mota = max(0, min(1, mota))  # Clamp to [0, 1]

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print("\n" + "="*70)
    print("MOTA METRICS WITH GROUND TRUTH")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"Frames evaluated: {len(common_frames)}")
    print(f"\nDetection Statistics:")
    print(f"  Ground Truth Objects: {total_gt_objects}")
    print(f"  True Positives:       {total_tp}")
    print(f"  False Positives:      {total_fp}")
    print(f"  False Negatives:      {total_fn}")
    print(f"  Identity Switches:    {id_switches} (not tracked)")

    print(f"\nMetrics:")
    print(f"  MOTA:      {mota:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print(f"\nPer-Class Statistics:")
    for class_name in class_names:
        if class_name in class_stats:
            stats = class_stats[class_name]
            print(f"  {class_name}: {stats['gt']} objects")

    print("\n" + "="*70 + "\n")

    return {
        'model': model_name,
        'mota': mota,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'gt_objects': total_gt_objects,
        'frames_evaluated': len(common_frames)
    }


def main():
    """Main function to calculate MOTA for all DCNv2 models"""

    print("="*70)
    print("MOTA CALCULATION WITH GROUND TRUTH - GATE3 DCNv2 MODELS")
    print("="*70)

    # Paths
    gt_dir = "/media/mydrive/GitHub/YOLO-20260129T044456Z-3-001/YOLO/G3-Merged-YOLO"
    video_path = "/media/mydrive/GitHub/ultralytics/videos/Gate3_Oct7.mp4"
    predictions_base_dir = "gate3_test_results_dcnv2"

    # Models
    models = ['DCNv2-Full', 'DCNv2-FPN', 'DCNv2-Pan', 'DCNv2-LIU']

    results = []

    for model_name in models:
        predictions_file = os.path.join(predictions_base_dir, model_name, "Gate3_Oct7_predictions.txt")

        if not os.path.exists(predictions_file):
            print(f"\n⚠ Skipping {model_name}: predictions not found at {predictions_file}")
            continue

        result = calculate_mota_with_groundtruth(gt_dir, predictions_file, video_path, model_name)
        if result:
            results.append(result)

    # Summary table
    if results:
        print("\n" + "="*70)
        print("SUMMARY - MOTA WITH GROUND TRUTH")
        print("="*70)
        print(f"\n{'Model':<20} {'MOTA':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Frames':<8}")
        print("-"*70)

        results.sort(key=lambda x: x['mota'], reverse=True)
        for r in results:
            print(f"{r['model']:<20} {r['mota']:<8.4f} {r['precision']:<10.4f} {r['recall']:<8.4f} "
                  f"{r['f1']:<8.4f} {r['frames_evaluated']:<8}")

        # Save summary
        output_dir = "gate3_mota_results"
        os.makedirs(output_dir, exist_ok=True)
        summary_file = os.path.join(output_dir, "mota_summary.txt")

        with open(summary_file, 'w') as f:
            f.write("MOTA Calculation with Ground Truth - Gate3_Oct7\n")
            f.write("="*70 + "\n\n")
            f.write(f"{'Model':<20} {'MOTA':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'TP':<8} {'FP':<8} {'FN':<8} {'GT':<8} {'Frames':<8}\n")
            f.write("-"*110 + "\n")
            for r in results:
                f.write(f"{r['model']:<20} {r['mota']:<8.4f} {r['precision']:<10.4f} {r['recall']:<8.4f} "
                       f"{r['f1']:<8.4f} {r['tp']:<8} {r['fp']:<8} {r['fn']:<8} {r['gt_objects']:<8} {r['frames_evaluated']:<8}\n")

        print(f"\n✓ Summary saved: {summary_file}")
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
