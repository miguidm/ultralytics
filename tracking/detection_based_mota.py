#!/usr/bin/env python3
"""
Detection-Based MOTA Evaluation
Since GT doesn't have track IDs, calculate MOTA based on detection metrics only
MOTA = 1 - (FN + FP) / GT_total
"""

import os
import sys
import numpy as np
from collections import defaultdict
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

            # Convert to absolute coordinates [x1, y1, x2, y2]
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
    """Parse MOT format predictions"""
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

            bbox = [x, y, x+w, y+h]  # Convert to [x1, y1, x2, y2]

            predictions[frame_num].append({
                'track_id': track_id,
                'class_id': class_id,
                'bbox': bbox,
                'conf': conf
            })

    return predictions


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max < x_min or y_max < y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def match_detections(gt_dets, pred_dets, iou_threshold=0.5):
    """Match predictions to ground truth using greedy matching"""
    if len(gt_dets) == 0 and len(pred_dets) == 0:
        return 0, 0, 0

    if len(gt_dets) == 0:
        return 0, len(pred_dets), 0

    if len(pred_dets) == 0:
        return 0, 0, len(gt_dets)

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(gt_dets), len(pred_dets)))

    for i, gt in enumerate(gt_dets):
        for j, pred in enumerate(pred_dets):
            # Only match if same class
            if gt['class_id'] == pred['class_id']:
                iou_matrix[i, j] = calculate_iou(gt['bbox'], pred['bbox'])

    # Greedy matching
    matched_gt = set()
    matched_pred = set()
    matches = []

    # Sort by IoU (highest first)
    for i in range(len(gt_dets)):
        for j in range(len(pred_dets)):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((iou_matrix[i, j], i, j))

    matches.sort(reverse=True)

    true_positives = 0
    for iou, gt_idx, pred_idx in matches:
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            true_positives += 1

    false_positives = len(pred_dets) - len(matched_pred)
    false_negatives = len(gt_dets) - len(matched_gt)

    return true_positives, false_positives, false_negatives


def evaluate_detection_mota(gt_dir, predictions_file, video_path, model_name):
    """Calculate detection-based MOTA"""

    print("="*70)
    print(f"Detection-Based MOTA Evaluation - {model_name}")
    print("="*70)

    class_names = ['Car', 'Motorcycle', 'Tricycle', 'Bus', 'Van', 'Truck']

    # Get video dimensions
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video")
        return None

    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"\nVideo: {video_path}")
    print(f"  Dimensions: {img_width}x{img_height}")
    print(f"  Total frames: {total_frames}")

    # Load ground truth
    print(f"\nLoading ground truth from: {gt_dir}")
    gt_data_dir = os.path.join(gt_dir, "obj_train_data")
    ground_truth = {}

    gt_files = [f for f in os.listdir(gt_data_dir) if f.endswith('.txt')]

    for gt_file in gt_files:
        basename = os.path.splitext(gt_file)[0]
        parts = basename.split('_')

        if len(parts) >= 3:
            try:
                frame_num = int(parts[-1])
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
        print("\n❌ No overlapping frames!")
        return None

    # Calculate metrics frame by frame
    print(f"\nCalculating detection-based MOTA...")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt_objects = 0

    for i, frame_num in enumerate(common_frames):
        gt_dets = ground_truth[frame_num]
        pred_dets = predictions[frame_num]

        tp, fp, fn = match_detections(gt_dets, pred_dets, iou_threshold=0.5)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt_objects += len(gt_dets)

        if (i + 1) % 100 == 0 or (i + 1) == len(common_frames):
            print(f"  Processed {i+1}/{len(common_frames)} frames", end='\r')

    print(f"\n  Processed {len(common_frames)}/{len(common_frames)} frames ✓")

    # Calculate MOTA
    # MOTA = 1 - (FN + FP + IDSW) / GT
    # Since we don't have track IDs in GT, IDSW = 0
    if total_gt_objects == 0:
        mota = 0.0
        precision = 0.0
        recall = 0.0
    else:
        # Detection-based MOTA (no ID switches)
        mota = 1 - (total_fn + total_fp) / total_gt_objects
        mota = max(-1, min(1, mota))  # Clamp to [-1, 1]

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print("\n" + "="*70)
    print("DETECTION-BASED MOTA RESULTS")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"Frames evaluated: {len(common_frames)}")

    print(f"\nDetection Statistics:")
    print(f"  Ground Truth Objects: {total_gt_objects}")
    print(f"  True Positives (TP):  {total_tp}")
    print(f"  False Positives (FP): {total_fp}")
    print(f"  False Negatives (FN): {total_fn}")

    print(f"\nMetrics:")
    print(f"  MOTA (Detection-based): {mota:.4f}")
    print(f"  Precision:              {precision:.4f}")
    print(f"  Recall:                 {recall:.4f}")
    print(f"  F1 Score:               {f1:.4f}")

    print(f"\nNote: MOTA = 1 - (FN + FP) / GT")
    print(f"      ID Switches not counted (GT has no track IDs)")

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
    """Main function"""

    print("="*70)
    print("DETECTION-BASED MOTA EVALUATION - GATE3 DCNv2 MODELS")
    print("="*70)
    print("\nNote: GT annotations don't have track IDs")
    print("Calculating MOTA based on detection metrics only\n")

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
            print(f"\n⚠ Skipping {model_name}: predictions not found")
            continue

        result = evaluate_detection_mota(gt_dir, predictions_file, video_path, model_name)
        if result:
            results.append(result)

    # Summary table
    if results:
        print("\n" + "="*70)
        print("SUMMARY - ALL MODELS")
        print("="*70)
        print(f"\n{'Model':<15} {'MOTA':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'TP':<8} {'FP':<8} {'FN':<8}")
        print("-"*85)

        results.sort(key=lambda x: x['mota'], reverse=True)
        for r in results:
            print(f"{r['model']:<15} {r['mota']:<8.4f} {r['precision']:<10.4f} {r['recall']:<8.4f} "
                  f"{r['f1']:<8.4f} {r['tp']:<8} {r['fp']:<8} {r['fn']:<8}")

        # Save summary
        output_dir = "gate3_detection_mota_results"
        os.makedirs(output_dir, exist_ok=True)
        summary_file = os.path.join(output_dir, "detection_mota_summary.txt")

        with open(summary_file, 'w') as f:
            f.write("Detection-Based MOTA Evaluation - Gate3_Oct7\n")
            f.write("="*85 + "\n\n")
            f.write("Note: MOTA calculated without ID switches (GT has no track IDs)\n")
            f.write("Formula: MOTA = 1 - (FN + FP) / GT\n\n")
            f.write(f"{'Model':<15} {'MOTA':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'TP':<8} {'FP':<8} {'FN':<8} {'GT':<8}\n")
            f.write("-"*95 + "\n")
            for r in results:
                f.write(f"{r['model']:<15} {r['mota']:<8.4f} {r['precision']:<10.4f} {r['recall']:<8.4f} "
                       f"{r['f1']:<8.4f} {r['tp']:<8} {r['fp']:<8} {r['fn']:<8} {r['gt_objects']:<8}\n")

        print(f"\n✓ Summary saved: {summary_file}")
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
