#!/usr/bin/env python3
"""
Calculate Detection mAP (mean Average Precision)
Proper metric for detection-only evaluation
"""

import os
import numpy as np
from collections import defaultdict
import cv2


def parse_yolo_annotation(label_file, img_width, img_height):
    """Parse YOLO format annotation"""
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

            # Convert to absolute coordinates
            x1 = (x_center - width/2) * img_width
            y1 = (y_center - height/2) * img_height
            x2 = (x_center + width/2) * img_width
            y2 = (y_center + height/2) * img_height

            detections.append({
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

            bbox = [x, y, x+w, y+h]

            predictions[frame_num].append({
                'bbox': bbox,
                'class_id': class_id,
                'conf': conf
            })

    return predictions


def calculate_iou(bbox1, bbox2):
    """Calculate IoU"""
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

    return intersection / union if union > 0 else 0.0


def calculate_ap(recalls, precisions):
    """Calculate Average Precision using 11-point interpolation"""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def evaluate_detection_map(gt_dir, predictions_file, model_name, iou_threshold=0.5):
    """Calculate mAP for detection"""

    print("="*70)
    print(f"Detection mAP Evaluation - {model_name}")
    print("="*70)

    class_names = ['Car', 'Motorcycle', 'Tricycle', 'Bus', 'Van', 'Truck']
    img_width = 1920
    img_height = 1080

    # Load ground truth
    print(f"\nLoading ground truth...")
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
                detections = parse_yolo_annotation(label_path, img_width, img_height)

                if detections:
                    ground_truth[frame_num] = detections
            except ValueError:
                continue

    print(f"✓ Loaded GT for {len(ground_truth)} frames")

    # Load predictions
    print(f"Loading predictions...")
    predictions = parse_mot_predictions(predictions_file)
    print(f"✓ Loaded predictions for {len(predictions)} frames")

    # Find common frames
    gt_frames = set(ground_truth.keys())
    pred_frames = set(predictions.keys())
    common_frames = sorted(gt_frames & pred_frames)

    print(f"\nFrame overlap: {len(common_frames)} frames")

    if len(common_frames) == 0:
        print("❌ No overlapping frames!")
        return None

    # Calculate mAP per class
    print(f"\nCalculating mAP@{iou_threshold}...")

    per_class_metrics = {}

    for class_id, class_name in enumerate(class_names):
        # Collect all GT and predictions for this class
        all_gt_boxes = []
        all_pred_boxes = []

        for frame_num in common_frames:
            # GT boxes for this class
            gt_dets = ground_truth[frame_num]
            gt_boxes_class = [det['bbox'] for det in gt_dets if det['class_id'] == class_id]

            # Predictions for this class
            pred_dets = predictions[frame_num]
            pred_boxes_class = [(det['bbox'], det['conf']) for det in pred_dets if det['class_id'] == class_id]

            all_gt_boxes.extend([(frame_num, box) for box in gt_boxes_class])
            all_pred_boxes.extend([(frame_num, box, conf) for box, conf in pred_boxes_class])

        if len(all_gt_boxes) == 0:
            print(f"  {class_name}: No GT objects")
            continue

        # Sort predictions by confidence (descending)
        all_pred_boxes.sort(key=lambda x: x[2], reverse=True)

        # Calculate precision-recall
        tp = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))
        gt_matched = set()

        for i, (pred_frame, pred_box, conf) in enumerate(all_pred_boxes):
            best_iou = 0
            best_gt_idx = -1

            # Find best matching GT in same frame
            for j, (gt_frame, gt_box) in enumerate(all_gt_boxes):
                if gt_frame != pred_frame:
                    continue

                if (pred_frame, j) in gt_matched:
                    continue

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1
                gt_matched.add((pred_frame, best_gt_idx))
            else:
                fp[i] = 1

        # Calculate cumulative precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(all_gt_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Calculate AP
        ap = calculate_ap(recalls, precisions)

        # Final precision/recall
        final_precision = precisions[-1] if len(precisions) > 0 else 0
        final_recall = recalls[-1] if len(recalls) > 0 else 0

        per_class_metrics[class_name] = {
            'ap': ap,
            'precision': final_precision,
            'recall': final_recall,
            'num_gt': len(all_gt_boxes),
            'num_pred': len(all_pred_boxes),
            'num_tp': int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
        }

        print(f"  {class_name}: AP@{iou_threshold}={ap:.4f}, P={final_precision:.4f}, R={final_recall:.4f} ({len(all_gt_boxes)} GT)")

    # Calculate mAP
    aps = [metrics['ap'] for metrics in per_class_metrics.values()]
    mAP = np.mean(aps) if aps else 0

    # Print summary
    print("\n" + "="*70)
    print("DETECTION mAP RESULTS")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"IoU Threshold: {iou_threshold}")
    print(f"Frames evaluated: {len(common_frames)}")

    print(f"\n📊 mAP@{iou_threshold}: {mAP:.4f}")

    print(f"\nPer-Class AP:")
    for class_name, metrics in per_class_metrics.items():
        print(f"  {class_name:<12}: AP={metrics['ap']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")

    print("\n" + "="*70 + "\n")

    return {
        'model': model_name,
        'mAP@0.5': mAP,
        'per_class': per_class_metrics,
        'frames_evaluated': len(common_frames)
    }


def main():
    """Main function"""

    print("="*70)
    print("DETECTION mAP EVALUATION - GATE3 DCNv2 MODELS")
    print("="*70)
    print("\nCalculating mAP (mean Average Precision) for detection quality\n")

    # Paths
    gt_dir = "/media/mydrive/GitHub/YOLO-20260129T044456Z-3-001/YOLO/G3-Merged-YOLO"
    predictions_base_dir = "gate3_test_results_dcnv2"

    # Models
    models = ['DCNv2-Full', 'DCNv2-FPN', 'DCNv2-Pan', 'DCNv2-LIU']

    results = []

    for model_name in models:
        pred_file = os.path.join(predictions_base_dir, model_name, "Gate3_Oct7_predictions.txt")

        if not os.path.exists(pred_file):
            print(f"\n⚠ Skipping {model_name}: predictions not found")
            continue

        result = evaluate_detection_map(gt_dir, pred_file, model_name, iou_threshold=0.5)
        if result:
            results.append(result)

    # Summary
    if results:
        print("\n" + "="*70)
        print("SUMMARY - All Models")
        print("="*70)

        results.sort(key=lambda x: x['mAP@0.5'], reverse=True)

        print(f"\n{'Model':<20} {'mAP@0.5':<10}")
        print("-"*30)
        for r in results:
            print(f"{r['model']:<20} {r['mAP@0.5']:<10.4f}")

        best_model = results[0]['model']
        best_map = results[0]['mAP@0.5']
        print(f"\n🏆 Best Model: {best_model} (mAP@0.5 = {best_map:.4f})")

        # Save summary
        output_dir = "gate3_detection_map_results"
        os.makedirs(output_dir, exist_ok=True)
        summary_file = os.path.join(output_dir, "map_summary.txt")

        with open(summary_file, 'w') as f:
            f.write("Detection mAP Evaluation - Gate3_Oct7\n")
            f.write("="*70 + "\n\n")
            f.write(f"{'Model':<20} {'mAP@0.5':<10}\n")
            f.write("-"*30 + "\n")
            for r in results:
                f.write(f"{r['model']:<20} {r['mAP@0.5']:<10.4f}\n")
            f.write(f"\nBest Model: {best_model} (mAP@0.5 = {best_map:.4f})\n")

        print(f"\n✓ Summary saved: {summary_file}")
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
