#!/usr/bin/env python3
"""
Proper MOTA Evaluation using motmetrics library
Calculates MOTA, MOTP, IDF1 and other MOT metrics using ground truth
"""

import os
import sys
import numpy as np
import motmetrics as mm
from collections import defaultdict
import cv2

def install_motmetrics():
    """Install motmetrics if not available"""
    try:
        import motmetrics
    except ImportError:
        print("Installing motmetrics library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "motmetrics"])
        print("✓ motmetrics installed")

# Ensure motmetrics is installed
install_motmetrics()
import motmetrics as mm


def calculate_iou_distance_matrix(objs, hyps, max_iou=0.5):
    """
    Custom IoU distance matrix calculation (workaround for numpy 2.0)
    Args:
        objs: Ground truth boxes [N x 4] in format [cx, cy, w, h]
        hyps: Predicted boxes [M x 4] in format [cx, cy, w, h]
        max_iou: Maximum IoU for a match (distances > max_iou are set to nan)
    Returns:
        Distance matrix [N x M] where distance = 1 - IoU
    """
    if len(objs) == 0 or len(hyps) == 0:
        return np.empty((len(objs), len(hyps)))

    objs = np.asarray(objs, dtype=float)
    hyps = np.asarray(hyps, dtype=float)

    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
    objs_x1 = objs[:, 0] - objs[:, 2] / 2
    objs_y1 = objs[:, 1] - objs[:, 3] / 2
    objs_x2 = objs[:, 0] + objs[:, 2] / 2
    objs_y2 = objs[:, 1] + objs[:, 3] / 2

    hyps_x1 = hyps[:, 0] - hyps[:, 2] / 2
    hyps_y1 = hyps[:, 1] - hyps[:, 3] / 2
    hyps_x2 = hyps[:, 0] + hyps[:, 2] / 2
    hyps_y2 = hyps[:, 1] + hyps[:, 3] / 2

    # Calculate intersection
    x1 = np.maximum(objs_x1[:, None], hyps_x1[None, :])
    y1 = np.maximum(objs_y1[:, None], hyps_y1[None, :])
    x2 = np.minimum(objs_x2[:, None], hyps_x2[None, :])
    y2 = np.minimum(objs_y2[:, None], hyps_y2[None, :])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate areas
    objs_area = (objs_x2 - objs_x1) * (objs_y2 - objs_y1)
    hyps_area = (hyps_x2 - hyps_x1) * (hyps_y2 - hyps_y1)

    # Calculate union and IoU
    union = objs_area[:, None] + hyps_area[None, :] - intersection
    iou = intersection / np.maximum(union, 1e-10)

    # Convert IoU to distance (1 - IoU)
    dist = 1 - iou

    # Set distances > max_iou to nan (no match)
    # max_iou is the maximum distance (e.g., 0.5 means IoU must be >= 0.5)
    dist[dist > max_iou] = np.nan

    return dist


def parse_yolo_annotation(label_file, img_width, img_height, class_names):
    """
    Parse YOLO format annotation file
    Returns list of detections
    """
    detections = []

    if not os.path.exists(label_file):
        return detections

    with open(label_file, 'r') as f:
        for idx, line in enumerate(f):
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

            # Use line index as pseudo-ID for ground truth (since GT doesn't have track IDs)
            # We'll assign unique IDs per frame
            detections.append({
                'class': class_name,
                'bbox': [x1, y1, x2, y2],
                'class_id': class_id,
                'gt_id': idx  # Pseudo ID for this frame
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

            # Store as center_x, center_y, width, height for motmetrics
            center_x = x + w/2
            center_y = y + h/2

            predictions[frame_num].append({
                'track_id': track_id,
                'class_id': class_id,
                'bbox': [center_x, center_y, w, h],  # motmetrics uses [x,y,w,h] format
                'conf': conf
            })

    return predictions


def evaluate_with_motmetrics(gt_dir, predictions_file, video_path, model_name):
    """
    Evaluate tracking using motmetrics library
    Returns proper MOTA, MOTP, IDF1 metrics
    """

    print("="*70)
    print(f"Proper MOTA Evaluation - {model_name}")
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

    # Create persistent GT IDs across frames
    gt_id_counter = 0
    gt_id_map = {}  # Map (frame, bbox_idx) to unique GT ID

    for gt_file in gt_files:
        basename = os.path.splitext(gt_file)[0]
        parts = basename.split('_')

        if len(parts) >= 3:
            try:
                frame_num = int(parts[-1])
                label_path = os.path.join(gt_data_dir, gt_file)
                detections = parse_yolo_annotation(label_path, img_width, img_height, class_names)

                # Assign unique IDs to GT objects
                for det in detections:
                    gt_id_counter += 1
                    det['gt_id'] = gt_id_counter

                if detections:
                    ground_truth[frame_num] = detections
            except ValueError:
                continue

    print(f"✓ Loaded ground truth for {len(ground_truth)} frames")
    print(f"  Total GT objects: {gt_id_counter}")

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
        return None

    # Create motmetrics accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    print(f"\nCalculating metrics using motmetrics library...")
    print(f"  Processing {len(common_frames)} frames...")

    # Process each frame
    for i, frame_num in enumerate(common_frames):
        gt_dets = ground_truth[frame_num]
        pred_dets = predictions[frame_num]

        # Extract GT IDs and boxes
        gt_ids = [det['gt_id'] for det in gt_dets]
        gt_boxes = [[det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3]] for det in gt_dets]

        # Extract prediction IDs and boxes
        pred_ids = [det['track_id'] for det in pred_dets]
        pred_boxes = [[det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3]] for det in pred_dets]

        # Convert boxes to [x, y, w, h] format for distance calculation
        # Calculate distances (using IoU-based distance)
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            # Calculate custom IoU distance matrix (workaround for numpy 2.0 compatibility)
            gt_array = np.array(gt_boxes, dtype=float)
            pred_array = np.array(pred_boxes, dtype=float)

            # Custom IoU calculation
            distances = calculate_iou_distance_matrix(gt_array, pred_array, max_iou=0.5)
        else:
            distances = np.empty((len(gt_boxes), len(pred_boxes)))

        # Update accumulator
        acc.update(
            gt_ids,           # Ground truth IDs
            pred_ids,         # Prediction IDs
            distances         # Distance matrix
        )

        # Progress update
        if (i + 1) % 100 == 0 or (i + 1) == len(common_frames):
            print(f"    Processed {i+1}/{len(common_frames)} frames", end='\r')

    print(f"\n    Processed {len(common_frames)}/{len(common_frames)} frames ✓")

    # Calculate metrics
    print(f"\nCalculating final metrics...")
    mh = mm.metrics.create()

    summary = mh.compute(
        acc,
        metrics=[
            'num_frames',
            'mota', 'motp', 'idf1',
            'num_switches', 'num_false_positives', 'num_misses',
            'num_detections', 'num_objects', 'num_predictions',
            'num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost',
            'precision', 'recall'
        ],
        name=model_name
    )

    # Print results
    print("\n" + "="*70)
    print("MOTMETRICS EVALUATION RESULTS")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"Frames evaluated: {int(summary['num_frames'].values[0])}")

    print(f"\nCore MOT Metrics:")
    print(f"  MOTA (Multiple Object Tracking Accuracy): {summary['mota'].values[0]:.4f}")
    print(f"  MOTP (Multiple Object Tracking Precision): {summary['motp'].values[0]:.4f}")
    print(f"  IDF1 (ID F1 Score):                       {summary['idf1'].values[0]:.4f}")

    print(f"\nDetection Metrics:")
    print(f"  Precision: {summary['precision'].values[0]:.4f}")
    print(f"  Recall:    {summary['recall'].values[0]:.4f}")

    print(f"\nError Breakdown:")
    print(f"  False Positives (FP): {int(summary['num_false_positives'].values[0])}")
    print(f"  Misses (FN):          {int(summary['num_misses'].values[0])}")
    print(f"  ID Switches (IDSW):   {int(summary['num_switches'].values[0])}")

    print(f"\nTracking Statistics:")
    print(f"  Ground Truth Objects:    {int(summary['num_objects'].values[0])}")
    print(f"  Unique GT Objects:       {int(summary['num_unique_objects'].values[0])}")
    print(f"  Total Predictions:       {int(summary['num_predictions'].values[0])}")
    print(f"  Total Detections:        {int(summary['num_detections'].values[0])}")

    print(f"\nTrack Quality:")
    print(f"  Mostly Tracked (MT):     {int(summary['mostly_tracked'].values[0])}")
    print(f"  Partially Tracked (PT):  {int(summary['partially_tracked'].values[0])}")
    print(f"  Mostly Lost (ML):        {int(summary['mostly_lost'].values[0])}")

    print("\n" + "="*70 + "\n")

    # Return results
    return {
        'model': model_name,
        'mota': summary['mota'].values[0],
        'motp': summary['motp'].values[0],
        'idf1': summary['idf1'].values[0],
        'precision': summary['precision'].values[0],
        'recall': summary['recall'].values[0],
        'num_switches': int(summary['num_switches'].values[0]),
        'num_false_positives': int(summary['num_false_positives'].values[0]),
        'num_misses': int(summary['num_misses'].values[0]),
        'mostly_tracked': int(summary['mostly_tracked'].values[0]),
        'mostly_lost': int(summary['mostly_lost'].values[0]),
        'frames_evaluated': int(summary['num_frames'].values[0])
    }


def main():
    """Main function to evaluate all DCNv2 models"""

    print("="*70)
    print("PROPER MOTA EVALUATION WITH MOTMETRICS - GATE3 DCNv2 MODELS")
    print("="*70)
    print("\nUsing py-motmetrics library for standard MOT evaluation\n")

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

        result = evaluate_with_motmetrics(gt_dir, predictions_file, video_path, model_name)
        if result:
            results.append(result)

    # Summary table
    if results:
        print("\n" + "="*70)
        print("SUMMARY - ALL MODELS")
        print("="*70)
        print(f"\n{'Model':<15} {'MOTA':<8} {'MOTP':<8} {'IDF1':<8} {'Precision':<10} {'Recall':<8} {'IDSW':<6} {'FP':<8} {'FN':<8}")
        print("-"*90)

        results.sort(key=lambda x: x['mota'], reverse=True)
        for r in results:
            print(f"{r['model']:<15} {r['mota']:<8.4f} {r['motp']:<8.4f} {r['idf1']:<8.4f} "
                  f"{r['precision']:<10.4f} {r['recall']:<8.4f} {r['num_switches']:<6} "
                  f"{r['num_false_positives']:<8} {r['num_misses']:<8}")

        # Save detailed summary
        output_dir = "gate3_proper_mota_results"
        os.makedirs(output_dir, exist_ok=True)
        summary_file = os.path.join(output_dir, "motmetrics_summary.txt")

        with open(summary_file, 'w') as f:
            f.write("Proper MOTA Evaluation with motmetrics - Gate3_Oct7\n")
            f.write("="*90 + "\n\n")
            f.write(f"{'Model':<15} {'MOTA':<8} {'MOTP':<8} {'IDF1':<8} {'Precision':<10} {'Recall':<8} {'IDSW':<6} {'FP':<8} {'FN':<8} {'MT':<6} {'ML':<6}\n")
            f.write("-"*100 + "\n")
            for r in results:
                f.write(f"{r['model']:<15} {r['mota']:<8.4f} {r['motp']:<8.4f} {r['idf1']:<8.4f} "
                       f"{r['precision']:<10.4f} {r['recall']:<8.4f} {r['num_switches']:<6} "
                       f"{r['num_false_positives']:<8} {r['num_misses']:<8} {r['mostly_tracked']:<6} {r['mostly_lost']:<6}\n")

        print(f"\n✓ Summary saved: {summary_file}")
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
