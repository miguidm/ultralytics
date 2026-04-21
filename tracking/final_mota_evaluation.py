#!/usr/bin/env python3
"""
Final Proper MOTA Evaluation
Uses generated tracking ground truth with track IDs and motmetrics library
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

# Install and import motmetrics
try:
    import motmetrics as mm
except ImportError:
    import subprocess
    print("Installing motmetrics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "motmetrics"])
    import motmetrics as mm


def custom_iou_distance_matrix(objs, hyps, max_iou=0.5):
    """
    Custom IoU distance matrix (numpy 2.0 compatible)
    Args:
        objs: Ground truth boxes [N x 4] in format [cx, cy, w, h]
        hyps: Predicted boxes [M x 4] in format [cx, cy, w, h]
        max_iou: Maximum IoU threshold (distance > max_iou set to nan)
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

    # Convert IoU to distance
    dist = 1 - iou

    # Set distances > max_iou to nan (no match)
    dist[dist > max_iou] = np.nan

    return dist


def parse_mot_format(mot_file):
    """
    Parse MOT format file
    Format: frame,id,x,y,w,h,conf,class,vis
    Returns: dict {frame_num: [(id, bbox), ...]}
    """
    data = defaultdict(list)

    if not os.path.exists(mot_file):
        return data

    with open(mot_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue

            frame_num = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            # Convert to center format [cx, cy, w, h] for motmetrics
            cx = x + w/2
            cy = y + h/2

            data[frame_num].append({
                'id': track_id,
                'bbox': [cx, cy, w, h]
            })

    return data


def evaluate_tracking(gt_file, pred_file, model_name):
    """Evaluate tracking using motmetrics"""

    print("="*70)
    print(f"Proper MOTA Evaluation - {model_name}")
    print("="*70)

    # Load ground truth
    print(f"\nLoading ground truth: {gt_file}")
    gt_data = parse_mot_format(gt_file)
    print(f"✓ Loaded GT for {len(gt_data)} frames")

    # Load predictions
    print(f"\nLoading predictions: {pred_file}")
    pred_data = parse_mot_format(pred_file)
    print(f"✓ Loaded predictions for {len(pred_data)} frames")

    # Find common frames
    gt_frames = set(gt_data.keys())
    pred_frames = set(pred_data.keys())
    common_frames = sorted(gt_frames & pred_frames)

    print(f"\nFrame overlap:")
    print(f"  GT frames: {len(gt_frames)}")
    print(f"  Prediction frames: {len(pred_frames)}")
    print(f"  Common frames: {len(common_frames)}")

    if len(common_frames) == 0:
        print("\n❌ No overlapping frames!")
        return None

    # Create accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    print(f"\nCalculating metrics...")

    # Process each frame
    for i, frame_num in enumerate(common_frames):
        gt_objects = gt_data[frame_num]
        pred_objects = pred_data[frame_num]

        # Extract IDs and bounding boxes
        gt_ids = [obj['id'] for obj in gt_objects]
        gt_boxes = np.array([obj['bbox'] for obj in gt_objects]) if gt_objects else np.empty((0, 4))

        pred_ids = [obj['id'] for obj in pred_objects]
        pred_boxes = np.array([obj['bbox'] for obj in pred_objects]) if pred_objects else np.empty((0, 4))

        # Calculate distances using custom IoU (numpy 2.0 compatibility)
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            distances = custom_iou_distance_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        else:
            distances = np.empty((len(gt_boxes), len(pred_boxes)))

        # Update accumulator
        acc.update(gt_ids, pred_ids, distances)

        if (i + 1) % 100 == 0 or (i + 1) == len(common_frames):
            print(f"  Processed {i+1}/{len(common_frames)} frames", end='\r')

    print(f"\n  Processed {len(common_frames)}/{len(common_frames)} frames ✓")

    # Calculate metrics
    mh = mm.metrics.create()

    try:
        summary = mh.compute(
            acc,
            metrics=[
                'num_frames', 'mota', 'motp', 'idf1',
                'num_switches', 'num_false_positives', 'num_misses',
                'num_detections', 'num_objects', 'num_predictions',
                'num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost',
                'precision', 'recall'
            ],
            name=model_name
        )

        # Print results
        print("\n" + "="*70)
        print("FINAL MOTA RESULTS (WITH TRACKING GROUND TRUTH)")
        print("="*70)
        print(f"\nModel: {model_name}")
        print(f"Frames evaluated: {int(summary['num_frames'].values[0])}")

        print(f"\n🎯 Core MOT Metrics:")
        mota_val = summary['mota'].values[0]
        motp_val = summary['motp'].values[0]
        idf1_val = summary['idf1'].values[0]

        print(f"  MOTA: {mota_val:.4f} {'✓ Good' if mota_val > 0.5 else '⚠ Poor' if mota_val > 0 else '❌ Very Poor'}")
        print(f"  MOTP: {motp_val:.4f}" if not np.isnan(motp_val) else "  MOTP: N/A")
        print(f"  IDF1: {idf1_val:.4f}")

        print(f"\n📊 Detection Metrics:")
        print(f"  Precision: {summary['precision'].values[0]:.4f}")
        print(f"  Recall:    {summary['recall'].values[0]:.4f}")

        print(f"\n🔴 Error Breakdown:")
        print(f"  False Positives (FP): {int(summary['num_false_positives'].values[0])}")
        print(f"  Misses (FN):          {int(summary['num_misses'].values[0])}")
        print(f"  ID Switches (IDSW):   {int(summary['num_switches'].values[0])}")

        print(f"\n📈 Tracking Statistics:")
        print(f"  GT Objects:       {int(summary['num_objects'].values[0])}")
        print(f"  Unique GT Tracks: {int(summary['num_unique_objects'].values[0])}")
        print(f"  Predictions:      {int(summary['num_predictions'].values[0])}")
        print(f"  Matched:          {int(summary['num_detections'].values[0])}")

        print(f"\n🎪 Track Quality:")
        mt = int(summary['mostly_tracked'].values[0])
        pt = int(summary['partially_tracked'].values[0])
        ml = int(summary['mostly_lost'].values[0])
        total_unique = int(summary['num_unique_objects'].values[0])

        print(f"  Mostly Tracked (MT):     {mt} ({mt/max(1,total_unique)*100:.1f}%)")
        print(f"  Partially Tracked (PT):  {pt} ({pt/max(1,total_unique)*100:.1f}%)")
        print(f"  Mostly Lost (ML):        {ml} ({ml/max(1,total_unique)*100:.1f}%)")

        print("\n" + "="*70 + "\n")

        return {
            'model': model_name,
            'mota': mota_val,
            'motp': motp_val if not np.isnan(motp_val) else 0,
            'idf1': idf1_val,
            'precision': summary['precision'].values[0],
            'recall': summary['recall'].values[0],
            'num_switches': int(summary['num_switches'].values[0]),
            'num_false_positives': int(summary['num_false_positives'].values[0]),
            'num_misses': int(summary['num_misses'].values[0]),
            'mostly_tracked': mt,
            'partially_tracked': pt,
            'mostly_lost': ml,
            'frames_evaluated': len(common_frames)
        }

    except Exception as e:
        print(f"\n❌ Error calculating metrics: {e}")
        return None


def main():
    """Main function"""

    print("="*70)
    print("FINAL PROPER MOTA EVALUATION - GATE3 DCNv2 MODELS")
    print("="*70)
    print("\nUsing generated tracking ground truth with track IDs\n")

    # Paths
    gt_file = "gate3_tracking_ground_truth/gt.txt"
    predictions_base_dir = "gate3_test_results_dcnv2"

    if not os.path.exists(gt_file):
        print(f"❌ Error: Ground truth file not found: {gt_file}")
        print("Run generate_tracking_gt.py first!")
        return

    # Models
    models = ['DCNv2-Full', 'DCNv2-FPN', 'DCNv2-Pan', 'DCNv2-LIU']

    results = []

    for model_name in models:
        pred_file = os.path.join(predictions_base_dir, model_name, "Gate3_Oct7_predictions.txt")

        if not os.path.exists(pred_file):
            print(f"\n⚠ Skipping {model_name}: predictions not found")
            continue

        result = evaluate_tracking(gt_file, pred_file, model_name)
        if result:
            results.append(result)

    # Summary table
    if results:
        print("\n" + "="*90)
        print("FINAL SUMMARY - ALL DCNv2 MODELS")
        print("="*90)
        print(f"\n{'Model':<15} {'MOTA':<8} {'MOTP':<8} {'IDF1':<8} {'Precision':<10} {'Recall':<8} {'IDSW':<6} {'FP':<8} {'FN':<8}")
        print("-"*90)

        results.sort(key=lambda x: x['mota'], reverse=True)
        for r in results:
            print(f"{r['model']:<15} {r['mota']:<8.4f} {r['motp']:<8.4f} {r['idf1']:<8.4f} "
                  f"{r['precision']:<10.4f} {r['recall']:<8.4f} {r['num_switches']:<6} "
                  f"{r['num_false_positives']:<8} {r['num_misses']:<8}")

        # Determine best model
        best_mota = max(r['mota'] for r in results)
        best_idf1 = max(r['idf1'] for r in results)

        best_mota_model = [r['model'] for r in results if r['mota'] == best_mota][0]
        best_idf1_model = [r['model'] for r in results if r['idf1'] == best_idf1][0]

        print("\n" + "-"*90)
        print(f"🏆 Best MOTA: {best_mota_model} ({best_mota:.4f})")
        print(f"🏆 Best IDF1: {best_idf1_model} ({best_idf1:.4f})")

        # Save detailed summary
        output_dir = "gate3_final_mota_results"
        os.makedirs(output_dir, exist_ok=True)
        summary_file = os.path.join(output_dir, "final_mota_summary.txt")

        with open(summary_file, 'w') as f:
            f.write("Final MOTA Evaluation with Tracking Ground Truth - Gate3_Oct7\n")
            f.write("="*100 + "\n\n")
            f.write(f"{'Model':<15} {'MOTA':<8} {'MOTP':<8} {'IDF1':<8} {'Precision':<10} {'Recall':<8} {'IDSW':<6} {'FP':<8} {'FN':<8} {'MT':<6} {'PT':<6} {'ML':<6}\n")
            f.write("-"*110 + "\n")
            for r in results:
                f.write(f"{r['model']:<15} {r['mota']:<8.4f} {r['motp']:<8.4f} {r['idf1']:<8.4f} "
                       f"{r['precision']:<10.4f} {r['recall']:<8.4f} {r['num_switches']:<6} "
                       f"{r['num_false_positives']:<8} {r['num_misses']:<8} {r['mostly_tracked']:<6} "
                       f"{r['partially_tracked']:<6} {r['mostly_lost']:<6}\n")
            f.write("\n" + "-"*110 + "\n")
            f.write(f"Best MOTA: {best_mota_model} ({best_mota:.4f})\n")
            f.write(f"Best IDF1: {best_idf1_model} ({best_idf1:.4f})\n")

        print(f"\n✓ Summary saved: {summary_file}")
        print("\n" + "="*90 + "\n")


if __name__ == "__main__":
    main()
