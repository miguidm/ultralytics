#!/usr/bin/env python3
"""
Evaluate All Gates with Proper MOTA
Runs motmetrics evaluation on all available gates using COCO-based ground truth
"""

import os
import sys
import numpy as np
from collections import defaultdict

# Install motmetrics if needed
try:
    import motmetrics as mm
except ImportError:
    import subprocess
    print("Installing motmetrics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "motmetrics"])
    import motmetrics as mm


def custom_iou_distance_matrix(objs, hyps, max_iou=0.5):
    """Custom IoU distance matrix (numpy 2.0 compatible)"""
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
    dist[dist > max_iou] = np.nan

    return dist


def parse_mot_format(mot_file):
    """Parse MOT format file"""
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

            # Convert to center format [cx, cy, w, h]
            cx = x + w/2
            cy = y + h/2

            data[frame_num].append({
                'id': track_id,
                'bbox': [cx, cy, w, h]
            })

    return data


def evaluate_gate(gt_file, pred_file, gate_name, model_name):
    """Evaluate tracking for a single gate"""

    # Load ground truth
    gt_data = parse_mot_format(gt_file)
    if len(gt_data) == 0:
        print(f"  ⚠ No ground truth data for {gate_name}")
        return None

    # Load predictions
    pred_data = parse_mot_format(pred_file)
    if len(pred_data) == 0:
        print(f"  ⚠ No prediction data for {gate_name}")
        return None

    # Find common frames
    gt_frames = set(gt_data.keys())
    pred_frames = set(pred_data.keys())
    common_frames = sorted(gt_frames & pred_frames)

    if len(common_frames) == 0:
        print(f"  ⚠ No overlapping frames for {gate_name}")
        return None

    # Create accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Process each frame
    for frame_num in common_frames:
        gt_objects = gt_data[frame_num]
        pred_objects = pred_data[frame_num]

        gt_ids = [obj['id'] for obj in gt_objects]
        gt_boxes = np.array([obj['bbox'] for obj in gt_objects]) if gt_objects else np.empty((0, 4))

        pred_ids = [obj['id'] for obj in pred_objects]
        pred_boxes = np.array([obj['bbox'] for obj in pred_objects]) if pred_objects else np.empty((0, 4))

        # Calculate distances
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            distances = custom_iou_distance_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        else:
            distances = np.empty((len(gt_boxes), len(pred_boxes)))

        acc.update(gt_ids, pred_ids, distances)

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
            name=f"{model_name}_{gate_name}"
        )

        mota_val = summary['mota'].values[0]
        motp_val = summary['motp'].values[0]
        idf1_val = summary['idf1'].values[0]

        return {
            'gate': gate_name,
            'model': model_name,
            'mota': mota_val,
            'motp': motp_val if not np.isnan(motp_val) else 0,
            'idf1': idf1_val,
            'precision': summary['precision'].values[0],
            'recall': summary['recall'].values[0],
            'num_switches': int(summary['num_switches'].values[0]),
            'num_false_positives': int(summary['num_false_positives'].values[0]),
            'num_misses': int(summary['num_misses'].values[0]),
            'mostly_tracked': int(summary['mostly_tracked'].values[0]),
            'partially_tracked': int(summary['partially_tracked'].values[0]),
            'mostly_lost': int(summary['mostly_lost'].values[0]),
            'gt_frames': len(gt_frames),
            'common_frames': len(common_frames),
            'num_unique_objects': int(summary['num_unique_objects'].values[0])
        }

    except Exception as e:
        print(f"  ❌ Error calculating metrics for {gate_name}: {e}")
        return None


def main():
    """Main function"""

    print("="*90)
    print("PROPER MOTA EVALUATION - ALL GATES COMPARISON")
    print("="*90)

    # Ground truth directory
    gt_base_dir = "tracking_ground_truth_all_gates"

    # Prediction directories and gate mappings
    gates_to_evaluate = [
        {
            'gate': 'gate2.9_oct',
            'video_name': 'Gate2.9_Oct7',
            'pred_base': 'tracking_metrics_results_dcnv2'
        },
        {
            'gate': 'gate2_oct',
            'video_name': 'Gate2_Oct7',
            'pred_base': 'tracking_metrics_results_dcnv2'
        },
        {
            'gate': 'gate3.5_oct',
            'video_name': 'Gate3.5_Oct7',
            'pred_base': 'tracking_metrics_results_dcnv2'
        },
        {
            'gate': 'gate3_oct',
            'video_name': 'Gate3_Oct7',
            'pred_base': 'gate3_test_results_dcnv2'
        }
    ]

    models = ['DCNv2-Full', 'DCNv2-FPN', 'DCNv2-Pan', 'DCNv2-LIU']

    all_results = []

    for gate_info in gates_to_evaluate:
        gate = gate_info['gate']
        video_name = gate_info['video_name']
        pred_base = gate_info['pred_base']

        print(f"\n{'='*90}")
        print(f"Evaluating: {gate} ({video_name})")
        print(f"{'='*90}")

        gt_file = os.path.join(gt_base_dir, gate, "gt.txt")

        if not os.path.exists(gt_file):
            print(f"  ⚠ Ground truth not found: {gt_file}")
            continue

        for model in models:
            # Try different prediction file locations
            pred_files = [
                os.path.join(pred_base, model, f"{video_name}_predictions.txt"),
                os.path.join(pred_base, model, video_name, f"{video_name}_predictions.txt")
            ]

            pred_file = None
            for pf in pred_files:
                if os.path.exists(pf):
                    pred_file = pf
                    break

            if pred_file is None:
                print(f"  ⚠ Predictions not found for {model}")
                continue

            print(f"\n  Evaluating {model}...")
            result = evaluate_gate(gt_file, pred_file, gate, model)

            if result:
                all_results.append(result)
                print(f"    MOTA: {result['mota']:.4f} | IDF1: {result['idf1']:.4f} | "
                      f"FP: {result['num_false_positives']} | FN: {result['num_misses']} | "
                      f"IDSW: {result['num_switches']}")

    # Summary table
    if all_results:
        print("\n" + "="*90)
        print("COMPREHENSIVE COMPARISON - ALL GATES")
        print("="*90)

        # Group by gate
        gates = sorted(set(r['gate'] for r in all_results))

        for gate in gates:
            gate_results = [r for r in all_results if r['gate'] == gate]
            gate_results.sort(key=lambda x: x['mota'], reverse=True)

            print(f"\n{gate.upper()}")
            print("-"*90)
            print(f"{'Model':<15} {'MOTA':<8} {'MOTP':<8} {'IDF1':<8} {'Precision':<10} {'Recall':<8} "
                  f"{'IDSW':<6} {'FP':<8} {'FN':<8}")
            print("-"*90)

            for r in gate_results:
                print(f"{r['model']:<15} {r['mota']:<8.4f} {r['motp']:<8.4f} {r['idf1']:<8.4f} "
                      f"{r['precision']:<10.4f} {r['recall']:<8.4f} {r['num_switches']:<6} "
                      f"{r['num_false_positives']:<8} {r['num_misses']:<8}")

        # Best models per gate
        print("\n" + "="*90)
        print("BEST PERFORMING MODELS PER GATE")
        print("="*90)
        print(f"{'Gate':<20} {'Best Model':<15} {'MOTA':<8} {'IDF1':<8} {'Precision':<10} {'Recall':<8}")
        print("-"*90)

        for gate in gates:
            gate_results = [r for r in all_results if r['gate'] == gate]
            best = max(gate_results, key=lambda x: x['mota'])
            print(f"{best['gate']:<20} {best['model']:<15} {best['mota']:<8.4f} {best['idf1']:<8.4f} "
                  f"{best['precision']:<10.4f} {best['recall']:<8.4f}")

        # Save detailed results
        output_file = "proper_mota_all_gates_comparison.txt"
        with open(output_file, 'w') as f:
            f.write("Proper MOTA Evaluation - All Gates Comparison\n")
            f.write("="*90 + "\n\n")

            for gate in gates:
                gate_results = [r for r in all_results if r['gate'] == gate]
                gate_results.sort(key=lambda x: x['mota'], reverse=True)

                f.write(f"\n{gate.upper()}\n")
                f.write("-"*90 + "\n")
                f.write(f"{'Model':<15} {'MOTA':<8} {'MOTP':<8} {'IDF1':<8} {'Precision':<10} {'Recall':<8} "
                       f"{'IDSW':<6} {'FP':<8} {'FN':<8} {'MT':<6} {'ML':<6}\n")
                f.write("-"*90 + "\n")

                for r in gate_results:
                    f.write(f"{r['model']:<15} {r['mota']:<8.4f} {r['motp']:<8.4f} {r['idf1']:<8.4f} "
                           f"{r['precision']:<10.4f} {r['recall']:<8.4f} {r['num_switches']:<6} "
                           f"{r['num_false_positives']:<8} {r['num_misses']:<8} "
                           f"{r['mostly_tracked']:<6} {r['mostly_lost']:<6}\n")

        print(f"\n✓ Detailed results saved: {output_file}")
        print("\n" + "="*90 + "\n")


if __name__ == "__main__":
    main()
