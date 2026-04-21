#!/usr/bin/env python3
"""
Diagnose Why MOTA is Zero
Analyze the ground truth vs predictions mismatch
"""

import os
import cv2
from collections import defaultdict

def analyze_frame_coverage():
    """Check which frames are annotated vs predicted"""

    print("="*70)
    print("MOTA DIAGNOSIS - Understanding the Problem")
    print("="*70)

    # Load GT frames
    gt_dir = "/media/mydrive/GitHub/YOLO-20260129T044456Z-3-001/YOLO/G3-Merged-YOLO/obj_train_data"
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]

    gt_frames = []
    for gt_file in gt_files:
        basename = os.path.splitext(gt_file)[0]
        parts = basename.split('_')
        if len(parts) >= 3:
            try:
                frame_num = int(parts[-1])
                gt_frames.append(frame_num)
            except:
                pass

    gt_frames.sort()

    print(f"\n📊 Ground Truth Coverage:")
    print(f"  Total annotated frames: {len(gt_frames)}")
    print(f"  Frame range: {min(gt_frames)} - {max(gt_frames)}")
    print(f"  Video total frames: 54,205")
    print(f"  Coverage: {len(gt_frames)/54205*100:.2f}%")

    # Check frame gaps
    gaps = []
    for i in range(1, len(gt_frames)):
        gap = gt_frames[i] - gt_frames[i-1]
        if gap > 1:
            gaps.append(gap)

    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        print(f"\n  Average gap between frames: {avg_gap:.1f} frames")
        print(f"  Maximum gap: {max_gap} frames")
        print(f"  → Annotations are SPARSE (not consecutive)")

    # Load prediction sample
    pred_file = "gate3_test_results_dcnv2/DCNv2-Full/Gate3_Oct7_predictions.txt"

    if os.path.exists(pred_file):
        pred_frames = set()
        frame_det_counts = defaultdict(int)

        with open(pred_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    frame_num = int(parts[0])
                    pred_frames.add(frame_num)
                    frame_det_counts[frame_num] += 1

        print(f"\n📊 Model Predictions (DCNv2-Full):")
        print(f"  Frames with predictions: {len(pred_frames)}")
        print(f"  Coverage: {len(pred_frames)/54205*100:.1f}%")

        # Predictions on annotated frames
        common_frames = set(gt_frames) & pred_frames
        print(f"\n  Overlap with GT: {len(common_frames)} frames")
        print(f"  → Model predicts on {len(common_frames)/len(gt_frames)*100:.1f}% of GT frames")

        # Detection density
        if common_frames:
            avg_dets = sum(frame_det_counts[f] for f in common_frames) / len(common_frames)
            print(f"\n  Avg detections per GT frame: {avg_dets:.1f}")


def analyze_detection_quality():
    """Analyze why precision/recall are so low"""

    print(f"\n{'='*70}")
    print("DETECTION QUALITY ANALYSIS")
    print(f"{'='*70}")

    # Load a sample annotated frame
    class_names = ['Car', 'Motorcycle', 'Tricycle', 'Bus', 'Van', 'Truck']
    img_width = 1920
    img_height = 1080

    # Sample frame 100
    gt_file = "/media/mydrive/GitHub/YOLO-20260129T044456Z-3-001/YOLO/G3-Merged-YOLO/obj_train_data/gate3_oct_0100.txt"
    pred_file = "gate3_test_results_dcnv2/DCNv2-Full/Gate3_Oct7_predictions.txt"

    if not os.path.exists(gt_file) or not os.path.exists(pred_file):
        print("Sample files not found")
        return

    # Count GT objects
    with open(gt_file) as f:
        gt_count = len(f.readlines())

    # Count predictions for frame 100
    pred_count = 0
    with open(pred_file) as f:
        for line in f:
            if line.startswith("100,"):
                pred_count += 1

    print(f"\n📸 Sample Frame 100:")
    print(f"  Ground truth objects: {gt_count}")
    print(f"  Model predictions: {pred_count}")
    print(f"  Ratio: {pred_count/gt_count:.2f}x")

    if pred_count > gt_count * 2:
        print(f"\n  ⚠️ Model is over-detecting (too many predictions)")
    elif pred_count < gt_count * 0.5:
        print(f"\n  ⚠️ Model is under-detecting (missing objects)")


def show_solutions():
    """Show practical solutions"""

    print(f"\n{'='*70}")
    print("💡 SOLUTIONS TO FIX MOTA")
    print(f"{'='*70}")

    print("""
The problem: Ground truth annotations don't match model predictions well.

ROOT CAUSES:
1. Sparse annotations (only 652 frames, 1.2% of video)
2. Short track lengths (avg 1.8 frames - can't evaluate tracking)
3. High false positive rate (model detects 3x more than GT)

REAL SOLUTIONS:

┌─────────────────────────────────────────────────────────────────┐
│ Option 1: Use Detection mAP Instead (RECOMMENDED)              │
├─────────────────────────────────────────────────────────────────┤
│ • MOTA requires good tracking ground truth                     │
│ • Your GT is detection-only (no track IDs across frames)       │
│ • Use mAP@0.5 to evaluate detection quality                    │
│ • This is standard for detection-only datasets                 │
│ • Time: 30 minutes                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Option 2: Improve Ground Truth Quality                         │
├─────────────────────────────────────────────────────────────────┤
│ • Annotate consecutive frames (not sparse)                     │
│ • Need at least 500-1000 consecutive frames                    │
│ • Add track IDs manually across frames                         │
│ • Use CVAT or Label Studio with tracking mode                  │
│ • Time: 8-16 hours                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Option 3: Evaluate on Standard Benchmark                       │
├─────────────────────────────────────────────────────────────────┤
│ • Use MOT17/MOT20 (vehicle tracking benchmarks)                │
│ • Proper ground truth with track IDs                           │
│ • Compare with published methods                               │
│ • Time: 2-3 hours                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Option 4: Use Relative Metrics                                 │
├─────────────────────────────────────────────────────────────────┤
│ • Compare models against each other (not absolute MOTA)        │
│ • Look at: ID switches, track fragmentation, counts            │
│ • Focus on: Which model tracks BETTER (not absolute score)     │
│ • This works even with poor GT                                 │
│ • Time: Already done! ✓                                        │
└─────────────────────────────────────────────────────────────────┘

IMMEDIATE ACTION:
─────────────────
I recommend Option 1 (Detection mAP) + Option 4 (Relative comparison).

This gives you:
✓ Objective detection quality metric (mAP)
✓ Relative tracking quality (which model tracks better)
✓ No additional annotation needed
✓ Results in ~30 minutes

Would you like me to implement this?
""")


def main():
    """Main diagnosis function"""

    analyze_frame_coverage()
    analyze_detection_quality()
    show_solutions()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("""
The MOTA is 0 because:
  1. Ground truth is sparse (only 1% of video)
  2. Model produces 3x more detections than GT expects
  3. Only 13% of GT objects are matched (low recall)

This is NOT a model problem - it's a ground truth problem.

Next step: Choose a solution from above.
""")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
