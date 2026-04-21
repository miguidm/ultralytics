#!/usr/bin/env python3
"""
Model Detection Evaluation Script
Compares detection performance of different YOLO architectures on COCO dataset
"""

import sys
import os
from pathlib import Path

# Setup DCNv2 environment
def setup_dcnv2_environment():
    """Configure environment for DCNv2 operations"""
    cuda_lib_path = "/home/migui/miniconda3/envs/dcn/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    torch_lib_path = "/home/migui/miniconda3/envs/dcn/lib/python3.10/site-packages/torch/lib"

    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}"

    ultralytics_root = "/media/mydrive/GitHub/ultralytics"
    if ultralytics_root not in sys.path:
        sys.path.insert(0, ultralytics_root)

setup_dcnv2_environment()

from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# Models to evaluate
models = {
    'DCNv2-Full': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Full.pt',
    'DCNv2-FPN': '/media/mydrive/GitHub/ultralytics/tracking/dcnv2-yolov8-neck-fpn.pt',
    'DCNv2-Pan': '/media/mydrive/GitHub/ultralytics/tracking/dcvn2-yolov8-neck-pan-best.pt',
    'DCNv2-Liu': '/media/mydrive/GitHub/ultralytics/tracking/dcnv2-yolov8n-liu-best.pt',
    'DCNv3-Liu': '/media/mydrive/GitHub/ultralytics/tracking/dcnv3-yolov8n-liu-best.pt',
    'YOLOv8n-Vanilla': '/media/mydrive/GitHub/ultralytics/tracking/yolov8n-vanilla-best.pt',
    'YOLOv8m-Vanilla': '/media/mydrive/GitHub/ultralytics/tracking/yolov8m-vanilla-best.pt',
}

# Dataset configuration
dataset_yaml = '/media/mydrive/GitHub/ultralytics/tracking/merged_dataset.yaml'

print("="*80)
print("MODEL DETECTION EVALUATION - Architecture Comparison")
print("="*80)
print(f"\nDataset: MergedAll COCO (14,843 images, 63,019 annotations)")
print(f"Evaluating {len(models)} models...\n")

results_summary = []

for model_name, model_path in models.items():
    print("\n" + "-"*80)
    print(f"Evaluating: {model_name}")
    print("-"*80)

    if not os.path.exists(model_path):
        print(f"⚠ Model not found: {model_path}")
        continue

    try:
        # Load model
        model = YOLO(model_path)

        # Run validation
        metrics = model.val(
            data=dataset_yaml,
            split='val',
            batch=16,
            imgsz=640,
            conf=0.001,  # Low conf for mAP calculation
            iou=0.6,
            plots=False,
            verbose=False
        )

        # Extract metrics
        map50_95 = metrics.box.map     # mAP@0.5:0.95
        map50 = metrics.box.map50       # mAP@0.5
        map75 = metrics.box.map75       # mAP@0.75
        precision = metrics.box.mp      # Mean precision
        recall = metrics.box.mr         # Mean recall

        # Per-class AP
        class_aps = metrics.box.ap_class_index
        class_maps = metrics.box.maps   # AP per class

        print(f"\n✓ Evaluation complete!")
        print(f"  mAP@0.5:0.95: {map50_95:.4f}")
        print(f"  mAP@0.5:     {map50:.4f}")
        print(f"  mAP@0.75:    {map75:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")

        # Store results
        results_summary.append({
            'model': model_name,
            'map50_95': map50_95,
            'map50': map50,
            'map75': map75,
            'precision': precision,
            'recall': recall
        })

    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")
        continue

print("\n" + "="*80)
print("SUMMARY - Detection Performance Comparison")
print("="*80)

if results_summary:
    # Sort by mAP@0.5:0.95
    results_summary.sort(key=lambda x: x['map50_95'], reverse=True)

    print("\nRanked by mAP@0.5:0.95:\n")
    print(f"{'Rank':<6} {'Model':<25} {'mAP50-95':<12} {'mAP50':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*80)

    for i, result in enumerate(results_summary, 1):
        print(f"{i:<6} {result['model']:<25} {result['map50_95']:<12.4f} {result['map50']:<12.4f} "
              f"{result['precision']:<12.4f} {result['recall']:<12.4f}")

    # Save to file
    output_file = 'model_detection_comparison.txt'
    with open(output_file, 'w') as f:
        f.write("MODEL DETECTION EVALUATION - Architecture Comparison\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: MergedAll COCO (14,843 images, 63,019 annotations)\n\n")
        f.write(f"{'Rank':<6} {'Model':<25} {'mAP50-95':<12} {'mAP50':<12} {'mAP75':<12} {'Precision':<12} {'Recall':<12}\n")
        f.write("-"*80 + "\n")
        for i, result in enumerate(results_summary, 1):
            f.write(f"{i:<6} {result['model']:<25} {result['map50_95']:<12.4f} {result['map50']:<12.4f} "
                   f"{result['map75']:<12.4f} {result['precision']:<12.4f} {result['recall']:<12.4f}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("\nConclusion:\n")
        f.write(f"Best model: {results_summary[0]['model']} (mAP@0.5:0.95 = {results_summary[0]['map50_95']:.4f})\n")
        f.write(f"\nImprovement vs worst: {((results_summary[0]['map50_95'] - results_summary[-1]['map50_95']) / results_summary[-1]['map50_95'] * 100):.2f}%\n")

    print(f"\n✓ Results saved to: {output_file}")
    print(f"\nConclusion:")
    print(f"  Best detection model: {results_summary[0]['model']}")
    print(f"  → This should provide the best tracking foundation")
else:
    print("\n⚠ No results to display")

print("\n" + "="*80 + "\n")
