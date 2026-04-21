#!/usr/bin/env python3
"""
Occlusion Detection Evaluation
Evaluate all models (DCNv2, DCNv3, Vanilla) on the OccludedYOLO dataset.
Outputs mAP, recall, precision per model and per class.

Usage:
  # DCNv2 models:
  /home/migui/miniconda3/envs/dcnv2/bin/python run_occlusion_eval.py --model-type dcnv2
  # DCNv3 models:
  /home/migui/miniconda3/envs/dcn/bin/python run_occlusion_eval.py --model-type dcnv3
  # Vanilla model:
  /home/migui/miniconda3/envs/dcnv2/bin/python run_occlusion_eval.py --model-type vanilla
  # Nano models (DCNv2n, DCNv3n, Vanilla-n):
  /home/migui/miniconda3/envs/dcnv2/bin/python run_occlusion_eval.py --model-type nano
"""

import sys
import os
import argparse
import csv
import time
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO

DATASET_YAML = os.path.join(os.path.dirname(__file__), 'occluded_dataset.yaml')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
CLASS_NAMES = ['car', 'motorcycle', 'tricycle', 'bus', 'van', 'truck']

MODELS = {
    'dcnv2': {
        'DCNv2-Full': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Full.pt',
        'DCNv2-FPN': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-FPN.pt',
        'DCNv2-Pan': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Pan.pt',
        'DCNv2-Liu': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-LIU.pt',
    },
    'dcnv3': {
        'DCNv3-Full': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_full_second/weights/best.pt',
        'DCNv3-FPN': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_fpn_second/weights/best.pt',
        'DCNv3-Pan': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_pan_second/weights/best.pt',
        'DCNv3-Liu': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_liu_second/weights/best.pt',
    },
    'vanilla': {
        'Vanilla-YOLOv8m': '/home/migui/Downloads/yolov8m-vanilla-20260211T133104Z-1-001/yolov8m-vanilla/weights/best.pt',
    },
    'nano': {
        'DCNv2n-Full': '/home/migui/Downloads/dcnv2-yolov8-neck-full-20260318T004120Z-1-001/dcnv2-yolov8-neck-full/weights/best.pt',
        'DCNv2n-FPN':  '/home/migui/Downloads/dcnv2-yolov8-neck-fpn-20260318T004118Z-1-001/dcnv2-yolov8-neck-fpn/weights/best.pt',
        'DCNv2n-Pan':  '/home/migui/Downloads/dcnv2-yolov8-neck-pan-20260318T004653Z-1-001/dcnv2-yolov8-neck-pan/weights/best.pt',
        'DCNv2n-Liu':  '/home/migui/Downloads/dcnv2-yolov8-liu-20260318T004538Z-1-001/dcnv2-yolov8-liu/weights/best.pt',
        'DCNv3n-Full': '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_full/weights/best.pt',
        'DCNv3n-FPN':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n-neck-fpn/weights/best.pt',
        'DCNv3n-Pan':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_pan/weights/best.pt',
        'DCNv3n-Liu':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_liu/weights/best.pt',
        'Vanilla-YOLOv8n': '/home/migui/Downloads/100_yolov8n_300epochs_b32-20260318T004620Z-1-001/100_yolov8n_300epochs_b32/weights/best.pt',
    },
}


def run_eval(model_path, model_name, dataset_yaml, output_dir):
    """Run YOLO val on the occluded dataset and extract metrics."""
    print(f"\n{'=' * 70}")
    print(f"  Evaluating: {model_name}")
    print(f"  Model: {model_path}")
    print(f"{'=' * 70}")

    if not os.path.exists(model_path):
        print(f"  SKIPPED: Model not found at {model_path}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    try:
        model = YOLO(model_path)
        print(f"  Model loaded")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return None

    start_time = time.time()

    try:
        results = model.val(
            data=dataset_yaml,
            conf=0.5,
            iou=0.5,
            batch=16,
            verbose=True,
            save_json=False,
            plots=True,
            project=output_dir,
            name=model_name,
        )
    except Exception as e:
        print(f"  ERROR during validation: {e}")
        return None

    elapsed = time.time() - start_time

    # Extract metrics
    metrics = {
        'model': model_name,
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr,
        'runtime': f"{elapsed:.1f}s",
    }

    # Per-class metrics
    ap50_per_class = results.box.ap50
    for i, cls_name in enumerate(CLASS_NAMES):
        if i < len(ap50_per_class):
            metrics[f'AP50_{cls_name}'] = ap50_per_class[i]
        else:
            metrics[f'AP50_{cls_name}'] = 0.0

    # Per-class precision and recall
    p_per_class = results.box.p
    r_per_class = results.box.r
    for i, cls_name in enumerate(CLASS_NAMES):
        if i < len(p_per_class):
            metrics[f'P_{cls_name}'] = p_per_class[i]
        else:
            metrics[f'P_{cls_name}'] = 0.0
        if i < len(r_per_class):
            metrics[f'R_{cls_name}'] = r_per_class[i]
        else:
            metrics[f'R_{cls_name}'] = 0.0

    # Print results
    print(f"\n  RESULTS: {model_name}")
    print(f"  {'Metric':<20} {'Value':<10}")
    print(f"  {'-' * 30}")
    print(f"  {'mAP@0.5':<20} {metrics['mAP50']:.4f}")
    print(f"  {'mAP@0.5:0.95':<20} {metrics['mAP50-95']:.4f}")
    print(f"  {'Precision':<20} {metrics['precision']:.4f}")
    print(f"  {'Recall':<20} {metrics['recall']:.4f}")
    print(f"  {'Runtime':<20} {metrics['runtime']}")

    print(f"\n  Per-Class AP@0.5:")
    for cls_name in CLASS_NAMES:
        ap = metrics.get(f'AP50_{cls_name}', 0)
        p = metrics.get(f'P_{cls_name}', 0)
        r = metrics.get(f'R_{cls_name}', 0)
        print(f"  {cls_name:<12} AP50={ap:.4f}  P={p:.4f}  R={r:.4f}")

    # Save individual report
    report_file = os.path.join(output_dir, model_name, f'{model_name}_occlusion_report.txt')
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w') as f:
        f.write(f"Occlusion Evaluation Report - {model_name}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: OccludedYOLO (1624 annotated occluded images)\n")
        f.write(f"Confidence: 0.5 | IoU: 0.5\n\n")
        f.write(f"Overall Metrics\n{'-' * 30}\n")
        f.write(f"mAP@0.5:       {metrics['mAP50']:.4f}\n")
        f.write(f"mAP@0.5:0.95:  {metrics['mAP50-95']:.4f}\n")
        f.write(f"Precision:     {metrics['precision']:.4f}\n")
        f.write(f"Recall:        {metrics['recall']:.4f}\n")
        f.write(f"Runtime:       {metrics['runtime']}\n\n")
        f.write(f"Per-Class Metrics\n{'-' * 50}\n")
        f.write(f"{'Class':<12} {'AP@0.5':<10} {'Precision':<12} {'Recall':<10}\n")
        for cls_name in CLASS_NAMES:
            ap = metrics.get(f'AP50_{cls_name}', 0)
            p = metrics.get(f'P_{cls_name}', 0)
            r = metrics.get(f'R_{cls_name}', 0)
            f.write(f"{cls_name:<12} {ap:<10.4f} {p:<12.4f} {r:<10.4f}\n")

    print(f"  Report saved: {report_file}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['dcnv2', 'dcnv3', 'vanilla', 'nano'], required=True)
    args = parser.parse_args()

    models = MODELS[args.model_type]
    output_dir = os.path.join(OUTPUT_DIR, args.model_type)

    print(f"\n{'=' * 70}")
    print(f"  OCCLUSION EVALUATION - {args.model_type.upper()} models")
    print(f"  Dataset: OccludedYOLO (1624 occluded images)")
    print(f"{'=' * 70}")

    all_results = []

    for model_name, model_path in models.items():
        result = run_eval(model_path, model_name, DATASET_YAML, output_dir)
        if result:
            all_results.append(result)

    # Combined summary
    if all_results:
        print(f"\n{'=' * 100}")
        print(f"  COMBINED SUMMARY - {args.model_type.upper()} on Occluded Dataset")
        print(f"{'=' * 100}")

        header = f"  {'Model':<18} {'mAP50':<10} {'mAP50-95':<10} {'Prec':<10} {'Recall':<10}"
        for cls in CLASS_NAMES:
            header += f" {cls[:5]:<8}"
        print(header)
        print(f"  {'-' * (58 + 8 * len(CLASS_NAMES))}")

        for r in all_results:
            row = f"  {r['model']:<18} {r['mAP50']:<10.4f} {r['mAP50-95']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f}"
            for cls in CLASS_NAMES:
                row += f" {r.get(f'AP50_{cls}', 0):<8.4f}"
            print(row)

        # Save summary CSV (merge with existing rows so two-env runs accumulate)
        summary_csv = os.path.join(output_dir, 'occlusion_summary.csv')
        os.makedirs(output_dir, exist_ok=True)
        fieldnames = ['Model', 'mAP50', 'mAP50-95', 'Precision', 'Recall']
        for cls in CLASS_NAMES:
            fieldnames.extend([f'AP50_{cls}', f'P_{cls}', f'R_{cls}'])
        fieldnames.append('Runtime')

        # Load existing rows keyed by model name
        existing = {}
        if os.path.exists(summary_csv):
            with open(summary_csv, newline='') as f:
                for row in csv.DictReader(f):
                    existing[row['Model']] = row

        # Only add new rows — never overwrite existing entries
        for r in all_results:
            if r['model'] in existing:
                continue
            row = {
                'Model': r['model'],
                'mAP50': f"{r['mAP50']:.4f}",
                'mAP50-95': f"{r['mAP50-95']:.4f}",
                'Precision': f"{r['precision']:.4f}",
                'Recall': f"{r['recall']:.4f}",
                'Runtime': r['runtime'],
            }
            for cls in CLASS_NAMES:
                row[f'AP50_{cls}'] = f"{r.get(f'AP50_{cls}', 0):.4f}"
                row[f'P_{cls}'] = f"{r.get(f'P_{cls}', 0):.4f}"
                row[f'R_{cls}'] = f"{r.get(f'R_{cls}', 0):.4f}"
            existing[r['model']] = row

        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing.values():
                writer.writerow(row)

        print(f"\n  Summary CSV: {summary_csv}")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
