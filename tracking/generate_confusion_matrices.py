#!/usr/bin/env python3
"""
Generate confusion matrices for all detection models using model.val()
on the OccludedYOLO dataset.

Usage:
  /home/migui/miniconda3/envs/dcnv2/bin/python generate_confusion_matrices.py --model-type dcnv2
  /home/migui/miniconda3/envs/dcn/bin/python generate_confusion_matrices.py --model-type dcnv3
  /home/migui/miniconda3/envs/dcnv2/bin/python generate_confusion_matrices.py --model-type vanilla
"""

import sys
import os
import argparse
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO

DATASET_YAML = '/media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/occluded_dataset.yaml'
OUTPUT_DIR   = '/media/mydrive/GitHub/ultralytics/tracking/confusion_matrices'


def run(model_path, model_name):
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    out = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(out, exist_ok=True)

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"  Error loading model: {e}")
        return

    model.val(
        data=DATASET_YAML,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        plots=True,
        save_json=False,
        project=OUTPUT_DIR,
        name=model_name,
        exist_ok=True,
        verbose=False,
    )
    print(f"  Saved to: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['dcnv2', 'dcnv3', 'vanilla'], required=True)
    args = parser.parse_args()

    if args.model_type == 'dcnv2':
        models = {
            'DCNv2-Full': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Full.pt',
            'DCNv2-FPN':  '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-FPN.pt',
            'DCNv2-Pan':  '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Pan.pt',
            'DCNv2-Liu':  '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-LIU.pt',
        }
    elif args.model_type == 'dcnv3':
        models = {
            'DCNv3-Full': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_full_second/weights/best.pt',
            'DCNv3-FPN':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_fpn_second/weights/best.pt',
            'DCNv3-Pan':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_pan_second/weights/best.pt',
            'DCNv3-Liu':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_liu_second/weights/best.pt',
        }
    else:
        models = {
            'Vanilla-YOLOv8m': '/home/migui/Downloads/yolov8m-vanilla-20260211T133104Z-1-001/yolov8m-vanilla/weights/best.pt',
        }

    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_name}: not found at {model_path}")
            continue
        run(model_path, model_name)

    print(f"\nAll confusion matrices saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
