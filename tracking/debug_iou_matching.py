#!/usr/bin/env python3
"""
Debug IoU matching to see why no detections are matching
"""

import os
import numpy as np

# Test with frame 100 which we know has matches
frame_num = 100

# GT file
gt_file = f'/media/mydrive/GitHub/YOLO-20260129T044456Z-3-001/YOLO/G3-Merged-YOLO/obj_train_data/gate3_oct_{frame_num:04d}.txt'
pred_file = 'gate3_test_results_dcnv2/DCNv2-Full/Gate3_Oct7_predictions.txt'

class_names = ['Car', 'Motorcycle', 'Tricycle', 'Bus', 'Van', 'Truck']
img_width = 1920
img_height = 1080

# Parse GT
gt_dets = []
with open(gt_file) as f:
    for idx, line in enumerate(f):
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Convert to absolute
        x1 = (x_center - width/2) * img_width
        y1 = (y_center - height/2) * img_height
        x2 = (x_center + width/2) * img_width
        y2 = (y_center + height/2) * img_height

        # For motmetrics format [cx, cy, w, h]
        abs_cx = x_center * img_width
        abs_cy = y_center * img_height
        abs_w = width * img_width
        abs_h = height * img_height

        gt_dets.append({
            'xyxy': [x1, y1, x2, y2],
            'xywh': [abs_cx, abs_cy, abs_w, abs_h],
            'class_id': class_id
        })

print(f"Frame {frame_num} - Ground Truth:")
print(f"Total: {len(gt_dets)}")
for i, gt in enumerate(gt_dets[:3]):
    print(f"  GT{i}: {class_names[gt['class_id']]}")
    print(f"    xyxy: {gt['xyxy']}")
    print(f"    xywh: {gt['xywh']}")

# Parse predictions
pred_dets = []
with open(pred_file) as f:
    for line in f:
        if line.startswith(f"{frame_num},"):
            parts = line.strip().split(',')
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            class_id = int(parts[7])

            # Current format: x,y,w,h where x,y is top-left
            # Convert to center format
            center_x = x + w/2
            center_y = y + h/2

            pred_dets.append({
                'tlwh': [x, y, w, h],
                'xywh': [center_x, center_y, w, h],
                'class_id': class_id,
                'track_id': track_id
            })

print(f"\nPredictions:")
print(f"Total: {len(pred_dets)}")
for i, pred in enumerate(pred_dets[:3]):
    print(f"  PRED{i}: {class_names[pred['class_id']]} (ID {pred['track_id']})")
    print(f"    tlwh: {pred['tlwh']}")
    print(f"    xywh: {pred['xywh']}")

# Test IoU calculation
print(f"\nTesting IoU calculation:")
if len(gt_dets) > 0 and len(pred_dets) > 0:
    gt_box = gt_dets[0]['xywh']
    pred_box = pred_dets[0]['xywh']

    print(f"GT box (xywh): {gt_box}")
    print(f"Pred box (xywh): {pred_box}")

    # Convert to xyxy for IoU
    def xywh_to_xyxy(box):
        cx, cy, w, h = box
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    gt_xyxy = xywh_to_xyxy(gt_box)
    pred_xyxy = xywh_to_xyxy(pred_box)

    print(f"GT box (xyxy): {gt_xyxy}")
    print(f"Pred box (xyxy): {pred_xyxy}")

    # Calculate IoU
    x1 = max(gt_xyxy[0], pred_xyxy[0])
    y1 = max(gt_xyxy[1], pred_xyxy[1])
    x2 = min(gt_xyxy[2], pred_xyxy[2])
    y2 = min(gt_xyxy[3], pred_xyxy[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    gt_area = (gt_xyxy[2] - gt_xyxy[0]) * (gt_xyxy[3] - gt_xyxy[1])
    pred_area = (pred_xyxy[2] - pred_xyxy[0]) * (pred_xyxy[3] - pred_xyxy[1])
    union = gt_area + pred_area - intersection
    iou = intersection / union if union > 0 else 0

    print(f"\nIoU = {iou:.3f}")
    print(f"Match (IoU >= 0.5): {iou >= 0.5}")
