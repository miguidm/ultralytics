#!/usr/bin/env python3
"""
Check raw model outputs to see if models are producing any predictions at all
"""
import sys
import os
import cv2
import torch

sys.path.insert(0, "/media/mydrive/GitHub/ultralytics")
dcnv3_path = '/media/mydrive/GitHub/ultralytics/ops_dcnv3'
if os.path.exists(dcnv3_path):
    sys.path.insert(0, dcnv3_path)

from ultralytics import YOLO
import numpy as np


def check_raw_predictions(model_path, video_path):
    """Check raw model predictions before NMS and confidence filtering"""
    model_name = os.path.basename(model_path)
    print(f"\n{'='*80}")
    print(f"CHECKING RAW OUTPUTS: {model_name}")
    print(f"{'='*80}")

    # Load model on GPU
    model = YOLO(model_path)
    model.to(0)
    print(f"✓ Model loaded on GPU")

    # Load video and get a few test frames
    cap = cv2.VideoCapture(video_path)
    test_frames = [100, 150, 200, 250, 300]

    all_predictions = []

    for frame_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Run inference with very low confidence
        results = model(frame, conf=0.001, device=0, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            print(f"\nFrame {frame_idx}:")
            print(f"  Total detections: {len(boxes)}")
            print(f"  Max confidence: {confidences.max():.4f}")
            print(f"  Min confidence: {confidences.min():.4f}")
            print(f"  Mean confidence: {confidences.mean():.4f}")

            # Show top 5 predictions
            sorted_indices = np.argsort(confidences)[::-1]
            print(f"  Top 5 predictions:")
            for i in range(min(5, len(boxes))):
                idx = sorted_indices[i]
                cls_id = int(classes[idx])
                conf = confidences[idx]
                cls_name = model.names[cls_id]
                print(f"    {i+1}. {cls_name}: {conf:.4f}")

            all_predictions.extend(confidences)
        else:
            print(f"\nFrame {frame_idx}: No predictions at all!")

    cap.release()

    if all_predictions:
        all_predictions = np.array(all_predictions)
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR {model_name}:")
        print(f"{'='*80}")
        print(f"  Total predictions across {len(test_frames)} frames: {len(all_predictions)}")
        print(f"  Overall max confidence: {all_predictions.max():.4f}")
        print(f"  Overall min confidence: {all_predictions.min():.4f}")
        print(f"  Overall mean confidence: {all_predictions.mean():.4f}")
        print(f"  Predictions above 0.25: {(all_predictions > 0.25).sum()}")
        print(f"  Predictions above 0.15: {(all_predictions > 0.15).sum()}")
        print(f"  Predictions above 0.10: {(all_predictions > 0.10).sum()}")
        print(f"  Predictions above 0.05: {(all_predictions > 0.05).sum()}")
    else:
        print(f"\n❌ NO PREDICTIONS AT ALL from {model_name}")
        print(f"   This model may not be properly trained or weights are corrupted!")

    print(f"{'='*80}\n")


def main():
    """Main function"""

    models = [
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-FPN.pt",
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-Full.pt",
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-Pan.pt",
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-Liu.pt",
    ]

    video_path = "gate3_feb_crop.mp4"

    print("\n" + "="*80)
    print("RAW MODEL OUTPUT ANALYSIS")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Testing with confidence threshold: 0.001 (very low)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    for model_path in models:
        if os.path.exists(model_path):
            check_raw_predictions(model_path, video_path)
        else:
            print(f"\n✗ Model not found: {model_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
