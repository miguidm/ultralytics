#!/usr/bin/env python3
"""
Diagnostic script to investigate why some DCNv3 models aren't detecting vehicles
"""
import sys
import os
import cv2

# Add paths for DCNv3
sys.path.insert(0, "/media/mydrive/GitHub/ultralytics")
dcnv3_path = '/media/mydrive/GitHub/ultralytics/ops_dcnv3'
if os.path.exists(dcnv3_path):
    sys.path.insert(0, dcnv3_path)

from ultralytics import YOLO
import torch

def diagnose_model(model_path, video_path, num_frames=5):
    """Diagnose a model by checking its predictions on sample frames"""
    print(f"\n{'='*80}")
    print(f"DIAGNOSING: {os.path.basename(model_path)}")
    print(f"{'='*80}")

    # Load model
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Print model info
    print(f"\nModel Information:")
    print(f"  - Class names: {model.names}")
    print(f"  - Number of classes: {len(model.names)}")
    print(f"  - Device: {model.device}")

    # Check model metadata
    if hasattr(model, 'ckpt') and model.ckpt is not None:
        ckpt = model.ckpt
        print(f"\nCheckpoint metadata:")
        if 'train_args' in ckpt:
            print(f"  - Training args: {ckpt['train_args']}")
        if 'epoch' in ckpt:
            print(f"  - Trained epochs: {ckpt['epoch']}")

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nVideo: {frame_width}x{frame_height}")

    # Test on first few frames with different confidence thresholds
    print(f"\n{'='*80}")
    print(f"Testing predictions on {num_frames} frames:")
    print(f"{'='*80}")

    for conf_threshold in [0.01, 0.05, 0.1, 0.25, 0.5]:
        print(f"\n--- Confidence threshold: {conf_threshold} ---")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning

        total_detections = 0
        detection_details = []

        for frame_idx in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Skip to middle frames for better chance of vehicles
            if frame_idx == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
                ret, frame = cap.read()
                if not ret:
                    break

            # Run inference
            results = model(frame, conf=conf_threshold, verbose=False)[0]

            if results.boxes is not None and len(results.boxes) > 0:
                num_detections = len(results.boxes)
                total_detections += num_detections

                # Get details
                for i in range(min(3, num_detections)):  # Show first 3 detections
                    box = results.boxes[i]
                    cls_id = int(box.cls.cpu().numpy())
                    conf = float(box.conf.cpu().numpy())
                    cls_name = model.names[cls_id]
                    detection_details.append(f"    Frame {frame_idx}: {cls_name} (conf: {conf:.3f})")

        print(f"  Total detections across {num_frames} frames: {total_detections}")
        if detection_details:
            print(f"  Sample detections:")
            for detail in detection_details[:5]:  # Show first 5
                print(detail)
        else:
            print(f"  No detections found!")

    cap.release()
    print(f"\n{'='*80}\n")


def main():
    """Main diagnostic function"""

    # Models to diagnose
    models_to_check = [
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-FPN.pt",
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-Full.pt",
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-Pan.pt",
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-Liu.pt",  # Working model for comparison
    ]

    video_path = "gate3_feb_crop.mp4"

    print("\n" + "="*80)
    print("DCNv3 MODEL DIAGNOSTIC TOOL")
    print("="*80)
    print(f"\nChecking models against: {video_path}")
    print(f"Will test with confidence thresholds: 0.01, 0.05, 0.1, 0.25, 0.5")

    for model_path in models_to_check:
        if os.path.exists(model_path):
            diagnose_model(model_path, video_path, num_frames=10)
        else:
            print(f"\n✗ Model not found: {model_path}")

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
