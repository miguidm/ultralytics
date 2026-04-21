#!/usr/bin/env python3
"""
Test GPU vs CPU inference to see if device affects detection
"""
import sys
import os
import cv2

sys.path.insert(0, "/media/mydrive/GitHub/ultralytics")
dcnv3_path = '/media/mydrive/GitHub/ultralytics/ops_dcnv3'
if os.path.exists(dcnv3_path):
    sys.path.insert(0, dcnv3_path)

from ultralytics import YOLO
import torch


def test_model_on_device(model_path, video_path, device='cuda', conf=0.1):
    """Test model inference on specified device"""
    model_name = os.path.basename(model_path)

    print(f"\n{'='*80}")
    print(f"Testing {model_name} on {device.upper()}")
    print(f"{'='*80}")

    # Load model on specific device
    model = YOLO(model_path)
    model.to(device)

    print(f"✓ Model loaded on {device}")
    print(f"  Model device: {next(model.model.parameters()).device}")

    # Load video and test on a few frames
    cap = cv2.VideoCapture(video_path)

    total_detections = 0
    test_frames = [50, 100, 150, 200, 250]  # Test on specific frames

    for frame_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Run inference
        results = model(frame, conf=conf, device=device, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            num_det = len(results.boxes)
            total_detections += num_det

            # Show first detection
            if num_det > 0:
                box = results.boxes[0]
                cls_id = int(box.cls.cpu().numpy()[0])
                confidence = float(box.conf.cpu().numpy()[0])
                cls_name = model.names[cls_id]
                print(f"  Frame {frame_idx}: Found {num_det} detections, top: {cls_name} ({confidence:.3f})")

    cap.release()

    print(f"\nTotal detections across {len(test_frames)} frames: {total_detections}")
    print(f"{'='*80}")

    return total_detections


def main():
    """Main test function"""

    models = [
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-FPN.pt",
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-Full.pt",
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-Pan.pt",
    ]

    video_path = "gate3_feb_crop.mp4"

    print("\n" + "="*80)
    print("GPU vs CPU INFERENCE COMPARISON")
    print("="*80)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    results = {}

    for model_path in models:
        model_name = os.path.basename(model_path)
        results[model_name] = {}

        # Test on CPU
        results[model_name]['CPU'] = test_model_on_device(model_path, video_path, device='cpu', conf=0.1)

        # Test on GPU if available
        if torch.cuda.is_available():
            results[model_name]['GPU'] = test_model_on_device(model_path, video_path, device='cuda', conf=0.1)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for model_name, devices in results.items():
        print(f"\n{model_name}:")
        for device, count in devices.items():
            print(f"  {device}: {count} detections")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
