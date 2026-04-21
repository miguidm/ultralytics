#!/usr/bin/env python3
"""
Simple test script to check if DCNv2 model can detect without tracking
"""

import sys
import warnings

# Patch MMCV before imports
print("Initializing MMCV patches...")
try:
    class DummyModule:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    sys.modules['mmcv._ext'] = DummyModule()
    sys.modules['mmcv.ops.bezier_align'] = DummyModule()
    sys.modules['mmcv.ops.bias_act'] = DummyModule()
    sys.modules['mmcv.ops.tin_shift'] = DummyModule()
    sys.modules['mmcv.ops.three_interpolate'] = DummyModule()
    sys.modules['mmcv.ops.three_nn'] = DummyModule()

    try:
        import mmcv.utils.ext_loader as ext_loader_module
        original_load_ext = ext_loader_module.load_ext

        def patched_load_ext(name, funcs):
            try:
                return original_load_ext(name, funcs)
            except (AssertionError, AttributeError):
                return DummyModule()

        ext_loader_module.load_ext = patched_load_ext
    except ImportError:
        pass

    print("✓ MMCV patches applied")
except Exception as e:
    print(f"⚠ Warning: Could not patch MMCV: {e}")

warnings.filterwarnings('ignore')
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2

# Test model
model_path = "/home/migui/YOLO_outputs/100_dcnv2-yolov8-neck-full_final/weights/DCNv2-Full.pt"
print(f"\nLoading model: {model_path}")

try:
    model = YOLO(model_path)
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Load one frame from video
video_path = "gate3_feb_crop.mp4"
print(f"\nLoading video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("✗ Cannot open video")
    sys.exit(1)

# Read first frame
ret, frame = cap.read()
if not ret:
    print("✗ Cannot read frame")
    sys.exit(1)

# Try detection at different confidence levels WITHOUT TRACKING
print("\n" + "="*60)
print("Testing DETECTION (no tracking) at various confidence levels:")
print("="*60)

for conf_threshold in [0.5, 0.3, 0.1, 0.01]:
    try:
        results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
        num_detections = len(results.boxes) if results.boxes is not None else 0
        print(f"\nconf={conf_threshold}: {num_detections} detections")

        if num_detections > 0:
            print(f"  Classes detected:")
            for box in results.boxes[:10]:  # Show first 10
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                cls_name = model.names[cls_id]
                print(f"    - {cls_name}: {conf:.3f}")

            # Save annotated frame
            annotated_frame = results.plot()
            output_name = f"test_detection_conf{conf_threshold}.jpg"
            cv2.imwrite(output_name, annotated_frame)
            print(f"  ✓ Saved: {output_name}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

cap.release()
print("\n" + "="*60)
print("Detection test complete")
print("="*60)
