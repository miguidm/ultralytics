#!/usr/bin/env python3
"""
DCNv3 YOLOv8m Inference Script
This script performs inference using DCNv3 YOLOv8m models.
"""
import cv2
import os
import sys
import torch
import glob

# Add local 'ultralytics' and 'ops_dcnv3' to Python path
sys.path.insert(0, "/media/mydrive/GitHub/ultralytics")
dcnv3_path = '/media/mydrive/GitHub/ultralytics/ops_dcnv3'
if os.path.exists(dcnv3_path):
    sys.path.insert(0, dcnv3_path)
else:
    print(f"⚠ Warning: DCNv3 path not found: {dcnv3_path}")
    print("  DCNv3 models may fail to load!")

# Verify DCNv3 module can be imported
try:
    import DCNv3
    print(f"✓ DCNv3 module loaded successfully from: {DCNv3.__file__}")
except ImportError as e:
    print(f"❌ ERROR: Cannot import DCNv3 module!")
    print(f"   Error: {e}")
    sys.exit(1)

from ultralytics import YOLO

def run_inference(model_path, source_video, output_video):
    """Main inference function"""
    print("\n" + "="*70)
    print(f"Running inference for model: {os.path.basename(model_path)}")
    print("="*70)

    # Load DCNv3 YOLOv8 model
    print(f"\n[1/4] Loading DCNv3 model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return

    try:
        model = YOLO(model_path)
        print(f"✓ DCNv3 model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading DCNv3 model: {e}")
        return

    # Load video
    print(f"\n[2/4] Loading video: {source_video}")
    if not os.path.exists(source_video):
        print(f"❌ Error: Video file not found: {source_video}")
        return

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {source_video}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # Setup output video
    print(f"\n[3/4] Setting up output: {output_video}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"❌ Error: Cannot create output video: {output_video}")
        cap.release()
        return

    # Start processing
    print(f"\n[4/4] Starting inference...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8-DCNv3 inference
        results = model(frame, verbose=False)[0]

        # Draw bounding boxes
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = confidences[i]
                class_id = class_ids[i]
                class_name = model.names[class_id]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        out.write(frame)

        # Display frame
        cv2.imshow("DCNv3 Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n" + "="*70)
    print(f"PROCESSING COMPLETE for {os.path.basename(model_path)}")
    print(f"\n✓ Output video saved: {output_video}")

def main():
    models_dir = '/media/mydrive/GitHub/ultralytics/tracking/DCNv3-models/'
    source_video = 'gate3_feb_crop.mp4'
    
    model_paths = glob.glob(os.path.join(models_dir, '*.pt'))
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.pt', '')
        output_video = f"{model_name}_inference_output.mp4"
        run_inference(model_path, source_video, output_video)

if __name__ == "__main__":
    main()
