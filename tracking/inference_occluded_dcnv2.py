#!/usr/bin/env python3
"""
DCNv2 Inference on OccludedYOLO Images

Runs DCNv2 model inference on the annotated occluded object images
and saves detection results with visualizations.
"""

import sys
import os
import argparse
from pathlib import Path
import warnings

# ============================================================================
# DCNv2 ENVIRONMENT SETUP - Must be done BEFORE importing YOLO!
# ============================================================================

def setup_dcnv2_environment():
    """Configure environment for DCNv2 operations"""

    # Set LD_LIBRARY_PATH to include CUDA runtime and PyTorch libraries
    cuda_lib_path = "/home/migui/miniconda3/envs/dcnv2/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    torch_lib_path = "/home/migui/miniconda3/envs/dcnv2/lib/python3.10/site-packages/torch/lib"

    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}"

    # Add ultralytics root to Python path
    ultralytics_root = "/media/mydrive/GitHub/ultralytics"
    if ultralytics_root not in sys.path:
        sys.path.insert(0, ultralytics_root)

# Setup environment FIRST
setup_dcnv2_environment()

warnings.filterwarnings('ignore', category=UserWarning)

from ultralytics import YOLO
import cv2
import numpy as np
import time


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DCNv2 Inference on Occluded Images')
    parser.add_argument('--model', type=str,
                        default='/media/mydrive/GitHub/ultralytics/tracking/best-dcnv2m.pt',
                        help='Path to DCNv2 model weights')
    parser.add_argument('--source', type=str,
                        default='/media/mydrive/GitHub/ultralytics/tracking/OccludedYOLO/inference',
                        help='Path to directory containing images')
    parser.add_argument('--output', type=str,
                        default='/media/mydrive/GitHub/ultralytics/tracking/OccludedYOLO/inference_results',
                        help='Path to output directory')
    parser.add_argument('--conf', type=float,
                        default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--save-txt', action='store_true',
                        default=True,
                        help='Save detection results as txt files')
    parser.add_argument('--save-img', action='store_true',
                        default=True,
                        help='Save images with detection boxes')
    parser.add_argument('--max-images', type=int,
                        default=None,
                        help='Maximum number of images to process (for testing)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*80)
    print("DCNv2 INFERENCE ON OCCLUDED IMAGES")
    print("="*80)

    # Check paths
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    model_path = Path(args.model)

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return

    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_img:
        (output_dir / 'images').mkdir(exist_ok=True)
    if args.save_txt:
        (output_dir / 'labels').mkdir(exist_ok=True)

    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in image_extensions:
        images.extend(sorted(source_dir.glob(ext)))

    if args.max_images:
        images = images[:args.max_images]

    if not images:
        print(f"ERROR: No images found in {source_dir}")
        return

    print(f"\nConfiguration:")
    print(f"  Model: {model_path.name}")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Images found: {len(images)}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  Save images: {args.save_img}")
    print(f"  Save labels: {args.save_txt}")
    print()

    # Load model
    print("Loading DCNv2 model...")
    try:
        model = YOLO(str(model_path))
        print(f"✓ Model loaded successfully")
        print(f"  Classes: {model.names}")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Define class colors (matching your obj.names)
    class_colors = {
        0: (55, 250, 250),    # Car - yellow
        1: (83, 179, 36),     # Motorcycle - green
        2: (83, 50, 250),     # Tricycle - red
        3: (245, 61, 184),    # Bus - pink
        4: (255, 221, 51),    # Van - cyan
        5: (49, 147, 245),    # Truck - orange
    }

    # Statistics
    total_detections = 0
    detection_by_class = {}
    for class_id in model.names.keys():
        detection_by_class[class_id] = 0

    processing_times = []

    print("\n" + "="*80)
    print("PROCESSING IMAGES")
    print("="*80 + "\n")

    start_time = time.time()

    # Process images
    for idx, image_path in enumerate(images, 1):
        img_start = time.time()

        # Run inference
        try:
            results = model.predict(
                source=str(image_path),
                conf=args.conf,
                verbose=False,
                save=False
            )[0]

            img_time = time.time() - img_start
            processing_times.append(img_time)

            # Get detections
            boxes = results.boxes
            num_detections = len(boxes) if boxes is not None else 0
            total_detections += num_detections

            # Count by class
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls)
                    detection_by_class[class_id] += 1

            # Save visualization
            if args.save_img and boxes is not None and len(boxes) > 0:
                # Read original image
                img = cv2.imread(str(image_path))

                # Draw boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    class_id = int(box.cls)
                    class_name = model.names[class_id]

                    # Get color
                    color = class_colors.get(class_id, (255, 255, 255))

                    # Draw box
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # Draw label
                    label = f"{class_name} {conf:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (int(x1), int(y1) - label_h - 10),
                                (int(x1) + label_w, int(y1)), color, -1)
                    cv2.putText(img, label, (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Save image
                output_img_path = output_dir / 'images' / image_path.name
                cv2.imwrite(str(output_img_path), img)

            # Save labels in YOLO format
            if args.save_txt and boxes is not None and len(boxes) > 0:
                label_file = output_dir / 'labels' / f"{image_path.stem}.txt"
                with open(label_file, 'w') as f:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf)
                        class_id = int(box.cls)

                        # Convert to YOLO format (normalized center x, y, width, height)
                        img_h, img_w = results.orig_shape
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h

                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")

            # Progress update
            if idx % 100 == 0 or idx == len(images):
                avg_time = np.mean(processing_times[-100:]) if processing_times else 0
                fps = 1 / avg_time if avg_time > 0 else 0
                eta = (len(images) - idx) * avg_time
                print(f"Progress: {idx}/{len(images)} | "
                      f"Detections: {num_detections:2d} | "
                      f"FPS: {fps:5.1f} | "
                      f"ETA: {eta:5.1f}s")

        except Exception as e:
            print(f"ERROR processing {image_path.name}: {e}")
            continue

    total_time = time.time() - start_time
    avg_fps = len(images) / total_time if total_time > 0 else 0

    # Create summary report
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Total images processed: {len(images)}")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average time per image: {total_time/len(images):.3f}s")
    print(f"\nDetections:")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per image: {total_detections/len(images):.2f}")
    print(f"\nDetections by class:")
    for class_id, count in detection_by_class.items():
        class_name = model.names[class_id]
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {class_name:12s}: {count:5d} ({percentage:5.1f}%)")

    # Save summary
    summary_file = output_dir / 'inference_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("DCNv2 INFERENCE ON OCCLUDED IMAGES - SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_path.name}\n")
        f.write(f"Source: {source_dir}\n")
        f.write(f"Confidence threshold: {args.conf}\n\n")
        f.write(f"Total images processed: {len(images)}\n")
        f.write(f"Total processing time: {total_time:.2f}s\n")
        f.write(f"Average FPS: {avg_fps:.2f}\n")
        f.write(f"Average time per image: {total_time/len(images):.3f}s\n\n")
        f.write(f"Total detections: {total_detections}\n")
        f.write(f"Average detections per image: {total_detections/len(images):.2f}\n\n")
        f.write("Detections by class:\n")
        for class_id, count in detection_by_class.items():
            class_name = model.names[class_id]
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            f.write(f"  {class_name:12s}: {count:5d} ({percentage:5.1f}%)\n")

    print(f"\n✓ Summary saved to: {summary_file}")
    if args.save_img:
        print(f"✓ Images saved to: {output_dir / 'images'}")
    if args.save_txt:
        print(f"✓ Labels saved to: {output_dir / 'labels'}")
    print()


if __name__ == "__main__":
    main()
