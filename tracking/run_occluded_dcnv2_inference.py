#!/usr/bin/env python3
"""
DCNv2 Inference on OccludedYOLO Images

Runs all DCNv2 models on the annotated occluded object images
and saves detection results with visualizations.

Based on: run_all_dcnv2_bytetrack.py
"""

import sys
import os
import warnings
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# CRITICAL: Use local custom ultralytics with DCNv2/DCNv3 modules
sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np


def run_model_on_images(model_path, model_name, images_dir, output_base_dir, conf_threshold=0.25):
    """Run inference for a single DCNv2 model on all images"""
    print("\n" + "="*70)
    print(f"Processing: {model_name} | Images: {images_dir.name}")
    print("="*70)

    # Create output directory structure
    output_dir = output_base_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(exist_ok=True)
    labels_output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load model
    print(f"Loading model from: {model_path}")
    try:
        model = YOLO(str(model_path))
        print("✓ Model loaded successfully")
        print(f"  Classes: {list(model.names.values())}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"  1. Make sure CUDA libraries are accessible")
        print(f"  2. Check that model file exists: {model_path.exists()}")
        print(f"  3. Try running with CPU mode if GPU is unavailable")
        import traceback
        traceback.print_exc()
        return None

    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in image_extensions:
        images.extend(sorted(images_dir.glob(ext)))

    if not images:
        print(f"✗ No images found in {images_dir}")
        return None

    print(f"✓ Found {len(images)} images to process")

    # Define class colors (matching YOLO vehicle classes)
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
    detection_by_class = {class_id: 0 for class_id in model.names.keys()}
    processing_times = []
    start_time = time.time()

    print(f"\nProcessing images with conf={conf_threshold}...\n")

    # Process images
    for idx, image_path in enumerate(images, 1):
        img_start = time.time()

        # Run inference
        try:
            results = model.predict(
                source=str(image_path),
                conf=conf_threshold,
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
            if boxes is not None and len(boxes) > 0:
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
                output_img_path = images_output_dir / image_path.name
                cv2.imwrite(str(output_img_path), img)

            # Save labels in YOLO format
            if boxes is not None and len(boxes) > 0:
                label_file = labels_output_dir / f"{image_path.stem}.txt"
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
                print(f"Progress: {idx:4d}/{len(images)} | "
                      f"Detections: {num_detections:2d} | "
                      f"FPS: {fps:5.1f} | "
                      f"ETA: {eta:6.1f}s")

        except Exception as e:
            print(f"✗ Error processing {image_path.name}: {e}")
            continue

    total_time = time.time() - start_time
    avg_fps = len(images) / total_time if total_time > 0 else 0

    # Create summary report
    print(f"\n✓ Model {model_name} complete!")
    print(f"  Total images: {len(images)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg detections/image: {total_detections/len(images):.2f}")

    # Save summary
    summary_file = output_dir / 'inference_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"DCNv2 INFERENCE ON OCCLUDED IMAGES - {model_name}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Source Directory: {images_dir}\n")
        f.write(f"Confidence Threshold: {conf_threshold}\n\n")

        f.write("="*70 + "\n")
        f.write("PROCESSING STATISTICS\n")
        f.write("="*70 + "\n")
        f.write(f"Total images processed: {len(images)}\n")
        f.write(f"Total processing time: {total_time:.2f}s\n")
        f.write(f"Average FPS: {avg_fps:.2f}\n")
        f.write(f"Average time per image: {total_time/len(images):.3f}s\n\n")

        f.write("="*70 + "\n")
        f.write("DETECTION STATISTICS\n")
        f.write("="*70 + "\n")
        f.write(f"Total detections: {total_detections}\n")
        f.write(f"Average detections per image: {total_detections/len(images):.2f}\n\n")

        f.write("Detections by class:\n")
        for class_id, count in detection_by_class.items():
            class_name = model.names[class_id]
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            f.write(f"  {class_name:12s}: {count:5d} ({percentage:5.1f}%)\n")

        f.write("\n" + "="*70 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("="*70 + "\n")
        f.write(f"Images: {images_output_dir}\n")
        f.write(f"Labels: {labels_output_dir}\n")
        f.write(f"Summary: {summary_file}\n")

    print(f"✓ Summary saved: {summary_file}")

    return {
        'model_name': model_name,
        'total_detections': total_detections,
        'images_processed': len(images),
        'avg_fps': avg_fps,
        'detection_by_class': detection_by_class
    }


def main():
    """Main batch processing function"""

    print("\n" + "="*70)
    print("DCNv2 Models - Occluded Images Inference")
    print("="*70)
    print("\nProcessing annotated occluded object images")
    print("Source: /media/mydrive/GitHub/ultralytics/tracking/OccludedYOLO/inference")
    print()

    # Configuration - matches run_all_dcnv2_bytetrack.py structure
    MODEL_DIR = Path("/media/mydrive/GitHub/ultralytics/modified_model")
    IMAGES_DIR = Path("/media/mydrive/GitHub/ultralytics/tracking/OccludedYOLO/inference")
    OUTPUT_BASE_DIR = Path("/media/mydrive/GitHub/ultralytics/tracking/OccludedYOLO/inference_results_dcnv2")
    CONF_THRESHOLD = 0.25

    # Get all DCNv2 model files from modified_model directory
    model_files = sorted(MODEL_DIR.glob("DCNv2-*.pt"))
    # Exclude _fixed versions
    model_files = [m for m in model_files if '_fixed' not in m.name]

    if not model_files:
        print(f"✗ No DCNv2 model files found in {MODEL_DIR}")
        print(f"  Looking for: DCNv2-*.pt")
        sys.exit(1)

    # Create models list as (path, name) tuples
    models = [(model_path, model_path.stem) for model_path in model_files]

    print(f"Found {len(models)} DCNv2 model(s) in {MODEL_DIR}:")
    for model_path, model_name in models:
        print(f"  - {model_name} ({model_path.stat().st_size / 1e6:.1f} MB)")
    print()

    # Check images directory
    if not IMAGES_DIR.exists():
        print(f"✗ Images directory not found: {IMAGES_DIR}")
        sys.exit(1)

    # Count images
    image_count = sum(1 for _ in IMAGES_DIR.glob('*.jpg'))
    print("="*70)
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Total images: {image_count}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print("="*70)

    # Track all results
    all_results = []

    # Process each model
    total_models = len(models)
    for current_model, (model_path, model_name) in enumerate(models, 1):
        print(f"\n{'='*70}")
        print(f"MODEL {current_model}/{total_models}")
        print(f"{'='*70}")

        result = run_model_on_images(
            model_path,
            model_name,
            IMAGES_DIR,
            OUTPUT_BASE_DIR,
            CONF_THRESHOLD
        )

        if result:
            all_results.append(result)

    # Print final summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"\nProcessed {len(all_results)}/{total_models} models successfully")
    print(f"Output directory: {OUTPUT_BASE_DIR}")

    if all_results:
        print("\nResults by model:")
        for result in all_results:
            print(f"\n  {result['model_name']}:")
            print(f"    - Images processed: {result['images_processed']}")
            print(f"    - Total detections: {result['total_detections']}")
            print(f"    - Avg FPS: {result['avg_fps']:.2f}")

            # Show top 3 detected classes
            top_classes = sorted(
                result['detection_by_class'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            print(f"    - Top detections:", end='')
            for class_id, count in top_classes:
                if count > 0:
                    print(f" {count}x class{class_id}", end='')
            print()

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
