#!/usr/bin/env python3
"""
Merge two YOLO datasets (G3-Aster-YOLO and G3-Migz-YOLO)
Combines images, labels, and creates unified train.txt
"""

import os
import shutil
from pathlib import Path

def merge_yolo_datasets():
    """Merge two YOLO datasets into one"""

    # Source datasets
    dataset1 = "/media/mydrive/GitHub/YOLO-20260129T044456Z-3-001/YOLO/G3-Aster-YOLO"
    dataset2 = "/media/mydrive/GitHub/YOLO-20260129T044456Z-3-001/YOLO/G3-Migz-YOLO"

    # Output merged dataset
    output_dir = "/media/mydrive/GitHub/YOLO-20260129T044456Z-3-001/YOLO/G3-Merged-YOLO"

    print("="*70)
    print("YOLO Dataset Merger")
    print("="*70)
    print(f"\nDataset 1: {dataset1}")
    print(f"Dataset 2: {dataset2}")
    print(f"Output:    {output_dir}")

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    output_train_data = os.path.join(output_dir, "obj_train_data")
    os.makedirs(output_train_data, exist_ok=True)

    print(f"\n✓ Created output directory: {output_dir}")

    # Copy obj.names and obj.data from first dataset
    shutil.copy2(
        os.path.join(dataset1, "obj.names"),
        os.path.join(output_dir, "obj.names")
    )
    shutil.copy2(
        os.path.join(dataset1, "obj.data"),
        os.path.join(output_dir, "obj.data")
    )

    print("✓ Copied obj.names and obj.data")

    # Track statistics
    total_images = 0
    total_labels = 0
    train_txt_lines = []

    # Process both datasets
    datasets = [
        (dataset1, "G3-Aster"),
        (dataset2, "G3-Migz")
    ]

    for dataset_path, dataset_name in datasets:
        print(f"\nProcessing {dataset_name}...")

        source_train_data = os.path.join(dataset_path, "obj_train_data")

        # Get all image files
        image_files = sorted([f for f in os.listdir(source_train_data) if f.endswith('.jpg')])

        dataset_images = 0
        dataset_labels = 0

        for img_file in image_files:
            # Image path
            source_img = os.path.join(source_train_data, img_file)
            dest_img = os.path.join(output_train_data, img_file)

            # Label path
            label_file = img_file.replace('.jpg', '.txt')
            source_label = os.path.join(source_train_data, label_file)
            dest_label = os.path.join(output_train_data, label_file)

            # Copy image
            shutil.copy2(source_img, dest_img)
            dataset_images += 1

            # Copy label if exists
            if os.path.exists(source_label):
                shutil.copy2(source_label, dest_label)
                dataset_labels += 1

            # Add to train.txt
            train_txt_lines.append(f"data/obj_train_data/{img_file}\n")

        total_images += dataset_images
        total_labels += dataset_labels

        print(f"  ✓ Copied {dataset_images} images")
        print(f"  ✓ Copied {dataset_labels} labels")

    # Write combined train.txt
    train_txt_path = os.path.join(output_dir, "train.txt")
    with open(train_txt_path, 'w') as f:
        f.writelines(sorted(train_txt_lines))

    print(f"\n✓ Created train.txt with {len(train_txt_lines)} entries")

    # Create data.yaml for modern YOLO (Ultralytics format)
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    with open(data_yaml_path, 'w') as f:
        f.write(f"""# G3 Merged Dataset
path: {output_dir}
train: obj_train_data
val: obj_train_data  # Using same as train, adjust if you have validation set

# Classes
names:
  0: Car
  1: Motorcycle
  2: Tricycle
  3: Bus
  4: Van
  5: Truck

# Number of classes
nc: 6
""")

    print(f"✓ Created data.yaml for Ultralytics YOLO")

    # Print summary
    print("\n" + "="*70)
    print("MERGE COMPLETE")
    print("="*70)
    print(f"\nMerged Dataset Statistics:")
    print(f"  Total Images:  {total_images}")
    print(f"  Total Labels:  {total_labels}")
    print(f"  Output Dir:    {output_dir}")
    print(f"\nFiles created:")
    print(f"  - {output_dir}/obj_train_data/ (images and labels)")
    print(f"  - {output_dir}/train.txt")
    print(f"  - {output_dir}/obj.names")
    print(f"  - {output_dir}/obj.data")
    print(f"  - {output_dir}/data.yaml (for Ultralytics YOLO)")

    print("\n" + "="*70 + "\n")

    return output_dir


if __name__ == "__main__":
    merge_yolo_datasets()
