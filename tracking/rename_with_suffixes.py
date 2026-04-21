#!/usr/bin/env python3
"""
Rename model report files to add proper suffixes:
- Add '-n' suffix for nano models
- Add '-m' suffix for medium models
"""

import os
import shutil

# Rename nano model files (add -n suffix)
nano_dir = "all_nano_reports"
if os.path.exists(nano_dir):
    print(f"Processing {nano_dir}...")
    for filename in os.listdir(nano_dir):
        if filename.endswith('.txt'):
            old_path = os.path.join(nano_dir, filename)

            # Skip if already has -n suffix or is Vanilla-YOLOv8n (already has 'n')
            if '-n.txt' in filename or 'Vanilla-YOLOv8n.txt' in filename:
                continue

            # Add -n suffix before .txt
            new_filename = filename.replace('.txt', '-n.txt')
            new_path = os.path.join(nano_dir, new_filename)

            try:
                shutil.move(old_path, new_path)
                print(f"  Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"  Error renaming {filename}: {e}")

# Rename medium model files (add -m suffix)
medium_dir = "model_reports"
if os.path.exists(medium_dir):
    print(f"\nProcessing {medium_dir}...")
    for filename in os.listdir(medium_dir):
        if filename.endswith('.txt') and filename != 'README.txt':
            old_path = os.path.join(medium_dir, filename)

            # Skip if already has -m suffix or is Vanilla-YOLOv8m (already has 'm')
            if '-m.txt' in filename or 'Vanilla-YOLOv8m.txt' in filename:
                continue

            # Add -m suffix before .txt
            new_filename = filename.replace('.txt', '-m.txt')
            new_path = os.path.join(medium_dir, new_filename)

            try:
                shutil.move(old_path, new_path)
                print(f"  Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"  Error renaming {filename}: {e}")

print("\nRenaming complete!")
print("\nExample filenames:")
print("  NANO:   gate2_DCNv2-FPN-n.txt")
print("  MEDIUM: Gate3.5_DCNv2-FPN-m.txt")
