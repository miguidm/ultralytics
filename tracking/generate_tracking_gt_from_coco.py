#!/usr/bin/env python3
"""
Generate Tracking Ground Truth from COCO Dataset
Converts COCO detection annotations to MOT format tracking ground truth
by matching objects across consecutive frames using IoU
"""

import os
import json
import numpy as np
from collections import defaultdict
import argparse


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes in [x, y, w, h] format"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to [x1, y1, x2, y2]
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    # Calculate intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max < x_min or y_max < y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def match_objects_between_frames(prev_objects, curr_objects, iou_threshold=0.3):
    """
    Match objects between consecutive frames using IoU
    Returns: dict mapping curr_idx -> prev_idx
    """
    if len(prev_objects) == 0 or len(curr_objects) == 0:
        return {}

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(curr_objects), len(prev_objects)))

    for i, curr_obj in enumerate(curr_objects):
        for j, prev_obj in enumerate(prev_objects):
            # Only match same class
            if curr_obj['category_id'] == prev_obj['category_id']:
                iou_matrix[i, j] = calculate_iou(curr_obj['bbox'], prev_obj['bbox'])

    # Greedy matching (highest IoU first)
    matches = {}
    matched_prev = set()

    # Get all potential matches above threshold
    candidates = []
    for i in range(len(curr_objects)):
        for j in range(len(prev_objects)):
            if iou_matrix[i, j] >= iou_threshold:
                candidates.append((iou_matrix[i, j], i, j))

    # Sort by IoU (highest first)
    candidates.sort(reverse=True)

    # Assign matches greedily
    for iou, curr_idx, prev_idx in candidates:
        if curr_idx not in matches and prev_idx not in matched_prev:
            matches[curr_idx] = prev_idx
            matched_prev.add(prev_idx)

    return matches


def extract_frame_number(filename):
    """Extract frame number from filename like 'gate3_oct_0001.jpg'"""
    parts = filename.replace('.jpg', '').split('_')
    if len(parts) >= 3:
        try:
            return int(parts[-1])
        except ValueError:
            return None
    return None


def generate_tracking_gt_for_gate(coco_data, gate_name, output_dir, iou_threshold=0.3):
    """
    Generate tracking ground truth for a specific gate

    Args:
        coco_data: COCO format data dictionary
        gate_name: Gate identifier (e.g., 'gate3_oct')
        output_dir: Output directory for MOT format files
        iou_threshold: IoU threshold for matching objects
    """

    print("="*80)
    print(f"Generating Tracking Ground Truth - {gate_name}")
    print("="*80)

    # Filter images for this gate
    gate_images = [img for img in coco_data['images']
                   if img['file_name'].startswith(gate_name)]

    if len(gate_images) == 0:
        print(f"⚠ No images found for {gate_name}")
        return None

    print(f"\n✓ Found {len(gate_images)} images for {gate_name}")

    # Create image_id to annotations mapping
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # Sort images by frame number
    images_with_frames = []
    for img in gate_images:
        frame_num = extract_frame_number(img['file_name'])
        if frame_num is not None:
            images_with_frames.append((frame_num, img))

    images_with_frames.sort(key=lambda x: x[0])
    print(f"✓ Sorted images by frame number: {images_with_frames[0][0]} - {images_with_frames[-1][0]}")

    # Build frame-by-frame detections
    frame_detections = {}
    for frame_num, img in images_with_frames:
        img_id = img['id']
        anns = annotations_by_image[img_id]

        if len(anns) > 0:
            detections = []
            for ann in anns:
                detections.append({
                    'bbox': ann['bbox'],  # [x, y, w, h] in COCO format
                    'category_id': ann['category_id'],
                    'area': ann['area']
                })
            frame_detections[frame_num] = detections

    print(f"✓ Loaded detections for {len(frame_detections)} frames")

    # Assign track IDs by matching across frames
    print(f"\nAssigning track IDs (IoU threshold: {iou_threshold})...")

    next_track_id = 1
    frame_tracks = {}  # {frame_num: [{bbox, category_id, track_id}, ...]}

    sorted_frames = sorted(frame_detections.keys())

    # Track statistics
    total_objects = 0
    total_tracks = 0

    for i, frame_num in enumerate(sorted_frames):
        curr_objects = frame_detections[frame_num]
        total_objects += len(curr_objects)

        # First frame: create new tracks for all objects
        if i == 0:
            frame_tracks[frame_num] = []
            for obj in curr_objects:
                track_id = next_track_id
                next_track_id += 1
                total_tracks += 1

                frame_tracks[frame_num].append({
                    'bbox': obj['bbox'],
                    'category_id': obj['category_id'],
                    'track_id': track_id
                })
        else:
            # Match with previous frame
            prev_frame_num = sorted_frames[i - 1]
            prev_tracks = frame_tracks[prev_frame_num]

            # Match current objects to previous tracks
            matches = match_objects_between_frames(prev_tracks, curr_objects, iou_threshold)

            frame_tracks[frame_num] = []

            for curr_idx, obj in enumerate(curr_objects):
                if curr_idx in matches:
                    # Matched to previous track
                    prev_idx = matches[curr_idx]
                    track_id = prev_tracks[prev_idx]['track_id']
                else:
                    # New track
                    track_id = next_track_id
                    next_track_id += 1
                    total_tracks += 1

                frame_tracks[frame_num].append({
                    'bbox': obj['bbox'],
                    'category_id': obj['category_id'],
                    'track_id': track_id
                })

        # Progress
        if (i + 1) % 100 == 0 or (i + 1) == len(sorted_frames):
            print(f"  Processed {i+1}/{len(sorted_frames)} frames, "
                  f"Tracks created: {next_track_id-1}", end='\r')

    print(f"\n  Processed {len(sorted_frames)}/{len(sorted_frames)} frames ✓")

    # Save tracking ground truth in MOT format
    gate_output_dir = os.path.join(output_dir, gate_name)
    os.makedirs(gate_output_dir, exist_ok=True)
    mot_gt_file = os.path.join(gate_output_dir, "gt.txt")

    print(f"\nSaving tracking ground truth...")
    print(f"  Output: {mot_gt_file}")

    with open(mot_gt_file, 'w') as f:
        for frame_num in sorted(frame_tracks.keys()):
            for track in frame_tracks[frame_num]:
                x, y, w, h = track['bbox']

                # MOT format: frame,id,x,y,w,h,conf,class,vis
                # conf=1 for GT, vis=1 for fully visible
                # COCO category_id starts from 1, so we use it directly
                f.write(f"{frame_num},{track['track_id']},{x:.2f},{y:.2f},"
                       f"{w:.2f},{h:.2f},1,{track['category_id']-1},1\n")

    print(f"✓ Saved tracking ground truth")

    # Calculate track statistics
    track_lengths = defaultdict(int)
    for frame_num in frame_tracks:
        for track in frame_tracks[frame_num]:
            track_lengths[track['track_id']] += 1

    avg_track_length = np.mean(list(track_lengths.values()))
    max_track_length = max(track_lengths.values())
    min_track_length = min(track_lengths.values())

    # Track lifespan distribution
    very_short = sum(1 for l in track_lengths.values() if l < 5)
    short_tracks = sum(1 for l in track_lengths.values() if 5 <= l < 20)
    medium_tracks = sum(1 for l in track_lengths.values() if 20 <= l < 50)
    long_tracks = sum(1 for l in track_lengths.values() if l >= 50)

    # Print statistics
    print("\n" + "="*80)
    print(f"TRACKING GROUND TRUTH STATISTICS - {gate_name}")
    print("="*80)
    print(f"\nFrames annotated: {len(sorted_frames)}")
    print(f"Frame range: {sorted_frames[0]} - {sorted_frames[-1]}")
    print(f"\nTotal objects: {total_objects}")
    print(f"Total tracks created: {total_tracks}")
    print(f"\nTrack Length Statistics:")
    print(f"  Average: {avg_track_length:.1f} frames")
    print(f"  Maximum: {max_track_length} frames")
    print(f"  Minimum: {min_track_length} frames")
    print(f"\nTrack Distribution:")
    print(f"  Very Short (<5 frames):  {very_short} tracks ({very_short/total_tracks*100:.1f}%)")
    print(f"  Short (5-20 frames):     {short_tracks} tracks ({short_tracks/total_tracks*100:.1f}%)")
    print(f"  Medium (20-50 frames):   {medium_tracks} tracks ({medium_tracks/total_tracks*100:.1f}%)")
    print(f"  Long (>=50 frames):      {long_tracks} tracks ({long_tracks/total_tracks*100:.1f}%)")
    print(f"\n✓ Tracking ground truth created successfully!")
    print(f"  Output file: {mot_gt_file}")
    print("\n" + "="*80 + "\n")

    return {
        'gate': gate_name,
        'frames': len(sorted_frames),
        'objects': total_objects,
        'tracks': total_tracks,
        'avg_track_length': avg_track_length,
        'output_file': mot_gt_file
    }


def main():
    """Main function"""

    parser = argparse.ArgumentParser(
        description='Generate tracking ground truth from COCO dataset for all gates'
    )
    parser.add_argument('--coco-json', type=str,
                        default='/media/mydrive/GitHub/ultralytics/References/MergedAll/annotations/instances_default.json',
                        help='Path to COCO annotations JSON file')
    parser.add_argument('--output-dir', type=str,
                        default='tracking_ground_truth_all_gates',
                        help='Output directory for tracking ground truth files')
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                        help='IoU threshold for matching objects across frames')
    parser.add_argument('--gates', type=str, nargs='+', default=None,
                        help='Specific gates to process (e.g., gate3_oct gate2_oct). If not specified, processes all.')

    args = parser.parse_args()

    print("="*80)
    print("GENERATE TRACKING GROUND TRUTH FROM COCO DATASET")
    print("="*80)
    print(f"\nCOCO JSON: {args.coco_json}")
    print(f"Output Directory: {args.output_dir}")
    print(f"IoU Threshold: {args.iou_threshold}")

    # Load COCO data
    print(f"\nLoading COCO annotations...")
    with open(args.coco_json, 'r') as f:
        coco_data = json.load(f)

    print(f"✓ Loaded COCO data")
    print(f"  Total images: {len(coco_data['images'])}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {[cat['name'] for cat in coco_data['categories']]}")

    # Identify all gates
    gate_groups = defaultdict(list)
    for img in coco_data['images']:
        filename = img['file_name']
        parts = filename.split('_')
        if len(parts) >= 2:
            gate = parts[0] + '_' + parts[1]
            gate_groups[gate].append(img)

    print(f"\n✓ Found {len(gate_groups)} gate groups:")
    for gate, images in sorted(gate_groups.items()):
        print(f"    {gate}: {len(images)} images")

    # Filter gates if specified
    if args.gates:
        gates_to_process = args.gates
        print(f"\nProcessing specified gates: {gates_to_process}")
    else:
        gates_to_process = sorted(gate_groups.keys())
        print(f"\nProcessing all gates")

    # Process each gate
    results = []
    for gate_name in gates_to_process:
        if gate_name not in gate_groups:
            print(f"\n⚠ Warning: Gate '{gate_name}' not found in dataset")
            continue

        result = generate_tracking_gt_for_gate(
            coco_data,
            gate_name,
            args.output_dir,
            args.iou_threshold
        )

        if result:
            results.append(result)

    # Summary
    if results:
        print("\n" + "="*80)
        print("GENERATION SUMMARY")
        print("="*80)
        print(f"\n{'Gate':<20} {'Frames':<10} {'Objects':<10} {'Tracks':<10} {'Avg Length':<12} {'Output File':<40}")
        print("-"*100)

        for r in results:
            print(f"{r['gate']:<20} {r['frames']:<10} {r['objects']:<10} {r['tracks']:<10} "
                  f"{r['avg_track_length']:<12.1f} {r['output_file']:<40}")

        print("\n" + "="*80)
        print(f"✓ Successfully generated tracking ground truth for {len(results)} gates")
        print(f"  Output directory: {args.output_dir}")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
