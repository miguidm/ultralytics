#!/usr/bin/env python3
"""
Generate Tracking Ground Truth with Track IDs
Takes YOLO detection annotations and creates tracking ground truth by matching objects across frames
"""

import os
import numpy as np
from collections import defaultdict
import cv2

def parse_yolo_annotation(label_file, img_width, img_height):
    """Parse YOLO format annotation"""
    detections = []

    if not os.path.exists(label_file):
        return detections

    with open(label_file, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert to absolute coordinates
            x1 = (x_center - width/2) * img_width
            y1 = (y_center - height/2) * img_height
            x2 = (x_center + width/2) * img_width
            y2 = (y_center + height/2) * img_height

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_id': class_id,
                'det_idx': idx  # Original detection index in file
            })

    return detections


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max < x_min or y_max < y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
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
            if curr_obj['class_id'] == prev_obj['class_id']:
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


def generate_tracking_ground_truth(gt_dir, output_dir, video_path):
    """
    Generate tracking ground truth from detection annotations
    Assigns track IDs by matching objects across frames
    """

    print("="*70)
    print("GENERATING TRACKING GROUND TRUTH FROM DETECTIONS")
    print("="*70)

    class_names = ['Car', 'Motorcycle', 'Tricycle', 'Bus', 'Van', 'Truck']

    # Get video dimensions
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video")
        return None

    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"\nVideo dimensions: {img_width}x{img_height}")

    # Load all annotations
    print(f"\nLoading annotations from: {gt_dir}")
    gt_data_dir = os.path.join(gt_dir, "obj_train_data")

    # Get all annotation files and extract frame numbers
    frame_annotations = {}
    gt_files = [f for f in os.listdir(gt_data_dir) if f.endswith('.txt')]

    for gt_file in gt_files:
        basename = os.path.splitext(gt_file)[0]
        parts = basename.split('_')

        if len(parts) >= 3:
            try:
                frame_num = int(parts[-1])
                label_path = os.path.join(gt_data_dir, gt_file)
                detections = parse_yolo_annotation(label_path, img_width, img_height)

                if detections:
                    frame_annotations[frame_num] = detections
            except ValueError:
                continue

    print(f"✓ Loaded annotations for {len(frame_annotations)} frames")

    # Sort frames
    sorted_frames = sorted(frame_annotations.keys())
    print(f"  Frame range: {sorted_frames[0]} - {sorted_frames[-1]}")

    # Assign track IDs by matching across frames
    print(f"\nAssigning track IDs by matching objects across frames...")

    next_track_id = 1
    frame_tracks = {}  # {frame_num: [{det, track_id}, ...]}

    # Track state: {track_id: last_object_info}
    active_tracks = {}
    track_id_map = {}  # Map from (frame, det_idx) to track_id

    # Statistics
    total_objects = 0
    total_tracks = 0
    id_switches_detected = 0

    for i, frame_num in enumerate(sorted_frames):
        curr_objects = frame_annotations[frame_num]
        total_objects += len(curr_objects)

        # For first frame, create new tracks for all objects
        if i == 0:
            frame_tracks[frame_num] = []
            for obj in curr_objects:
                track_id = next_track_id
                next_track_id += 1
                total_tracks += 1

                frame_tracks[frame_num].append({
                    'bbox': obj['bbox'],
                    'class_id': obj['class_id'],
                    'track_id': track_id
                })

                active_tracks[track_id] = obj
        else:
            # Match with previous frame
            prev_frame_num = sorted_frames[i - 1]
            prev_tracks = frame_tracks[prev_frame_num]

            # Match current objects to previous tracks
            matches = match_objects_between_frames(prev_tracks, curr_objects, iou_threshold=0.3)

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
                    'class_id': obj['class_id'],
                    'track_id': track_id
                })

        # Progress
        if (i + 1) % 100 == 0 or (i + 1) == len(sorted_frames):
            print(f"  Processed {i+1}/{len(sorted_frames)} frames, Active tracks: {next_track_id-1}", end='\r')

    print(f"\n  Processed {len(sorted_frames)}/{len(sorted_frames)} frames ✓")

    # Save tracking ground truth in MOT format
    os.makedirs(output_dir, exist_ok=True)
    mot_gt_file = os.path.join(output_dir, "gt.txt")

    print(f"\nSaving tracking ground truth...")
    print(f"  Output: {mot_gt_file}")

    with open(mot_gt_file, 'w') as f:
        for frame_num in sorted(frame_tracks.keys()):
            for track in frame_tracks[frame_num]:
                x1, y1, x2, y2 = track['bbox']
                w = x2 - x1
                h = y2 - y1

                # MOT format: frame,id,x,y,w,h,conf,class,vis
                # conf=1 for GT, vis=1 for fully visible
                f.write(f"{frame_num},{track['track_id']},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,{track['class_id']},1\n")

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
    print("\n" + "="*70)
    print("TRACKING GROUND TRUTH STATISTICS")
    print("="*70)
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
    print("\n" + "="*70 + "\n")

    return mot_gt_file


def main():
    """Main function"""

    # Paths
    gt_dir = "/media/mydrive/GitHub/YOLO-20260129T044456Z-3-001/YOLO/G3-Merged-YOLO"
    video_path = "/media/mydrive/GitHub/ultralytics/videos/Gate3_Oct7.mp4"
    output_dir = "gate3_tracking_ground_truth"

    # Generate tracking GT
    mot_gt_file = generate_tracking_ground_truth(gt_dir, output_dir, video_path)

    if mot_gt_file:
        print(f"Next step: Run proper MOTA evaluation using this ground truth")
        print(f"  GT file: {mot_gt_file}")
        print(f"  Prediction files: gate3_test_results_dcnv2/*/Gate3_Oct7_predictions.txt")


if __name__ == "__main__":
    main()
