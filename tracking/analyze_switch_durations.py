#!/usr/bin/env python3
"""
Analyze how long classification switches last.
For each track with switches, measure the duration of each class state.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

# Class mapping
CLASS_NAMES = {0: 'car', 1: 'motorcycle', 2: 'tricycle', 3: 'bus', 4: 'truck', 5: 'van'}

def analyze_switch_durations(predictions_file):
    """Analyze duration of each classification state for tracks with switches."""

    # Load predictions (MOT format: frame,id,x,y,w,h,conf,class,-1,-1)
    df = pd.read_csv(predictions_file, header=None,
                     names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'x1', 'x2'])

    print(f"Loaded {len(df)} detections")
    print(f"Unique tracks: {df['id'].nunique()}")
    print(f"Frame range: {df['frame'].min()} - {df['frame'].max()}")
    print()

    # Analyze each track
    all_durations = []  # List of (track_id, class_name, duration_frames)
    tracks_with_switches = []

    for track_id in df['id'].unique():
        track_data = df[df['id'] == track_id].sort_values('frame')

        if len(track_data) < 2:
            continue

        # Build list of (frame, class) for this track
        frames = track_data['frame'].values
        classes = track_data['class'].values

        # Check if this track has any switches
        unique_classes = np.unique(classes)
        if len(unique_classes) == 1:
            continue  # No switches in this track

        # Track has switches - analyze durations
        segments = []  # List of (start_frame, end_frame, class_id)

        current_class = classes[0]
        segment_start = frames[0]

        for i in range(1, len(frames)):
            if classes[i] != current_class:
                # End of segment
                segments.append({
                    'start_frame': segment_start,
                    'end_frame': frames[i-1],
                    'class_id': current_class,
                    'class_name': CLASS_NAMES.get(current_class, f'unknown_{current_class}'),
                    'duration': frames[i-1] - segment_start + 1
                })
                # Start new segment
                current_class = classes[i]
                segment_start = frames[i]

        # Don't forget the last segment
        segments.append({
            'start_frame': segment_start,
            'end_frame': frames[-1],
            'class_id': current_class,
            'class_name': CLASS_NAMES.get(current_class, f'unknown_{current_class}'),
            'duration': frames[-1] - segment_start + 1
        })

        tracks_with_switches.append({
            'track_id': track_id,
            'total_frames': len(track_data),
            'num_switches': len(segments) - 1,
            'segments': segments
        })

        for seg in segments:
            all_durations.append({
                'track_id': track_id,
                'class_name': seg['class_name'],
                'duration': seg['duration']
            })

    return tracks_with_switches, all_durations


def print_duration_analysis(tracks_with_switches, all_durations):
    """Print analysis of switch durations."""

    print("=" * 80)
    print("CLASSIFICATION SWITCH DURATION ANALYSIS")
    print("=" * 80)
    print()

    if not all_durations:
        print("No tracks with classification switches found.")
        return

    # Convert to DataFrame for easier analysis
    dur_df = pd.DataFrame(all_durations)

    # Overall duration statistics
    print("OVERALL DURATION STATISTICS (frames)")
    print("-" * 40)
    print(f"Total segments analyzed: {len(dur_df)}")
    print(f"Mean duration: {dur_df['duration'].mean():.1f} frames")
    print(f"Median duration: {dur_df['duration'].median():.1f} frames")
    print(f"Min duration: {dur_df['duration'].min()} frames")
    print(f"Max duration: {dur_df['duration'].max()} frames")
    print()

    # Duration distribution buckets
    print("DURATION DISTRIBUTION")
    print("-" * 40)
    buckets = [
        (1, 1, "1 frame (instant flicker)"),
        (2, 2, "2 frames"),
        (3, 5, "3-5 frames (brief)"),
        (6, 10, "6-10 frames"),
        (11, 30, "11-30 frames (~1 sec at 30fps)"),
        (31, 100, "31-100 frames"),
        (101, float('inf'), "100+ frames (sustained)")
    ]

    for min_d, max_d, label in buckets:
        if max_d == float('inf'):
            count = len(dur_df[dur_df['duration'] >= min_d])
        else:
            count = len(dur_df[(dur_df['duration'] >= min_d) & (dur_df['duration'] <= max_d)])
        pct = 100 * count / len(dur_df)
        print(f"  {label}: {count} ({pct:.1f}%)")
    print()

    # Per-class analysis
    print("DURATION BY CLASS")
    print("-" * 40)
    for class_name in dur_df['class_name'].unique():
        class_dur = dur_df[dur_df['class_name'] == class_name]['duration']
        print(f"  {class_name}: mean={class_dur.mean():.1f}, median={class_dur.median():.1f}, "
              f"min={class_dur.min()}, max={class_dur.max()}")
    print()

    # Short flickers (likely misclassification noise)
    print("SHORT FLICKERS (1-5 frames) - Likely Noise")
    print("-" * 40)
    short_flickers = dur_df[dur_df['duration'] <= 5]
    print(f"Total short flickers: {len(short_flickers)} ({100*len(short_flickers)/len(dur_df):.1f}%)")

    # Count by class
    for class_name in short_flickers['class_name'].unique():
        count = len(short_flickers[short_flickers['class_name'] == class_name])
        print(f"  {class_name}: {count}")
    print()

    # Detailed per-track analysis (top 5 worst)
    print("TOP 5 TRACKS WITH MOST SWITCHES")
    print("-" * 40)
    sorted_tracks = sorted(tracks_with_switches, key=lambda x: x['num_switches'], reverse=True)[:5]

    for track in sorted_tracks:
        print(f"\nTrack ID: {track['track_id']}")
        print(f"  Total frames: {track['total_frames']}")
        print(f"  Number of switches: {track['num_switches']}")
        print(f"  Segment durations:")

        for seg in track['segments'][:15]:  # Show first 15 segments
            print(f"    {seg['class_name']}: {seg['duration']} frames "
                  f"(frame {seg['start_frame']} → {seg['end_frame']})")

        if len(track['segments']) > 15:
            print(f"    ... and {len(track['segments']) - 15} more segments")


def main():
    # Use DCNv2-Pan predictions for Gate2.9_Oct7
    predictions_file = Path("/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv2/DCNv2-Pan/Gate2.9_Oct7/Gate2.9_Oct7_predictions.txt")

    if not predictions_file.exists():
        print(f"Predictions file not found: {predictions_file}")
        return

    print(f"Analyzing: {predictions_file}")
    print()

    tracks_with_switches, all_durations = analyze_switch_durations(predictions_file)
    print_duration_analysis(tracks_with_switches, all_durations)


if __name__ == "__main__":
    main()
