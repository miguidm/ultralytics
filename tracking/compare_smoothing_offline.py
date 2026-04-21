#!/usr/bin/env python3
"""
Compare raw vs smoothed classification switches using existing predictions.
This is a fast offline analysis that doesn't require re-running tracking.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from classification_smoother import ClassificationSmoother, create_smoother

# Class mapping
CLASS_NAMES = {0: 'car', 1: 'motorcycle', 2: 'tricycle', 3: 'bus', 4: 'truck', 5: 'van'}


def analyze_predictions(predictions_file: str, strategy: str = 'hysteresis'):
    """Analyze predictions with and without smoothing."""

    # Load predictions
    df = pd.read_csv(predictions_file, header=None,
                     names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'x1', 'x2'])

    print(f"Loaded {len(df)} detections from {Path(predictions_file).name}")
    print(f"Unique tracks: {df['id'].nunique()}")
    print(f"Frame range: {df['frame'].min()} - {df['frame'].max()}")
    print()

    # Initialize smoother
    smoother = create_smoother(strategy)

    # Process each detection and get smoothed class
    smoothed_classes = []

    for _, row in df.iterrows():
        track_id = int(row['id'])
        raw_class = CLASS_NAMES.get(int(row['class']), 'unknown')
        conf = row['conf']
        frame = int(row['frame'])

        smooth_class = smoother.get_stable_class(
            track_id=track_id,
            raw_class=raw_class,
            confidence=conf,
            frame_num=frame
        )
        smoothed_classes.append(smooth_class)

    df['raw_class_name'] = df['class'].map(CLASS_NAMES)
    df['smooth_class_name'] = smoothed_classes

    # Analyze switches
    raw_switches = 0
    smooth_switches = 0
    tracks_with_raw_switches = 0
    tracks_with_smooth_switches = 0

    raw_switch_details = []
    smooth_switch_details = []

    for track_id in df['id'].unique():
        track = df[df['id'] == track_id].sort_values('frame')

        if len(track) < 2:
            continue

        # Count raw switches
        raw_classes = track['raw_class_name'].values
        track_raw_switches = sum(1 for i in range(1, len(raw_classes)) if raw_classes[i] != raw_classes[i-1])
        raw_switches += track_raw_switches
        if track_raw_switches > 0:
            tracks_with_raw_switches += 1

        # Count smooth switches
        smooth_classes = track['smooth_class_name'].values
        track_smooth_switches = sum(1 for i in range(1, len(smooth_classes)) if smooth_classes[i] != smooth_classes[i-1])
        smooth_switches += track_smooth_switches
        if track_smooth_switches > 0:
            tracks_with_smooth_switches += 1

        # Record details for worst tracks
        if track_raw_switches > 5:
            raw_switch_details.append({
                'track_id': track_id,
                'raw_switches': track_raw_switches,
                'smooth_switches': track_smooth_switches,
                'frames': len(track)
            })

    # Sort by most switches
    raw_switch_details.sort(key=lambda x: -x['raw_switches'])

    return {
        'total_detections': len(df),
        'unique_tracks': df['id'].nunique(),
        'raw_switches': raw_switches,
        'smooth_switches': smooth_switches,
        'tracks_with_raw_switches': tracks_with_raw_switches,
        'tracks_with_smooth_switches': tracks_with_smooth_switches,
        'reduction_pct': 100 * (raw_switches - smooth_switches) / max(1, raw_switches),
        'top_tracks': raw_switch_details[:10],
        'smoother_stats': smoother.get_statistics()
    }


def main():
    predictions_file = "/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv2/DCNv2-Pan/Gate2.9_Oct7/Gate2.9_Oct7_predictions.txt"

    print("=" * 70)
    print("CLASSIFICATION SMOOTHING COMPARISON")
    print("=" * 70)
    print()

    # Test different strategies
    strategies = ['majority_vote', 'hysteresis', 'lock_after_n']

    results = {}
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        print("-" * 40)
        results[strategy] = analyze_predictions(predictions_file, strategy)

        r = results[strategy]
        print(f"  Raw switches:      {r['raw_switches']}")
        print(f"  Smooth switches:   {r['smooth_switches']}")
        print(f"  Reduction:         {r['reduction_pct']:.1f}%")
        print(f"  Tracks affected:   {r['tracks_with_raw_switches']} → {r['tracks_with_smooth_switches']}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Strategy':<20} {'Raw':<10} {'Smooth':<10} {'Reduction':<12} {'Tracks Fixed':<15}")
    print("-" * 67)

    for strategy in strategies:
        r = results[strategy]
        tracks_fixed = r['tracks_with_raw_switches'] - r['tracks_with_smooth_switches']
        print(f"{strategy:<20} {r['raw_switches']:<10} {r['smooth_switches']:<10} {r['reduction_pct']:.1f}%{'':<6} {tracks_fixed:<15}")

    # Show worst tracks and how smoothing helps
    print("\n" + "=" * 70)
    print("TOP 10 WORST TRACKS (using 'hysteresis' strategy)")
    print("=" * 70)
    print()
    print(f"{'Track ID':<12} {'Frames':<10} {'Raw Switches':<15} {'Smooth Switches':<18} {'Reduction':<10}")
    print("-" * 65)

    for track in results['hysteresis']['top_tracks']:
        reduction = 100 * (track['raw_switches'] - track['smooth_switches']) / max(1, track['raw_switches'])
        print(f"{track['track_id']:<12} {track['frames']:<10} {track['raw_switches']:<15} {track['smooth_switches']:<18} {reduction:.0f}%")

    print("\n" + "=" * 70)
    print("RECOMMENDATION: Use 'hysteresis' strategy for best results")
    print("=" * 70)


if __name__ == "__main__":
    main()
