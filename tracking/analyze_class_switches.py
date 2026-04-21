#!/usr/bin/env python3
"""
Analyze classification switches in tracking predictions
Detects when a track ID changes its detected class during tracking
and determines the class when crossing the counting line
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Class ID to name mapping (YOLO COCO classes)
CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
    4: "van",
    5: "tricycle",
    # Add more if needed
}

# Gate counting line configurations
GATE_COUNTING_LINES = {
    'Gate2_Oct7': {
        'x_positions': [0.625, 0.781],
        'y_positions': [1.0, 0.278]
    },
    'Gate2.9_Oct7': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.42, 0.65]
    },
    'Gate3_Oct7': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.47, 0.60]
    },
    'Gate3.5_Oct7': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.547, 0.578]
    },
    'Gate3.1_Oct7': {
        'x_positions': [0.885, 0.885],
        'y_positions': [0.0, 0.194]
    },
    'Gate3_Feb20': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.47, 0.60]
    },
    'Gate3_Apr3': {
        'x_positions': [0.0, 1.0],
        'y_positions': [0.47, 0.60]
    },
}

# Video properties (for line calculation)
VIDEO_PROPERTIES = {
    'Gate2_Oct7': {'width': 1920, 'height': 1080},
    'Gate2.9_Oct7': {'width': 1920, 'height': 1080},
    'Gate3_Oct7': {'width': 1920, 'height': 1080},
    'Gate3.5_Oct7': {'width': 1920, 'height': 1080},
    'Gate3.1_Oct7': {'width': 1920, 'height': 1080},
    'Gate3_Feb20': {'width': 1920, 'height': 1080},
    'Gate3_Apr3': {'width': 1920, 'height': 1080},
}


def parse_predictions_file(predictions_file):
    """Parse MOT format predictions file"""
    try:
        # MOT format: frame,id,x,y,w,h,conf,class,-1,-1
        df = pd.read_csv(predictions_file, header=None,
                        names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'v1', 'v2'])
        return df
    except Exception as e:
        print(f"Error reading {predictions_file}: {e}")
        return None


def analyze_class_switches(predictions_df, gate_name):
    """Analyze classification switches for each track ID"""

    # Group by track ID
    switches = []
    track_histories = defaultdict(list)

    for track_id in predictions_df['id'].unique():
        track_data = predictions_df[predictions_df['id'] == track_id].sort_values('frame')

        # Track class history
        class_history = []
        prev_class = None
        switch_points = []

        for idx, row in track_data.iterrows():
            frame = int(row['frame'])
            class_id = int(row['class'])
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

            track_histories[track_id].append({
                'frame': frame,
                'class_id': class_id,
                'class_name': class_name,
                'x': row['x'],
                'y': row['y'],
                'w': row['w'],
                'h': row['h'],
                'conf': row['conf']
            })

            # Detect class switch
            if prev_class is not None and class_id != prev_class:
                prev_class_name = CLASS_NAMES.get(prev_class, f"class_{prev_class}")
                switch_points.append({
                    'frame': frame,
                    'from_class': prev_class_name,
                    'to_class': class_name,
                    'from_class_id': prev_class,
                    'to_class_id': class_id
                })

            prev_class = class_id

        # If there were switches, record them
        if switch_points:
            first_class = CLASS_NAMES.get(int(track_data.iloc[0]['class']), f"class_{int(track_data.iloc[0]['class'])}")
            last_class = CLASS_NAMES.get(int(track_data.iloc[-1]['class']), f"class_{int(track_data.iloc[-1]['class'])}")

            switches.append({
                'track_id': track_id,
                'first_frame': int(track_data.iloc[0]['frame']),
                'last_frame': int(track_data.iloc[-1]['frame']),
                'first_class': first_class,
                'last_class': last_class,
                'num_switches': len(switch_points),
                'switch_points': switch_points,
                'track_length': len(track_data)
            })

    return switches, track_histories


def detect_line_crossings(track_histories, gate_name):
    """Detect when tracks cross the counting line and their class at that moment"""

    gate_config = GATE_COUNTING_LINES.get(gate_name, GATE_COUNTING_LINES.get('Gate3_Oct7'))
    video_props = VIDEO_PROPERTIES.get(gate_name, {'width': 1920, 'height': 1080})

    frame_width = video_props['width']
    frame_height = video_props['height']

    line_x_positions = [
        int(frame_width * gate_config['x_positions'][0]),
        int(frame_width * gate_config['x_positions'][1])
    ]
    line_y_positions = [
        int(frame_height * gate_config['y_positions'][0]),
        int(frame_height * gate_config['y_positions'][1])
    ]

    crossings = []

    for track_id, history in track_histories.items():
        prev_position = None

        for detection in history:
            # Calculate center point
            center_x = detection['x'] + detection['w'] / 2
            center_y = detection['y'] + detection['h'] / 2

            # Determine position relative to line
            line_y_at_x = np.interp(center_x, line_x_positions, line_y_positions)
            current_position = 'above' if center_y < line_y_at_x else 'below'

            # Detect crossing
            if prev_position is not None and prev_position != current_position:
                crossings.append({
                    'track_id': track_id,
                    'frame': detection['frame'],
                    'class': detection['class_name'],
                    'class_id': detection['class_id'],
                    'confidence': detection['conf'],
                    'direction': f"{prev_position}_to_{current_position}"
                })

            prev_position = current_position

    return crossings


def create_switch_report(switches, crossings, gate_name, output_dir):
    """Create detailed report of classification switches"""

    report_file = os.path.join(output_dir, f"{gate_name}_class_switches_report.txt")

    with open(report_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"CLASSIFICATION SWITCH ANALYSIS - {gate_name}\n")
        f.write("="*100 + "\n\n")

        f.write(f"Total tracks analyzed: {len(switches) + sum(1 for _ in crossings)}\n")
        f.write(f"Tracks with class switches: {len(switches)}\n")
        f.write(f"Total line crossings detected: {len(crossings)}\n\n")

        if switches:
            f.write("="*100 + "\n")
            f.write("TRACKS WITH CLASSIFICATION SWITCHES\n")
            f.write("="*100 + "\n\n")

            for switch_info in sorted(switches, key=lambda x: x['num_switches'], reverse=True):
                f.write(f"\nTrack ID: {switch_info['track_id']}\n")
                f.write(f"  Duration: Frame {switch_info['first_frame']} → {switch_info['last_frame']} ({switch_info['track_length']} frames)\n")
                f.write(f"  Initial Class: {switch_info['first_class']}\n")
                f.write(f"  Final Class: {switch_info['last_class']}\n")
                f.write(f"  Number of switches: {switch_info['num_switches']}\n")
                f.write(f"  Switch points:\n")

                for sp in switch_info['switch_points']:
                    f.write(f"    Frame {sp['frame']}: {sp['from_class']} → {sp['to_class']}\n")
        else:
            f.write("\n✓ No classification switches detected! All tracks maintained consistent classes.\n")

        # Crossings analysis
        f.write("\n" + "="*100 + "\n")
        f.write("LINE CROSSING CLASSIFICATIONS\n")
        f.write("="*100 + "\n\n")

        if crossings:
            # Group by class
            class_counts = defaultdict(int)
            for crossing in crossings:
                class_counts[crossing['class']] += 1

            f.write(f"Total crossings: {len(crossings)}\n\n")
            f.write("Classification at line crossing:\n")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {class_name}: {count} ({count/len(crossings)*100:.1f}%)\n")

            f.write("\n" + "-"*100 + "\n")
            f.write("Detailed crossing information:\n")
            f.write(f"{'Track ID':<12} {'Frame':<8} {'Class':<15} {'Confidence':<12} {'Direction':<20}\n")
            f.write("-"*100 + "\n")

            for crossing in sorted(crossings, key=lambda x: x['frame']):
                f.write(f"{crossing['track_id']:<12} {crossing['frame']:<8} {crossing['class']:<15} "
                       f"{crossing['confidence']:<12.4f} {crossing['direction']:<20}\n")

        # Find tracks that switched AND crossed
        f.write("\n" + "="*100 + "\n")
        f.write("TRACKS WITH SWITCHES THAT CROSSED THE LINE\n")
        f.write("="*100 + "\n\n")

        switched_track_ids = {s['track_id'] for s in switches}
        crossing_track_ids = {c['track_id'] for c in crossings}
        both = switched_track_ids & crossing_track_ids

        if both:
            f.write(f"Found {len(both)} tracks that had class switches AND crossed the line:\n\n")

            for track_id in sorted(both):
                switch_info = next(s for s in switches if s['track_id'] == track_id)
                crossing_info = [c for c in crossings if c['track_id'] == track_id]

                f.write(f"\nTrack ID {track_id}:\n")
                f.write(f"  Class switches: {switch_info['num_switches']}\n")
                f.write(f"  Initial class: {switch_info['first_class']}\n")
                f.write(f"  Final class: {switch_info['last_class']}\n")
                f.write(f"  Line crossings: {len(crossing_info)}\n")

                for crossing in crossing_info:
                    f.write(f"    Crossed at frame {crossing['frame']} as {crossing['class']}\n")

                f.write(f"  Switch timeline:\n")
                for sp in switch_info['switch_points']:
                    f.write(f"    Frame {sp['frame']}: {sp['from_class']} → {sp['to_class']}\n")
        else:
            f.write("✓ No tracks with switches crossed the line.\n")

        f.write("\n" + "="*100 + "\n")

    print(f"Saved: {report_file}")
    return report_file


def create_visualization(switches, gate_name, output_dir):
    """Create visualization of classification switches"""

    if not switches:
        print(f"No switches to visualize for {gate_name}")
        return

    # Create timeline visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Top plot: Number of switches per track
    track_ids = [s['track_id'] for s in sorted(switches, key=lambda x: x['num_switches'], reverse=True)[:20]]
    num_switches = [s['num_switches'] for s in sorted(switches, key=lambda x: x['num_switches'], reverse=True)[:20]]

    ax1.barh(range(len(track_ids)), num_switches, color='coral', edgecolor='black')
    ax1.set_yticks(range(len(track_ids)))
    ax1.set_yticklabels([f"Track {tid}" for tid in track_ids])
    ax1.set_xlabel('Number of Class Switches', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top 20 Tracks with Most Class Switches - {gate_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Bottom plot: Switch distribution over time
    all_switch_frames = []
    for s in switches:
        for sp in s['switch_points']:
            all_switch_frames.append(sp['frame'])

    if all_switch_frames:
        ax2.hist(all_switch_frames, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Class Switches', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution of Class Switches Over Video Timeline', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    viz_file = os.path.join(output_dir, f"{gate_name}_class_switches_viz.png")
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {viz_file}")


def main():
    """Main function to analyze classification switches"""

    print("="*100)
    print("CLASSIFICATION SWITCH ANALYSIS FOR TRACKING PREDICTIONS")
    print("="*100)

    # Analyze Vanilla model predictions
    predictions_base = "/media/mydrive/GitHub/ultralytics/tracking/yolov8m_vanilla_gate3_tracking_results"
    output_dir = "/media/mydrive/GitHub/ultralytics/tracking/class_switch_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Also analyze from tracking_metrics_results_vanilla
    predictions_base2 = "/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_vanilla/Vanilla-YOLOv8m"

    all_results = []

    # Process all prediction files
    for base_dir in [predictions_base, predictions_base2]:
        if not os.path.exists(base_dir):
            continue

        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('_predictions.txt'):
                    predictions_file = os.path.join(root, file)
                    gate_name = file.replace('_predictions.txt', '').replace('GATE 3 ENTRANCE #1 - 1920 x 1080 - 15fps_', '')

                    # Try to map to standard gate names
                    if '20250220' in gate_name:
                        gate_name = 'Gate3_Feb20'
                    elif '20250403' in gate_name:
                        gate_name = 'Gate3_Apr3'
                    elif '20251007' in gate_name:
                        gate_name = 'Gate3_Oct7'

                    print(f"\nAnalyzing: {gate_name}")
                    print(f"File: {predictions_file}")

                    # Parse predictions
                    predictions_df = parse_predictions_file(predictions_file)
                    if predictions_df is None:
                        continue

                    print(f"  Total detections: {len(predictions_df)}")
                    print(f"  Unique track IDs: {predictions_df['id'].nunique()}")

                    # Analyze switches
                    switches, track_histories = analyze_class_switches(predictions_df, gate_name)
                    print(f"  Tracks with class switches: {len(switches)}")

                    # Detect line crossings
                    crossings = detect_line_crossings(track_histories, gate_name)
                    print(f"  Line crossings detected: {len(crossings)}")

                    # Create report
                    report_file = create_switch_report(switches, crossings, gate_name, output_dir)

                    # Create visualization
                    if switches:
                        create_visualization(switches, gate_name, output_dir)

                    all_results.append({
                        'gate': gate_name,
                        'total_tracks': predictions_df['id'].nunique(),
                        'tracks_with_switches': len(switches),
                        'total_switches': sum(s['num_switches'] for s in switches),
                        'crossings': len(crossings)
                    })

    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"\n{'Gate':<20} {'Total Tracks':<15} {'With Switches':<15} {'Total Switches':<15} {'Crossings':<15}")
    print("-"*100)
    for r in all_results:
        print(f"{r['gate']:<20} {r['total_tracks']:<15} {r['tracks_with_switches']:<15} "
              f"{r['total_switches']:<15} {r['crossings']:<15}")

    print(f"\n✓ All reports saved to: {output_dir}")
    print("="*100)


if __name__ == "__main__":
    main()
