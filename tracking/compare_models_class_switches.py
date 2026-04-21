#!/usr/bin/env python3
"""
Compare classification switches across all models (DCNv2, DCNv3, Vanilla) per gate
Analyzes when switches occur (temporal distribution within track lifetime)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Class ID to name mapping
CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
    4: "van",
    5: "tricycle",
}

# Model directories
MODEL_TRACKING_DIRS = {
    'DCNv2-FPN': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv2/DCNv2-FPN',
    'DCNv2-Full': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv2/DCNv2-Full',
    'DCNv2-Liu': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv2/DCNv2-Liu',
    'DCNv2-Pan': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv2/DCNv2-Pan',
    'DCNv3-FPN': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv3/DCNv3-FPN',
    'DCNv3-Full': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv3/DCNv3-Full',
    'DCNv3-Liu': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv3/DCNv3-Liu',
    'DCNv3-Pan': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv3/DCNv3-Pan',
    'Vanilla-YOLOv8m': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_vanilla/Vanilla-YOLOv8m',
}

# Gates to analyze (5 gates - best Gate3 only)
GATES_TO_ANALYZE = ['Gate2_Oct7', 'Gate2.9_Oct7', 'Gate3_Feb20', 'Gate3.1_Oct7', 'Gate3.5_Oct7']


def parse_predictions_file(predictions_file):
    """Parse MOT format predictions file"""
    try:
        df = pd.read_csv(predictions_file, header=None,
                        names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'v1', 'v2'])
        return df
    except Exception as e:
        print(f"  Error reading {predictions_file}: {e}")
        return None


def analyze_switches_with_temporal(predictions_df):
    """Analyze switches and determine when they occur in track lifetime"""

    switches_summary = {
        'total_tracks': predictions_df['id'].nunique(),
        'tracks_with_switches': 0,
        'total_switches': 0,
        'switches_early': 0,   # First 1/3 of track
        'switches_middle': 0,  # Middle 1/3 of track
        'switches_late': 0,    # Last 1/3 of track
        'switch_details': []
    }

    for track_id in predictions_df['id'].unique():
        track_data = predictions_df[predictions_df['id'] == track_id].sort_values('frame')

        if len(track_data) < 2:
            continue

        track_length = len(track_data)
        prev_class = None
        num_switches = 0

        for idx, (row_idx, row) in enumerate(track_data.iterrows()):
            class_id = int(row['class'])

            if prev_class is not None and class_id != prev_class:
                num_switches += 1

                # Determine temporal position
                position_ratio = idx / track_length
                if position_ratio < 0.33:
                    switches_summary['switches_early'] += 1
                    temporal_position = 'early'
                elif position_ratio < 0.67:
                    switches_summary['switches_middle'] += 1
                    temporal_position = 'middle'
                else:
                    switches_summary['switches_late'] += 1
                    temporal_position = 'late'

                from_class = CLASS_NAMES.get(prev_class, f"class_{prev_class}")
                to_class = CLASS_NAMES.get(class_id, f"class_{class_id}")

                switches_summary['switch_details'].append({
                    'track_id': track_id,
                    'frame': int(row['frame']),
                    'from_class': from_class,
                    'to_class': to_class,
                    'temporal_position': temporal_position,
                    'position_ratio': position_ratio,
                    'track_length': track_length
                })

            prev_class = class_id

        if num_switches > 0:
            switches_summary['tracks_with_switches'] += 1
            switches_summary['total_switches'] += num_switches

    return switches_summary


def collect_all_model_switches():
    """Collect switch statistics for all models and gates"""

    all_results = []

    for model_name, model_dir in MODEL_TRACKING_DIRS.items():
        if not os.path.exists(model_dir):
            print(f"⚠ Directory not found: {model_dir}")
            continue

        print(f"\nAnalyzing {model_name}:")

        for gate in GATES_TO_ANALYZE:
            predictions_file = os.path.join(model_dir, gate, f"{gate}_predictions.txt")

            if not os.path.exists(predictions_file):
                print(f"  ✗ {gate}: predictions file not found")
                continue

            print(f"  Processing {gate}...", end=' ')

            # Parse and analyze
            predictions_df = parse_predictions_file(predictions_file)
            if predictions_df is None:
                continue

            switches = analyze_switches_with_temporal(predictions_df)

            # Calculate percentages
            switch_rate = (switches['tracks_with_switches'] / switches['total_tracks'] * 100) if switches['total_tracks'] > 0 else 0

            result = {
                'Model': model_name,
                'Gate': gate,
                'Total_Tracks': switches['total_tracks'],
                'Tracks_With_Switches': switches['tracks_with_switches'],
                'Total_Switches': switches['total_switches'],
                'Switch_Rate_%': switch_rate,
                'Switches_Early': switches['switches_early'],
                'Switches_Middle': switches['switches_middle'],
                'Switches_Late': switches['switches_late'],
                'Avg_Switches_Per_Track': switches['total_switches'] / switches['total_tracks'] if switches['total_tracks'] > 0 else 0,
                'switch_details': switches['switch_details']
            }

            all_results.append(result)
            print(f"✓ {switches['tracks_with_switches']}/{switches['total_tracks']} tracks with {switches['total_switches']} switches ({switch_rate:.1f}%)")

    return pd.DataFrame(all_results)


def create_comparison_visualizations(results_df, output_dir):
    """Create comparative visualizations"""

    # 1. Switch Rate by Model and Gate (Heatmap)
    plt.figure(figsize=(14, 8))
    pivot_data = results_df.pivot(index='Model', columns='Gate', values='Switch_Rate_%')

    # Sort by average switch rate
    pivot_data['Average'] = pivot_data.mean(axis=1)
    pivot_data = pivot_data.sort_values('Average', ascending=False)
    pivot_data = pivot_data.drop('Average', axis=1)

    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Classification Switch Rate (%)'},
                linewidths=0.5, linecolor='gray')
    plt.title('Classification Switch Rate by Model and Gate (%)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Gate', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_switch_rate_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/comparison_switch_rate_heatmap.png")

    # 2. Temporal Distribution of Switches (Stacked Bar)
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate temporal percentages per model
    temporal_data = results_df.groupby('Model')[['Switches_Early', 'Switches_Middle', 'Switches_Late']].sum()

    # Sort by total switches
    temporal_data['Total'] = temporal_data.sum(axis=1)
    temporal_data = temporal_data.sort_values('Total', ascending=False).drop('Total', axis=1)

    # Create stacked bar chart
    temporal_data.plot(kind='barh', stacked=True, ax=ax,
                       color=['#3498db', '#f39c12', '#e74c3c'],
                       edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Number of Classification Switches', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('When Do Classification Switches Occur?\n(Early/Middle/Late in Track Lifetime)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(['Early (0-33%)', 'Middle (33-67%)', 'Late (67-100%)'],
              loc='best', fontsize=11, title='Track Lifetime Position')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_temporal_switches.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/comparison_temporal_switches.png")

    # 3. Per-Gate Comparison (Grouped Bar Chart)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, gate in enumerate(GATES_TO_ANALYZE):
        gate_data = results_df[results_df['Gate'] == gate].sort_values('Switch_Rate_%', ascending=False)

        if len(gate_data) == 0:
            continue

        colors = ['#2ecc71' if 'Vanilla' in m else '#e74c3c' if 'DCNv3' in m else '#3498db'
                  for m in gate_data['Model']]

        axes[idx].barh(range(len(gate_data)), gate_data['Switch_Rate_%'], color=colors,
                       edgecolor='black', linewidth=1)
        axes[idx].set_yticks(range(len(gate_data)))
        axes[idx].set_yticklabels(gate_data['Model'], fontsize=9)
        axes[idx].set_xlabel('Switch Rate (%)', fontsize=10, fontweight='bold')
        axes[idx].set_title(f'{gate}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (model, rate) in enumerate(zip(gate_data['Model'], gate_data['Switch_Rate_%'])):
            axes[idx].text(rate + 0.5, i, f'{rate:.1f}%', va='center', fontsize=8)

    # Hide the 6th subplot (we only have 5 gates)
    axes[5].axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='DCNv2'),
        Patch(facecolor='#e74c3c', label='DCNv3'),
        Patch(facecolor='#2ecc71', label='Vanilla-YOLOv8m')
    ]
    axes[5].legend(handles=legend_elements, loc='center', fontsize=14, title='Model Family',
                   title_fontsize=16, frameon=True, fancybox=True, shadow=True)

    fig.suptitle('Classification Switch Rate Comparison by Gate', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_per_gate_switches.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/comparison_per_gate_switches.png")

    # 4. Average Switches per Track (Box Plot)
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for box plot
    model_families = {
        'DCNv2': ['DCNv2-FPN', 'DCNv2-Full', 'DCNv2-Liu', 'DCNv2-Pan'],
        'DCNv3': ['DCNv3-FPN', 'DCNv3-Full', 'DCNv3-Liu', 'DCNv3-Pan'],
        'Vanilla': ['Vanilla-YOLOv8m']
    }

    box_data = []
    box_labels = []
    colors_box = []

    for family, models in model_families.items():
        family_data = results_df[results_df['Model'].isin(models)]['Avg_Switches_Per_Track'].tolist()
        if family_data:
            box_data.append(family_data)
            box_labels.append(family)
            colors_box.append('#3498db' if family == 'DCNv2' else '#e74c3c' if family == 'DCNv3' else '#2ecc71')

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='red'),
                    meanprops=dict(linewidth=2, color='blue', linestyle='--'))

    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Average Switches per Track', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Family', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Average Switches per Track by Model Family',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_switches_per_track_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/comparison_switches_per_track_boxplot.png")


def create_summary_report(results_df, output_dir):
    """Create comprehensive text summary"""

    report_file = os.path.join(output_dir, 'model_comparison_switches_summary.txt')

    with open(report_file, 'w') as f:
        f.write("="*120 + "\n")
        f.write("CLASSIFICATION SWITCH COMPARISON - ALL MODELS\n")
        f.write("="*120 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*120 + "\n")
        total_tracks = results_df['Total_Tracks'].sum()
        total_with_switches = results_df['Tracks_With_Switches'].sum()
        total_switches = results_df['Total_Switches'].sum()

        f.write(f"Total tracks analyzed: {total_tracks:,}\n")
        f.write(f"Tracks with switches: {total_with_switches:,} ({total_with_switches/total_tracks*100:.1f}%)\n")
        f.write(f"Total classification switches: {total_switches:,}\n")
        f.write(f"Average switches per track: {total_switches/total_tracks:.2f}\n\n")

        # Per model summary
        f.write("="*120 + "\n")
        f.write("PER MODEL SUMMARY (Average Across All Gates)\n")
        f.write("="*120 + "\n\n")

        model_summary = results_df.groupby('Model').agg({
            'Total_Tracks': 'sum',
            'Tracks_With_Switches': 'sum',
            'Total_Switches': 'sum',
            'Switch_Rate_%': 'mean',
            'Switches_Early': 'sum',
            'Switches_Middle': 'sum',
            'Switches_Late': 'sum'
        }).sort_values('Switch_Rate_%')

        f.write(f"{'Model':<20} {'Tot Tracks':<12} {'With Switch':<12} {'Tot Switch':<12} {'Avg Rate %':<12} {'Early':<8} {'Mid':<8} {'Late':<8}\n")
        f.write("-"*120 + "\n")

        for model, row in model_summary.iterrows():
            f.write(f"{model:<20} {int(row['Total_Tracks']):<12} {int(row['Tracks_With_Switches']):<12} "
                   f"{int(row['Total_Switches']):<12} {row['Switch_Rate_%']:<12.1f} "
                   f"{int(row['Switches_Early']):<8} {int(row['Switches_Middle']):<8} {int(row['Switches_Late']):<8}\n")

        # Per gate summary
        f.write("\n" + "="*120 + "\n")
        f.write("PER GATE SUMMARY\n")
        f.write("="*120 + "\n\n")

        for gate in GATES_TO_ANALYZE:
            f.write(f"\n{gate}\n")
            f.write("-"*120 + "\n")

            gate_data = results_df[results_df['Gate'] == gate].sort_values('Switch_Rate_%')

            f.write(f"{'Model':<20} {'Tracks':<10} {'Switches':<10} {'Rate %':<10} {'Early':<8} {'Mid':<8} {'Late':<8}\n")
            f.write("-"*120 + "\n")

            for _, row in gate_data.iterrows():
                f.write(f"{row['Model']:<20} {int(row['Total_Tracks']):<10} {int(row['Total_Switches']):<10} "
                       f"{row['Switch_Rate_%']:<10.1f} {int(row['Switches_Early']):<8} "
                       f"{int(row['Switches_Middle']):<8} {int(row['Switches_Late']):<8}\n")

        # Key insights
        f.write("\n" + "="*120 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*120 + "\n\n")

        best_model = model_summary.index[0]
        worst_model = model_summary.index[-1]

        f.write(f"✓ Model with LOWEST switch rate: {best_model} ({model_summary.loc[best_model, 'Switch_Rate_%']:.1f}%)\n")
        f.write(f"✗ Model with HIGHEST switch rate: {worst_model} ({model_summary.loc[worst_model, 'Switch_Rate_%']:.1f}%)\n\n")

        # Temporal analysis
        total_early = results_df['Switches_Early'].sum()
        total_middle = results_df['Switches_Middle'].sum()
        total_late = results_df['Switches_Late'].sum()
        total_temporal = total_early + total_middle + total_late

        if total_temporal > 0:
            f.write("Temporal Distribution of Switches:\n")
            f.write(f"  Early (first 1/3 of track): {total_early} ({total_early/total_temporal*100:.1f}%)\n")
            f.write(f"  Middle (middle 1/3): {total_middle} ({total_middle/total_temporal*100:.1f}%)\n")
            f.write(f"  Late (last 1/3 of track): {total_late} ({total_late/total_temporal*100:.1f}%)\n\n")

        f.write("="*120 + "\n")

    print(f"Saved: {report_file}")


def main():
    """Main function"""

    print("="*120)
    print("CLASSIFICATION SWITCH COMPARISON - ALL MODELS AND GATES")
    print("="*120)

    output_dir = '/media/mydrive/GitHub/ultralytics/tracking/model_comparison_switches'
    os.makedirs(output_dir, exist_ok=True)

    # Collect data
    print("\nCollecting switch statistics for all models...")
    results_df = collect_all_model_switches()

    # Save to CSV
    csv_file = os.path.join(output_dir, 'model_switch_comparison.csv')
    results_df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved data: {csv_file}")

    # Create visualizations
    print("\nCreating comparison visualizations...")
    create_comparison_visualizations(results_df, output_dir)

    # Create summary report
    print("\nCreating summary report...")
    create_summary_report(results_df, output_dir)

    print("\n" + "="*120)
    print("COMPLETE - All analysis files saved to:")
    print(f"  {output_dir}")
    print("="*120)


if __name__ == "__main__":
    main()
