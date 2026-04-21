#!/usr/bin/env python3
"""
Final comprehensive analysis with scatter plots
- 5 gates for Vanilla model (best Gate3 performance only)
- Clean gate names (no dates)
- Scatter plot visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Gate name mapping (remove dates)
GATE_NAME_MAPPING = {
    'Gate2_Oct7': 'Gate2',
    'Gate2.9_Oct7': 'Gate2.9',
    'Gate3_Oct7': 'Gate3',
    'Gate3_Apr3': 'Gate3',
    'Gate3_Feb20': 'Gate3',
    'Gate3.1_Oct7': 'Gate3.1',
    'Gate3.5_Oct7': 'Gate3.5',
}

# Detection metrics (from final epochs)
detection_data = {
    'DCNv2-FPN': {'mAP50-95': 0.7342, 'Precision': 0.9035, 'Recall': 0.8096, 'mAP50': 0.8793},
    'DCNv2-Full': {'mAP50-95': 0.7529, 'Precision': 0.9122, 'Recall': 0.8505, 'mAP50': 0.8941},
    'DCNv2-Liu': {'mAP50-95': 0.7362, 'Precision': 0.9229, 'Recall': 0.7994, 'mAP50': 0.8730},
    'DCNv2-Pan': {'mAP50-95': 0.7329, 'Precision': 0.8880, 'Recall': 0.8103, 'mAP50': 0.8743},
    'DCNv3-Base': {'mAP50-95': 0.7468, 'Precision': 0.8929, 'Recall': 0.8331, 'mAP50': 0.8902},
    'DCNv3-FPN': {'mAP50-95': 0.7451, 'Precision': 0.9213, 'Recall': 0.7967, 'mAP50': 0.8976},
    'DCNv3-Full': {'mAP50-95': 0.7301, 'Precision': 0.9256, 'Recall': 0.8079, 'mAP50': 0.8733},
    'DCNv3-Liu': {'mAP50-95': 0.7612, 'Precision': 0.9360, 'Recall': 0.8024, 'mAP50': 0.9011},
    'DCNv3-Pan': {'mAP50-95': 0.7409, 'Precision': 0.8828, 'Recall': 0.8425, 'mAP50': 0.8762},
    'Vanilla-YOLOv8m': {'mAP50-95': 0.7530, 'Precision': 0.9031, 'Recall': 0.8156, 'mAP50': 0.8905},
}

# Model directories
detection_dirs = {
    'DCNv2-FPN': '/home/migui/YOLO_outputs/100_dcnv2-yolov8-neck-fpn',
    'DCNv2-Full': '/home/migui/YOLO_outputs/100_dcnv2-yolov8-neck-full_final',
    'DCNv2-Pan': '/home/migui/YOLO_outputs/100_dcnv2-yolov8-neck-pan',
    'DCNv2-Liu': '/home/migui/YOLO_outputs/100_dcnv2_yolov8m_liu_final',
    'DCNv3-Base': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m',
    'DCNv3-FPN': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_fpn_second',
    'DCNv3-Full': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_full_second',
    'DCNv3-Liu': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_liu_second',
    'DCNv3-Pan': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_pan_second',
    'Vanilla-YOLOv8m': '/home/migui/Downloads/yolov8m-vanilla-20260211T133104Z-1-001/yolov8m-vanilla',
}

tracking_base_dirs = {
    'DCNv2': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv2',
    'DCNv3': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv3',
    'Vanilla': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_vanilla',
    'Vanilla_Gate3': '/media/mydrive/GitHub/ultralytics/tracking/yolov8m_vanilla_gate3_tracking_results',
}

# Vanilla gates to KEEP (5 total - best Gate3 only)
VANILLA_GATES_TO_KEEP = ['Gate2.9_Oct7', 'Gate2_Oct7', 'Gate3.5_Oct7', 'Gate3.1_Oct7', 'Gate3_Feb20']


def parse_tracking_metrics_file(metrics_file):
    """Parse individual tracking metrics file"""
    try:
        with open(metrics_file, 'r') as f:
            content = f.read()

        # Extract model name
        model_match = re.search(r'Model:\s+(\S+)', content)
        model = model_match.group(1) if model_match else ''

        # Extract gate name
        gate_match = re.search(r'Video:\s+(\S+)', content)
        gate = gate_match.group(1) if gate_match else ''

        # Extract metrics
        idf1_match = re.search(r'(?:^\d+\.\s+)?IDF1\s*(?:\(.*?\))?\s*:\s+([\d.]+)', content, re.MULTILINE)
        mota_match = re.search(r'(?:^\d+\.\s+)?MOTA\s*(?:\(.*?\))?\s*:\s+([\d.]+)', content, re.MULTILINE)
        idsw_match = re.search(r'(?:^\d+\.\s+)?IDSW\s*(?:\(.*?\))?\s*:\s+(\d+)', content, re.MULTILINE)
        fps_match = re.search(r'(?:^\d+\.\s+FPS\s*\(.*?\)|^FPS)\s*:\s+([\d.]+)', content, re.MULTILINE)
        total_match = re.search(r'TOTAL:\s+(\d+)', content)
        detections_match = re.search(r'Total Detections:\s+(\d+)', content)
        tracks_match = re.search(r'Total Tracks[:\s]+(\d+)', content)

        return {
            'Model': model,
            'Gate': gate,
            'IDF1': float(idf1_match.group(1)) if idf1_match else 0.0,
            'MOTA': float(mota_match.group(1)) if mota_match else 0.0,
            'IDSW': int(idsw_match.group(1)) if idsw_match else 0,
            'FPS': float(fps_match.group(1)) if fps_match else 0.0,
            'Count': int(total_match.group(1)) if total_match else 0,
            'Detections': int(detections_match.group(1)) if detections_match else 0,
            'Tracks': int(tracks_match.group(1)) if tracks_match else 0,
        }
    except Exception as e:
        print(f"Error parsing {metrics_file}: {e}")
        return None


def collect_all_tracking_metrics(base_dirs):
    """Collect all tracking metrics from individual files"""
    all_metrics = []

    for version, base_dir in base_dirs.items():
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('_metrics.txt'):
                    metrics_file = os.path.join(root, file)
                    metrics = parse_tracking_metrics_file(metrics_file)
                    if metrics:
                        # Filter vanilla to only keep specified gates
                        if metrics['Model'] == 'Vanilla-YOLOv8m':
                            if metrics['Gate'] in VANILLA_GATES_TO_KEEP:
                                # Map gate name to clean version
                                metrics['Gate_Clean'] = GATE_NAME_MAPPING.get(metrics['Gate'], metrics['Gate'])
                                all_metrics.append(metrics)
                                print(f"  Parsed (Vanilla): {metrics['Model']} - {metrics['Gate']} → {metrics['Gate_Clean']}")
                        else:
                            # For DCN models, keep all gates and map names
                            metrics['Gate_Clean'] = GATE_NAME_MAPPING.get(metrics['Gate'], metrics['Gate'])
                            all_metrics.append(metrics)
                            print(f"  Parsed: {metrics['Model']} - {metrics['Gate']} → {metrics['Gate_Clean']}")

    return pd.DataFrame(all_metrics)


def create_scatter_plots(tracking_metrics, output_dir):
    """Create scatter plot visualizations"""

    # Calculate average metrics per model
    avg_metrics = tracking_metrics.groupby('Model').agg({
        'IDF1': 'mean',
        'MOTA': 'mean',
        'FPS': 'mean',
        'Detections': 'sum',
        'Tracks': 'sum'
    }).reset_index()

    # Add detection metrics
    avg_metrics['mAP50-95'] = avg_metrics['Model'].map(lambda x: detection_data.get(x, {}).get('mAP50-95', 0))
    avg_metrics['Precision'] = avg_metrics['Model'].map(lambda x: detection_data.get(x, {}).get('Precision', 0))
    avg_metrics['Recall'] = avg_metrics['Model'].map(lambda x: detection_data.get(x, {}).get('Recall', 0))

    # Assign colors by model family
    def get_model_color(model):
        if 'DCNv2' in model:
            return '#3498db'  # Blue
        elif 'DCNv3' in model:
            return '#e74c3c'  # Red
        else:
            return '#2ecc71'  # Green (Vanilla)

    avg_metrics['Color'] = avg_metrics['Model'].apply(get_model_color)

    # 1. IDF1 vs FPS (Tracking Quality vs Efficiency)
    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, row in avg_metrics.iterrows():
        ax.scatter(row['FPS'], row['IDF1'], s=200, c=row['Color'], alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(row['Model'], (row['FPS'], row['IDF1']), fontsize=9, ha='center', va='bottom')

    ax.set_xlabel('Average FPS (Processing Speed)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average IDF1 (Identity Tracking Quality)', fontsize=14, fontweight='bold')
    ax.set_title('Tracking Quality vs Efficiency Trade-off', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='DCNv2'),
        Patch(facecolor='#e74c3c', label='DCNv3'),
        Patch(facecolor='#2ecc71', label='Vanilla-YOLOv8m')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_idf1_vs_fps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/scatter_idf1_vs_fps.png")

    # 2. mAP50-95 vs IDF1 (Detection Quality vs Tracking Quality)
    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, row in avg_metrics.iterrows():
        ax.scatter(row['mAP50-95'], row['IDF1'], s=200, c=row['Color'], alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(row['Model'], (row['mAP50-95'], row['IDF1']), fontsize=9, ha='center', va='bottom')

    ax.set_xlabel('mAP@0.5:0.95 (Detection Quality)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average IDF1 (Tracking Quality)', fontsize=14, fontweight='bold')
    ax.set_title('Detection Quality vs Tracking Quality', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, loc='best', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_map_vs_idf1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/scatter_map_vs_idf1.png")

    # 3. mAP50-95 vs MOTA (Detection vs Overall Tracking)
    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, row in avg_metrics.iterrows():
        ax.scatter(row['mAP50-95'], row['MOTA'], s=200, c=row['Color'], alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(row['Model'], (row['mAP50-95'], row['MOTA']), fontsize=9, ha='center', va='bottom')

    ax.set_xlabel('mAP@0.5:0.95 (Detection Quality)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average MOTA (Overall Tracking Accuracy)', fontsize=14, fontweight='bold')
    ax.set_title('Detection Quality vs Tracking Accuracy', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, loc='best', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_map_vs_mota.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/scatter_map_vs_mota.png")

    # 4. IDF1 vs MOTA (Identity Preservation vs Overall Accuracy)
    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, row in avg_metrics.iterrows():
        ax.scatter(row['MOTA'], row['IDF1'], s=200, c=row['Color'], alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(row['Model'], (row['MOTA'], row['IDF1']), fontsize=9, ha='center', va='bottom')

    ax.set_xlabel('Average MOTA (Overall Tracking Accuracy)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average IDF1 (Identity Preservation)', fontsize=14, fontweight='bold')
    ax.set_title('Tracking Performance: Identity vs Accuracy', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, loc='best', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_idf1_vs_mota.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/scatter_idf1_vs_mota.png")

    # 5. Bubble Chart: IDF1 vs mAP50-95 (bubble size = FPS)
    fig, ax = plt.subplots(figsize=(14, 10))

    # Normalize FPS for bubble sizes
    fps_normalized = (avg_metrics['FPS'] - avg_metrics['FPS'].min()) / (avg_metrics['FPS'].max() - avg_metrics['FPS'].min())
    bubble_sizes = 100 + fps_normalized * 500  # Scale between 100 and 600

    for idx, row in avg_metrics.iterrows():
        ax.scatter(row['mAP50-95'], row['IDF1'], s=bubble_sizes.iloc[idx],
                  c=row['Color'], alpha=0.6, edgecolors='black', linewidth=2)
        ax.annotate(row['Model'], (row['mAP50-95'], row['IDF1']),
                   fontsize=10, ha='center', va='center', fontweight='bold')

    ax.set_xlabel('mAP@0.5:0.95 (Detection Quality)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average IDF1 (Tracking Quality)', fontsize=14, fontweight='bold')
    ax.set_title('Multi-Dimensional Performance (Bubble Size = FPS)', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, loc='best', fontsize=11)

    # Add FPS legend
    fps_legend_sizes = [100, 300, 600]
    fps_legend_values = [avg_metrics['FPS'].min(), avg_metrics['FPS'].mean(), avg_metrics['FPS'].max()]
    for size, fps_val in zip(fps_legend_sizes, fps_legend_values):
        ax.scatter([], [], s=size, c='gray', alpha=0.5, edgecolors='black',
                  label=f'FPS: {fps_val:.1f}')
    ax.legend(loc='lower right', fontsize=10, title='Processing Speed')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_bubble_multidimensional.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/scatter_bubble_multidimensional.png")


def main():
    """Main function"""
    print("="*80)
    print("FINAL COMPREHENSIVE ANALYSIS WITH SCATTER PLOTS")
    print("5 Gates for Vanilla Model (Best Gate3 Only)")
    print("Clean Gate Names (No Dates)")
    print("="*80)

    output_dir = '/media/mydrive/GitHub/ultralytics/tracking/metrics_summary_complete'
    os.makedirs(output_dir, exist_ok=True)

    # Collect tracking metrics
    print("\nCollecting tracking metrics...")
    tracking_metrics = collect_all_tracking_metrics(tracking_base_dirs)

    # Save to CSV with clean names
    csv_file = os.path.join(output_dir, 'tracking_metrics_final.csv')
    tracking_metrics.to_csv(csv_file, index=False)
    print(f"\nSaved: {csv_file}")

    # Create scatter plots
    print("\nCreating scatter plot visualizations...")
    create_scatter_plots(tracking_metrics, output_dir)

    # Summary statistics
    print("\n" + "="*80)
    print("VANILLA MODEL SUMMARY (5 Gates)")
    print("="*80)
    vanilla_metrics = tracking_metrics[tracking_metrics['Model'] == 'Vanilla-YOLOv8m']
    print(f"\nGates included: {sorted(vanilla_metrics['Gate_Clean'].unique())}")
    print(f"Average IDF1: {vanilla_metrics['IDF1'].mean():.4f}")
    print(f"Average MOTA: {vanilla_metrics['MOTA'].mean():.4f}")
    print(f"Average FPS: {vanilla_metrics['FPS'].mean():.2f}")
    print(f"Total IDSW: {vanilla_metrics['IDSW'].sum()}")

    print("\n" + "="*80)
    print("Analysis complete! All files saved to:")
    print(f"  {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
