#!/usr/bin/env python3
"""
Script to analyze and visualize detection and tracking metrics for DCNv2 and DCNv3 models
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Define model directories
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
}

tracking_dirs = {
    'DCNv2': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv2',
    'DCNv3': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv3',
}

def extract_final_detection_metrics(results_csv_path):
    """Extract final epoch metrics from results.csv"""
    try:
        df = pd.read_csv(results_csv_path)
        # Get the last row (final epoch)
        final_metrics = df.iloc[-1]

        metrics = {
            'epoch': int(final_metrics['epoch']) if 'epoch' in final_metrics else len(df),
            'precision': final_metrics.get('metrics/precision(B)', 0),
            'recall': final_metrics.get('metrics/recall(B)', 0),
            'mAP50': final_metrics.get('metrics/mAP50(B)', 0),
            'mAP50-95': final_metrics.get('metrics/mAP50-95(B)', 0),
            'box_loss': final_metrics.get('val/box_loss', 0),
            'cls_loss': final_metrics.get('val/cls_loss', 0),
        }
        return metrics
    except Exception as e:
        print(f"Error reading {results_csv_path}: {e}")
        return None

def parse_tracking_summary(summary_file):
    """Parse tracking metrics summary file"""
    tracking_data = []

    try:
        with open(summary_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if 'DCNv' in line and 'Gate' in line:
                parts = line.split()
                if len(parts) >= 7:
                    model = parts[0]
                    gate = parts[1]
                    try:
                        idf1 = float(parts[2])
                        mota = float(parts[3])
                        idsw = int(parts[4])
                        fps = float(parts[5])
                        count = int(parts[6])

                        tracking_data.append({
                            'Model': model,
                            'Gate': gate,
                            'IDF1': idf1,
                            'MOTA': mota,
                            'IDSW': idsw,
                            'FPS': fps,
                            'Count': count
                        })
                    except (ValueError, IndexError):
                        continue

        return pd.DataFrame(tracking_data)
    except Exception as e:
        print(f"Error parsing {summary_file}: {e}")
        return pd.DataFrame()

def create_summary_text(detection_metrics, tracking_metrics):
    """Create comprehensive text summary"""
    summary = []
    summary.append("=" * 100)
    summary.append("COMPREHENSIVE DETECTION AND TRACKING METRICS SUMMARY")
    summary.append("=" * 100)
    summary.append("")

    # Detection Metrics Summary
    summary.append("=" * 100)
    summary.append("DETECTION METRICS (Final Epoch)")
    summary.append("=" * 100)
    summary.append("")
    summary.append(f"{'Model':<20} {'Epoch':<8} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
    summary.append("-" * 100)

    for model, metrics in detection_metrics.items():
        if metrics:
            summary.append(f"{model:<20} {metrics['epoch']:<8} {metrics['precision']:<12.4f} "
                         f"{metrics['recall']:<12.4f} {metrics['mAP50']:<12.4f} {metrics['mAP50-95']:<12.4f}")

    summary.append("")
    summary.append("=" * 100)
    summary.append("TRACKING METRICS BY GATE")
    summary.append("=" * 100)
    summary.append("")

    # Group tracking metrics by gate
    gates = sorted(tracking_metrics['Gate'].unique())

    for gate in gates:
        summary.append(f"\n{gate}")
        summary.append("-" * 100)
        summary.append(f"{'Model':<20} {'IDF1':<12} {'MOTA':<12} {'IDSW':<8} {'FPS':<12} {'Count':<8}")
        summary.append("-" * 100)

        gate_data = tracking_metrics[tracking_metrics['Gate'] == gate].sort_values('IDF1', ascending=False)
        for _, row in gate_data.iterrows():
            summary.append(f"{row['Model']:<20} {row['IDF1']:<12.4f} {row['MOTA']:<12.4f} "
                         f"{row['IDSW']:<8} {row['FPS']:<12.2f} {row['Count']:<8}")

    summary.append("")
    summary.append("=" * 100)
    summary.append("OVERALL TRACKING PERFORMANCE")
    summary.append("=" * 100)
    summary.append("")

    # Calculate average metrics per model
    avg_metrics = tracking_metrics.groupby('Model').agg({
        'IDF1': 'mean',
        'MOTA': 'mean',
        'IDSW': 'sum',
        'FPS': 'mean',
        'Count': 'sum'
    }).sort_values('IDF1', ascending=False)

    summary.append(f"{'Model':<20} {'Avg IDF1':<12} {'Avg MOTA':<12} {'Total IDSW':<12} {'Avg FPS':<12} {'Total Count':<12}")
    summary.append("-" * 100)

    for model, row in avg_metrics.iterrows():
        summary.append(f"{model:<20} {row['IDF1']:<12.4f} {row['MOTA']:<12.4f} "
                     f"{row['IDSW']:<12.0f} {row['FPS']:<12.2f} {row['Count']:<12.0f}")

    summary.append("")
    summary.append("=" * 100)

    return "\n".join(summary)

def create_detection_graphs(detection_metrics, output_dir):
    """Create detection metrics comparison graphs"""
    # Prepare data
    models = []
    precision = []
    recall = []
    mAP50 = []
    mAP50_95 = []

    for model, metrics in detection_metrics.items():
        if metrics:
            models.append(model)
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            mAP50.append(metrics['mAP50'])
            mAP50_95.append(metrics['mAP50-95'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detection Metrics Comparison', fontsize=16, fontweight='bold')

    # Precision comparison
    axes[0, 0].bar(range(len(models)), precision, color='skyblue')
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision Comparison')
    axes[0, 0].grid(True, alpha=0.3)

    # Recall comparison
    axes[0, 1].bar(range(len(models)), recall, color='lightgreen')
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall Comparison')
    axes[0, 1].grid(True, alpha=0.3)

    # mAP50 comparison
    axes[1, 0].bar(range(len(models)), mAP50, color='coral')
    axes[1, 0].set_xticks(range(len(models)))
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].set_ylabel('mAP50')
    axes[1, 0].set_title('mAP@0.5 Comparison')
    axes[1, 0].grid(True, alpha=0.3)

    # mAP50-95 comparison
    axes[1, 1].bar(range(len(models)), mAP50_95, color='mediumpurple')
    axes[1, 1].set_xticks(range(len(models)))
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].set_ylabel('mAP50-95')
    axes[1, 1].set_title('mAP@0.5:0.95 Comparison')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/detection_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/detection_metrics_comparison.png")

def create_tracking_graphs_per_gate(tracking_metrics, output_dir):
    """Create tracking metrics graphs per gate"""
    gates = sorted(tracking_metrics['Gate'].unique())

    for gate in gates:
        gate_data = tracking_metrics[tracking_metrics['Gate'] == gate].sort_values('IDF1', ascending=False)

        if len(gate_data) == 0:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Tracking Metrics - {gate}', fontsize=16, fontweight='bold')

        models = gate_data['Model'].tolist()

        # IDF1
        axes[0, 0].barh(models, gate_data['IDF1'], color='skyblue')
        axes[0, 0].set_xlabel('IDF1')
        axes[0, 0].set_title('IDF1 Score')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # MOTA
        axes[0, 1].barh(models, gate_data['MOTA'], color='lightgreen')
        axes[0, 1].set_xlabel('MOTA')
        axes[0, 1].set_title('MOTA Score')
        axes[0, 1].grid(True, alpha=0.3, axis='x')

        # FPS
        axes[1, 0].barh(models, gate_data['FPS'], color='coral')
        axes[1, 0].set_xlabel('FPS')
        axes[1, 0].set_title('Frames Per Second')
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        # Count
        axes[1, 1].barh(models, gate_data['Count'], color='mediumpurple')
        axes[1, 1].set_xlabel('Count')
        axes[1, 1].set_title('Detection Count')
        axes[1, 1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        safe_gate_name = gate.replace('/', '_').replace('.', '_')
        plt.savefig(f'{output_dir}/tracking_metrics_{safe_gate_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_dir}/tracking_metrics_{safe_gate_name}.png")

def create_overall_tracking_graphs(tracking_metrics, output_dir):
    """Create overall tracking performance graphs"""
    # Calculate average metrics per model
    avg_metrics = tracking_metrics.groupby('Model').agg({
        'IDF1': 'mean',
        'MOTA': 'mean',
        'FPS': 'mean',
        'Count': 'sum'
    }).sort_values('IDF1', ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Overall Tracking Performance', fontsize=16, fontweight='bold')

    models = avg_metrics.index.tolist()

    # Average IDF1
    axes[0, 0].barh(models, avg_metrics['IDF1'], color='skyblue')
    axes[0, 0].set_xlabel('Average IDF1')
    axes[0, 0].set_title('Average IDF1 Across All Gates')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # Average MOTA
    axes[0, 1].barh(models, avg_metrics['MOTA'], color='lightgreen')
    axes[0, 1].set_xlabel('Average MOTA')
    axes[0, 1].set_title('Average MOTA Across All Gates')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # Average FPS
    axes[1, 0].barh(models, avg_metrics['FPS'], color='coral')
    axes[1, 0].set_xlabel('Average FPS')
    axes[1, 0].set_title('Average FPS Across All Gates')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Total Count
    axes[1, 1].barh(models, avg_metrics['Count'], color='mediumpurple')
    axes[1, 1].set_xlabel('Total Count')
    axes[1, 1].set_title('Total Detection Count')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_tracking_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/overall_tracking_performance.png")

def main():
    print("Starting metrics analysis...")

    # Create output directory
    output_dir = '/media/mydrive/GitHub/ultralytics/tracking/metrics_summary'
    os.makedirs(output_dir, exist_ok=True)

    # Extract detection metrics
    print("\nExtracting detection metrics...")
    detection_metrics = {}
    for model_name, dir_path in detection_dirs.items():
        results_file = os.path.join(dir_path, 'results.csv')
        if os.path.exists(results_file):
            metrics = extract_final_detection_metrics(results_file)
            if metrics:
                detection_metrics[model_name] = metrics
                print(f"  {model_name}: ✓")
        else:
            print(f"  {model_name}: File not found")

    # Extract tracking metrics
    print("\nExtracting tracking metrics...")
    all_tracking_data = []
    for version, dir_path in tracking_dirs.items():
        summary_file = os.path.join(dir_path, 'summary.txt')
        if os.path.exists(summary_file):
            df = parse_tracking_summary(summary_file)
            if not df.empty:
                all_tracking_data.append(df)
                print(f"  {version}: {len(df)} entries")
        else:
            print(f"  {version}: File not found")

    tracking_metrics = pd.concat(all_tracking_data, ignore_index=True) if all_tracking_data else pd.DataFrame()

    # Create text summary
    print("\nCreating text summary...")
    summary_text = create_summary_text(detection_metrics, tracking_metrics)
    summary_file = os.path.join(output_dir, 'comprehensive_metrics_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    print(f"Saved: {summary_file}")

    # Create graphs
    print("\nCreating detection comparison graphs...")
    create_detection_graphs(detection_metrics, output_dir)

    if not tracking_metrics.empty:
        print("\nCreating tracking graphs per gate...")
        create_tracking_graphs_per_gate(tracking_metrics, output_dir)

        print("\nCreating overall tracking performance graphs...")
        create_overall_tracking_graphs(tracking_metrics, output_dir)

    print(f"\n{'='*80}")
    print("Analysis complete! All files saved to:")
    print(f"  {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
