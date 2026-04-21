#!/usr/bin/env python3
"""
Complete script to analyze detection and tracking metrics for DCNv2 and DCNv3 models
Reads individual metrics files for comprehensive analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns
import re

# Set style
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
    'Vanilla-YOLOv8m': '/home/migui/Downloads/yolov8m-vanilla-20260211T133104Z-1-001/yolov8m-vanilla',
}

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

# Gates to KEEP for ALL models (5 total - best Gate3 only = Gate3_Feb20)
# Exclude Gate3_Oct7 and Gate3_Apr3
GATES_TO_KEEP = ['Gate2.9_Oct7', 'Gate2_Oct7', 'Gate3.5_Oct7', 'Gate3.1_Oct7', 'Gate3_Feb20']

tracking_base_dirs = {
    'DCNv2': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv2',
    'DCNv3': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv3',
    'Vanilla': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_vanilla',
    'Vanilla_Gate3': '/media/mydrive/GitHub/ultralytics/tracking/yolov8m_vanilla_gate3_tracking_results',
}

def extract_final_detection_metrics(results_csv_path):
    """Extract final epoch metrics from results.csv"""
    try:
        df = pd.read_csv(results_csv_path)
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

        # Extract metrics (handle both "IDF1:" and "IDF1 (..." formats)
        idf1_match = re.search(r'(?:^\d+\.\s+)?IDF1\s*(?:\(.*?\))?\s*:\s+([\d.]+)', content, re.MULTILINE)
        mota_match = re.search(r'(?:^\d+\.\s+)?MOTA\s*(?:\(.*?\))?\s*:\s+([\d.]+)', content, re.MULTILINE)
        idsw_match = re.search(r'(?:^\d+\.\s+)?IDSW\s*(?:\(.*?\))?\s*:\s+(\d+)', content, re.MULTILINE)
        # FPS: match both "FPS:  125.81" (DCN) and "5. FPS (Frames Per Second): 73.20" (Vanilla)
        # For vanilla files, look for the numbered metrics section FPS, not the video properties FPS
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
        # Find all metrics files
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('_metrics.txt'):
                    metrics_file = os.path.join(root, file)
                    metrics = parse_tracking_metrics_file(metrics_file)
                    if metrics:
                        # Filter ALL models to only keep specified gates (best Gate3 only)
                        if metrics['Gate'] in GATES_TO_KEEP:
                            # Map gate name to clean version
                            metrics['Gate_Clean'] = GATE_NAME_MAPPING.get(metrics['Gate'], metrics['Gate'])
                            all_metrics.append(metrics)
                            print(f"  Parsed: {metrics['Model']} - {metrics['Gate']} → {metrics['Gate_Clean']}")

    return pd.DataFrame(all_metrics)

def create_enhanced_summary_text(detection_metrics, tracking_metrics):
    """Create comprehensive text summary with additional insights"""
    summary = []
    summary.append("=" * 120)
    summary.append("COMPREHENSIVE DETECTION AND TRACKING METRICS SUMMARY")
    summary.append("=" * 120)
    summary.append("")

    # Detection Metrics Summary
    summary.append("=" * 120)
    summary.append("DETECTION METRICS (Final Epoch)")
    summary.append("=" * 120)
    summary.append("")
    summary.append(f"{'Model':<20} {'Epoch':<8} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
    summary.append("-" * 120)

    for model, metrics in sorted(detection_metrics.items()):
        if metrics:
            summary.append(f"{model:<20} {metrics['epoch']:<8} {metrics['precision']:<12.4f} "
                         f"{metrics['recall']:<12.4f} {metrics['mAP50']:<12.4f} {metrics['mAP50-95']:<12.4f}")

    summary.append("")
    summary.append("=" * 120)
    summary.append("TRACKING METRICS BY GATE (with Detection Counts)")
    summary.append("=" * 120)
    summary.append("")

    # Group tracking metrics by gate
    gates = sorted(tracking_metrics['Gate'].unique())

    for gate in gates:
        summary.append(f"\n{gate}")
        summary.append("-" * 120)
        summary.append(f"{'Model':<20} {'IDF1':<10} {'MOTA':<10} {'IDSW':<8} {'FPS':<10} {'Count':<10} {'Detections':<12} {'Tracks':<8}")
        summary.append("-" * 120)

        gate_data = tracking_metrics[tracking_metrics['Gate'] == gate].sort_values('IDF1', ascending=False)
        for _, row in gate_data.iterrows():
            summary.append(f"{row['Model']:<20} {row['IDF1']:<10.4f} {row['MOTA']:<10.4f} "
                         f"{row['IDSW']:<8} {row['FPS']:<10.2f} {row['Count']:<10} "
                         f"{row['Detections']:<12} {row['Tracks']:<8}")

    summary.append("")
    summary.append("=" * 120)
    summary.append("OVERALL TRACKING PERFORMANCE")
    summary.append("=" * 120)
    summary.append("")

    # Calculate average metrics per model
    avg_metrics = tracking_metrics.groupby('Model').agg({
        'IDF1': 'mean',
        'MOTA': 'mean',
        'IDSW': 'sum',
        'FPS': 'mean',
        'Count': 'sum',
        'Detections': 'sum',
        'Tracks': 'sum'
    }).sort_values('IDF1', ascending=False)

    summary.append(f"{'Model':<20} {'Avg IDF1':<12} {'Avg MOTA':<12} {'Tot IDSW':<10} {'Avg FPS':<12} {'Tot Count':<12} {'Tot Detects':<14} {'Tot Tracks':<12}")
    summary.append("-" * 120)

    for model, row in avg_metrics.iterrows():
        summary.append(f"{model:<20} {row['IDF1']:<12.4f} {row['MOTA']:<12.4f} "
                     f"{row['IDSW']:<10.0f} {row['FPS']:<12.2f} {row['Count']:<12.0f} "
                     f"{row['Detections']:<14.0f} {row['Tracks']:<12.0f}")

    summary.append("")
    summary.append("=" * 120)
    summary.append("KEY INSIGHTS")
    summary.append("=" * 120)
    summary.append("")

    # Detection insights
    best_map = max(detection_metrics.items(), key=lambda x: x[1]['mAP50-95'] if x[1] else 0)
    summary.append(f"Best Detection mAP50-95: {best_map[0]} ({best_map[1]['mAP50-95']:.4f})")

    # Tracking insights
    best_tracking = avg_metrics.iloc[0]
    summary.append(f"Best Overall Tracking (IDF1): {avg_metrics.index[0]} ({best_tracking['IDF1']:.4f})")

    fastest = avg_metrics.nlargest(1, 'FPS')
    summary.append(f"Fastest Tracking (FPS): {fastest.index[0]} ({fastest['FPS'].iloc[0]:.2f} FPS)")

    # Models with zero detections
    zero_det_models = tracking_metrics[tracking_metrics['Detections'] == 0].groupby('Model').size()
    if len(zero_det_models) > 0:
        summary.append("\nModels with Zero Detections on some gates:")
        for model, count in zero_det_models.items():
            summary.append(f"  {model}: {count} gate(s)")

    summary.append("")
    summary.append("=" * 120)

    return "\n".join(summary)

def main():
    print("Starting complete metrics analysis...")

    # Create output directory
    output_dir = '/media/mydrive/GitHub/ultralytics/tracking/metrics_summary_complete'
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

    # Extract tracking metrics from individual files
    print("\nExtracting tracking metrics from individual files...")
    tracking_metrics = collect_all_tracking_metrics(tracking_base_dirs)

    # Create enhanced text summary
    print("\nCreating enhanced text summary...")
    summary_text = create_enhanced_summary_text(detection_metrics, tracking_metrics)
    summary_file = os.path.join(output_dir, 'comprehensive_metrics_summary_complete.txt')
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    print(f"Saved: {summary_file}")

    # Save tracking metrics to CSV for further analysis
    csv_file = os.path.join(output_dir, 'tracking_metrics_complete.csv')
    tracking_metrics.to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")

    print(f"\n{'='*80}")
    print("Complete analysis finished! All files saved to:")
    print(f"  {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
