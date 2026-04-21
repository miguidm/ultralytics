#!/usr/bin/env python3
"""
Create enhanced visualization graphs for detection and tracking metrics comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load data
tracking_df = pd.read_csv('/media/mydrive/GitHub/ultralytics/tracking/metrics_summary_complete/tracking_metrics_complete.csv')
output_dir = '/media/mydrive/GitHub/ultralytics/tracking/metrics_summary_complete'

# Use Gate_Clean if available, otherwise use Gate
if 'Gate_Clean' in tracking_df.columns:
    tracking_df['Gate'] = tracking_df['Gate_Clean']

# Detection metrics (from final epochs)
detection_data = {
    'DCNv2-FPN': {'mAP50-95': 0.7342, 'Precision': 0.9035, 'Recall': 0.8096},
    'DCNv2-Full': {'mAP50-95': 0.7529, 'Precision': 0.9122, 'Recall': 0.8505},
    'DCNv2-Liu': {'mAP50-95': 0.7362, 'Precision': 0.9229, 'Recall': 0.7994},
    'DCNv2-Pan': {'mAP50-95': 0.7329, 'Precision': 0.8880, 'Recall': 0.8103},
    'DCNv3-Base': {'mAP50-95': 0.7468, 'Precision': 0.8929, 'Recall': 0.8331},
    'DCNv3-FPN': {'mAP50-95': 0.7451, 'Precision': 0.9213, 'Recall': 0.7967},
    'DCNv3-Full': {'mAP50-95': 0.7301, 'Precision': 0.9256, 'Recall': 0.8079},
    'DCNv3-Liu': {'mAP50-95': 0.7612, 'Precision': 0.9360, 'Recall': 0.8024},
    'DCNv3-Pan': {'mAP50-95': 0.7409, 'Precision': 0.8828, 'Recall': 0.8425},
    'Vanilla-YOLOv8m': {'mAP50-95': 0.7530, 'Precision': 0.9031, 'Recall': 0.8156},
}

def create_dcnv2_vs_dcnv3_comparison():
    """Compare DCNv2 vs DCNv3 overall performance"""

    # Calculate averages for DCNv2 vs DCNv3
    dcnv2_models = ['DCNv2-FPN', 'DCNv2-Full', 'DCNv2-Liu', 'DCNv2-Pan']
    dcnv3_models = ['DCNv3-FPN', 'DCNv3-Full', 'DCNv3-Liu', 'DCNv3-Pan']

    # Tracking metrics
    dcnv2_tracking = tracking_df[tracking_df['Model'].isin(dcnv2_models)]
    dcnv3_tracking = tracking_df[tracking_df['Model'].isin(dcnv3_models)]

    dcnv2_avg_idf1 = dcnv2_tracking.groupby('Model')['IDF1'].mean().mean()
    dcnv3_avg_idf1 = dcnv3_tracking[dcnv3_tracking['IDF1'] > 0].groupby('Model')['IDF1'].mean().mean()

    dcnv2_avg_fps = dcnv2_tracking.groupby('Model')['FPS'].mean().mean()
    dcnv3_avg_fps = dcnv3_tracking.groupby('Model')['FPS'].mean().mean()

    # Detection metrics
    dcnv2_avg_map = np.mean([detection_data[m]['mAP50-95'] for m in dcnv2_models])
    dcnv3_avg_map = np.mean([detection_data[m]['mAP50-95'] for m in dcnv3_models])

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('DCNv2 vs DCNv3 Overall Comparison', fontsize=16, fontweight='bold')

    # IDF1 Comparison
    categories = ['DCNv2\n(Avg)', 'DCNv3\n(Avg)']
    idf1_values = [dcnv2_avg_idf1, dcnv3_avg_idf1]
    bars1 = axes[0].bar(categories, idf1_values, color=['#3498db', '#e74c3c'], width=0.6)
    axes[0].set_ylabel('Average IDF1 Score', fontsize=12)
    axes[0].set_title('Tracking Performance (IDF1)', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, idf1_values)):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # FPS Comparison
    fps_values = [dcnv2_avg_fps, dcnv3_avg_fps]
    bars2 = axes[1].bar(categories, fps_values, color=['#3498db', '#e74c3c'], width=0.6)
    axes[1].set_ylabel('Average FPS', fontsize=12)
    axes[1].set_title('Processing Speed (FPS)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars2, fps_values)):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # mAP Comparison
    map_values = [dcnv2_avg_map, dcnv3_avg_map]
    bars3 = axes[2].bar(categories, map_values, color=['#3498db', '#e74c3c'], width=0.6)
    axes[2].set_ylabel('Average mAP@0.5:0.95', fontsize=12)
    axes[2].set_title('Detection Performance (mAP50-95)', fontsize=14, fontweight='bold')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars3, map_values)):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dcnv2_vs_dcnv3_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/dcnv2_vs_dcnv3_comparison.png")

def create_heatmap_idf1_by_gate():
    """Create heatmap of IDF1 scores by model and gate"""

    # Pivot data for heatmap
    pivot_data = tracking_df.pivot(index='Model', columns='Gate', values='IDF1')

    # Sort by average IDF1
    pivot_data['Average'] = pivot_data.mean(axis=1)
    pivot_data = pivot_data.sort_values('Average', ascending=False)
    pivot_data = pivot_data.drop('Average', axis=1)

    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', center=0.5,
                cbar_kws={'label': 'IDF1 Score'}, linewidths=0.5, linecolor='gray')
    plt.title('IDF1 Score Heatmap: Models vs Gates', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Gate', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_idf1_by_gate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/heatmap_idf1_by_gate.png")

def create_heatmap_fps_by_gate():
    """Create heatmap of FPS by model and gate"""

    pivot_data = tracking_df.pivot(index='Model', columns='Gate', values='FPS')
    pivot_data['Average'] = pivot_data.mean(axis=1)
    pivot_data = pivot_data.sort_values('Average', ascending=False)
    pivot_data = pivot_data.drop('Average', axis=1)

    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'FPS'}, linewidths=0.5, linecolor='gray')
    plt.title('FPS Heatmap: Models vs Gates', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Gate', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_fps_by_gate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/heatmap_fps_by_gate.png")

def create_detection_count_comparison():
    """Create detection count comparison across models and gates"""

    # Filter out zero detection models
    valid_tracking = tracking_df[tracking_df['Detections'] > 0]

    # Group by model
    model_detections = valid_tracking.groupby('Model').agg({
        'Detections': 'sum',
        'Tracks': 'sum',
        'Count': 'sum'
    }).sort_values('Detections', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Detection and Track Counts by Model', fontsize=16, fontweight='bold')

    # Total Detections
    models = model_detections.index.tolist()
    detections = model_detections['Detections'].tolist()

    bars1 = axes[0].barh(models, detections, color='steelblue')
    axes[0].set_xlabel('Total Detections', fontsize=12)
    axes[0].set_title('Total Detections Across All Gates', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    for i, (bar, val) in enumerate(zip(bars1, detections)):
        axes[0].text(val + max(detections)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=10)

    # Total Tracks
    tracks = model_detections['Tracks'].tolist()

    bars2 = axes[1].barh(models, tracks, color='coral')
    axes[1].set_xlabel('Total Tracks', fontsize=12)
    axes[1].set_title('Total Tracks Across All Gates', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    for i, (bar, val) in enumerate(zip(bars2, tracks)):
        axes[1].text(val + max(tracks)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/detection_track_counts_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/detection_track_counts_comparison.png")

def create_model_architecture_comparison():
    """Compare different architectures (FPN, Full, Liu, Pan)"""

    architectures = ['FPN', 'Full', 'Liu', 'Pan']

    # Prepare data
    dcnv2_data = {}
    dcnv3_data = {}

    for arch in architectures:
        dcnv2_model = f'DCNv2-{arch}'
        dcnv3_model = f'DCNv3-{arch}'

        # Tracking metrics (excluding zero detections)
        dcnv2_tracking = tracking_df[tracking_df['Model'] == dcnv2_model]
        dcnv3_tracking = tracking_df[(tracking_df['Model'] == dcnv3_model) & (tracking_df['IDF1'] > 0)]

        dcnv2_data[arch] = {
            'IDF1': dcnv2_tracking['IDF1'].mean() if len(dcnv2_tracking) > 0 else 0,
            'FPS': dcnv2_tracking['FPS'].mean() if len(dcnv2_tracking) > 0 else 0,
            'mAP': detection_data.get(dcnv2_model, {}).get('mAP50-95', 0)
        }

        dcnv3_data[arch] = {
            'IDF1': dcnv3_tracking['IDF1'].mean() if len(dcnv3_tracking) > 0 else 0,
            'FPS': dcnv3_tracking['FPS'].mean() if len(dcnv3_tracking) > 0 else 0,
            'mAP': detection_data.get(dcnv3_model, {}).get('mAP50-95', 0)
        }

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Architecture Comparison: DCNv2 vs DCNv3', fontsize=16, fontweight='bold')

    x = np.arange(len(architectures))
    width = 0.35

    # IDF1 Comparison
    dcnv2_idf1 = [dcnv2_data[arch]['IDF1'] for arch in architectures]
    dcnv3_idf1 = [dcnv3_data[arch]['IDF1'] for arch in architectures]

    bars1 = axes[0].bar(x - width/2, dcnv2_idf1, width, label='DCNv2', color='#3498db')
    bars2 = axes[0].bar(x + width/2, dcnv3_idf1, width, label='DCNv3', color='#e74c3c')

    axes[0].set_ylabel('Average IDF1', fontsize=12)
    axes[0].set_title('Tracking Performance (IDF1)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(architectures)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # FPS Comparison
    dcnv2_fps = [dcnv2_data[arch]['FPS'] for arch in architectures]
    dcnv3_fps = [dcnv3_data[arch]['FPS'] for arch in architectures]

    axes[1].bar(x - width/2, dcnv2_fps, width, label='DCNv2', color='#3498db')
    axes[1].bar(x + width/2, dcnv3_fps, width, label='DCNv3', color='#e74c3c')

    axes[1].set_ylabel('Average FPS', fontsize=12)
    axes[1].set_title('Processing Speed (FPS)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(architectures)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # mAP Comparison
    dcnv2_map = [dcnv2_data[arch]['mAP'] for arch in architectures]
    dcnv3_map = [dcnv3_data[arch]['mAP'] for arch in architectures]

    axes[2].bar(x - width/2, dcnv2_map, width, label='DCNv2', color='#3498db')
    axes[2].bar(x + width/2, dcnv3_map, width, label='DCNv3', color='#e74c3c')

    axes[2].set_ylabel('mAP@0.5:0.95', fontsize=12)
    axes[2].set_title('Detection Performance (mAP50-95)', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(architectures)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/architecture_comparison.png")

def create_scatter_plots():
    """Create scatter plot visualizations"""

    # Calculate average metrics per model
    avg_metrics = tracking_df.groupby('Model').agg({
        'IDF1': 'mean',
        'MOTA': 'mean',
        'FPS': 'mean',
        'Detections': 'sum',
        'Tracks': 'sum'
    }).reset_index()

    # Add detection metrics
    avg_metrics['mAP50-95'] = avg_metrics['Model'].map(lambda x: detection_data.get(x, {}).get('mAP50-95', 0))

    # Assign colors by model family
    def get_model_color(model):
        if 'DCNv2' in model:
            return '#3498db'  # Blue
        elif 'DCNv3' in model:
            return '#e74c3c'  # Red
        else:
            return '#2ecc71'  # Green (Vanilla)

    avg_metrics['Color'] = avg_metrics['Model'].apply(get_model_color)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='DCNv2'),
        Patch(facecolor='#e74c3c', label='DCNv3'),
        Patch(facecolor='#2ecc71', label='Vanilla-YOLOv8m')
    ]

    # 1. IDF1 vs FPS
    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, row in avg_metrics.iterrows():
        ax.scatter(row['FPS'], row['IDF1'], s=200, c=row['Color'], alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(row['Model'], (row['FPS'], row['IDF1']), fontsize=9, ha='center', va='bottom')
    ax.set_xlabel('Average FPS (Processing Speed)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average IDF1 (Identity Tracking Quality)', fontsize=14, fontweight='bold')
    ax.set_title('Tracking Quality vs Efficiency Trade-off', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_idf1_vs_fps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/scatter_idf1_vs_fps.png")

    # 2. mAP50-95 vs IDF1
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

    # 3. mAP50-95 vs MOTA
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

    # 4. IDF1 vs MOTA
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

    # 5. Bubble Chart
    fig, ax = plt.subplots(figsize=(14, 10))
    fps_normalized = (avg_metrics['FPS'] - avg_metrics['FPS'].min()) / (avg_metrics['FPS'].max() - avg_metrics['FPS'].min())
    bubble_sizes = 100 + fps_normalized * 500

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

    fps_legend_sizes = [100, 300, 600]
    fps_legend_values = [avg_metrics['FPS'].min(), avg_metrics['FPS'].mean(), avg_metrics['FPS'].max()]
    for size, fps_val in zip(fps_legend_sizes, fps_legend_values):
        ax.scatter([], [], s=size, c='gray', alpha=0.5, edgecolors='black', label=f'FPS: {fps_val:.1f}')
    ax.legend(loc='lower right', fontsize=10, title='Processing Speed')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_bubble_multidimensional.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/scatter_bubble_multidimensional.png")


def main():
    print("Creating enhanced visualization graphs...")

    print("\n1. Creating DCNv2 vs DCNv3 overall comparison...")
    create_dcnv2_vs_dcnv3_comparison()

    print("\n2. Creating IDF1 heatmap by gate...")
    create_heatmap_idf1_by_gate()

    print("\n3. Creating FPS heatmap by gate...")
    create_heatmap_fps_by_gate()

    print("\n4. Creating detection count comparison...")
    create_detection_count_comparison()

    print("\n5. Creating architecture comparison...")
    create_model_architecture_comparison()

    print("\n6. Creating scatter plot visualizations...")
    create_scatter_plots()

    print(f"\n{'='*80}")
    print("Enhanced graphs created successfully!")
    print(f"All files saved to: {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
