#!/usr/bin/env python3
"""
Comprehensive comparison of DCNv2 vs DCNv3 models for Gate 3 Feb 20
Using multiple visualization methods to compare 2+ metrics
"""

import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def parse_metrics_file(filepath):
    """Parse a metrics file and extract key metrics"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Extract metrics
        model_match = re.search(r'Model:\s+(\S+)', content)
        idf1_match = re.search(r'IDF1:\s+([\d.]+)', content)
        mota_match = re.search(r'MOTA:\s+([\d.]+)', content)
        idsw_match = re.search(r'IDSW:\s+(\d+)', content)
        fps_match = re.search(r'FPS:\s+([\d.]+)', content)
        frames_match = re.search(r'Total Frames:\s+(\d+)', content)
        tracks_match = re.search(r'Total Tracks:\s+(\d+)', content)
        detections_match = re.search(r'Total Detections:\s+(\d+)', content)
        count_match = re.search(r'TOTAL:\s+(\d+)', content)

        if not all([model_match, idf1_match, fps_match]):
            return None

        return {
            'Model': model_match.group(1),
            'IDF1': float(idf1_match.group(1)),
            'MOTA': float(mota_match.group(1)) if mota_match else 0,
            'IDSW': int(idsw_match.group(1)) if idsw_match else 0,
            'FPS': float(fps_match.group(1)),
            'Frames': int(frames_match.group(1)) if frames_match else 0,
            'Tracks': int(tracks_match.group(1)) if tracks_match else 0,
            'Detections': int(detections_match.group(1)) if detections_match else 0,
            'Count': int(count_match.group(1)) if count_match else 0
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def collect_gate3_feb20_metrics():
    """Collect all metrics for Gate3_Feb20"""
    base_dirs = {
        'DCNv2': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv2',
        'DCNv3': '/media/mydrive/GitHub/ultralytics/tracking/tracking_metrics_results_dcnv3'
    }

    models = ['FPN', 'Full', 'Liu', 'Pan']
    all_metrics = []

    for version, base_dir in base_dirs.items():
        for model in models:
            model_name = f'{version}-{model}'
            metrics_file = os.path.join(base_dir, model_name, 'Gate3_Feb20', 'Gate3_Feb20_metrics.txt')

            if os.path.exists(metrics_file):
                metrics = parse_metrics_file(metrics_file)
                if metrics:
                    metrics['Version'] = version
                    metrics['Architecture'] = model
                    all_metrics.append(metrics)
                    print(f"✓ Parsed: {model_name}")
            else:
                print(f"✗ Not found: {metrics_file}")

    return pd.DataFrame(all_metrics)

def create_scatter_plot(df, output_dir):
    """Scatter plot: IDF1 vs FPS (Speed-Accuracy Trade-off)"""
    plt.figure(figsize=(12, 8))

    # Separate DCNv2 and DCNv3
    dcnv2 = df[df['Version'] == 'DCNv2']
    dcnv3 = df[df['Version'] == 'DCNv3']

    # Plot points
    plt.scatter(dcnv2['IDF1'], dcnv2['FPS'], s=200, alpha=0.7,
                color='#3498db', marker='o', label='DCNv2', edgecolors='black', linewidth=2)
    plt.scatter(dcnv3['IDF1'], dcnv3['FPS'], s=200, alpha=0.7,
                color='#e74c3c', marker='s', label='DCNv3', edgecolors='black', linewidth=2)

    # Add labels for each point
    for idx, row in df.iterrows():
        plt.annotate(row['Architecture'],
                    (row['IDF1'], row['FPS']),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    # Add quadrant lines at median
    median_idf1 = df['IDF1'].median()
    median_fps = df['FPS'].median()
    plt.axvline(median_idf1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axhline(median_fps, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Add quadrant labels
    max_idf1 = df['IDF1'].max()
    max_fps = df['FPS'].max()
    plt.text(max_idf1, max_fps, 'Best Overall\n(High Speed + Accuracy)',
             ha='right', va='top', fontsize=10, alpha=0.6, style='italic')

    plt.xlabel('IDF1 Score (Tracking Accuracy)', fontsize=14, fontweight='bold')
    plt.ylabel('FPS (Processing Speed)', fontsize=14, fontweight='bold')
    plt.title('Speed vs Accuracy Trade-off - Gate 3 Feb 20', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower left', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_idf1_vs_fps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: scatter_idf1_vs_fps.png")

def create_normalized_comparison(df, output_dir):
    """Normalized multi-metric comparison with composite scores"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Metrics to normalize
    metrics = ['IDF1', 'MOTA', 'FPS']

    # Normalize metrics (0-1 scale)
    df_norm = df.copy()
    for metric in metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        if max_val > min_val:
            df_norm[f'{metric}_norm'] = (df[metric] - min_val) / (max_val - min_val)
        else:
            df_norm[f'{metric}_norm'] = 0

    # Plot 1: Normalized metrics comparison
    x = np.arange(len(df))
    width = 0.25

    bars1 = axes[0].bar(x - width, df_norm['IDF1_norm'], width, label='IDF1', color='#2ecc71', alpha=0.8)
    bars2 = axes[0].bar(x, df_norm['MOTA_norm'], width, label='MOTA', color='#3498db', alpha=0.8)
    bars3 = axes[0].bar(x + width, df_norm['FPS_norm'], width, label='FPS', color='#e74c3c', alpha=0.8)

    axes[0].set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
    axes[0].set_title('Normalized Performance Metrics - Gate 3 Feb 20', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df['Model'], rotation=45, ha='right')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 1.1)

    # Plot 2: Composite scores with different weights
    # Define 3 scenarios
    scenarios = {
        'Accuracy Priority\n(70% IDF1, 20% MOTA, 10% FPS)': [0.7, 0.2, 0.1],
        'Balanced\n(40% IDF1, 30% MOTA, 30% FPS)': [0.4, 0.3, 0.3],
        'Speed Priority\n(20% IDF1, 10% MOTA, 70% FPS)': [0.2, 0.1, 0.7]
    }

    colors_scenario = ['#2ecc71', '#3498db', '#e74c3c']
    x_pos = np.arange(len(df))
    width_scenario = 0.25

    for idx, (scenario_name, weights) in enumerate(scenarios.items()):
        composite_scores = (
            df_norm['IDF1_norm'] * weights[0] +
            df_norm['MOTA_norm'] * weights[1] +
            df_norm['FPS_norm'] * weights[2]
        )

        offset = (idx - 1) * width_scenario
        bars = axes[1].bar(x_pos + offset, composite_scores, width_scenario,
                          label=scenario_name, color=colors_scenario[idx], alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    axes[1].set_ylabel('Composite Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Composite Scores for Different Use Cases', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(df['Model'], rotation=45, ha='right')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/normalized_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: normalized_comparison.png")

def create_radar_chart(df, output_dir):
    """Radar chart for multi-metric comparison"""
    # Normalize metrics for radar chart
    metrics_to_plot = ['IDF1', 'MOTA', 'FPS']
    df_norm = df.copy()

    for metric in metrics_to_plot:
        min_val = df[metric].min()
        max_val = df[metric].max()
        if max_val > min_val:
            df_norm[metric] = (df[metric] - min_val) / (max_val - min_val)
        else:
            df_norm[metric] = 0

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

    for idx, row in df_norm.iterrows():
        values = [row[metric] for metric in metrics_to_plot]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.title('Multi-Metric Performance Comparison - Gate 3 Feb 20',
              fontsize=16, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: radar_chart.png")

def create_pareto_analysis(df, output_dir):
    """Identify Pareto frontier (optimal models)"""
    plt.figure(figsize=(12, 8))

    # For Pareto frontier: maximize both IDF1 and FPS
    # A point is on the Pareto frontier if no other point is better in both dimensions

    def is_pareto_efficient(costs):
        """Find Pareto efficient points (maximize both dimensions)"""
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Point is efficient if no other point is better in both dimensions
                is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
                is_efficient[i] = True
        return is_efficient

    # Prepare data (higher is better for both)
    costs = df[['IDF1', 'FPS']].values
    pareto_mask = is_pareto_efficient(costs)

    # Separate Pareto and non-Pareto points
    pareto_df = df[pareto_mask]
    non_pareto_df = df[~pareto_mask]

    # Plot non-Pareto points
    dcnv2_non = non_pareto_df[non_pareto_df['Version'] == 'DCNv2']
    dcnv3_non = non_pareto_df[non_pareto_df['Version'] == 'DCNv3']

    plt.scatter(dcnv2_non['IDF1'], dcnv2_non['FPS'], s=150, alpha=0.4,
                color='#3498db', marker='o', label='DCNv2 (Sub-optimal)')
    plt.scatter(dcnv3_non['IDF1'], dcnv3_non['FPS'], s=150, alpha=0.4,
                color='#e74c3c', marker='s', label='DCNv3 (Sub-optimal)')

    # Plot Pareto points
    dcnv2_pareto = pareto_df[pareto_df['Version'] == 'DCNv2']
    dcnv3_pareto = pareto_df[pareto_df['Version'] == 'DCNv3']

    plt.scatter(dcnv2_pareto['IDF1'], dcnv2_pareto['FPS'], s=250, alpha=0.9,
                color='#3498db', marker='o', edgecolors='gold', linewidth=3,
                label='DCNv2 (Pareto Optimal)', zorder=5)
    plt.scatter(dcnv3_pareto['IDF1'], dcnv3_pareto['FPS'], s=250, alpha=0.9,
                color='#e74c3c', marker='s', edgecolors='gold', linewidth=3,
                label='DCNv3 (Pareto Optimal)', zorder=5)

    # Draw Pareto frontier line
    pareto_sorted = pareto_df.sort_values('IDF1')
    plt.plot(pareto_sorted['IDF1'], pareto_sorted['FPS'],
             'k--', linewidth=2, alpha=0.5, label='Pareto Frontier', zorder=4)

    # Annotate all points
    for idx, row in df.iterrows():
        is_pareto = pareto_mask[idx]
        plt.annotate(row['Architecture'],
                    (row['IDF1'], row['FPS']),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold' if is_pareto else 'normal',
                    color='darkgreen' if is_pareto else 'black')

    plt.xlabel('IDF1 Score (Tracking Accuracy)', fontsize=14, fontweight='bold')
    plt.ylabel('FPS (Processing Speed)', fontsize=14, fontweight='bold')
    plt.title('Pareto Optimal Models - Gate 3 Feb 20\n(Models on frontier offer best trade-offs)',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower left', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/pareto_frontier.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: pareto_frontier.png")

    return pareto_df

def create_summary_report(df, pareto_df, output_dir):
    """Generate comprehensive text summary"""
    report = []
    report.append("="*80)
    report.append("GATE 3 FEB 20 - COMPREHENSIVE MODEL COMPARISON")
    report.append("="*80)
    report.append("")

    # Overall statistics
    report.append("Dataset Information:")
    report.append("-" * 80)
    sample = df.iloc[0]
    report.append(f"Total Frames Analyzed: {sample['Frames']:,}")
    report.append(f"Total Models Compared: {len(df)}")
    report.append(f"  - DCNv2 Models: {len(df[df['Version'] == 'DCNv2'])}")
    report.append(f"  - DCNv3 Models: {len(df[df['Version'] == 'DCNv3'])}")
    report.append("")

    # Key metrics comparison
    report.append("Performance Metrics Summary:")
    report.append("-" * 80)
    report.append(f"{'Model':<15} {'IDF1':<8} {'MOTA':<8} {'FPS':<8} {'Count':<8} {'Detections':<12}")
    report.append("-" * 80)

    for idx, row in df.sort_values('IDF1', ascending=False).iterrows():
        report.append(f"{row['Model']:<15} {row['IDF1']:<8.4f} {row['MOTA']:<8.4f} "
                     f"{row['FPS']:<8.2f} {row['Count']:<8} {row['Detections']:<12,}")
    report.append("")

    # Rankings
    report.append("Rankings by Metric:")
    report.append("-" * 80)

    report.append("\n🏆 Best IDF1 (Tracking Accuracy):")
    top3_idf1 = df.nlargest(3, 'IDF1')
    for i, row in enumerate(top3_idf1.itertuples(), 1):
        report.append(f"  {i}. {row.Model}: {row.IDF1:.4f}")

    report.append("\n🏆 Best MOTA:")
    top3_mota = df.nlargest(3, 'MOTA')
    for i, row in enumerate(top3_mota.itertuples(), 1):
        report.append(f"  {i}. {row.Model}: {row.MOTA:.4f}")

    report.append("\n🚀 Fastest (FPS):")
    top3_fps = df.nlargest(3, 'FPS')
    for i, row in enumerate(top3_fps.itertuples(), 1):
        report.append(f"  {i}. {row.Model}: {row.FPS:.2f} FPS")

    report.append("\n📊 Most Detections:")
    top3_det = df.nlargest(3, 'Detections')
    for i, row in enumerate(top3_det.itertuples(), 1):
        report.append(f"  {i}. {row.Model}: {row.Detections:,}")

    # Pareto optimal models
    report.append("\n" + "="*80)
    report.append("PARETO OPTIMAL MODELS (Best Trade-offs)")
    report.append("="*80)
    report.append("These models cannot be improved in one metric without sacrificing another:\n")
    for idx, row in pareto_df.iterrows():
        report.append(f"✓ {row['Model']}: IDF1={row['IDF1']:.4f}, FPS={row['FPS']:.2f}")
    report.append("")

    # DCNv2 vs DCNv3 comparison
    report.append("="*80)
    report.append("DCNv2 vs DCNv3 COMPARISON")
    report.append("="*80)

    dcnv2_avg = df[df['Version'] == 'DCNv2'][['IDF1', 'MOTA', 'FPS']].mean()
    dcnv3_avg = df[df['Version'] == 'DCNv3'][['IDF1', 'MOTA', 'FPS']].mean()

    report.append(f"\nAverage IDF1:")
    report.append(f"  DCNv2: {dcnv2_avg['IDF1']:.4f}")
    report.append(f"  DCNv3: {dcnv3_avg['IDF1']:.4f}")
    diff_pct = ((dcnv3_avg['IDF1'] - dcnv2_avg['IDF1']) / dcnv2_avg['IDF1']) * 100
    winner = "DCNv3" if diff_pct > 0 else "DCNv2"
    report.append(f"  Winner: {winner} ({abs(diff_pct):.1f}% {'better' if diff_pct > 0 else 'worse'})")

    report.append(f"\nAverage MOTA:")
    report.append(f"  DCNv2: {dcnv2_avg['MOTA']:.4f}")
    report.append(f"  DCNv3: {dcnv3_avg['MOTA']:.4f}")
    diff_pct = ((dcnv3_avg['MOTA'] - dcnv2_avg['MOTA']) / dcnv2_avg['MOTA']) * 100 if dcnv2_avg['MOTA'] > 0 else 0
    winner = "DCNv3" if diff_pct > 0 else "DCNv2"
    report.append(f"  Winner: {winner} ({abs(diff_pct):.1f}% {'better' if diff_pct > 0 else 'worse'})")

    report.append(f"\nAverage FPS:")
    report.append(f"  DCNv2: {dcnv2_avg['FPS']:.2f}")
    report.append(f"  DCNv3: {dcnv3_avg['FPS']:.2f}")
    diff_pct = ((dcnv3_avg['FPS'] - dcnv2_avg['FPS']) / dcnv2_avg['FPS']) * 100
    winner = "DCNv2" if diff_pct < 0 else "DCNv3"
    report.append(f"  Winner: {winner} ({abs(diff_pct):.1f}% {'faster' if winner == 'DCNv2' else 'slower'})")

    # Recommendations
    report.append("\n" + "="*80)
    report.append("RECOMMENDATIONS")
    report.append("="*80)

    best_accuracy = df.nlargest(1, 'IDF1').iloc[0]
    best_speed = df.nlargest(1, 'FPS').iloc[0]

    # Composite score (balanced)
    df_temp = df.copy()
    df_temp['IDF1_norm'] = (df_temp['IDF1'] - df_temp['IDF1'].min()) / (df_temp['IDF1'].max() - df_temp['IDF1'].min())
    df_temp['FPS_norm'] = (df_temp['FPS'] - df_temp['FPS'].min()) / (df_temp['FPS'].max() - df_temp['FPS'].min())
    df_temp['composite'] = 0.5 * df_temp['IDF1_norm'] + 0.5 * df_temp['FPS_norm']
    best_balanced = df_temp.nlargest(1, 'composite').iloc[0]

    report.append(f"\n🎯 Best for Accuracy-Critical Applications:")
    report.append(f"   → {best_accuracy['Model']} (IDF1: {best_accuracy['IDF1']:.4f}, FPS: {best_accuracy['FPS']:.2f})")

    report.append(f"\n⚡ Best for Real-Time Applications:")
    report.append(f"   → {best_speed['Model']} (FPS: {best_speed['FPS']:.2f}, IDF1: {best_speed['IDF1']:.4f})")

    report.append(f"\n⚖️  Best Balanced Performance:")
    report.append(f"   → {best_balanced['Model']} (IDF1: {best_balanced['IDF1']:.4f}, FPS: {best_balanced['FPS']:.2f})")

    report.append("\n" + "="*80)
    report.append("Analysis complete!")
    report.append("="*80)

    # Save report
    report_text = "\n".join(report)
    with open(f'{output_dir}/comparison_summary.txt', 'w') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n✓ Saved: comparison_summary.txt")

def main():
    print("="*80)
    print("GATE 3 FEB 20 - COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    print("\nCollecting metrics from both DCNv2 and DCNv3...")

    # Collect metrics
    df = collect_gate3_feb20_metrics()

    if len(df) == 0:
        print("❌ No metrics found!")
        return

    print(f"\n✓ Collected metrics for {len(df)} models")
    print(f"  - DCNv2: {len(df[df['Version'] == 'DCNv2'])} models")
    print(f"  - DCNv3: {len(df[df['Version'] == 'DCNv3'])} models")

    # Create output directory
    output_dir = '/media/mydrive/GitHub/ultralytics/tracking/gate3_feb20_comparison'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output directory: {output_dir}")

    # Save raw data
    df.to_csv(f'{output_dir}/gate3_feb20_metrics.csv', index=False)
    print(f"✓ Saved: gate3_feb20_metrics.csv")

    # Create visualizations
    print("\nGenerating visualizations...")
    print("-" * 80)

    print("\n1. Creating scatter plot (IDF1 vs FPS)...")
    create_scatter_plot(df, output_dir)

    print("\n2. Creating normalized comparison...")
    create_normalized_comparison(df, output_dir)

    print("\n3. Creating radar chart...")
    create_radar_chart(df, output_dir)

    print("\n4. Creating Pareto frontier analysis...")
    pareto_df = create_pareto_analysis(df, output_dir)

    print("\n5. Generating summary report...")
    create_summary_report(df, pareto_df, output_dir)

    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print(f"All files saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
