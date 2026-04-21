#!/usr/bin/env python3
"""
Generate individual model performance reports for each gate-model combination.
Total: 45 files (5 gates × 9 models)
"""

import os

# Data structure containing all metrics
data = {
    "Gate3.5": {
        "DCNv2-Backbone": {"IDF1": 0.8619, "MOTA": 0.9800, "IDSW": 506, "FPS": 83.52, "ML": 11.8, "MT": 84.5, "GFLOPS": 63.92, "Count": 4115, "Time": "2.88h"},
        "DCNv2-FPN": {"IDF1": 0.8453, "MOTA": 0.9748, "IDSW": 557, "FPS": 81.69, "ML": 11.8, "MT": 84.2, "GFLOPS": 66.96, "Count": 4116, "Time": "2.95h"},
        "DCNv2-Full": {"IDF1": 0.8724, "MOTA": 0.9417, "IDSW": 508, "FPS": 81.69, "ML": 12.1, "MT": 83.8, "GFLOPS": 62.36, "Count": 4105, "Time": "3.05h"},
        "DCNv2-Pan": {"IDF1": 0.8503, "MOTA": 0.9692, "IDSW": 586, "FPS": 80.65, "ML": 11.9, "MT": 84.3, "GFLOPS": 63.31, "Count": 4111, "Time": "2.91h"},
        "DCNv3-Backbone": {"IDF1": 0.8773, "MOTA": 0.8876, "IDSW": 558, "FPS": 45.0, "ML": 12.4, "MT": 83.1, "GFLOPS": 71.08, "Count": 4110, "Time": "3.98h"},
        "DCNv3-FPN": {"IDF1": 0.8793, "MOTA": 0.9528, "IDSW": 459, "FPS": 38.9, "ML": 12.0, "MT": 83.4, "GFLOPS": 58.57, "Count": 4123, "Time": "4.61h"},
        "DCNv3-Full": {"IDF1": 0.8637, "MOTA": 0.9416, "IDSW": 627, "FPS": 39.86, "ML": 11.8, "MT": 83.8, "GFLOPS": 63.16, "Count": 4112, "Time": "6.48h"},
        "DCNv3-Pan": {"IDF1": 0.8572, "MOTA": 0.9908, "IDSW": 533, "FPS": 39.9, "ML": 12.2, "MT": 83.7, "GFLOPS": 65.96, "Count": 4112, "Time": "4.50h"},
        "Vanilla-YOLOv8m": {"IDF1": 0.7545, "MOTA": 0.8689, "IDSW": 547, "FPS": 42.94, "ML": 12.0, "MT": 83.7, "GFLOPS": 59.92, "Count": 4097, "Time": "2.48h"},
    },
    "Gate2": {
        "Vanilla-YOLOv8m": {"IDF1": 0.8401, "MOTA": 0.9791, "IDSW": 508, "FPS": 56.5, "ML": 10.4, "MT": 82.5, "GFLOPS": 33.59, "Count": 3641, "Time": "2h27m"},
        "DCNv3-Pan": {"IDF1": 0.8373, "MOTA": 0.9795, "IDSW": 660, "FPS": 49.20, "ML": 11.9, "MT": 81.4, "GFLOPS": 42.84, "Count": 3652, "Time": "2h55m"},
        "DCNv3-Full": {"IDF1": 0.8511, "MOTA": 0.9939, "IDSW": 470, "FPS": 54.3, "ML": 8.9, "MT": 84.1, "GFLOPS": 36.77, "Count": 3639, "Time": "3h15m"},
        "DCNv3-FPN": {"IDF1": 0.8413, "MOTA": 0.9947, "IDSW": 557, "FPS": 58.3, "ML": 9.4, "MT": 83.6, "GFLOPS": 37.10, "Count": 3673, "Time": "3h04m"},
        "DCNv3-Backbone": {"IDF1": 0.8362, "MOTA": 0.9974, "IDSW": 546, "FPS": 61.6, "ML": 11.4, "MT": 81.4, "GFLOPS": 37.91, "Count": 3690, "Time": "2h53m"},
        "DCNv2-Pan": {"IDF1": 0.8474, "MOTA": 0.8997, "IDSW": 547, "FPS": 62.9, "ML": 10.1, "MT": 83.6, "GFLOPS": 36.25, "Count": 3710, "Time": "2h53m"},
        "DCNv2-Full": {"IDF1": 0.8399, "MOTA": 0.9879, "IDSW": 486, "FPS": 60.4, "ML": 11.5, "MT": 81.5, "GFLOPS": 35.29, "Count": 3633, "Time": "2h59m"},
        "DCNv2-FPN": {"IDF1": 0.8324, "MOTA": 0.8870, "IDSW": 601, "FPS": 62.1, "ML": 9.7, "MT": 83.8, "GFLOPS": 39.73, "Count": 3668, "Time": "2h55m"},
        "DCNv2-Backbone": {"IDF1": 0.8319, "MOTA": 0.9257, "IDSW": 521, "FPS": 64.5, "ML": 9.8, "MT": 82.8, "GFLOPS": 38.07, "Count": 3737, "Time": "2h47m"},
    },
    "Gate2.9": {
        "DCNv2-Backbone": {"IDF1": 0.9550, "MOTA": 0.9301, "IDSW": 882, "FPS": 62.5, "ML": 12.5, "MT": 74.1, "GFLOPS": 46.13, "Count": 5194, "Time": "2.85h"},
        "DCNv2-FPN": {"IDF1": 0.9652, "MOTA": 0.9714, "IDSW": 903, "FPS": 60.0, "ML": 12.0, "MT": 75.7, "GFLOPS": 46.30, "Count": 5147, "Time": "2.98h"},
        "DCNv2-Full": {"IDF1": 0.9553, "MOTA": 0.8830, "IDSW": 771, "FPS": 58.3, "ML": 15.6, "MT": 70.7, "GFLOPS": 43.94, "Count": 5234, "Time": "3.06h"},
        "DCNv2-Pan": {"IDF1": 0.9607, "MOTA": 0.8911, "IDSW": 881, "FPS": 60.5, "ML": 13.2, "MT": 74.4, "GFLOPS": 50.67, "Count": 5225, "Time": "2.95h"},
        "DCNv3-Backbone": {"IDF1": 0.9675, "MOTA": 0.8666, "IDSW": 859, "FPS": 60.1, "ML": 13.9, "MT": 71.8, "GFLOPS": 47.21, "Count": 5402, "Time": "2.96h"},
        "DCNv3-FPN": {"IDF1": 0.9688, "MOTA": 0.9072, "IDSW": 877, "FPS": 56.8, "ML": 15.9, "MT": 71.2, "GFLOPS": 46.53, "Count": 5307, "Time": "3.15h"},
        "DCNv3-Full": {"IDF1": 0.9625, "MOTA": 0.9258, "IDSW": 831, "FPS": 53.2, "ML": 15.6, "MT": 71.2, "GFLOPS": 49.31, "Count": 5183, "Time": "3.36h"},
        "DCNv3-Pan": {"IDF1": 0.9667, "MOTA": 0.8331, "IDSW": 902, "FPS": 59.6, "ML": 14.8, "MT": 72.7, "GFLOPS": 47.51, "Count": 5413, "Time": "3.0h"},
        "Vanilla-YOLOv8m": {"IDF1": 0.9504, "MOTA": 0.8475, "IDSW": 783, "FPS": 60.0, "ML": 13.3, "MT": 72.2, "GFLOPS": 41.7, "Count": 5130, "Time": "2.53h"},
    },
    "Gate3": {
        "DCNv2-Backbone": {"IDF1": 0.9196, "MOTA": 0.7182, "IDSW": 642, "FPS": 64.4, "ML": 11.8, "MT": 84.5, "GFLOPS": 49.65, "Count": 6550, "Time": "2.78h"},
        "DCNv2-FPN": {"IDF1": 0.9217, "MOTA": 0.7186, "IDSW": 499, "FPS": 61.8, "ML": 11.8, "MT": 84.2, "GFLOPS": 53.68, "Count": 6347, "Time": "2.9h"},
        "DCNv2-Full": {"IDF1": 0.9375, "MOTA": 0.7056, "IDSW": 492, "FPS": 60.3, "ML": 12.1, "MT": 83.6, "GFLOPS": 42.22, "Count": 6396, "Time": "2.96h"},
        "DCNv2-Pan": {"IDF1": 0.9195, "MOTA": 0.7179, "IDSW": 605, "FPS": 62.0, "ML": 11.9, "MT": 84.5, "GFLOPS": 48.73, "Count": 6295, "Time": "2.9h"},
        "DCNv3-Backbone": {"IDF1": 0.9212, "MOTA": 0.7045, "IDSW": 658, "FPS": 61.6, "ML": 12.4, "MT": 83.1, "GFLOPS": 49.87, "Count": 6195, "Time": "2.91h"},
        "DCNv3-FPN": {"IDF1": 0.9387, "MOTA": 0.7063, "IDSW": 611, "FPS": 58.6, "ML": 12.4, "MT": 83.4, "GFLOPS": 51.25, "Count": 6490, "Time": "3.06h"},
        "DCNv3-Full": {"IDF1": 0.9355, "MOTA": 0.7055, "IDSW": 493, "FPS": 54.8, "ML": 11.9, "MT": 83.8, "GFLOPS": 43.47, "Count": 6396, "Time": "3.28h"},
        "DCNv3-Pan": {"IDF1": 0.9306, "MOTA": 0.7040, "IDSW": 680, "FPS": 61.2, "ML": 12.2, "MT": 83.7, "GFLOPS": 46.05, "Count": 6346, "Time": "2.93h"},
        "Vanilla-YOLOv8m": {"IDF1": 0.9209, "MOTA": 0.7060, "IDSW": 533, "FPS": 63.22, "ML": 12.0, "MT": 83.7, "GFLOPS": 48.66, "Count": 6426, "Time": "2.45h"},
    },
    "Gate3.2": {
        "DCNv3-FPN": {"IDF1": 0.8792, "MOTA": 0.9528, "IDSW": 442, "FPS": 57.9, "ML": 11.0, "MT": 70.5, "GFLOPS": 54.4, "Count": 2365, "Time": "3h06m"},
        "DCNv3-Backbone": {"IDF1": 0.8774, "MOTA": 0.8876, "IDSW": 378, "FPS": 60.2, "ML": 12.5, "MT": 65.8, "GFLOPS": 53.7, "Count": 2365, "Time": "2h59m"},
        "Vanilla-YOLOv8m": {"IDF1": 0.8767, "MOTA": 0.9334, "IDSW": 467, "FPS": 73.39, "ML": 13.4, "MT": 69.4, "GFLOPS": 48.5, "Count": 2351, "Time": "2h27m"},
        "DCNv2-Full": {"IDF1": 0.8724, "MOTA": 0.9416, "IDSW": 368, "FPS": 60.45, "ML": 12.4, "MT": 72.4, "GFLOPS": 51.2, "Count": 2292, "Time": "2h50m"},
        "DCNv3-Full": {"IDF1": 0.8637, "MOTA": 0.9528, "IDSW": 456, "FPS": 53.0, "ML": 11.4, "MT": 68.4, "GFLOPS": 52.6, "Count": 2388, "Time": "3h23m"},
        "DCNv2-Backbone": {"IDF1": 0.8619, "MOTA": 0.9801, "IDSW": 543, "FPS": 65.27, "ML": 10.4, "MT": 67.6, "GFLOPS": 53.8, "Count": 2362, "Time": "2h48m"},
        "DCNv3-Pan": {"IDF1": 0.8572, "MOTA": 0.9908, "IDSW": 367, "FPS": 60.0, "ML": 11.2, "MT": 68.0, "GFLOPS": 54.2, "Count": 2379, "Time": "3h16m"},
        "DCNv2-Pan": {"IDF1": 0.8503, "MOTA": 0.9693, "IDSW": 453, "FPS": 63.25, "ML": 13.0, "MT": 69.0, "GFLOPS": 53.1, "Count": 2610, "Time": "2h39m"},
        "DCNv2-FPN": {"IDF1": 0.8453, "MOTA": 0.9747, "IDSW": 418, "FPS": 62.57, "ML": 12.5, "MT": 65.0, "GFLOPS": 53.8, "Count": 2601, "Time": "2h53m"},
    }
}

# Architecture descriptions
arch_descriptions = {
    "DCNv2-Backbone": "YOLOv8m + DCNv2 Deformable Convolution (Backbone Only)",
    "DCNv2-FPN": "YOLOv8m + DCNv2 Deformable Convolution (FPN Neck)",
    "DCNv2-Full": "YOLOv8m + DCNv2 Deformable Convolution (Full Integration)",
    "DCNv2-Pan": "YOLOv8m + DCNv2 Deformable Convolution (PAN Neck)",
    "DCNv3-Backbone": "YOLOv8m + DCNv3 Deformable Convolution (Backbone Only)",
    "DCNv3-FPN": "YOLOv8m + DCNv3 Deformable Convolution (FPN Neck)",
    "DCNv3-Full": "YOLOv8m + DCNv3 Deformable Convolution (Full Integration)",
    "DCNv3-Pan": "YOLOv8m + DCNv3 Deformable Convolution (PAN Neck)",
    "Vanilla-YOLOv8m": "Baseline YOLOv8m (No Deformable Convolutions)"
}

def create_report(gate, model, metrics):
    """Create individual performance report."""

    report = f"""================================================================
TRACKING PERFORMANCE REPORT
================================================================

Gate:           {gate.replace('Gate', '')}
Model:          {model}
Architecture:   {arch_descriptions[model]}

================================================================
PERFORMANCE METRICS
================================================================

Identity Tracking:
  IDF1 (Identity F1 Score):        {metrics['IDF1']:.4f}
  ID Switches (IDSW):              {metrics['IDSW']}

Tracking Accuracy:
  MOTA (Multi-Object Tracking):    {metrics['MOTA']:.4f}
  Mostly Tracked (MT):             {metrics['MT']:.1f}%
  Mostly Lost (ML):                {metrics['ML']:.1f}%

Computational Performance:
  FPS (Frames Per Second):         {metrics['FPS']:.2f}
  GFLOPs:                          {metrics['GFLOPS']:.2f}
  Processing Time:                 {metrics['Time']}

Counting Results:
  Total Vehicle Count:             {metrics['Count']}

================================================================
GATE CONTEXT - {gate.replace('Gate', '')}
================================================================

"""

    # Gate-specific context
    gate_contexts = {
        "Gate3.5": """Characteristics:
  - High-speed processing scenario
  - Average count: ~4,110 vehicles
  - Processing time range: 2.48h - 6.48h
  - Excellent tracking accuracy overall
  - Moderate traffic density
""",
        "Gate2": """Characteristics:
  - Extended monitoring period (2-3 hours)
  - Average count: ~3,670 vehicles
  - Exceptional MOTA scores (>0.88 all models)
  - Long-duration tracking stress test
  - Moderate traffic volume
""",
        "Gate2.9": """Characteristics:
  - High IDF1 scores (>0.95 all models)
  - Average count: ~5,250 vehicles
  - Higher ID switch rates (771-903)
  - Variable traffic patterns
  - Challenging tracking conditions
""",
        "Gate3": """Characteristics:
  - Highest vehicle volume (~6,400 avg)
  - Challenging MOTA scores (~0.70-0.72)
  - Difficult tracking environment
  - Occlusions and complex traffic patterns
  - Extended processing duration
""",
        "Gate3.2": """Characteristics:
  - Long monitoring period (2.5-3.5 hours)
  - Lower volume (~2,400 vehicles)
  - Good MOTA performance (>0.88 most models)
  - Lower traffic density
  - Favorable tracking conditions
"""
    }

    report += gate_contexts.get(gate, "")

    # Rankings
    gate_data = data[gate]
    models_list = list(gate_data.keys())

    # Sort by different metrics
    idf1_sorted = sorted(models_list, key=lambda m: gate_data[m]['IDF1'], reverse=True)
    mota_sorted = sorted(models_list, key=lambda m: gate_data[m]['MOTA'], reverse=True)
    idsw_sorted = sorted(models_list, key=lambda m: gate_data[m]['IDSW'])
    fps_sorted = sorted(models_list, key=lambda m: gate_data[m]['FPS'], reverse=True)

    idf1_rank = idf1_sorted.index(model) + 1
    mota_rank = mota_sorted.index(model) + 1
    idsw_rank = idsw_sorted.index(model) + 1
    fps_rank = fps_sorted.index(model) + 1

    report += f"""
================================================================
RANKING (Among {len(models_list)} models on {gate.replace('Gate', 'Gate ')})
================================================================

  IDF1 (Identity):     #{idf1_rank} out of {len(models_list)} ({metrics['IDF1']:.4f})
  MOTA (Accuracy):     #{mota_rank} out of {len(models_list)} ({metrics['MOTA']:.4f})
  IDSW (Lower Better): #{idsw_rank} out of {len(models_list)} ({metrics['IDSW']} switches)
  FPS (Speed):         #{fps_rank} out of {len(models_list)} ({metrics['FPS']:.2f})
  MT% (Coverage):      {metrics['MT']:.1f}%
  ML% (Loss):          {metrics['ML']:.1f}%

"""

    # Performance assessment
    report += """================================================================
PERFORMANCE ASSESSMENT
================================================================

"""

    # Determine strengths and weaknesses
    strengths = []
    weaknesses = []

    if idf1_rank <= 3:
        strengths.append(f"✓ Excellent identity tracking (IDF1 rank: #{idf1_rank})")
    elif idf1_rank >= 7:
        weaknesses.append(f"✗ Poor identity tracking (IDF1 rank: #{idf1_rank})")

    if mota_rank <= 3:
        strengths.append(f"✓ High tracking accuracy (MOTA rank: #{mota_rank})")
    elif mota_rank >= 7:
        weaknesses.append(f"✗ Lower tracking accuracy (MOTA rank: #{mota_rank})")

    if idsw_rank <= 3:
        strengths.append(f"✓ Minimal ID switches ({metrics['IDSW']} - rank #{idsw_rank})")
    elif idsw_rank >= 7:
        weaknesses.append(f"✗ High ID switches ({metrics['IDSW']} - rank #{idsw_rank})")

    if fps_rank <= 3:
        strengths.append(f"✓ Fast inference speed ({metrics['FPS']:.2f} FPS - rank #{fps_rank})")
    elif fps_rank >= 7:
        weaknesses.append(f"✗ Slow inference speed ({metrics['FPS']:.2f} FPS - rank #{fps_rank})")

    if metrics['MT'] >= 80:
        strengths.append(f"✓ High track coverage (MT: {metrics['MT']:.1f}%)")

    if metrics['ML'] <= 10:
        strengths.append(f"✓ Low track loss (ML: {metrics['ML']:.1f}%)")
    elif metrics['ML'] >= 13:
        weaknesses.append(f"✗ Higher track loss (ML: {metrics['ML']:.1f}%)")

    report += "STRENGTHS:\n"
    for s in strengths[:5]:
        report += f"  {s}\n"

    report += "\nWEAKNESSES:\n"
    if weaknesses:
        for w in weaknesses[:5]:
            report += f"  {w}\n"
    else:
        report += "  None significant\n"

    # Recommendations
    report += f"""
================================================================
USE CASE RECOMMENDATIONS
================================================================

"""

    if fps_rank <= 3 and idf1_rank <= 5:
        report += "RECOMMENDED FOR:\n"
        report += "  → Real-time traffic monitoring (high FPS + good accuracy)\n"
        report += "  → Production deployment requiring speed + quality\n"
        report += "  → Multi-stream processing scenarios\n"
    elif idf1_rank <= 2:
        report += "RECOMMENDED FOR:\n"
        report += "  → Maximum identity tracking quality\n"
        report += "  → Long-term trajectory analysis\n"
        report += "  → Re-identification applications\n"
        report += "  → Forensic video analysis\n"
    elif fps_rank <= 2:
        report += "RECOMMENDED FOR:\n"
        report += "  → Real-time applications requiring maximum speed\n"
        report += "  → Resource-constrained environments\n"
        report += "  → Live monitoring dashboards\n"
    elif idsw_rank <= 2:
        report += "RECOMMENDED FOR:\n"
        report += "  → Applications requiring minimal ID confusion\n"
        report += "  → Long-duration tracking scenarios\n"
        report += "  → Behavior pattern analysis\n"
    else:
        report += "SUITABLE FOR:\n"
        report += "  → General-purpose tracking applications\n"
        report += "  → Balanced speed/accuracy requirements\n"

    if idf1_rank >= 7 or mota_rank >= 7:
        report += "\nNOT RECOMMENDED FOR:\n"
        if idf1_rank >= 7:
            report += "  → Identity-critical applications\n"
            report += "  → Long-term trajectory tracking\n"
        if mota_rank >= 7:
            report += "  → High-accuracy detection requirements\n"
        if fps_rank >= 7:
            report += "  → Real-time processing needs\n"

    # Comparison with best model
    best_idf1_model = idf1_sorted[0]
    if model != best_idf1_model:
        best_metrics = gate_data[best_idf1_model]
        report += f"""
================================================================
COMPARISON WITH BEST MODEL ({best_idf1_model})
================================================================

IDF1:     {metrics['IDF1']:.4f} vs {best_metrics['IDF1']:.4f} ({metrics['IDF1'] - best_metrics['IDF1']:+.4f})
MOTA:     {metrics['MOTA']:.4f} vs {best_metrics['MOTA']:.4f} ({metrics['MOTA'] - best_metrics['MOTA']:+.4f})
IDSW:     {metrics['IDSW']} vs {best_metrics['IDSW']} ({metrics['IDSW'] - best_metrics['IDSW']:+d})
FPS:      {metrics['FPS']:.2f} vs {best_metrics['FPS']:.2f} ({metrics['FPS'] - best_metrics['FPS']:+.2f})

"""

    report += """================================================================
CONCLUSION
================================================================

"""

    # Overall assessment
    avg_rank = (idf1_rank + mota_rank + idsw_rank + fps_rank) / 4

    if avg_rank <= 2.5:
        assessment = "EXCELLENT - Top tier performance across metrics"
    elif avg_rank <= 4:
        assessment = "GOOD - Strong performance overall"
    elif avg_rank <= 6:
        assessment = "MODERATE - Acceptable for general use"
    else:
        assessment = "LIMITED - Consider alternatives"

    report += f"Overall Performance: {assessment}\n"
    report += f"Average Rank: #{avg_rank:.1f} out of {len(models_list)}\n\n"

    if idf1_rank == 1:
        report += f"🏆 BEST identity tracking on {gate.replace('Gate', 'Gate ')}\n"
    if mota_rank == 1:
        report += f"🏆 BEST tracking accuracy on {gate.replace('Gate', 'Gate ')}\n"
    if idsw_rank == 1:
        report += f"🏆 FEWEST ID switches on {gate.replace('Gate', 'Gate ')}\n"
    if fps_rank == 1:
        report += f"🏆 FASTEST inference on {gate.replace('Gate', 'Gate ')}\n"

    report += f"""
================================================================
Report Generated: March 2026
Dataset: {gate.replace('Gate', 'Gate ')} Traffic Monitoring
Architecture: {model}
Total Models Tested: {len(models_list)}
================================================================
"""

    return report

# Create output directory
output_dir = "model_reports"
os.makedirs(output_dir, exist_ok=True)

# Generate all reports
count = 0
for gate, models in data.items():
    for model, metrics in models.items():
        filename = f"{gate}_{model}.txt"
        filepath = os.path.join(output_dir, filename)

        report = create_report(gate, model, metrics)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        count += 1
        print(f"Created: {filename}")

print(f"\n✓ Successfully created {count} model reports in '{output_dir}/' directory")
