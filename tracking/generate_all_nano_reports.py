#!/usr/bin/env python3
"""
Generate comprehensive performance reports for ALL YOLOv8n (nano) models:
- Vanilla YOLOv8n
- DCNv2 variants (Full, FPN, Pan, Liu)
- DCNv3 variants (Full, FPN, Pan, Liu)

Total: 9 models × 5 gates = 45 nano model reports
"""

import os

# Complete nano model data across all gates
nano_data = {
    "Gate2": {
        "Vanilla-YOLOv8n": {
            "IDF1": 0.9975, "MOTA": 0.9950, "IDSW": 1406, "FPS": 135.01, "ML": 0.6, "MT": 85.4,
            "Count": 3586, "Time": "1h20m", "Fragmentations": 4227
        },
        "DCNv2-Full": {
            "IDF1": 0.9975, "MOTA": 0.9950, "IDSW": 1340, "FPS": 103.40, "ML": 0.7, "MT": 84.3,
            "Count": 3578, "Time": "1h44m", "Fragmentations": 4432
        },
        "DCNv2-FPN": {
            "IDF1": 0.9976, "MOTA": 0.9952, "IDSW": 1305, "FPS": 107.45, "ML": 0.6, "MT": 85.0,
            "Count": 3555, "Time": "1h40m", "Fragmentations": 4272
        },
        "DCNv2-Pan": {
            "IDF1": 0.9976, "MOTA": 0.9952, "IDSW": 1313, "FPS": 107.47, "ML": 0.5, "MT": 86.0,
            "Count": 3591, "Time": "1h40m", "Fragmentations": 4156
        },
        "DCNv2-Liu": {
            "IDF1": 0.9975, "MOTA": 0.9950, "IDSW": 1619, "FPS": 114.78, "ML": 0.7, "MT": 85.6,
            "Count": 3610, "Time": "1h34m", "Fragmentations": 4334
        },
        "DCNv3-Full": {
            "IDF1": 0.9975, "MOTA": 0.9950, "IDSW": 1436, "FPS": 83.73, "ML": 0.6, "MT": 85.6,
            "Count": 3583, "Time": "2h09m", "Fragmentations": 4386
        },
        "DCNv3-FPN": {
            "IDF1": 0.9981, "MOTA": 0.9962, "IDSW": 1120, "FPS": 58.67, "ML": 0.7, "MT": 87.5,
            "Count": 3636, "Time": "3h04m", "Fragmentations": 3778
        },
        "DCNv3-Pan": {
            "IDF1": 0.9980, "MOTA": 0.9959, "IDSW": 1267, "FPS": 60.74, "ML": 0.8, "MT": 86.9,
            "Count": 3669, "Time": "2h58m", "Fragmentations": 4041
        },
        "DCNv3-Liu": {
            "IDF1": 0.9975, "MOTA": 0.9950, "IDSW": 1526, "FPS": 100.19, "ML": 0.6, "MT": 85.7,
            "Count": 3587, "Time": "1h48m", "Fragmentations": 4244
        }
    },
    "Gate3": {
        "Vanilla-YOLOv8n": {
            "IDF1": 0.9984, "MOTA": 0.9968, "IDSW": 1006, "FPS": 134.28, "ML": 0.2, "MT": 90.4,
            "Count": 5723, "Time": "1h10m", "Fragmentations": 4211
        },
        "DCNv2-Full": {
            "IDF1": 0.9987, "MOTA": 0.9973, "IDSW": 643, "FPS": 102.06, "ML": 0.2, "MT": 91.5,
            "Count": 5743, "Time": "1h32m", "Fragmentations": 3754
        },
        "DCNv2-FPN": {
            "IDF1": 0.9985, "MOTA": 0.9969, "IDSW": 1312, "FPS": 107.18, "ML": 0.2, "MT": 92.6,
            "Count": 5777, "Time": "1h27m", "Fragmentations": 3681
        },
        "DCNv2-Pan": {
            "IDF1": 0.9980, "MOTA": 0.9961, "IDSW": 1427, "FPS": 106.02, "ML": 0.3, "MT": 90.0,
            "Count": 5809, "Time": "1h28m", "Fragmentations": 5172
        },
        "DCNv2-Liu": {
            "IDF1": 0.9981, "MOTA": 0.9961, "IDSW": 1091, "FPS": 111.89, "ML": 0.2, "MT": 88.6,
            "Count": 5795, "Time": "1h24m", "Fragmentations": 5228
        },
        "DCNv3-Full": {
            "IDF1": 0.9987, "MOTA": 0.9973, "IDSW": 784, "FPS": 81.99, "ML": 0.2, "MT": 90.7,
            "Count": 5743, "Time": "1h54m", "Fragmentations": 3612
        },
        "DCNv3-FPN": {
            "IDF1": 0.9987, "MOTA": 0.9973, "IDSW": 844, "FPS": 57.75, "ML": 0.3, "MT": 91.2,
            "Count": 5763, "Time": "2h42m", "Fragmentations": 3809
        },
        "DCNv3-Pan": {
            "IDF1": 0.9987, "MOTA": 0.9974, "IDSW": 915, "FPS": 60.13, "ML": 0.2, "MT": 91.2,
            "Count": 5789, "Time": "2h36m", "Fragmentations": 3806
        },
        "DCNv3-Liu": {
            "IDF1": 0.9982, "MOTA": 0.9965, "IDSW": 863, "FPS": 99.42, "ML": 0.2, "MT": 88.7,
            "Count": 5781, "Time": "1h34m", "Fragmentations": 4862
        }
    },
    "Gate2.9": {
        "Vanilla-YOLOv8n": {
            "IDF1": 0.9993, "MOTA": 0.9985, "IDSW": 1590, "FPS": 125.56, "ML": 0.4, "MT": 88.0,
            "Count": 5128, "Time": "1h26m", "Fragmentations": 5957
        },
        "DCNv2-Full": {
            "IDF1": 0.9992, "MOTA": 0.9985, "IDSW": 1367, "FPS": 97.66, "ML": 0.6, "MT": 87.2,
            "Count": 5162, "Time": "1h51m", "Fragmentations": 6340
        },
        "DCNv2-FPN": {
            "IDF1": 0.9993, "MOTA": 0.9986, "IDSW": 1297, "FPS": 102.98, "ML": 0.5, "MT": 87.4,
            "Count": 5191, "Time": "1h45m", "Fragmentations": 6147
        },
        "DCNv2-Pan": {
            "IDF1": 0.9992, "MOTA": 0.9984, "IDSW": 1517, "FPS": 102.37, "ML": 0.4, "MT": 86.0,
            "Count": 5138, "Time": "1h46m", "Fragmentations": 6679
        },
        "DCNv2-Liu": {
            "IDF1": 0.9992, "MOTA": 0.9984, "IDSW": 1622, "FPS": 108.31, "ML": 0.4, "MT": 86.4,
            "Count": 5141, "Time": "1h40m", "Fragmentations": 6392
        },
        "DCNv3-Full": {
            "IDF1": 0.9991, "MOTA": 0.9982, "IDSW": 1918, "FPS": 79.83, "ML": 0.4, "MT": 85.5,
            "Count": 5173, "Time": "2h15m", "Fragmentations": 7348
        },
        "DCNv3-FPN": {
            "IDF1": 0.9994, "MOTA": 0.9988, "IDSW": 1634, "FPS": 56.59, "ML": 0.5, "MT": 89.1,
            "Count": 5151, "Time": "3h11m", "Fragmentations": 5336
        },
        "DCNv3-Pan": {
            "IDF1": 0.9994, "MOTA": 0.9988, "IDSW": 1666, "FPS": 58.68, "ML": 0.4, "MT": 89.0,
            "Count": 5157, "Time": "3h04m", "Fragmentations": 5464
        },
        "DCNv3-Liu": {
            "IDF1": 0.9993, "MOTA": 0.9985, "IDSW": 1457, "FPS": 94.34, "ML": 0.3, "MT": 87.5,
            "Count": 5136, "Time": "1h54m", "Fragmentations": 5786
        }
    },
    "Gate3.5": {
        "Vanilla-YOLOv8n": {
            "IDF1": 0.9989, "MOTA": 0.9977, "IDSW": 1796, "FPS": 127.62, "ML": 0.8, "MT": 86.6,
            "Count": 4090, "Time": "1h25m", "Fragmentations": 6695
        },
        "DCNv2-Full": {
            "IDF1": 0.9988, "MOTA": 0.9977, "IDSW": 2051, "FPS": 83.98, "ML": 0.6, "MT": 87.7,
            "Count": 4087, "Time": "2h09m", "Fragmentations": 6532
        },
        "DCNv2-FPN": {
            "IDF1": 0.9989, "MOTA": 0.9979, "IDSW": 1707, "FPS": 86.82, "ML": 0.9, "MT": 87.7,
            "Count": 4110, "Time": "2h04m", "Fragmentations": 6215
        },
        "DCNv2-Pan": {
            "IDF1": 0.9987, "MOTA": 0.9974, "IDSW": 2570, "FPS": 87.60, "ML": 0.8, "MT": 86.4,
            "Count": 4098, "Time": "2h03m", "Fragmentations": 7268
        },
        "DCNv2-Liu": {
            "IDF1": 0.9988, "MOTA": 0.9977, "IDSW": 2184, "FPS": 91.94, "ML": 0.6, "MT": 87.7,
            "Count": 4072, "Time": "1h57m", "Fragmentations": 6453
        },
        "DCNv3-Full": {
            "IDF1": 0.9988, "MOTA": 0.9977, "IDSW": 2324, "FPS": 80.14, "ML": 0.9, "MT": 86.4,
            "Count": 4113, "Time": "2h15m", "Fragmentations": 6540
        },
        "DCNv3-FPN": {
            "IDF1": 0.9991, "MOTA": 0.9982, "IDSW": 1757, "FPS": 56.43, "ML": 0.9, "MT": 88.6,
            "Count": 4104, "Time": "3h11m", "Fragmentations": 5225
        },
        "DCNv3-Pan": {
            "IDF1": 0.9990, "MOTA": 0.9981, "IDSW": 2026, "FPS": 58.18, "ML": 0.8, "MT": 87.5,
            "Count": 4105, "Time": "3h06m", "Fragmentations": 5726
        },
        "DCNv3-Liu": {
            "IDF1": 0.9989, "MOTA": 0.9978, "IDSW": 2105, "FPS": 95.42, "ML": 0.8, "MT": 87.7,
            "Count": 4096, "Time": "1h53m", "Fragmentations": 6260
        }
    },
    "Gate3.1": {
        "Vanilla-YOLOv8n": {
            "IDF1": 0.9989, "MOTA": 0.9977, "IDSW": 1796, "FPS": 126.90, "ML": 0.8, "MT": 86.6,
            "Count": 2308, "Time": "1h25m", "Fragmentations": 6695
        },
        "DCNv2-Full": {
            "IDF1": 0.9988, "MOTA": 0.9977, "IDSW": 2051, "FPS": 98.88, "ML": 0.6, "MT": 87.7,
            "Count": 2295, "Time": "1h49m", "Fragmentations": 6532
        },
        "DCNv2-FPN": {
            "IDF1": 0.9989, "MOTA": 0.9979, "IDSW": 1707, "FPS": 103.01, "ML": 0.9, "MT": 87.7,
            "Count": 2307, "Time": "1h45m", "Fragmentations": 6215
        },
        "DCNv2-Pan": {
            "IDF1": 0.9987, "MOTA": 0.9974, "IDSW": 2570, "FPS": 103.14, "ML": 0.8, "MT": 86.4,
            "Count": 2300, "Time": "1h45m", "Fragmentations": 7268
        },
        "DCNv2-Liu": {
            "IDF1": 0.9988, "MOTA": 0.9977, "IDSW": 2184, "FPS": 108.06, "ML": 0.6, "MT": 87.7,
            "Count": 2318, "Time": "1h40m", "Fragmentations": 6453
        },
        "DCNv3-Full": {
            "IDF1": 0.9988, "MOTA": 0.9977, "IDSW": 2324, "FPS": 79.78, "ML": 0.9, "MT": 86.4,
            "Count": 2317, "Time": "2h15m", "Fragmentations": 6540
        },
        "DCNv3-FPN": {
            "IDF1": 0.9991, "MOTA": 0.9982, "IDSW": 1757, "FPS": 56.65, "ML": 0.9, "MT": 88.6,
            "Count": 2350, "Time": "3h11m", "Fragmentations": 5225
        },
        "DCNv3-Pan": {
            "IDF1": 0.9990, "MOTA": 0.9981, "IDSW": 2026, "FPS": 58.90, "ML": 0.8, "MT": 87.5,
            "Count": 2351, "Time": "3h03m", "Fragmentations": 5726
        },
        "DCNv3-Liu": {
            "IDF1": 0.9989, "MOTA": 0.9978, "IDSW": 2105, "FPS": 95.45, "ML": 0.8, "MT": 87.7,
            "Count": 2292, "Time": "1h53m", "Fragmentations": 6260
        }
    }
}

# Model architecture descriptions
arch_descriptions = {
    "Vanilla-YOLOv8n": "Baseline YOLOv8n Nano (6.2M parameters)",
    "DCNv2-Full": "YOLOv8n + DCNv2 Full Integration",
    "DCNv2-FPN": "YOLOv8n + DCNv2 FPN Neck",
    "DCNv2-Pan": "YOLOv8n + DCNv2 PAN Neck",
    "DCNv2-Liu": "YOLOv8n + DCNv2 Liu Architecture",
    "DCNv3-Full": "YOLOv8n + DCNv3 Full Integration",
    "DCNv3-FPN": "YOLOv8n + DCNv3 FPN Neck",
    "DCNv3-Pan": "YOLOv8n + DCNv3 PAN Neck",
    "DCNv3-Liu": "YOLOv8n + DCNv3 Liu Architecture"
}

def create_nano_report(gate, model, metrics):
    """Create individual performance report for nano model."""

    # Determine model family
    if "DCNv2" in model:
        family = "DCNv2"
    elif "DCNv3" in model:
        family = "DCNv3"
    else:
        family = "Vanilla"

    report = f"""================================================================
YOLOv8n (NANO) TRACKING PERFORMANCE REPORT
================================================================

Gate:           {gate.replace('Gate', '')}
Model:          {model}
Architecture:   {arch_descriptions[model]}
Model Family:   {family}

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
  Fragmentations:                  {metrics['Fragmentations']}

Computational Performance:
  FPS (Frames Per Second):         {metrics['FPS']:.2f}
  Processing Time:                 {metrics['Time']}

Counting Results:
  Line Crossing Count:             {metrics['Count']}

================================================================
RANKING ON {gate.replace('Gate', 'GATE ')} (Among 9 Nano Models)
================================================================

"""

    # Get all models for this gate
    gate_models = nano_data[gate]
    all_models = list(gate_models.keys())

    # Rankings
    idf1_sorted = sorted(all_models, key=lambda m: gate_models[m]['IDF1'], reverse=True)
    mota_sorted = sorted(all_models, key=lambda m: gate_models[m]['MOTA'], reverse=True)
    idsw_sorted = sorted(all_models, key=lambda m: gate_models[m]['IDSW'])
    fps_sorted = sorted(all_models, key=lambda m: gate_models[m]['FPS'], reverse=True)

    idf1_rank = idf1_sorted.index(model) + 1
    mota_rank = mota_sorted.index(model) + 1
    idsw_rank = idsw_sorted.index(model) + 1
    fps_rank = fps_sorted.index(model) + 1

    report += f"""  IDF1 (Identity):     #{idf1_rank} out of 9 ({metrics['IDF1']:.4f})
  MOTA (Accuracy):     #{mota_rank} out of 9 ({metrics['MOTA']:.4f})
  IDSW (Lower Better): #{idsw_rank} out of 9 ({metrics['IDSW']} switches)
  FPS (Speed):         #{fps_rank} out of 9 ({metrics['FPS']:.2f})
  MT% (Coverage):      {metrics['MT']:.1f}%
  ML% (Loss):          {metrics['ML']:.1f}%

================================================================
PERFORMANCE ASSESSMENT
================================================================

"""

    # Strengths and weaknesses
    strengths = []
    weaknesses = []

    if idf1_rank <= 3:
        strengths.append(f"✓ Excellent identity tracking (IDF1 rank: #{idf1_rank})")
    if mota_rank <= 3:
        strengths.append(f"✓ Top tracking accuracy (MOTA rank: #{mota_rank})")
    if idsw_rank <= 3:
        strengths.append(f"✓ Minimal ID switches ({metrics['IDSW']} - rank #{idsw_rank})")
    if fps_rank <= 3:
        strengths.append(f"✓ Fast inference speed ({metrics['FPS']:.2f} FPS - rank #{fps_rank})")
    if metrics['FPS'] >= 100:
        strengths.append(f"✓ Ultra-fast real-time processing ({metrics['FPS']:.2f} FPS)")
    if metrics['MT'] >= 88:
        strengths.append(f"✓ Excellent track coverage (MT: {metrics['MT']:.1f}%)")
    if metrics['ML'] <= 0.5:
        strengths.append(f"✓ Minimal track loss (ML: {metrics['ML']:.1f}%)")

    if idf1_rank >= 7:
        weaknesses.append(f"✗ Lower identity tracking (IDF1 rank: #{idf1_rank})")
    if idsw_rank >= 7:
        weaknesses.append(f"✗ Higher ID switches ({metrics['IDSW']} - rank #{idsw_rank})")
    if fps_rank >= 7:
        weaknesses.append(f"✗ Slower inference (rank #{fps_rank})")

    report += "STRENGTHS:\n"
    for s in strengths[:6]:
        report += f"  {s}\n"

    report += "\nWEAKNESSES:\n"
    if weaknesses:
        for w in weaknesses:
            report += f"  {w}\n"
    else:
        report += "  None significant\n"

    # Comparison with best
    best_model = idf1_sorted[0]
    if model != best_model:
        best_metrics = gate_models[best_model]
        report += f"""
================================================================
COMPARISON WITH BEST MODEL ({best_model})
================================================================

IDF1:     {metrics['IDF1']:.4f} vs {best_metrics['IDF1']:.4f} ({metrics['IDF1'] - best_metrics['IDF1']:+.4f})
MOTA:     {metrics['MOTA']:.4f} vs {best_metrics['MOTA']:.4f} ({metrics['MOTA'] - best_metrics['MOTA']:+.4f})
IDSW:     {metrics['IDSW']} vs {best_metrics['IDSW']} ({metrics['IDSW'] - best_metrics['IDSW']:+d})
FPS:      {metrics['FPS']:.2f} vs {best_metrics['FPS']:.2f} ({metrics['FPS'] - best_metrics['FPS']:+.2f})

"""

    # Overall conclusion
    avg_rank = (idf1_rank + mota_rank + idsw_rank + fps_rank) / 4

    report += """================================================================
CONCLUSION
================================================================

"""

    if avg_rank <= 2.5:
        assessment = "EXCELLENT - Top tier performance"
    elif avg_rank <= 4:
        assessment = "GOOD - Strong performance overall"
    elif avg_rank <= 6:
        assessment = "MODERATE - Acceptable performance"
    else:
        assessment = "ACCEPTABLE - Consider top performers"

    report += f"Overall Performance: {assessment}\n"
    report += f"Average Rank: #{avg_rank:.1f} out of 9\n\n"

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
Model: {model} (Nano - 6.2M parameters)
Rank: #{avg_rank:.1f} out of 9 models
================================================================
"""

    return report

# Create output directory
output_dir = "all_nano_reports"
os.makedirs(output_dir, exist_ok=True)

# Generate all reports
count = 0
for gate, models in nano_data.items():
    for model, metrics in models.items():
        filename = f"{gate}_{model}.txt"
        filepath = os.path.join(output_dir, filename)

        report = create_nano_report(gate, model, metrics)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        count += 1
        print(f"Created: {filename}")

print(f"\nSuccessfully created {count} nano model reports in '{output_dir}/' directory")
print(f"Total: 9 models × 5 gates = {count} reports")
