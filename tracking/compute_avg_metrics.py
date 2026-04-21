#!/usr/bin/env python3
"""Compute average MOTA/IDF1 from 1hr and 12hr tracking data, per gate per model."""

import csv
import os

# Gate mapping: 1hr name -> 12hr name
GATE_MAP = {
    'Gate2': 'g2',
    'Gate2.9': 'g29',
    'Gate3': 'g3',
    'Gate3.5': 'g35',
}

# 1hr data from metrics_summary_complete
hr1_csv = '/media/mydrive/GitHub/ultralytics/tracking/metrics_summary_complete/tracking_metrics_complete.csv'

# 12hr data directories
hr12_base = '/media/mydrive/GitHub/ultralytics/tracking'
hr12_dirs = {
    'g2': {'dcnv2': 'counting_6to6_g2/dcnv2/summary.csv', 'dcnv3': 'counting_6to6_g2/dcnv3/summary.csv', 'vanilla': 'counting_6to6_g2/vanilla/summary.csv'},
    'g3': {'dcnv2': 'counting_6to6_g3/dcnv2/summary.csv', 'dcnv3': 'counting_6to6_g3/dcnv3/summary.csv', 'vanilla': 'counting_6to6_g3/vanilla/summary.csv'},
    'g29': {'dcnv2': 'counting_6to6_g29/dcnv2/summary.csv', 'dcnv3': 'counting_6to6_g29/dcnv3/summary.csv', 'vanilla': 'counting_6to6_g29/vanilla/summary.csv'},
    'g35': {'dcnv2': 'counting_6to6_g35/dcnv2/summary.csv', 'dcnv3': 'counting_6to6_g35/dcnv3/summary.csv', 'vanilla': 'counting_6to6_g35/vanilla/summary.csv'},
}

# Parse 1hr data
hr1_data = {}  # {(model, gate_clean): {idf1, mota}}
with open(hr1_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        model = row['Model']
        gate_clean = row['Gate_Clean']
        if gate_clean in GATE_MAP:
            key = (model, gate_clean)
            hr1_data[key] = {
                'idf1': float(row['IDF1']),
                'mota': float(row['MOTA']),
            }

# Parse 12hr data (including FPS and Count)
hr12_data = {}  # {(model, gate): {idf1, mota, fps, count}}
for gate, type_csvs in hr12_dirs.items():
    for model_type, csv_path in type_csvs.items():
        full_path = os.path.join(hr12_base, csv_path)
        if not os.path.exists(full_path):
            continue
        with open(full_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row['Model']
                gate_1hr = [k for k, v in GATE_MAP.items() if v == gate][0]
                key = (model, gate_1hr)
                hr12_data[key] = {
                    'idf1': float(row['IDF1']),
                    'mota': float(row['MOTA']),
                    'fps': float(row.get('FPS', 0)),
                    'count': int(row.get('Count', 0)),
                }

# Compute and display per gate, per model
gates_order = ['Gate2', 'Gate2.9', 'Gate3', 'Gate3.5']
models_order = [
    'DCNv2-Full', 'DCNv2-FPN', 'DCNv2-Pan', 'DCNv2-Liu',
    'DCNv3-Full', 'DCNv3-FPN', 'DCNv3-Pan', 'DCNv3-Liu',
    'Vanilla-YOLOv8m',
]

output_lines = []

for gate in gates_order:
    gate_12hr = GATE_MAP[gate]
    output_lines.append(f"\n{'='*130}")
    output_lines.append(f"  {gate} (1hr) / 6to6_{gate_12hr} (12hr)")
    output_lines.append(f"{'='*130}")
    header = f"  {'Model':<18} {'MOTA_1hr':<10} {'MOTA_12hr':<11} {'MOTA_avg':<10} {'IDF1_1hr':<10} {'IDF1_12hr':<11} {'IDF1_avg':<10} {'FPS_12hr':<10} {'Count_12hr':<10}"
    output_lines.append(header)
    output_lines.append(f"  {'-'*120}")

    for model in models_order:
        key = (model, gate)
        h1 = hr1_data.get(key)
        h12 = hr12_data.get(key)

        mota_1hr = f"{h1['mota']:.4f}" if h1 else "N/A"
        idf1_1hr = f"{h1['idf1']:.4f}" if h1 else "N/A"
        mota_12hr = f"{h12['mota']:.4f}" if h12 else "N/A"
        idf1_12hr = f"{h12['idf1']:.4f}" if h12 else "N/A"
        fps_12hr = f"{h12['fps']:.2f}" if h12 else "N/A"
        count_12hr = f"{h12['count']}" if h12 else "N/A"

        if h1 and h12:
            mota_avg = f"{(h1['mota'] + h12['mota']) / 2:.4f}"
            idf1_avg = f"{(h1['idf1'] + h12['idf1']) / 2:.4f}"
        else:
            mota_avg = "N/A"
            idf1_avg = "N/A"

        row = f"  {model:<18} {mota_1hr:<10} {mota_12hr:<11} {mota_avg:<10} {idf1_1hr:<10} {idf1_12hr:<11} {idf1_avg:<10} {fps_12hr:<10} {count_12hr:<10}"
        output_lines.append(row)

# Overall average per model (across all gates where both 1hr and 12hr exist)
output_lines.append(f"\n{'='*130}")
output_lines.append(f"  OVERALL AVERAGE (across all gates with both 1hr and 12hr data)")
output_lines.append(f"{'='*130}")
header = f"  {'Model':<18} {'Avg MOTA':<12} {'Avg IDF1':<12} {'Avg FPS':<10} {'Total Count':<12} {'Gates Used'}"
output_lines.append(header)
output_lines.append(f"  {'-'*80}")

for model in models_order:
    mota_avgs = []
    idf1_avgs = []
    fps_vals = []
    total_count = 0
    gates_used = []
    for gate in gates_order:
        key = (model, gate)
        h1 = hr1_data.get(key)
        h12 = hr12_data.get(key)
        if h1 and h12:
            mota_avgs.append((h1['mota'] + h12['mota']) / 2)
            idf1_avgs.append((h1['idf1'] + h12['idf1']) / 2)
            fps_vals.append(h12['fps'])
            total_count += h12['count']
            gates_used.append(GATE_MAP[gate])

    if mota_avgs:
        overall_mota = sum(mota_avgs) / len(mota_avgs)
        overall_idf1 = sum(idf1_avgs) / len(idf1_avgs)
        avg_fps = sum(fps_vals) / len(fps_vals)
        row = f"  {model:<18} {overall_mota:<12.4f} {overall_idf1:<12.4f} {avg_fps:<10.2f} {total_count:<12} {','.join(gates_used)}"
    else:
        row = f"  {model:<18} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<12} none"
    output_lines.append(row)

# Save and print
output_text = '\n'.join(output_lines)
print(output_text)

output_file = '/media/mydrive/GitHub/ultralytics/tracking/avg_mota_idf1_1hr_12hr.txt'
with open(output_file, 'w') as f:
    f.write(output_text + '\n')

# Also save CSV
csv_file = '/media/mydrive/GitHub/ultralytics/tracking/avg_mota_idf1_1hr_12hr.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Gate', 'MOTA_1hr', 'MOTA_12hr', 'MOTA_avg', 'IDF1_1hr', 'IDF1_12hr', 'IDF1_avg', 'FPS_12hr', 'Count_12hr'])
    for gate in gates_order:
        for model in models_order:
            key = (model, gate)
            h1 = hr1_data.get(key)
            h12 = hr12_data.get(key)
            mota_1hr = f"{h1['mota']:.4f}" if h1 else ""
            idf1_1hr = f"{h1['idf1']:.4f}" if h1 else ""
            mota_12hr = f"{h12['mota']:.4f}" if h12 else ""
            idf1_12hr = f"{h12['idf1']:.4f}" if h12 else ""
            fps_12hr = f"{h12['fps']:.2f}" if h12 else ""
            count_12hr = f"{h12['count']}" if h12 else ""
            if h1 and h12:
                mota_avg = f"{(h1['mota'] + h12['mota']) / 2:.4f}"
                idf1_avg = f"{(h1['idf1'] + h12['idf1']) / 2:.4f}"
            else:
                mota_avg = ""
                idf1_avg = ""
            writer.writerow([model, f"{gate} / {GATE_MAP[gate]}", mota_1hr, mota_12hr, mota_avg, idf1_1hr, idf1_12hr, idf1_avg, fps_12hr, count_12hr])

print(f"\n\nSaved: {output_file}")
print(f"Saved: {csv_file}")
