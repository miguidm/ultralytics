#!/usr/bin/env python3
"""
Compare SORT vs ByteTrack Performance
Evaluate both trackers with MOTA and compare results
"""

import os
import sys
import subprocess

def run_sort_tracker():
    """Run SORT tracker on all models"""
    print("="*70)
    print("STEP 1: Running SORT Tracker on DCNv2 Models")
    print("="*70)

    result = subprocess.run(
        ["python3", "run_sort_tracker_gate3.py"],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print("❌ Error running SORT tracker")
        return False

    return True


def evaluate_sort_with_mota():
    """Evaluate SORT results with MOTA"""
    print("\n" + "="*70)
    print("STEP 2: Evaluating SORT Tracker with MOTA")
    print("="*70)

    # Import the evaluation function
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from final_mota_evaluation import evaluate_tracking

    gt_file = "gate3_tracking_ground_truth/gt.txt"
    predictions_base_dir = "gate3_sort_results_dcnv2"

    if not os.path.exists(gt_file):
        print(f"❌ Error: Ground truth not found: {gt_file}")
        return []

    models = ['DCNv2-Full', 'DCNv2-FPN', 'DCNv2-Pan', 'DCNv2-LIU']
    sort_results = []

    for model_name in models:
        pred_file = os.path.join(predictions_base_dir, model_name, "Gate3_Oct7_predictions.txt")

        if not os.path.exists(pred_file):
            print(f"\n⚠ Skipping {model_name}: predictions not found")
            continue

        result = evaluate_tracking(gt_file, pred_file, f"{model_name}-SORT")
        if result:
            sort_results.append(result)

    return sort_results


def load_bytetrack_results():
    """Load ByteTrack results from previous evaluation"""
    results_file = "gate3_final_mota_results/final_mota_summary.txt"

    if not os.path.exists(results_file):
        print(f"⚠ ByteTrack results not found: {results_file}")
        return []

    # Parse results from file
    bytetrack_results = []

    with open(results_file, 'r') as f:
        lines = f.readlines()

        # Find data lines (after header)
        data_started = False
        for line in lines:
            if line.startswith('-'):
                data_started = True
                continue

            if data_started and line.strip() and not line.startswith('Best'):
                parts = line.split()
                if len(parts) >= 12:
                    model = parts[0]
                    mota = float(parts[1])
                    motp = float(parts[2])
                    idf1 = float(parts[3])
                    precision = float(parts[4])
                    recall = float(parts[5])
                    idsw = int(parts[6])
                    fp = int(parts[7])
                    fn = int(parts[8])
                    mt = int(parts[9])
                    pt = int(parts[10])
                    ml = int(parts[11])

                    bytetrack_results.append({
                        'model': model,
                        'tracker': 'ByteTrack',
                        'mota': mota,
                        'motp': motp,
                        'idf1': idf1,
                        'precision': precision,
                        'recall': recall,
                        'num_switches': idsw,
                        'num_false_positives': fp,
                        'num_misses': fn,
                        'mostly_tracked': mt,
                        'partially_tracked': pt,
                        'mostly_lost': ml
                    })

    return bytetrack_results


def compare_results(sort_results, bytetrack_results):
    """Compare SORT vs ByteTrack results"""

    print("\n" + "="*90)
    print("COMPARISON: SORT vs ByteTrack")
    print("="*90)

    if not sort_results:
        print("❌ No SORT results to compare")
        return

    if not bytetrack_results:
        print("❌ No ByteTrack results to compare")
        return

    # Combine results
    all_results = []

    # Add SORT results
    for r in sort_results:
        model_base = r['model'].replace('-SORT', '')
        all_results.append({
            'model': model_base,
            'tracker': 'SORT',
            **r
        })

    # Add ByteTrack results
    for r in bytetrack_results:
        all_results.append({
            'model': r['model'],
            'tracker': 'ByteTrack',
            **r
        })

    # Print comparison table
    print(f"\n{'Model':<15} {'Tracker':<12} {'MOTA':<8} {'IDF1':<8} {'Precision':<10} {'Recall':<8} {'IDSW':<6}")
    print("-"*75)

    # Group by model
    models = sorted(set(r['model'] for r in all_results))

    for model in models:
        model_results = [r for r in all_results if r['model'] == model]
        model_results.sort(key=lambda x: x['tracker'])

        for r in model_results:
            tracker_symbol = "🟢" if r['tracker'] == 'SORT' else "🔵"
            print(f"{r['model']:<15} {tracker_symbol} {r['tracker']:<10} {r['mota']:<8.4f} {r['idf1']:<8.4f} "
                  f"{r['precision']:<10.4f} {r['recall']:<8.4f} {r['num_switches']:<6}")

    # Calculate average improvements
    print("\n" + "-"*75)
    print("Average Performance:")
    print("-"*75)

    sort_avg_mota = sum(r['mota'] for r in sort_results) / len(sort_results)
    sort_avg_idf1 = sum(r['idf1'] for r in sort_results) / len(sort_results)
    sort_avg_idsw = sum(r['num_switches'] for r in sort_results) / len(sort_results)

    bt_avg_mota = sum(r['mota'] for r in bytetrack_results) / len(bytetrack_results)
    bt_avg_idf1 = sum(r['idf1'] for r in bytetrack_results) / len(bytetrack_results)
    bt_avg_idsw = sum(r['num_switches'] for r in bytetrack_results) / len(bytetrack_results)

    print(f"{'Tracker':<15} {'Avg MOTA':<12} {'Avg IDF1':<12} {'Avg IDSW':<12}")
    print(f"SORT            {sort_avg_mota:<12.4f} {sort_avg_idf1:<12.4f} {sort_avg_idsw:<12.1f}")
    print(f"ByteTrack       {bt_avg_mota:<12.4f} {bt_avg_idf1:<12.4f} {bt_avg_idsw:<12.1f}")

    # Determine winner
    print("\n" + "-"*75)
    print("Winner:")
    print("-"*75)

    if sort_avg_mota > bt_avg_mota:
        diff = sort_avg_mota - bt_avg_mota
        print(f"🏆 SORT wins on MOTA ({sort_avg_mota:.4f} vs {bt_avg_mota:.4f}, +{diff:.4f})")
    elif bt_avg_mota > sort_avg_mota:
        diff = bt_avg_mota - sort_avg_mota
        print(f"🏆 ByteTrack wins on MOTA ({bt_avg_mota:.4f} vs {sort_avg_mota:.4f}, +{diff:.4f})")
    else:
        print("🤝 Tie on MOTA")

    if sort_avg_idf1 > bt_avg_idf1:
        diff = sort_avg_idf1 - bt_avg_idf1
        print(f"🏆 SORT wins on IDF1 ({sort_avg_idf1:.4f} vs {bt_avg_idf1:.4f}, +{diff:.4f})")
    elif bt_avg_idf1 > sort_avg_idf1:
        diff = bt_avg_idf1 - sort_avg_idf1
        print(f"🏆 ByteTrack wins on IDF1 ({bt_avg_idf1:.4f} vs {sort_avg_idf1:.4f}, +{diff:.4f})")
    else:
        print("🤝 Tie on IDF1")

    if sort_avg_idsw < bt_avg_idsw:
        diff = bt_avg_idsw - sort_avg_idsw
        print(f"🏆 SORT has fewer ID switches ({sort_avg_idsw:.1f} vs {bt_avg_idsw:.1f}, -{diff:.1f})")
    elif bt_avg_idsw < sort_avg_idsw:
        diff = sort_avg_idsw - bt_avg_idsw
        print(f"🏆 ByteTrack has fewer ID switches ({bt_avg_idsw:.1f} vs {sort_avg_idsw:.1f}, -{diff:.1f})")
    else:
        print("🤝 Tie on ID switches")

    # Save comparison
    output_dir = "gate3_tracker_comparison"
    os.makedirs(output_dir, exist_ok=True)
    comparison_file = os.path.join(output_dir, "sort_vs_bytetrack.txt")

    with open(comparison_file, 'w') as f:
        f.write("SORT vs ByteTrack Comparison - Gate3_Oct7\n")
        f.write("="*90 + "\n\n")
        f.write(f"{'Model':<15} {'Tracker':<12} {'MOTA':<8} {'IDF1':<8} {'Precision':<10} {'Recall':<8} {'IDSW':<6} {'FP':<8} {'FN':<8}\n")
        f.write("-"*100 + "\n")

        for model in models:
            model_results = [r for r in all_results if r['model'] == model]
            model_results.sort(key=lambda x: x['tracker'])

            for r in model_results:
                f.write(f"{r['model']:<15} {r['tracker']:<12} {r['mota']:<8.4f} {r['idf1']:<8.4f} "
                       f"{r['precision']:<10.4f} {r['recall']:<8.4f} {r['num_switches']:<6} "
                       f"{r['num_false_positives']:<8} {r['num_misses']:<8}\n")

        f.write("\n" + "-"*100 + "\n")
        f.write("Average Performance:\n")
        f.write(f"  SORT:      MOTA={sort_avg_mota:.4f}, IDF1={sort_avg_idf1:.4f}, IDSW={sort_avg_idsw:.1f}\n")
        f.write(f"  ByteTrack: MOTA={bt_avg_mota:.4f}, IDF1={bt_avg_idf1:.4f}, IDSW={bt_avg_idsw:.1f}\n")

    print(f"\n✓ Comparison saved: {comparison_file}")
    print("\n" + "="*90 + "\n")


def main():
    """Main comparison function"""

    print("="*90)
    print("SORT vs ByteTrack Comparison - Gate3 DCNv2 Models")
    print("="*90)
    print()

    # Step 1: Run SORT tracker
    if not run_sort_tracker():
        return

    # Step 2: Evaluate SORT with MOTA
    sort_results = evaluate_sort_with_mota()

    # Step 3: Load ByteTrack results
    bytetrack_results = load_bytetrack_results()

    # Step 4: Compare
    compare_results(sort_results, bytetrack_results)


if __name__ == "__main__":
    main()
