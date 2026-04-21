#!/usr/bin/env python3
"""
Statistical significance analysis for tracking (MOTA, IDF1), IDSW/ClassSwitches, and detection (mAP50, Precision, Recall).

Approach:
  1. Friedman test       — are any models significantly different? (non-parametric repeated measures)
  2. Paired t-test       — parametric pairwise comparisons with Bonferroni correction
  3. Wilcoxon signed-rank — non-parametric pairwise comparisons with Bonferroni correction
  4. Cliff's delta       — effect size (not affected by n)
  5. Bootstrap 95% CI   — mean difference with confidence interval

Tracking repeated measure: 4 gates (g2, g29, g3, g35)
Detection repeated measure: 6 classes (car, motorcycle, tricycle, bus, van, truck)
"""

import csv
import os
import sys
import itertools
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# DATA SOURCES
# ─────────────────────────────────────────────
TRACKING_BASE = '/media/mydrive/GitHub/ultralytics/tracking'
OCCLUSION_BASE = '/media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/results'

GATE_MAP = {'Gate2': 'g2', 'Gate2.9': 'g29', 'Gate3': 'g3', 'Gate3.5': 'g35'}
GATES_1HR = list(GATE_MAP.keys())
GATES_12HR = list(GATE_MAP.values())
CLASSES = ['car', 'motorcycle', 'tricycle', 'bus', 'van', 'truck']

MODELS_ORDER = [
    'DCNv2-Full', 'DCNv2-FPN', 'DCNv2-Pan', 'DCNv2-Liu',
    'DCNv3-Full', 'DCNv3-FPN', 'DCNv3-Pan', 'DCNv3-Liu',
    'Vanilla-YOLOv8m',
]

MODEL_TYPE = {
    'DCNv2-Full': 'dcnv2', 'DCNv2-FPN': 'dcnv2', 'DCNv2-Pan': 'dcnv2', 'DCNv2-Liu': 'dcnv2',
    'DCNv3-Full': 'dcnv3', 'DCNv3-FPN': 'dcnv3', 'DCNv3-Pan': 'dcnv3', 'DCNv3-Liu': 'dcnv3',
    'Vanilla-YOLOv8m': 'vanilla',
}

OCCLUSION_TYPE = {
    'DCNv2-Full': 'dcnv2', 'DCNv2-FPN': 'dcnv2', 'DCNv2-Pan': 'dcnv2', 'DCNv2-Liu': 'dcnv2',
    'DCNv3-Full': 'dcnv3', 'DCNv3-FPN': 'dcnv3', 'DCNv3-Pan': 'dcnv3', 'DCNv3-Liu': 'dcnv3',
    'Vanilla-YOLOv8m': 'vanilla',
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def cliffs_delta(x, y):
    """Cliff's delta effect size. Range [-1, 1]. |d|<0.147 negligible, <0.33 small, <0.474 medium, else large."""
    n1, n2 = len(x), len(y)
    dom = sum(1 for xi in x for yj in y if xi > yj) - sum(1 for xi in x for yj in y if xi < yj)
    return dom / (n1 * n2)


def cliffs_magnitude(d):
    ad = abs(d)
    if ad < 0.147: return 'negligible'
    if ad < 0.330: return 'small'
    if ad < 0.474: return 'medium'
    return 'large'


def bootstrap_ci(x, y, n_boot=10000, ci=0.95):
    """Bootstrap 95% CI for mean(x) - mean(y)."""
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        diffs.append(np.mean(bx) - np.mean(by))
    lo = np.percentile(diffs, (1 - ci) / 2 * 100)
    hi = np.percentile(diffs, (1 + ci) / 2 * 100)
    return lo, hi


def safe_wilcoxon(x, y):
    """Wilcoxon signed-rank. Returns (stat, p). Returns (nan, nan) if all differences are zero."""
    diffs = [xi - yi for xi, yi in zip(x, y)]
    if all(d == 0 for d in diffs):
        return float('nan'), float('nan')
    try:
        stat, p = wilcoxon(x, y, alternative='two-sided')
        return stat, p
    except Exception:
        return float('nan'), float('nan')


def safe_ttest(x, y):
    """Paired t-test. Returns (stat, p). Returns (nan, nan) on error."""
    try:
        stat, p = stats.ttest_rel(x, y)
        return stat, p
    except Exception:
        return float('nan'), float('nan')


def bonferroni(p_values, n_comparisons):
    return [min(p * n_comparisons, 1.0) if not np.isnan(p) else float('nan') for p in p_values]


# ─────────────────────────────────────────────
# LOAD TRACKING DATA
# ─────────────────────────────────────────────
def load_tracking_data():
    """Returns {model: {gate: {mota, idf1}}} using pre-computed 1hr+12hr averages from compute_avg_metrics.py output."""
    avg_csv = os.path.join(TRACKING_BASE, 'avg_mota_idf1_1hr_12hr.csv')
    tracking = {}
    with open(avg_csv) as f:
        for row in csv.DictReader(f):
            model = row['Model']
            # Gate column format: "Gate2 / g2" — extract the 1hr gate name
            gate = row['Gate'].split(' / ')[0].strip()
            if model not in MODELS_ORDER or gate not in GATES_1HR:
                continue
            mota_avg = row.get('MOTA_avg', '').strip()
            idf1_avg = row.get('IDF1_avg', '').strip()
            if mota_avg and idf1_avg:
                if model not in tracking:
                    tracking[model] = {}
                tracking[model][gate] = {
                    'mota': float(mota_avg),
                    'idf1': float(idf1_avg),
                }
    return tracking


# ─────────────────────────────────────────────
# LOAD IDSW / CLASS SWITCH DATA (12hr)
# ─────────────────────────────────────────────
def load_idsw_clssw_data():
    """Returns {model: {gate: {idsw_rate, clssw_rate}}} where rate = metric/TrackIDs."""
    dirs = ['dcnv2', 'dcnv3', 'vanilla']
    result = {}
    for gate in GATES_12HR:
        for d in dirs:
            f = os.path.join(TRACKING_BASE, f'counting_6to6_{gate}/{d}/summary.csv')
            if not os.path.exists(f):
                continue
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    model = row['Model']
                    if model not in MODELS_ORDER:
                        continue
                    tracks = int(row['TrackIDs'])
                    idsw = int(row['IDSW'])
                    cls_sw = int(row['ClassSwitches'])
                    if tracks == 0:
                        continue
                    gate_1hr = [k for k, v in GATE_MAP.items() if v == gate][0]
                    if model not in result:
                        result[model] = {}
                    result[model][gate_1hr] = {
                        'idsw_rate': idsw / tracks,
                        'clssw_rate': cls_sw / tracks,
                    }
    return result


# ─────────────────────────────────────────────
# LOAD DETECTION DATA
# ─────────────────────────────────────────────
def load_detection_data():
    """Returns {model: {class: {ap50, precision, recall}}}"""
    detection = {}
    for model in MODELS_ORDER:
        mtype = OCCLUSION_TYPE[model]
        csv_path = os.path.join(OCCLUSION_BASE, mtype, 'occlusion_summary.csv')
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row['Model'] == model:
                    detection[model] = {}
                    for cls in CLASSES:
                        detection[model][cls] = {
                            'ap50': float(row.get(f'AP50_{cls}', 0)),
                            'precision': float(row.get(f'P_{cls}', 0)),
                            'recall': float(row.get(f'R_{cls}', 0)),
                        }
    return detection


# ─────────────────────────────────────────────
# STATISTICAL ANALYSIS
# ─────────────────────────────────────────────
def analyze(data, models, repeated_measure_keys, metric_name, domain_name):
    """
    data: {model: {key: {metric_name: value}}}
    Returns analysis dict and formatted lines.
    """
    lines = []
    lines.append(f"\n{'='*110}")
    lines.append(f"  {domain_name} — {metric_name.upper()}")
    lines.append(f"{'='*110}")

    # Build matrix: models x repeated_measures
    matrix = {}
    for model in models:
        vals = []
        for key in repeated_measure_keys:
            v = data.get(model, {}).get(key, {}).get(metric_name)
            if v is not None:
                vals.append(v)
        if len(vals) == len(repeated_measure_keys):
            matrix[model] = vals

    valid_models = [m for m in models if m in matrix]
    if len(valid_models) < 2:
        lines.append("  Not enough data.")
        return lines

    n = len(repeated_measure_keys)

    # ── Model means
    lines.append(f"\n  Model means (n={n} {domain_name.lower().split()[0]}s):")
    lines.append(f"  {'Model':<20} {'Mean':>8}  {'Std':>8}  Values")
    lines.append(f"  {'-'*80}")
    for model in valid_models:
        vals = matrix[model]
        lines.append(f"  {model:<20} {np.mean(vals):>8.4f}  {np.std(vals):>8.4f}  {[round(v,4) for v in vals]}")

    # ── Friedman test
    arrays = [matrix[m] for m in valid_models]
    if len(arrays) >= 3 and all(len(a) == n for a in arrays):
        try:
            stat, p = friedmanchisquare(*arrays)
            lines.append(f"\n  Friedman test (all {len(valid_models)} models): χ²={stat:.4f}, p={p:.4f} {'*' if p<0.05 else '(ns)'}")
        except Exception as e:
            lines.append(f"\n  Friedman test: ERROR {e}")

    # ── Pairwise comparisons
    pairs = list(itertools.combinations(valid_models, 2))
    n_pairs = len(pairs)

    raw_t_ps, raw_w_ps, pair_results = [], [], []
    for mA, mB in pairs:
        vA, vB = np.array(matrix[mA]), np.array(matrix[mB])
        diff = np.mean(vA) - np.mean(vB)
        lo, hi = bootstrap_ci(vA, vB)
        tstat, tp = safe_ttest(vA, vB)
        wstat, wp = safe_wilcoxon(vA, vB)
        d = cliffs_delta(vA, vB)
        mag = cliffs_magnitude(d)
        raw_t_ps.append(tp)
        raw_w_ps.append(wp)
        pair_results.append((mA, mB, diff, lo, hi, tstat, tp, wstat, wp, d, mag))

    adj_t_ps = bonferroni(raw_t_ps, n_pairs)
    adj_w_ps = bonferroni(raw_w_ps, n_pairs)

    lines.append(f"\n  Pairwise — Paired t-test ({n_pairs} pairs, Bonferroni-corrected):")
    lines.append(f"  {'Model A':<20} {'Model B':<20} {'Mean diff':>10} {'95% CI':>20} {'t-stat':>8} {'p-raw':>8} {'p-adj':>8} {'Cliff d':>8} {'Effect':>10}")
    lines.append(f"  {'-'*120}")
    for (mA, mB, diff, lo, hi, tstat, tp, wstat, wp, d, mag), padj in zip(pair_results, adj_t_ps):
        sig = '**' if (not np.isnan(padj) and padj < 0.01) else ('*' if (not np.isnan(padj) and padj < 0.05) else '')
        tstr = f"{tstat:.3f}" if not np.isnan(tstat) else "N/A"
        pstr = f"{tp:.4f}" if not np.isnan(tp) else "N/A"
        padjstr = f"{padj:.4f}" if not np.isnan(padj) else "N/A"
        lines.append(f"  {mA:<20} {mB:<20} {diff:>10.4f} [{lo:>7.4f},{hi:>7.4f}] {tstr:>8} {pstr:>8} {padjstr:>8} {d:>8.3f} {mag:>10} {sig}")

    lines.append(f"\n  Pairwise — Wilcoxon signed-rank ({n_pairs} pairs, Bonferroni-corrected):")
    lines.append(f"  {'Model A':<20} {'Model B':<20} {'Mean diff':>10} {'95% CI':>20} {'W-stat':>8} {'p-raw':>8} {'p-adj':>8} {'Cliff d':>8} {'Effect':>10}")
    lines.append(f"  {'-'*120}")
    for (mA, mB, diff, lo, hi, tstat, tp, wstat, wp, d, mag), padj in zip(pair_results, adj_w_ps):
        sig = '**' if (not np.isnan(padj) and padj < 0.01) else ('*' if (not np.isnan(padj) and padj < 0.05) else '')
        wstr = f"{wstat:.1f}" if not np.isnan(wstat) else "N/A"
        pstr = f"{wp:.4f}" if not np.isnan(wp) else "N/A"
        padjstr = f"{padj:.4f}" if not np.isnan(padj) else "N/A"
        lines.append(f"  {mA:<20} {mB:<20} {diff:>10.4f} [{lo:>7.4f},{hi:>7.4f}] {wstr:>8} {pstr:>8} {padjstr:>8} {d:>8.3f} {mag:>10} {sig}")

    lines.append(f"\n  Note: With n={n}, Wilcoxon min achievable p (two-tailed) = {1.0/2**(n-1):.3f}. t-test and effect sizes are more informative at small n.")
    return lines


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("Loading tracking data...")
    tracking = load_tracking_data()
    print("Loading detection data...")
    detection = load_detection_data()

    all_lines = []
    all_lines.append("=" * 110)
    all_lines.append("  STATISTICAL ANALYSIS — TRACKING & DETECTION")
    all_lines.append("  Methods: Friedman test | Wilcoxon signed-rank (Bonferroni) | Cliff's delta | Bootstrap 95% CI")
    all_lines.append("=" * 110)

    # ── TRACKING
    all_lines.append("\n\n" + "█"*110)
    all_lines.append("  TRACKING METRICS (averaged 1hr + 12hr per gate)")
    all_lines.append("█"*110)
    for metric in ['mota', 'idf1']:
        lines = analyze(tracking, MODELS_ORDER, GATES_1HR, metric, "Tracking (4 gates)")
        all_lines.extend(lines)

    # ── IDSW / CLASS SWITCHES
    print("Loading IDSW/ClassSwitch data...")
    idsw_data = load_idsw_clssw_data()
    all_lines.append("\n\n" + "█"*110)
    all_lines.append("  IDENTITY SWITCH & CLASS SWITCH RATES (12hr, rate = metric / TrackIDs)")
    all_lines.append("█"*110)
    for metric in ['idsw_rate', 'clssw_rate']:
        lines = analyze(idsw_data, MODELS_ORDER, GATES_1HR, metric, "Tracking (4 gates)")
        all_lines.extend(lines)

    # ── DETECTION
    all_lines.append("\n\n" + "█"*110)
    all_lines.append("  DETECTION METRICS (OccludedYOLO — 6 classes)")
    all_lines.append("█"*110)
    det_models = [m for m in MODELS_ORDER if m in detection]
    for metric in ['ap50', 'precision', 'recall']:
        lines = analyze(detection, det_models, CLASSES, metric, "Detection (6 classes)")
        all_lines.extend(lines)

    output = '\n'.join(all_lines)
    print(output)

    out_txt = os.path.join(OUTPUT_DIR, 'statistical_analysis_results.txt')
    out_csv = os.path.join(OUTPUT_DIR, 'statistical_analysis_results.csv')

    with open(out_txt, 'w') as f:
        f.write(output + '\n')
    print(f"\n\nSaved: {out_txt}")

    # Save pairwise CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Domain', 'Metric', 'Model_A', 'Model_B', 'Mean_A', 'Mean_B', 'Mean_diff',
                         'CI_low', 'CI_high',
                         't_stat', 't_p_raw', 't_p_adj_bonferroni',
                         'Wilcoxon_stat', 'w_p_raw', 'w_p_adj_bonferroni',
                         'Cliffs_delta', 'Effect_magnitude'])

        def write_pairs(domain, metric, source_data, models, keys):
            matrix = {}
            for model in models:
                vals = [source_data.get(model, {}).get(k, {}).get(metric) for k in keys]
                if all(v is not None for v in vals):
                    matrix[model] = vals
            valid = [m for m in models if m in matrix]
            pairs = list(itertools.combinations(valid, 2))
            raw_t = [safe_ttest(np.array(matrix[mA]), np.array(matrix[mB]))[1] for mA, mB in pairs]
            raw_w = [safe_wilcoxon(np.array(matrix[mA]), np.array(matrix[mB]))[1] for mA, mB in pairs]
            adj_t = bonferroni(raw_t, len(pairs))
            adj_w = bonferroni(raw_w, len(pairs))
            for (mA, mB), tp, tp_adj, wp, wp_adj in zip(pairs, raw_t, adj_t, raw_w, adj_w):
                vA, vB = np.array(matrix[mA]), np.array(matrix[mB])
                diff = np.mean(vA) - np.mean(vB)
                lo, hi = bootstrap_ci(vA, vB)
                tstat, _ = safe_ttest(vA, vB)
                wstat, _ = safe_wilcoxon(vA, vB)
                d = cliffs_delta(vA, vB)
                writer.writerow([domain, metric.upper(), mA, mB,
                                 f"{np.mean(vA):.4f}", f"{np.mean(vB):.4f}", f"{diff:.4f}",
                                 f"{lo:.4f}", f"{hi:.4f}",
                                 f"{tstat:.3f}" if not np.isnan(tstat) else "N/A",
                                 f"{tp:.4f}" if not np.isnan(tp) else "N/A",
                                 f"{tp_adj:.4f}" if not np.isnan(tp_adj) else "N/A",
                                 f"{wstat:.1f}" if not np.isnan(wstat) else "N/A",
                                 f"{wp:.4f}" if not np.isnan(wp) else "N/A",
                                 f"{wp_adj:.4f}" if not np.isnan(wp_adj) else "N/A",
                                 f"{d:.3f}", cliffs_magnitude(d)])

        for metric in ['mota', 'idf1']:
            write_pairs('Tracking', metric, tracking, MODELS_ORDER, GATES_1HR)
        for metric in ['idsw_rate', 'clssw_rate']:
            write_pairs('IDSW_ClsSw', metric, idsw_data, MODELS_ORDER, GATES_1HR)
        det_models = [m for m in MODELS_ORDER if m in detection]
        for metric in ['ap50', 'precision', 'recall']:
            write_pairs('Detection', metric, detection, det_models, CLASSES)

    print(f"Saved: {out_csv}")


if __name__ == '__main__':
    main()
