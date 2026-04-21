#!/usr/bin/env python3
"""
Focused statistical comparison: DCN-modified models vs Vanilla-YOLOv8m (base).

Groups:
  - DCNv2  (Full, FPN, Pan, Liu)
  - DCNv3  (Full, FPN, Pan, Liu)
  - DCN-All (all 8 DCN variants combined)

Each group is averaged per repeated measure (gate / class) to produce a single
vector, then compared against Vanilla with:
  1. Paired t-test (Bonferroni-corrected)
  2. Wilcoxon signed-rank
  3. Cliff's delta (effect size)
  4. Bootstrap 95% CI

Metrics covered:
  - Tracking : MOTA, IDF1  (n=4 gates, averaged across 1hr+12hr)
  - Switch rates: IDSW/TrackID, ClassSW/TrackID  (n=4 gates, 12hr)
  - Detection : AP50, Precision, Recall  (n=6 classes)
"""

import csv, os, itertools
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

TRACKING_BASE = '/media/mydrive/GitHub/ultralytics/tracking'
OCCLUSION_BASE = os.path.join(TRACKING_BASE, 'occlusion_eval/results')
OUTPUT_DIR    = TRACKING_BASE

GATE_MAP  = {'Gate2': 'g2', 'Gate2.9': 'g29', 'Gate3': 'g3', 'Gate3.5': 'g35'}
GATES_1HR = list(GATE_MAP.keys())
CLASSES   = ['car', 'motorcycle', 'tricycle', 'bus', 'van', 'truck']

DCNv2_MODELS   = ['DCNv2-Full', 'DCNv2-FPN', 'DCNv2-Pan', 'DCNv2-Liu']
DCNv3_MODELS   = ['DCNv3-Full', 'DCNv3-FPN', 'DCNv3-Pan', 'DCNv3-Liu']
DCN_ALL_MODELS = DCNv2_MODELS + DCNv3_MODELS
VANILLA        = 'Vanilla-YOLOv8m'

GROUPS = {
    'DCNv2':   DCNv2_MODELS,
    'DCNv3':   DCNv3_MODELS,
    'DCN-All': DCN_ALL_MODELS,
}

MODEL_TYPE = {m: 'dcnv2' for m in DCNv2_MODELS}
MODEL_TYPE.update({m: 'dcnv3' for m in DCNv3_MODELS})
MODEL_TYPE[VANILLA] = 'vanilla'


# ─────────────────────────────────────────────
# STATS HELPERS
# ─────────────────────────────────────────────
def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    dom = sum(1 for xi in x for yj in y if xi > yj) - \
          sum(1 for xi in x for yj in y if xi < yj)
    return dom / (n1 * n2)

def cliffs_magnitude(d):
    ad = abs(d)
    if ad < 0.147: return 'negligible'
    if ad < 0.330: return 'small'
    if ad < 0.474: return 'medium'
    return 'large'

def bootstrap_ci(x, y, n_boot=10000, ci=0.95):
    rng = np.random.default_rng(42)
    diffs = [np.mean(rng.choice(x, len(x), replace=True)) -
             np.mean(rng.choice(y, len(y), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(diffs, (1 - ci) / 2 * 100)
    hi = np.percentile(diffs, (1 + ci) / 2 * 100)
    return lo, hi

def safe_ttest(x, y):
    try:    return stats.ttest_rel(x, y)
    except: return float('nan'), float('nan')

def safe_wilcoxon(x, y):
    diffs = [xi - yi for xi, yi in zip(x, y)]
    if all(d == 0 for d in diffs):
        return float('nan'), float('nan')
    try:    return wilcoxon(x, y, alternative='two-sided')
    except: return float('nan'), float('nan')

def fmt(v, dec=4):
    return f"{v:.{dec}f}" if not np.isnan(v) else 'N/A'


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
def load_tracking():
    avg_csv = os.path.join(TRACKING_BASE, 'avg_mota_idf1_1hr_12hr.csv')
    data = {}
    with open(avg_csv) as f:
        for row in csv.DictReader(f):
            model = row['Model']
            gate  = row['Gate'].split(' / ')[0].strip()
            if gate not in GATES_1HR:
                continue
            mota = row.get('MOTA_avg', '').strip()
            idf1 = row.get('IDF1_avg', '').strip()
            if mota and idf1:
                data.setdefault(model, {})[gate] = {
                    'mota': float(mota), 'idf1': float(idf1)
                }
    return data


def load_idsw():
    data = {}
    for gate in GATE_MAP.values():
        for d in ['dcnv2', 'dcnv3', 'vanilla']:
            f = os.path.join(TRACKING_BASE, f'counting_6to6_{gate}/{d}/summary.csv')
            if not os.path.exists(f):
                continue
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    model  = row['Model']
                    tracks = int(row['TrackIDs'])
                    if tracks == 0:
                        continue
                    gate_1hr = [k for k, v in GATE_MAP.items() if v == gate][0]
                    data.setdefault(model, {})[gate_1hr] = {
                        'idsw_rate':  int(row['IDSW'])          / tracks,
                        'clssw_rate': int(row['ClassSwitches']) / tracks,
                    }
    return data


def load_detection():
    data = {}
    all_models = DCN_ALL_MODELS + [VANILLA]
    for model in all_models:
        mtype    = MODEL_TYPE[model]
        csv_path = os.path.join(OCCLUSION_BASE, mtype, 'occlusion_summary.csv')
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row['Model'] == model:
                    data[model] = {}
                    for cls in CLASSES:
                        data[model][cls] = {
                            'ap50':      float(row.get(f'AP50_{cls}', 0)),
                            'precision': float(row.get(f'P_{cls}',    0)),
                            'recall':    float(row.get(f'R_{cls}',    0)),
                        }
    return data


# ─────────────────────────────────────────────
# GROUP AVERAGING
# ─────────────────────────────────────────────
def group_avg(data, models, keys, metric):
    """For each key (gate/class), average metric across models in the group."""
    result = []
    for key in keys:
        vals = [data.get(m, {}).get(key, {}).get(metric) for m in models]
        vals = [v for v in vals if v is not None]
        if vals:
            result.append(np.mean(vals))
        else:
            result.append(None)
    return result


def vanilla_vec(data, keys, metric):
    return [data.get(VANILLA, {}).get(k, {}).get(metric) for k in keys]


# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
def analyze_metric(data, keys, metric, domain, lines, csv_rows):
    n = len(keys)
    van = vanilla_vec(data, keys, metric)
    if any(v is None for v in van):
        lines.append(f"  [{metric}] Vanilla data missing, skipping.")
        return

    van = np.array(van, dtype=float)
    van_mean = np.mean(van)

    lines.append(f"\n  {'─'*100}")
    lines.append(f"  Metric: {metric.upper()}   |   Vanilla mean = {van_mean:.4f}   "
                 f"|   n = {n} {domain}")
    lines.append(f"  {'─'*100}")
    lines.append(f"  {'Group':<12} {'Mean':>8}  {'Diff vs Van':>12}  "
                 f"{'95% CI':>20}  {'t-stat':>8}  {'t-p':>7}  "
                 f"{'W-stat':>8}  {'w-p':>7}  {'Cliff d':>8}  {'Effect'}")
    lines.append(f"  {'-'*110}")

    for gname, gmodels in GROUPS.items():
        gvec = group_avg(data, gmodels, keys, metric)
        if any(v is None for v in gvec):
            lines.append(f"  {gname:<12}  -- missing data --")
            continue
        gvec = np.array(gvec, dtype=float)
        g_mean = np.mean(gvec)
        diff   = g_mean - van_mean
        lo, hi = bootstrap_ci(gvec, van)
        tstat, tp = safe_ttest(gvec, van)
        wstat, wp = safe_wilcoxon(gvec, van)
        d  = cliffs_delta(gvec, van)
        mag = cliffs_magnitude(d)

        # significance marker
        sig = ''
        if not np.isnan(tp):
            if tp < 0.001: sig = '***'
            elif tp < 0.01: sig = '**'
            elif tp < 0.05: sig = '*'

        lines.append(
            f"  {gname:<12} {g_mean:>8.4f}  {diff:>+12.4f}  "
            f"[{lo:>7.4f},{hi:>7.4f}]  {fmt(tstat):>8}  {fmt(tp):>7}  "
            f"{fmt(wstat,1):>8}  {fmt(wp):>7}  {d:>8.3f}  {mag:<12} {sig}"
        )
        csv_rows.append([domain, metric, gname, VANILLA,
                         fmt(g_mean), fmt(van_mean), fmt(diff),
                         fmt(lo), fmt(hi),
                         fmt(tstat), fmt(tp), fmt(wstat,1), fmt(wp),
                         fmt(d), mag, sig])

    lines.append(f"\n  Individual DCN models vs Vanilla:")
    lines.append(f"  {'Model':<20} {'Mean':>8}  {'Diff':>10}  "
                 f"{'t-p':>7}  {'w-p':>7}  {'Cliff d':>8}  {'Effect'}")
    lines.append(f"  {'-'*80}")
    for model in DCN_ALL_MODELS:
        mvec_raw = [data.get(model, {}).get(k, {}).get(metric) for k in keys]
        if any(v is None for v in mvec_raw):
            continue
        mvec = np.array(mvec_raw, dtype=float)
        m_mean = np.mean(mvec)
        diff   = m_mean - van_mean
        _, tp  = safe_ttest(mvec, van)
        _, wp  = safe_wilcoxon(mvec, van)
        d      = cliffs_delta(mvec, van)
        mag    = cliffs_magnitude(d)
        sig = ''
        if not np.isnan(tp):
            if tp < 0.001: sig = '***'
            elif tp < 0.01: sig = '**'
            elif tp < 0.05: sig = '*'
        lines.append(
            f"  {model:<20} {m_mean:>8.4f}  {diff:>+10.4f}  "
            f"{fmt(tp):>7}  {fmt(wp):>7}  {d:>8.3f}  {mag:<12} {sig}"
        )
        csv_rows.append([domain, metric, model, VANILLA,
                         fmt(m_mean), fmt(van_mean), fmt(diff),
                         'N/A', 'N/A',
                         'N/A', fmt(tp), 'N/A', fmt(wp),
                         fmt(d), mag, sig])


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("Loading data...")
    tracking  = load_tracking()
    idsw_data = load_idsw()
    detection = load_detection()

    lines    = []
    csv_rows = []

    header = ['Domain', 'Metric', 'Model_A', 'Model_B',
              'Mean_A', 'Mean_B', 'Mean_diff',
              'CI_low', 'CI_high',
              't_stat', 't_p', 'W_stat', 'w_p',
              'Cliffs_delta', 'Effect', 'Sig']

    lines.append("=" * 110)
    lines.append("  DCN MODIFIED vs VANILLA (BASE MODEL) — Statistical Analysis")
    lines.append("  Tests: paired t-test | Wilcoxon signed-rank | Cliff's delta | Bootstrap 95% CI")
    lines.append("  Groups: DCNv2 (4 variants), DCNv3 (4 variants), DCN-All (8 variants)")
    lines.append("  * p<0.05   ** p<0.01   *** p<0.001  (raw p, not Bonferroni-corrected)")
    lines.append("=" * 110)

    # ── TRACKING
    lines.append("\n\n" + "█" * 110)
    lines.append("  TRACKING METRICS  (n=4 gates, averaged 1hr + 12hr)")
    lines.append("█" * 110)
    for metric in ['mota', 'idf1']:
        analyze_metric(tracking, GATES_1HR, metric, 'gates', lines, csv_rows)

    # ── SWITCH RATES
    lines.append("\n\n" + "█" * 110)
    lines.append("  SWITCH RATES  (n=4 gates, 12hr runs)")
    lines.append("█" * 110)
    for metric in ['idsw_rate', 'clssw_rate']:
        analyze_metric(idsw_data, GATES_1HR, metric, 'gates', lines, csv_rows)

    # ── DETECTION
    lines.append("\n\n" + "█" * 110)
    lines.append("  DETECTION METRICS  (n=6 classes, OccludedYOLO dataset)")
    lines.append("█" * 110)
    for metric in ['ap50', 'precision', 'recall']:
        analyze_metric(detection, CLASSES, metric, 'classes', lines, csv_rows)

    lines.append("\n\nNote: With n=4 (tracking/switch) or n=6 (detection), Wilcoxon min achievable")
    lines.append("p (two-tailed) = 0.125 (n=4) or 0.031 (n=6). t-test and Cliff's delta are")
    lines.append("more informative at these small sample sizes.")

    output = '\n'.join(lines)
    print(output)

    txt_path = os.path.join(OUTPUT_DIR, 'dcn_vs_vanilla_analysis.txt')
    csv_path = os.path.join(OUTPUT_DIR, 'dcn_vs_vanilla_analysis.csv')

    with open(txt_path, 'w') as f:
        f.write(output + '\n')

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(csv_rows)

    print(f"\nSaved: {txt_path}")
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
