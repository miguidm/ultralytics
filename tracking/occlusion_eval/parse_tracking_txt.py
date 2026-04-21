import os
import re
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

def robust_parse(filepath):
    data = []
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Split by Gate
    parts = re.split(r'(?i)GATE\s+([\d\.]+)', content)
    
    for i in range(1, len(parts), 2):
        gate = parts[i].strip()
        table_text = parts[i+1]
        
        # Tokenize by whitespace
        tokens = re.split(r'\s+', table_text)
        
        current_model = None
        current_floats = []
        
        for t in tokens:
            if 'DCN' in t or 'Vanilla' in t:
                # Save previous
                if current_model and len(current_floats) >= 2:
                    data.append({
                        'Gate': gate,
                        'Model_raw': current_model,
                        'IDF1': current_floats[0],
                        'MOTA': current_floats[1]
                    })
                current_model = t
                current_floats = []
            else:
                if current_model:
                    try:
                        # handle comma decimals
                        t_clean = t.replace(',', '.')
                        if t_clean.startswith('.'): t_clean = '0' + t_clean
                        val = float(t_clean)
                        current_floats.append(val)
                    except ValueError:
                        pass
        
        # Save last
        if current_model and len(current_floats) >= 2:
            data.append({
                'Gate': gate,
                'Model_raw': current_model,
                'IDF1': current_floats[0],
                'MOTA': current_floats[1]
            })

    cleaned_data = []
    for d in data:
        model_raw = d['Model_raw']
        if 'Vanilla' in model_raw or 'vanilla' in model_raw.lower():
            version = 'Vanilla'
            placement = 'None'
            type_mod = 'Vanilla Baseline'
            standard_model = 'Vanilla-YOLOv8'
        else:
            version = 'DCNv2' if 'v2' in model_raw.lower() else 'DCNv3'
            if 'Full' in model_raw or 'FULL' in model_raw: placement = 'Neck-Full'
            elif 'FPN' in model_raw: placement = 'Neck-FPN'
            elif 'Pan' in model_raw or 'PAN' in model_raw: placement = 'Neck-PAN'
            elif 'Backbone' in model_raw or 'Liu' in model_raw or '-B' in model_raw: placement = 'Backbone'
            else: placement = 'Unknown'
            type_mod = 'With DCN Modifications'
            standard_model = f"{version}-{placement}"
            
        cleaned_data.append({
            'Gate': d['Gate'],
            'Model': standard_model,
            'Version': version,
            'Placement': placement,
            'Type': type_mod,
            'IDF1': d['IDF1'],
            'MOTA': d['MOTA']
        })
        
    return pd.DataFrame(cleaned_data)

def run_anova(df, size_label):
    out_lines = []
    out_lines.append("================================================================================")
    out_lines.append(f" FINAL TRACKING STATISTICAL ANALYSIS RESULTS ({size_label.upper()} MODELS)")
    out_lines.append(f" Source: {size_label}.txt (Gates as N=5 Replicates)")
    out_lines.append("================================================================================\n")
    
    if len(df) == 0:
        out_lines.append("No data parsed!")
        return "\n".join(out_lines)

    # Dedup just in case there's multiple runs per gate
    # df = df.drop_duplicates(subset=['Gate', 'Model'])

    # PART 1
    out_lines.append("--------------------------------------------------------------------------------")
    out_lines.append(" PART 1: VANILLA BASELINE VS. WITH DCN MODIFICATIONS (ONE-WAY ANOVA)")
    out_lines.append("--------------------------------------------------------------------------------\n")
    for metric in ['MOTA', 'IDF1']:
        out_lines.append(f">>> Dependent Variable: {metric}")
        model_t1 = ols(f'{metric} ~ C(Type)', data=df).fit()
        anova_t1 = sm.stats.anova_lm(model_t1, typ=2)
        out_lines.append(anova_t1.to_string())
        p_val = anova_t1.loc['C(Type)', 'PR(>F)']
        sig = "Significant" if p_val < 0.05 else "Not Significant"
        out_lines.append(f"Conclusion: p = {p_val:.4f} ({sig})\n")

    # PART 2
    out_lines.append("--------------------------------------------------------------------------------")
    out_lines.append(" PART 2: TWO-WAY ANOVA (DCN VERSION x PLACEMENT) - Main Effects")
    out_lines.append(" Note: Vanilla included, interaction excluded to prevent rank deficiency.")
    out_lines.append("--------------------------------------------------------------------------------\n")
    for metric in ['MOTA', 'IDF1']:
        out_lines.append(f">>> Dependent Variable: {metric}")
        model_t2 = ols(f'{metric} ~ C(Version) + C(Placement)', data=df).fit()
        anova_t2 = sm.stats.anova_lm(model_t2, typ=2)
        out_lines.append(anova_t2.to_string())
        out_lines.append(f"Version Effect p-value:   {anova_t2.loc['C(Version)', 'PR(>F)']:.6f}")
        out_lines.append(f"Placement Effect p-value: {anova_t2.loc['C(Placement)', 'PR(>F)']:.6f}\n")

    # PART 3
    out_lines.append("--------------------------------------------------------------------------------")
    out_lines.append(" PART 3: SPECIFIC MODIFICATIONS VS. VANILLA BASELINE (TUKEY HSD POST-HOC)")
    out_lines.append("--------------------------------------------------------------------------------\n")
    for metric in ['MOTA', 'IDF1']:
        out_lines.append(f">>> Dependent Variable: {metric}")
        tukey = pairwise_tukeyhsd(endog=df[metric], groups=df['Model'], alpha=0.05)
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        
        vanilla_comparisons = tukey_df[(tukey_df['group1'] == 'Vanilla-YOLOv8') | (tukey_df['group2'] == 'Vanilla-YOLOv8')]
        
        def format_comparison(row):
            mod = row['group2'] if row['group1'] == 'Vanilla-YOLOv8' else row['group1']
            diff = row['meandiff'] if row['group1'] == 'Vanilla-YOLOv8' else -row['meandiff']
            p_adj = row['p-adj']
            reject = row['reject']
            sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
            return f"Vanilla vs {mod:<15} | Mean Diff: {diff:>7.4f} | p-adj: {p_adj:>6.4f} | Significant: {str(reject):<5} {sig}"
            
        for _, row in vanilla_comparisons.iterrows():
            out_lines.append(format_comparison(row))
        out_lines.append("")

    return "\n".join(out_lines)

nano_df = robust_parse('/media/mydrive/GitHub/ultralytics/tracking/nano.txt')
med_df = robust_parse('/media/mydrive/GitHub/ultralytics/tracking/medium.txt')

print(f"Data points parsed -> Nano: {len(nano_df)}, Medium: {len(med_df)}")

nano_res = run_anova(nano_df, "nano")
med_res = run_anova(med_df, "medium")

with open('/media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/tracking_anova_txt_results.txt', 'w') as f:
    f.write(nano_res)
    f.write("\n\n" + "="*80 + "\n\n")
    f.write(med_res)

print("Results saved to /media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/tracking_anova_txt_results.txt")
