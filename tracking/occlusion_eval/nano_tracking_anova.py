import os
import re
import glob
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR = '/media/mydrive/GitHub/ultralytics/tracking/all_nano_reports'
OUTPUT_FILE = '/media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/nano_tracking_anova_results.txt'

def parse_reports():
    data = []
    
    # We expect files like Gate2_DCNv2-FPN-n.txt
    for filepath in glob.glob(os.path.join(REPORTS_DIR, '*.txt')):
        filename = os.path.basename(filepath)
        
        # Skip summary files
        if 'SUMMARY' in filename:
            continue
            
        parts = filename.replace('.txt', '').split('_')
        if len(parts) < 2:
            continue
            
        gate = parts[0]
        model_raw = '_'.join(parts[1:])
        
        # Standardize Model names to match the detection ANOVA format
        if 'Vanilla' in model_raw:
            model = 'Vanilla-YOLOv8n'
            version = 'Vanilla'
            placement = 'None'
            type_mod = 'Vanilla Baseline'
        else:
            # model_raw is e.g. DCNv2-FPN-n
            model = model_raw.replace('-n', '')
            version = 'DCNv2' if 'DCNv2' in model else 'DCNv3'
            
            if 'Full' in model: placement = 'Neck-Full'
            elif 'FPN' in model: placement = 'Neck-FPN'
            elif 'Pan' in model: placement = 'Neck-PAN'
            elif 'Liu' in model: placement = 'Backbone'
            else: placement = 'Unknown'
            
            type_mod = 'With DCN Modifications'
            
        # Parse metrics from file
        idf1 = None
        mota = None
        idsw = None
        
        with open(filepath, 'r') as f:
            content = f.read()
            
            idf1_match = re.search(r'IDF1 \(Identity F1 Score\):\s+([\d.]+)', content)
            if idf1_match: idf1 = float(idf1_match.group(1))
            
            mota_match = re.search(r'MOTA \(Multi-Object Tracking\):\s+([\d.]+)', content)
            if mota_match: mota = float(mota_match.group(1))
            
            idsw_match = re.search(r'ID Switches \(IDSW\):\s+(\d+)', content)
            if idsw_match: idsw = int(idsw_match.group(1))
            
        if idf1 is not None and mota is not None:
            data.append({
                'Gate': gate,
                'Model': model,
                'Version': version,
                'Placement': placement,
                'Type': type_mod,
                'IDF1': idf1,
                'MOTA': mota,
                'IDSW': idsw
            })
            
    return pd.DataFrame(data)

def main():
    df = parse_reports()
    
    if df.empty:
        print("No tracking data found.")
        return
        
    print(f"Parsed {len(df)} tracking report files successfully.")

    with open(OUTPUT_FILE, 'w') as f:
        f.write("================================================================================\n")
        f.write(" FINAL TRACKING STATISTICAL ANALYSIS RESULTS (N=5 Gates per model)\n")
        f.write(" Dataset: 5 Gates (Gate2, Gate2.9, Gate3, Gate3.1, Gate3.5)\n")
        f.write("================================================================================\n\n")

        f.write("================================================================================\n")
        f.write(" METHODOLOGY: HOW SIGNIFICANCE TESTING WAS CONDUCTED\n")
        f.write("================================================================================\n")
        f.write("1. Variance Generation (Across 5 Cameras/Gates):\n")
        f.write("   ANOVA mathematically requires variance (multiple replicates per model) to calculate\n")
        f.write("   significance. For object tracking, we utilized the real-world tracking metrics\n")
        f.write("   (MOTA, IDF1) gathered across 5 distinct camera angles (Gates 2, 2.9, 3, 3.1, 3.5).\n")
        f.write("   Every model was evaluated on these exact same 5 video feeds.\n")
        f.write("   This yielded 5 MOTA and 5 IDF1 scores per model (N=5), establishing the required\n")
        f.write("   variance for ANOVA while demonstrating robustness across varying perspectives.\n\n")
        f.write("2. Testing Framework:\n")
        f.write("   - Part 1 (One-Way ANOVA): Tests if applying ANY modification generally improves tracking.\n")
        f.write("   - Part 2 (Two-Way ANOVA): A main-effects factorial analysis on the 9 variants to\n")
        f.write("     isolate the impacts of DCN Version (Vanilla vs v2 vs v3) and Placement \n")
        f.write("     (Backbone, FPN, PAN, Full). Interaction is excluded to prevent matrix rank deficiency.\n")
        f.write("   - Part 3 (Tukey HSD Post-Hoc): Directly compares every specific modified model against the\n")
        f.write("     Vanilla baseline to see exactly WHICH modifications achieved statistical significance.\n\n")

        # -------------------------------------------------------------------------
        # PART 1: VANILLA BASELINE VS. WITH DCN MODIFICATIONS
        # -------------------------------------------------------------------------
        f.write("--------------------------------------------------------------------------------\n")
        f.write(" PART 1: VANILLA BASELINE VS. WITH DCN MODIFICATIONS (ONE-WAY ANOVA)\n")
        f.write(" Hypothesis: Does applying DCN modifications generally improve tracking performance?\n")
        f.write(" Explanation: This groups all 40 DCN inference runs against the 5 Vanilla runs.\n")
        f.write("--------------------------------------------------------------------------------\n\n")
        
        for metric in ['MOTA', 'IDF1']:
            f.write(f">>> Dependent Variable: {metric}\n")
            model_t1 = ols(f'{metric} ~ C(Type)', data=df).fit()
            anova_t1 = sm.stats.anova_lm(model_t1, typ=2)
            f.write(anova_t1.to_string() + "\n")
            p_val = anova_t1.loc['C(Type)', 'PR(>F)']
            sig = "Significant" if p_val < 0.05 else "Not Significant"
            f.write(f"Conclusion: p = {p_val:.4f} ({sig})\n\n")


        # -------------------------------------------------------------------------
        # PART 2: TWO-WAY ANOVA (ISOLATING DCN ARCHITECTURE)
        # -------------------------------------------------------------------------
        f.write("--------------------------------------------------------------------------------\n")
        f.write(" PART 2: TWO-WAY ANOVA (DCN VERSION x PLACEMENT)\n")
        f.write(" Hypothesis: Do Version (Vanilla, DCNv2, DCNv3) and Placement matter?\n")
        f.write(" Explanation: A main-effects ANOVA measuring the individual impacts of structural \n")
        f.write(" version and architectural placement on tracking performance.\n")
        f.write("--------------------------------------------------------------------------------\n\n")
        
        for metric in ['MOTA', 'IDF1']:
            f.write(f">>> Dependent Variable: {metric}\n")
            model_t2 = ols(f'{metric} ~ C(Version) + C(Placement)', data=df).fit()
            anova_t2 = sm.stats.anova_lm(model_t2, typ=2)
            f.write(anova_t2.to_string() + "\n")
            f.write(f"Version Effect p-value:   {anova_t2.loc['C(Version)', 'PR(>F)']:.6f}\n")
            f.write(f"Placement Effect p-value: {anova_t2.loc['C(Placement)', 'PR(>F)']:.6f}\n\n")

        f.write("===================\n")
        f.write("ACADEMIC INTERPRETATION FOR PART 2 (TWO-WAY ANOVA):\n")
        f.write("A main-effects Two-Way ANOVA was conducted across all models to isolate the independent impacts\n")
        f.write("of structural version and architectural placement on multi-object tracking. Due to the structurally\n")
        f.write("incomplete nature of the Vanilla model (lacking FPN/PAN modifiers), the interaction term was\n")
        f.write("excluded. The results indicate whether the Architecture Version (Vanilla vs. DCNv2 vs. DCNv3)\n")
        f.write("and the specific Placement (Backbone vs Neck) had a statistically significant main effect on\n")
        f.write("tracking accuracy (MOTA) and identity preservation (IDF1) across varying camera angles.\n")
        f.write("===================\n\n")

        # -------------------------------------------------------------------------
        # PART 3: SPECIFIC MODIFICATIONS VS. VANILLA BASELINE (POST-HOC)
        # -------------------------------------------------------------------------
        f.write("--------------------------------------------------------------------------------\n")
        f.write(" PART 3: SPECIFIC MODIFICATIONS VS. VANILLA BASELINE (TUKEY HSD POST-HOC)\n")
        f.write(" Hypothesis: Which specific modified models significantly outperform the Vanilla baseline?\n")
        f.write(" Explanation: A strict pairwise comparison of every modification directly against the \n")
        f.write(" Vanilla baseline to see which ones are statistically valid improvements (p < 0.05).\n")
        f.write("--------------------------------------------------------------------------------\n\n")
        
        for metric in ['MOTA', 'IDF1']:
            f.write(f">>> Dependent Variable: {metric}\n")
            tukey = pairwise_tukeyhsd(endog=df[metric], groups=df['Model'], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            
            vanilla_comparisons = tukey_df[(tukey_df['group1'] == 'Vanilla-YOLOv8n') | (tukey_df['group2'] == 'Vanilla-YOLOv8n')]
            
            def format_comparison(row):
                mod = row['group2'] if row['group1'] == 'Vanilla-YOLOv8n' else row['group1']
                diff = row['meandiff'] if row['group1'] == 'Vanilla-YOLOv8n' else -row['meandiff']
                p_adj = row['p-adj']
                reject = row['reject']
                sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
                return f"Vanilla vs {mod:<15} | Mean Diff: {diff:>7.4f} | p-adj: {p_adj:>6.4f} | Significant: {str(reject):<5} {sig}"
                
            for _, row in vanilla_comparisons.iterrows():
                f.write(format_comparison(row) + "\n")
            f.write("\n")

        f.write("===================\n")
        f.write("ACADEMIC INTERPRETATION FOR PART 3 (POST-HOC):\n")
        f.write("To evaluate the specific impact of structural modifications against the baseline tracking\n")
        f.write("performance, a post-hoc analysis using Tukey's Honestly Significant Difference (HSD) test\n")
        f.write("was conducted. This analysis reveals which exact structural modifications (if any) provided\n")
        f.write("statistically significant gains in tracking accuracy (MOTA) and identity stability (IDF1)\n")
        f.write("over the standard Vanilla YOLOv8n baseline when deployed across multiple operational cameras.\n")
        f.write("===================\n")

    print(f"Tracking ANOVA results successfully saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
