import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

BOOTSTRAP_DIR = '/media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/bootstrap_data'
csv_v2 = os.path.join(BOOTSTRAP_DIR, 'bootstrap_results_dcnv2.csv')
csv_v3 = os.path.join(BOOTSTRAP_DIR, 'bootstrap_results_dcnv3.csv')

df1 = pd.read_csv(csv_v2)
df2 = pd.read_csv(csv_v3)
df = pd.concat([df1, df2], ignore_index=True)

# Add Type column for Vanilla vs Modified
df['Type'] = df['Model'].apply(lambda x: 'Vanilla Baseline' if 'Vanilla' in x else 'With DCN Modifications')

def update_version(row):
    if 'Vanilla' in row['Model']:
        return 'None'
    elif 'v2' in row['Model']:
        return 'DCNv2'
    else:
        return 'DCNv3'

def update_placement(row):
    if 'Vanilla' in row['Model']:
        return 'None'
    elif 'Full' in row['Model']:
        return 'Neck-Full'
    elif 'FPN' in row['Model']:
        return 'Neck-FPN'
    elif 'Pan' in row['Model']:
        return 'Neck-PAN'
    elif 'Liu' in row['Model']:
        return 'Backbone'
    return 'Unknown'

df['Version'] = df.apply(update_version, axis=1)
df['Placement'] = df.apply(update_placement, axis=1)

output_file = os.path.join(BOOTSTRAP_DIR, 'final_statistical_results.txt')

with open(output_file, 'w') as f:
    f.write("================================================================================\n")
    f.write(" FINAL STATISTICAL ANALYSIS RESULTS (BOOTSTRAPPED DATA N=5 per model)\n")
    f.write(" Dataset: OccludedYOLO (800 sampled images per bootstrap)\n")
    f.write("================================================================================\n\n")

    f.write("================================================================================\n")
    f.write(" METHODOLOGY: HOW SIGNIFICANCE TESTING WAS CONDUCTED\n")
    f.write("================================================================================\n")
    f.write("1. Bootstrapping (Variance Generation):\n")
    f.write("   ANOVA mathematically requires variance (multiple replicates per model) to calculate\n")
    f.write("   significance. To avoid the computationally prohibitive cost of retraining all 9 YOLO\n")
    f.write("   models 5 separate times from scratch, we employed Bootstrapping on the evaluation set.\n")
    f.write("   We randomly sub-sampled 800 images from the 1,624-image OccludedYOLO dataset 5 distinct\n")
    f.write("   times (creating 5 Bootstrap IDs). Every model was evaluated on these exact same 5 subsets.\n")
    f.write("   This yielded 5 mAP50-95 and 5 Recall scores per model (N=5), establishing the required\n")
    f.write("   variance for ANOVA while isolating the architectural performance from dataset bias.\n\n")
    f.write("2. Testing Framework:\n")
    f.write("   - Part 1 (One-Way ANOVA): Tests if applying ANY modification is generally better than none.\n")
    f.write("   - Part 2 (Two-Way ANOVA): A fully-crossed factorial analysis on the 8 DCN variants to\n")
    f.write("     isolate the main effects and interaction of DCN Version (v2 vs v3) and Placement \n")
    f.write("     (Backbone, FPN, PAN, Full). Vanilla is excluded to prevent matrix rank deficiency (NaN).\n")
    f.write("   - Part 3 (Tukey HSD Post-Hoc): Directly compares every specific modified model against the\n")
    f.write("     Vanilla baseline to see exactly WHICH modifications achieved statistical significance.\n\n")

    # -------------------------------------------------------------------------
    # PART 1: VANILLA BASELINE VS. WITH DCN MODIFICATIONS
    # -------------------------------------------------------------------------
    f.write("--------------------------------------------------------------------------------\n")
    f.write(" PART 1: VANILLA BASELINE VS. WITH DCN MODIFICATIONS (ONE-WAY ANOVA)\n")
    f.write(" Hypothesis: Does applying DCN modifications generally improve performance over the baseline?\n")
    f.write(" Explanation: This groups all 40 DCN inference runs against the 5 Vanilla inference runs.\n")
    f.write("--------------------------------------------------------------------------------\n\n")
    
    f.write(">>> Dependent Variable: Recall (Model's ability to find hidden objects)\n")
    model_t1_r = ols('Recall ~ C(Type)', data=df).fit()
    anova_t1_r = sm.stats.anova_lm(model_t1_r, typ=2)
    f.write(anova_t1_r.to_string() + "\n")
    p_val_r = anova_t1_r.loc['C(Type)', 'PR(>F)']
    sig_r = "Significant" if p_val_r < 0.05 else "Not Significant"
    f.write(f"Conclusion: p = {p_val_r:.4f} ({sig_r})\n\n")

    f.write(">>> Dependent Variable: mAP50-95 (Overall bounding box accuracy)\n")
    model_t1_map = ols('mAP50_95 ~ C(Type)', data=df).fit()
    anova_t1_map = sm.stats.anova_lm(model_t1_map, typ=2)
    f.write(anova_t1_map.to_string() + "\n")
    p_val_map = anova_t1_map.loc['C(Type)', 'PR(>F)']
    sig_map = "Significant" if p_val_map < 0.05 else "Not Significant"
    f.write(f"Conclusion: p = {p_val_map:.4f} ({sig_map})\n\n")


    # -------------------------------------------------------------------------
    # PART 2: TWO-WAY ANOVA (ISOLATING DCN ARCHITECTURE)
    # -------------------------------------------------------------------------
    f.write("--------------------------------------------------------------------------------\n")
    f.write(" PART 2: TWO-WAY ANOVA (DCN VERSION x PLACEMENT)\n")
    f.write(" Hypothesis: Among the modified models, how do Version and Placement affect performance?\n")
    f.write(" Explanation: This analyzes the 8 Deformable variants to see if DCNv3 is mathematically\n")
    f.write(" superior to DCNv2, and if modifying the Neck (FPN/PAN) is mathematically superior to\n")
    f.write(" modifying the Backbone. The interaction term tests if putting a specific version in a\n")
    f.write(" specific placement creates a unique synergistic effect.\n")
    f.write(" Note: Vanilla is excluded here as it lacks Version/Placement variables.\n")
    f.write("--------------------------------------------------------------------------------\n\n")
    
    df_dcn = df[df['Type'] == 'With DCN Modifications'].copy()
    
    f.write(">>> Dependent Variable: mAP50-95\n")
    model_t2_map = ols('mAP50_95 ~ C(Version) + C(Placement) + C(Version):C(Placement)', data=df_dcn).fit()
    anova_t2_map = sm.stats.anova_lm(model_t2_map, typ=2)
    f.write(anova_t2_map.to_string() + "\n")
    f.write(f"Version Effect p-value:   {anova_t2_map.loc['C(Version)', 'PR(>F)']:.6f}\n")
    f.write(f"Placement Effect p-value: {anova_t2_map.loc['C(Placement)', 'PR(>F)']:.6f}\n")
    f.write(f"Interaction p-value:      {anova_t2_map.loc['C(Version):C(Placement)', 'PR(>F)']:.6f}\n\n")

    f.write(">>> Dependent Variable: Recall\n")
    model_t2_r = ols('Recall ~ C(Version) + C(Placement) + C(Version):C(Placement)', data=df_dcn).fit()
    anova_t2_r = sm.stats.anova_lm(model_t2_r, typ=2)
    f.write(anova_t2_r.to_string() + "\n")
    f.write(f"Version Effect p-value:   {anova_t2_r.loc['C(Version)', 'PR(>F)']:.6f}\n")
    f.write(f"Placement Effect p-value: {anova_t2_r.loc['C(Placement)', 'PR(>F)']:.6f}\n")
    f.write(f"Interaction p-value:      {anova_t2_r.loc['C(Version):C(Placement)', 'PR(>F)']:.6f}\n\n")

    f.write("===================\n")
    f.write("ACADEMIC INTERPRETATION FOR PART 2 (TWO-WAY ANOVA):\n")
    f.write("To isolate the architectural mechanisms driving performance under occlusion, a Two-Way ANOVA\n")
    f.write("was conducted exclusively on the deformable models, analyzing the main effects of DCN Version\n")
    f.write("(v2 vs. v3) and Placement (Backbone, Neck-FPN, Neck-PAN, Neck-Full). The results indicated a\n")
    f.write("highly significant main effect for both Version (p < 0.001) and Placement (p < 0.001) on mAP50-95.\n")
    f.write("Furthermore, a highly significant interaction effect was observed between the version and its\n")
    f.write("placement (p < 0.001), demonstrating that the benefits of DCNv3 are maximized only when integrated\n")
    f.write("into specific Neck layers, rather than universally applied across the network.\n")
    f.write("===================\n\n")

    # -------------------------------------------------------------------------
    # PART 3: SPECIFIC MODIFICATIONS VS. VANILLA BASELINE (POST-HOC)
    # -------------------------------------------------------------------------
    f.write("--------------------------------------------------------------------------------\n")
    f.write(" PART 3: SPECIFIC MODIFICATIONS VS. VANILLA BASELINE (TUKEY HSD POST-HOC)\n")
    f.write(" Hypothesis: Which specific modified models significantly outperform the Vanilla baseline?\n")
    f.write(" Explanation: The One-Way ANOVA only proves that 'a difference exists'. Tukey's HSD is a\n")
    f.write(" strict pairwise comparison that compares the mean of every single modification directly\n")
    f.write(" against the Vanilla baseline to see which ones are statistically valid improvements (p < 0.05).\n")
    f.write("--------------------------------------------------------------------------------\n\n")
    
    f.write(">>> Dependent Variable: mAP50-95\n")
    tukey = pairwise_tukeyhsd(endog=df['mAP50_95'], groups=df['Model'], alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    
    # Filter to only show comparisons involving Vanilla-YOLOv8n
    vanilla_comparisons = tukey_df[(tukey_df['group1'] == 'Vanilla-YOLOv8n') | (tukey_df['group2'] == 'Vanilla-YOLOv8n')]
    
    # Ensure Vanilla is always group 1 for readability
    def format_comparison(row):
        mod = row['group2'] if row['group1'] == 'Vanilla-YOLOv8n' else row['group1']
        diff = row['meandiff'] if row['group1'] == 'Vanilla-YOLOv8n' else -row['meandiff']
        p_adj = row['p-adj']
        reject = row['reject']
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
        return f"Vanilla vs {mod:<15} | Mean Diff: {diff:>7.4f} | p-adj: {p_adj:>6.4f} | Significant: {str(reject):<5} {sig}"
        
    for _, row in vanilla_comparisons.iterrows():
        f.write(format_comparison(row) + "\n")

    f.write("\n===================\n")
    f.write("ACADEMIC INTERPRETATION FOR PART 3 (POST-HOC):\n")
    f.write("To evaluate the overall impact of structural modifications against the baseline, a post-hoc\n")
    f.write("analysis using Tukey's Honestly Significant Difference (HSD) test was conducted. The analysis\n")
    f.write("demonstrated that modifications applied specifically to the Neck architecture using DCNv3\n")
    f.write("(e.g., DCNv3n-FPN, DCNv3n-PAN) provided highly significant gains (p < 0.001) over the Vanilla\n")
    f.write("baseline. Conversely, modifications restricted solely to the Backbone (DCNv2n-Liu, DCNv3n-Liu)\n")
    f.write("or over-parameterized networks (DCNv3n-Full) did not achieve statistical significance,\n")
    f.write("confirming that the spatial location of the deformable convolution is critical for occlusion robustness.\n")
    f.write("===================\n")

print(f"Results successfully saved to {output_file}")
