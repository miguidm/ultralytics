import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

BOOTSTRAP_DIR = '/media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/bootstrap_data'
csv_v2 = os.path.join(BOOTSTRAP_DIR, 'bootstrap_results_dcnv2.csv')
csv_v3 = os.path.join(BOOTSTRAP_DIR, 'bootstrap_results_dcnv3.csv')

df1 = pd.read_csv(csv_v2)
df2 = pd.read_csv(csv_v3)
df = pd.concat([df1, df2], ignore_index=True)

output_file = os.path.join(BOOTSTRAP_DIR, 'anova_statistical_results.txt')

with open(output_file, 'w') as f:
    f.write("================================================================================\n")
    f.write(" STATISTICAL ANALYSIS RESULTS (BOOTSTRAPPED DATA N=5 per model)\n")
    f.write("================================================================================\n\n")

    # TEST 1: Vanilla vs. All Deformable Models
    f.write("--- TEST 1: ONE-WAY ANOVA (Vanilla vs. All Deformable Models) ---\n")
    f.write("Hypothesis: Do Deformable models generally perform better than Vanilla?\n\n")
    df['Type'] = df['Model'].apply(lambda x: 'Vanilla' if 'Vanilla' in x else 'Deformable')
    
    f.write("Dependent Variable: Recall\n")
    model_t1 = ols('Recall ~ C(Type)', data=df).fit()
    anova_t1 = sm.stats.anova_lm(model_t1, typ=2)
    f.write(anova_t1.to_string() + "\n\n")
    
    # TEST 2: Two-Way ANOVA for DCN Architecture
    f.write("--- TEST 2: TWO-WAY ANOVA (Isolating the DCN Architecture) ---\n")
    f.write("Hypothesis: Do DCN Version (v2 vs v3) and Placement (Backbone, FPN, PAN, Full) matter?\n\n")
    
    df_dcn = df[df['Version'] != 'Vanilla']
    
    f.write("Dependent Variable: mAP50-95\n")
    model_t2_map = ols('mAP50_95 ~ C(Version) + C(Placement) + C(Version):C(Placement)', data=df_dcn).fit()
    anova_t2_map = sm.stats.anova_lm(model_t2_map, typ=2)
    f.write(anova_t2_map.to_string() + "\n\n")
    
    f.write("Dependent Variable: Recall\n")
    model_t2_r = ols('Recall ~ C(Version) + C(Placement) + C(Version):C(Placement)', data=df_dcn).fit()
    anova_t2_r = sm.stats.anova_lm(model_t2_r, typ=2)
    f.write(anova_t2_r.to_string() + "\n\n")

    # TEST 3: Best DCN vs Vanilla
    f.write("--- TEST 3: ONE-WAY ANOVA (Best DCN vs Vanilla YOLOv8n) ---\n")
    f.write("Hypothesis: Does the best optimized DCN architecture outperform standard convolutions?\n\n")
    
    best_dcn_model = df_dcn.groupby('Model')['mAP50_95'].mean().idxmax()
    df_compare = df[df['Model'].isin([best_dcn_model, 'Vanilla-YOLOv8n'])]
    
    f.write(f"Comparing: {best_dcn_model} vs Vanilla-YOLOv8n\n")
    f.write("Dependent Variable: mAP50-95\n")
    model_t3 = ols('mAP50_95 ~ C(Model)', data=df_compare).fit()
    anova_t3 = sm.stats.anova_lm(model_t3, typ=2)
    f.write(anova_t3.to_string() + "\n")

print(f"Results successfully saved to {output_file}")
