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

# Format for the Two-Way ANOVA as exactly requested:
# Independent Variable 1: DCN Version (None, DCNv2, DCNv3).
# Independent Variable 2: Placement (None, Backbone, Neck-FPN, Neck-PAN, Neck-Full).

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

print("="*80)
print(" TWO-WAY ANOVA: DCN Version (None, DCNv2, DCNv3) x Placement (None, Backbone, FPN, PAN, Full)")
print(" Replicates: 5 Sub-sampled datasets (N=5 per model)")
print("="*80)

# 1. mAP50-95
print("\n--- Dependent Variable: mAP50-95 ---")
model_map = ols('mAP50_95 ~ C(Version) + C(Placement) + C(Version):C(Placement)', data=df).fit()
# We use typ=2 for unbalanced designs (since Vanilla doesn't have FPN, PAN, etc.)
anova_map = sm.stats.anova_lm(model_map, typ=2)
print(anova_map.to_string())

print("\nInterpretation (mAP50-95):")
print(f"Version Effect p-value:   {anova_map.loc['C(Version)', 'PR(>F)']:.6f}")
print(f"Placement Effect p-value: {anova_map.loc['C(Placement)', 'PR(>F)']:.6f}")
print(f"Interaction p-value:      {anova_map.loc['C(Version):C(Placement)', 'PR(>F)']:.6f}")

# 2. Recall
print("\n" + "-"*80)
print("\n--- Dependent Variable: Recall ---")
model_r = ols('Recall ~ C(Version) + C(Placement) + C(Version):C(Placement)', data=df).fit()
anova_r = sm.stats.anova_lm(model_r, typ=2)
print(anova_r.to_string())

print("\nInterpretation (Recall):")
print(f"Version Effect p-value:   {anova_r.loc['C(Version)', 'PR(>F)']:.6f}")
print(f"Placement Effect p-value: {anova_r.loc['C(Placement)', 'PR(>F)']:.6f}")
print(f"Interaction p-value:      {anova_r.loc['C(Version):C(Placement)', 'PR(>F)']:.6f}")
