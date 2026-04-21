import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

base_dir = '/media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/results/nano'
csv_path = os.path.join(base_dir, 'occlusion_summary.csv')

if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
    exit(1)

df_all = pd.read_csv(csv_path)

data_rows = []
classes = ['car', 'motorcycle', 'tricycle', 'bus', 'van', 'truck']

for _, row in df_all.iterrows():
    model_name = row['Model']
    
    # Exclude Vanilla from the Two-Way ANOVA because it doesn't have Version/Placement factors
    if 'Vanilla' in model_name:
        continue
        
    version = 'DCNv2' if 'v2' in model_name else 'DCNv3'
    
    if 'Full' in model_name: placement = 'Neck-Full'
    elif 'FPN' in model_name: placement = 'Neck-FPN'
    elif 'Pan' in model_name: placement = 'Neck-PAN'
    elif 'Liu' in model_name: placement = 'Backbone'
    else: continue

    for cls in classes:
        r_val = float(row[f'R_{cls}'])
        ap_val = float(row[f'AP50_{cls}']) # We use AP50 as proxy since AP50-95 per class isn't saved in CSV
        
        data_rows.append({
            'Model': model_name,
            'Version': version,
            'Placement': placement,
            'Class': cls,
            'Recall': r_val,
            'AP50': ap_val
        })

df_anova = pd.DataFrame(data_rows)

print("="*80)
print(" TWO-WAY ANOVA: DCN Version x Placement (Nano Models)")
print(" Data Source: occlusion_summary.csv")
print(" Replicates: 6 Classes (N=6 per model)")
print("="*80)

# RECALL ANOVA
print("\n--- Dependent Variable: RECALL ---")
model_recall = ols('Recall ~ C(Version) + C(Placement) + C(Version):C(Placement)', data=df_anova).fit()
anova_recall = sm.stats.anova_lm(model_recall, typ=2)
print(anova_recall.to_string())
print("\nInterpretation (Recall):")
print(f"Version Effect p-value:   {anova_recall.loc['C(Version)', 'PR(>F)']:.4f}")
print(f"Placement Effect p-value: {anova_recall.loc['C(Placement)', 'PR(>F)']:.4f}")
print(f"Interaction p-value:      {anova_recall.loc['C(Version):C(Placement)', 'PR(>F)']:.4f}")

# AP50 ANOVA (Proxy for mAP50-95)
print("\n" + "-"*80)
print("\n--- Dependent Variable: AP50 (Proxy for mAP50-95) ---")
model_ap50 = ols('AP50 ~ C(Version) + C(Placement) + C(Version):C(Placement)', data=df_anova).fit()
anova_ap50 = sm.stats.anova_lm(model_ap50, typ=2)
print(anova_ap50.to_string())
print("\nInterpretation (AP50):")
print(f"Version Effect p-value:   {anova_ap50.loc['C(Version)', 'PR(>F)']:.4f}")
print(f"Placement Effect p-value: {anova_ap50.loc['C(Placement)', 'PR(>F)']:.4f}")
print(f"Interaction p-value:      {anova_ap50.loc['C(Version):C(Placement)', 'PR(>F)']:.4f}")
