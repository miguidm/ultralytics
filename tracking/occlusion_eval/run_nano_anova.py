import os
import sys
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

DATASET_YAML = '/media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/occluded_dataset.yaml'
CLASS_NAMES = ['car', 'motorcycle', 'tricycle', 'bus', 'van', 'truck']

# Nano models dictionary
MODELS = {
    'DCNv2n-Full': '/home/migui/Downloads/dcnv2-yolov8-neck-full-20260318T004120Z-1-001/dcnv2-yolov8-neck-full/weights/best.pt',
    'DCNv2n-FPN':  '/home/migui/Downloads/dcnv2-yolov8-neck-fpn-20260318T004118Z-1-001/dcnv2-yolov8-neck-fpn/weights/best.pt',
    'DCNv2n-Pan':  '/home/migui/Downloads/dcnv2-yolov8-neck-pan-20260318T004653Z-1-001/dcnv2-yolov8-neck-pan/weights/best.pt',
    'DCNv2n-Liu':  '/home/migui/Downloads/dcnv2-yolov8-liu-20260318T004538Z-1-001/dcnv2-yolov8-liu/weights/best.pt',
    'DCNv3n-Full': '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_full/weights/best.pt',
    'DCNv3n-FPN':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n-neck-fpn/weights/best.pt',
    'DCNv3n-Pan':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_pan/weights/best.pt',
    'DCNv3n-Liu':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_liu/weights/best.pt',
    'Vanilla-YOLOv8n': '/home/migui/Downloads/100_yolov8n_300epochs_b32-20260318T004620Z-1-001/100_yolov8n_300epochs_b32/weights/best.pt',
}

data_rows = []

print("="*80)
print(" Extracting Per-Class Metrics (mAP50-95 and Recall) for Nano Models")
print("="*80)

for model_name, model_path in MODELS.items():
    if not os.path.exists(model_path):
        print(f"Skipping {model_name}, file not found: {model_path}")
        continue
        
    print(f"  Evaluating: {model_name}...")
    try:
        model = YOLO(model_path)
        # Using verbose=False, save_json=False, plots=False to speed up extraction
        results = model.val(data=DATASET_YAML, conf=0.5, iou=0.5, batch=16, verbose=False, plots=False, save_json=False)
        
        # AP50-95 per class is results.box.ap (array of length equal to number of valid classes)
        ap_per_class = results.box.ap
        # Recall per class is results.box.r
        r_per_class = results.box.r
        
        # We need to map the output array back to the 6 classes
        # YOLO only returns results for classes present in the val set predictions
        # To be safe, we iterate through results.names
        ap_dict = {model.names[c]: ap for c, ap in zip(results.box.ap_class_index, ap_per_class)}
        r_dict = {model.names[c]: r for c, r in zip(results.box.ap_class_index, r_per_class)}
        
        if 'Vanilla' in model_name:
            version = 'Vanilla'
            placement = 'Vanilla'
        else:
            version = 'DCNv2' if 'v2' in model_name else 'DCNv3'
            if 'Full' in model_name: placement = 'Neck-Full'
            elif 'FPN' in model_name: placement = 'Neck-FPN'
            elif 'Pan' in model_name: placement = 'Neck-PAN'
            elif 'Liu' in model_name: placement = 'Backbone'
            else: placement = 'Unknown'
            
        for cls in CLASS_NAMES:
            ap_val = ap_dict.get(cls, 0.0)
            r_val = r_dict.get(cls, 0.0)
            
            data_rows.append({
                'Model': model_name,
                'Version': version,
                'Placement': placement,
                'Class': cls,
                'mAP50_95': float(ap_val),
                'Recall': float(r_val)
            })
    except Exception as e:
        print(f"  Error evaluating {model_name}: {e}")

df = pd.DataFrame(data_rows)
csv_path = '/media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/results/nano_anova_data.csv'
df.to_csv(csv_path, index=False)
print(f"\nData saved to {csv_path}\n")

# ---------------------------------------------------------
# STATISTICAL ANALYSIS
# ---------------------------------------------------------
# Isolate DCN models for Two-Way ANOVA (Vanilla cannot be included since it lacks "Placement")
df_dcn = df[df['Version'] != 'Vanilla'].copy()

print("="*80)
print(" 1. TWO-WAY ANOVA: DCN Version (v2, v3) x Placement (Backbone, FPN, PAN, Full)")
print(" Dependent Variable: mAP50-95 (Per-class, N=6 per model)")
print("="*80)
model_map = ols('mAP50_95 ~ C(Version) + C(Placement) + C(Version):C(Placement)', data=df_dcn).fit()
anova_map = sm.stats.anova_lm(model_map, typ=2)
print(anova_map.to_string())

print("\n" + "="*80)
print(" 2. TWO-WAY ANOVA: DCN Version (v2, v3) x Placement (Backbone, FPN, PAN, Full)")
print(" Dependent Variable: Recall (Per-class, N=6 per model)")
print("="*80)
model_recall = ols('Recall ~ C(Version) + C(Placement) + C(Version):C(Placement)', data=df_dcn).fit()
anova_recall = sm.stats.anova_lm(model_recall, typ=2)
print(anova_recall.to_string())

print("\n" + "="*80)
print(" 3. ONE-WAY ANOVA: All 9 Models (Including Vanilla Baseline)")
print(" To determine if there is a significant difference across ALL architectures.")
print("="*80)
# mAP50-95
model_one_map = ols('mAP50_95 ~ C(Model)', data=df).fit()
anova_one_map = sm.stats.anova_lm(model_one_map, typ=2)
print("--- Dependent Variable: mAP50-95 ---")
print(anova_one_map.to_string())

# Recall
model_one_recall = ols('Recall ~ C(Model)', data=df).fit()
anova_one_recall = sm.stats.anova_lm(model_one_recall, typ=2)
print("\n--- Dependent Variable: Recall ---")
print(anova_one_recall.to_string())
