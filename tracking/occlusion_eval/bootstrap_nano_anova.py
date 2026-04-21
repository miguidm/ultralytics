import sys
import os
import argparse
import random
import csv
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')
from ultralytics import YOLO

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/migui/Downloads/OccludedYOLO/OccludedYOLO/inference/images'
BASE_YAML = '/media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/occluded_dataset.yaml'
BOOTSTRAP_DIR = '/media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/bootstrap_data'

N_BOOTSTRAPS = 5
SAMPLE_SIZE = 800

MODELS = {
    'dcnv2': {
        'DCNv2n-Full': '/home/migui/Downloads/dcnv2-yolov8-neck-full-20260318T004120Z-1-001/dcnv2-yolov8-neck-full/weights/best.pt',
        'DCNv2n-FPN':  '/home/migui/Downloads/dcnv2-yolov8-neck-fpn-20260318T004118Z-1-001/dcnv2-yolov8-neck-fpn/weights/best.pt',
        'DCNv2n-Pan':  '/home/migui/Downloads/dcnv2-yolov8-neck-pan-20260318T004653Z-1-001/dcnv2-yolov8-neck-pan/weights/best.pt',
        'DCNv2n-Liu':  '/home/migui/Downloads/dcnv2-yolov8-liu-20260318T004538Z-1-001/dcnv2-yolov8-liu/weights/best.pt',
        'Vanilla-YOLOv8n': '/home/migui/Downloads/100_yolov8n_300epochs_b32-20260318T004620Z-1-001/100_yolov8n_300epochs_b32/weights/best.pt',
    },
    'dcnv3': {
        'DCNv3n-Full': '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_full/weights/best.pt',
        'DCNv3n-FPN':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n-neck-fpn/weights/best.pt',
        'DCNv3n-Pan':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_pan/weights/best.pt',
        'DCNv3n-Liu':  '/home/migui/YOLO_outputs/100_dcnv3_yolov8n_liu/weights/best.pt',
    }
}

def create_bootstrap_datasets():
    os.makedirs(BOOTSTRAP_DIR, exist_ok=True)
    all_images = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(('.jpg', '.png'))]
    
    yaml_paths = []
    for i in range(N_BOOTSTRAPS):
        subset = random.sample(all_images, SAMPLE_SIZE)
        txt_path = os.path.join(BOOTSTRAP_DIR, f'boot_{i}.txt')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(subset))
            
        yaml_path = os.path.join(BOOTSTRAP_DIR, f'boot_{i}.yaml')
        with open(yaml_path, 'w') as f:
            f.write(f"path: /home/migui/Downloads/OccludedYOLO/OccludedYOLO/inference\n")
            f.write(f"train: {txt_path}\n")
            f.write(f"val: {txt_path}\n")
            f.write("names:\n  0: car\n  1: motorcycle\n  2: tricycle\n  3: bus\n  4: van\n  5: truck\n")
        yaml_paths.append(yaml_path)
    return yaml_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['dcnv2', 'dcnv3', 'anova_only'], required=True)
    args = parser.parse_args()

    if args.model_type == 'anova_only':
        run_anova()
        return

    yaml_paths = create_bootstrap_datasets()
    
    csv_file = os.path.join(BOOTSTRAP_DIR, f'bootstrap_results_{args.model_type}.csv')
    fieldnames = ['Model', 'Version', 'Placement', 'Bootstrap_ID', 'mAP50_95', 'Recall']
    
    # Check if CSV exists, write header if not
    write_header = not os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        for model_name, model_path in MODELS[args.model_type].items():
            print(f"\n==========================================")
            print(f" Evaluating {model_name}...")
            print(f"==========================================")
            
            if not os.path.exists(model_path):
                print(f"Skipping {model_name}, file not found: {model_path}")
                continue
                
            try:
                model = YOLO(model_path)
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                continue

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

            for i, yaml_path in enumerate(yaml_paths):
                print(f"  [{model_name}] Bootstrap {i+1}/{N_BOOTSTRAPS} ... ", end='', flush=True)
                try:
                    # project= and name= prevent Path /media/migui/New Volume errors
                    results = model.val(
                        data=yaml_path, conf=0.5, iou=0.5, batch=16, 
                        verbose=False, plots=False, save_json=False,
                        project=os.path.join(BOOTSTRAP_DIR, 'runs'),
                        name=f'{model_name}_{i}',
                        exist_ok=True
                    )
                    map50_95 = float(results.box.map)
                    recall = float(results.box.mr)
                    
                    writer.writerow({
                        'Model': model_name,
                        'Version': version,
                        'Placement': placement,
                        'Bootstrap_ID': i,
                        'mAP50_95': map50_95,
                        'Recall': recall
                    })
                    f.flush()
                    print(f"mAP: {map50_95:.4f}, R: {recall:.4f}")
                except Exception as e:
                    print(f"Error: {e}")

    print(f"\nFinished evaluating {args.model_type} models.")
    print(f"Results saved to {csv_file}")

def run_anova():
    csv_v2 = os.path.join(BOOTSTRAP_DIR, 'bootstrap_results_dcnv2.csv')
    csv_v3 = os.path.join(BOOTSTRAP_DIR, 'bootstrap_results_dcnv3.csv')
    
    if not os.path.exists(csv_v2) or not os.path.exists(csv_v3):
        print("Please run evaluation for both dcnv2 and dcnv3 before running ANOVA.")
        return
        
    df1 = pd.read_csv(csv_v2)
    df2 = pd.read_csv(csv_v3)
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Exclude Vanilla for the Two-Way ANOVA
    df_dcn = df[df['Version'] != 'Vanilla']
    
    print("\n" + "="*80)
    print(f" TWO-WAY ANOVA: DCN Version x Placement")
    print(f" Replicates: {N_BOOTSTRAPS} Sub-sampled datasets (N={N_BOOTSTRAPS} per model)")
    print("="*80)

    # 1. mAP50-95
    print("\n--- Dependent Variable: mAP50-95 ---")
    model_map = ols('mAP50_95 ~ C(Version) + C(Placement) + C(Version):C(Placement)', data=df_dcn).fit()
    anova_map = sm.stats.anova_lm(model_map, typ=2)
    print(anova_map.to_string())
    print("\nInterpretation (mAP50-95):")
    print(f"Version Effect p-value:   {anova_map.loc['C(Version)', 'PR(>F)']:.4f}")
    print(f"Placement Effect p-value: {anova_map.loc['C(Placement)', 'PR(>F)']:.4f}")
    print(f"Interaction p-value:      {anova_map.loc['C(Version):C(Placement)', 'PR(>F)']:.4f}")

    # 2. Recall
    print("\n" + "-"*80)
    print("\n--- Dependent Variable: Recall ---")
    model_r = ols('Recall ~ C(Version) + C(Placement) + C(Version):C(Placement)', data=df_dcn).fit()
    anova_r = sm.stats.anova_lm(model_r, typ=2)
    print(anova_r.to_string())
    print("\nInterpretation (Recall):")
    print(f"Version Effect p-value:   {anova_r.loc['C(Version)', 'PR(>F)']:.4f}")
    print(f"Placement Effect p-value: {anova_r.loc['C(Placement)', 'PR(>F)']:.4f}")
    print(f"Interaction p-value:      {anova_r.loc['C(Version):C(Placement)', 'PR(>F)']:.4f}")
    
    # 3. Best DCN vs Vanilla
    best_dcn_model = df_dcn.groupby('Model')['mAP50_95'].mean().idxmax()
    print("\n" + "="*80)
    print(f" ONE-WAY ANOVA: Best DCN ({best_dcn_model}) vs Vanilla-YOLOv8n")
    print("="*80)
    df_compare = df[df['Model'].isin([best_dcn_model, 'Vanilla-YOLOv8n'])]
    model_comp = ols('mAP50_95 ~ C(Model)', data=df_compare).fit()
    anova_comp = sm.stats.anova_lm(model_comp, typ=2)
    print(anova_comp.to_string())
    print(f"\np-value for Difference: {anova_comp.loc['C(Model)', 'PR(>F)']:.4f}")

if __name__ == "__main__":
    main()
