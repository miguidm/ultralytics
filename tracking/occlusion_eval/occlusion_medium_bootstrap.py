import sys
import os
import argparse
import csv
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sys.path.insert(0, '/media/mydrive/GitHub/ultralytics')
from ultralytics import YOLO

import warnings
warnings.filterwarnings('ignore')

BOOTSTRAP_DIR = '/media/mydrive/GitHub/ultralytics/tracking/occlusion_eval/bootstrap_data'
N_BOOTSTRAPS = 5

MODELS = {
    'dcnv2': {
        'DCNv2m-Full': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Full.pt',
        'DCNv2m-FPN': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-FPN.pt',
        'DCNv2m-Pan': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-Pan.pt',
        'DCNv2m-Liu': '/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-LIU.pt',
        'Vanilla-YOLOv8m': '/home/migui/Downloads/yolov8m-vanilla-20260211T133104Z-1-001/yolov8m-vanilla/weights/best.pt',
    },
    'dcnv3': {
        'DCNv3m-Full': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_full_second/weights/best.pt',
        'DCNv3m-FPN': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_fpn_second/weights/best.pt',
        'DCNv3m-Pan': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_pan_second/weights/best.pt',
        'DCNv3m-Liu': '/home/migui/YOLO_outputs/100_dcnv3_yolov8m_liu_second/weights/best.pt',
    }
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['dcnv2', 'dcnv3', 'anova_only'], required=True)
    args = parser.parse_args()

    if args.model_type == 'anova_only':
        run_anova()
        return

    csv_file = os.path.join(BOOTSTRAP_DIR, f'bootstrap_results_medium_{args.model_type}_occ.csv')
    fieldnames = ['Model', 'Version', 'Placement', 'Bootstrap_ID', 'mAP50_95', 'Recall']
    
    write_header = not os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        for model_name, model_path in MODELS[args.model_type].items():
            print(f"\n==========================================")
            print(f" Evaluating {model_name} on Occluded Data...")
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
                placement = 'None'
            else:
                version = 'DCNv2' if 'v2' in model_name else 'DCNv3'
                if 'Full' in model_name: placement = 'Neck-Full'
                elif 'FPN' in model_name: placement = 'Neck-FPN'
                elif 'Pan' in model_name: placement = 'Neck-PAN'
                elif 'Liu' in model_name: placement = 'Backbone'
                else: placement = 'Unknown'

            # We use the EXACT SAME 5 bootstrapped YAML files generated for the Nano models
            for i in range(N_BOOTSTRAPS):
                yaml_path = os.path.join(BOOTSTRAP_DIR, f'boot_{i}.yaml')
                print(f"  [{model_name}] Bootstrap {i+1}/{N_BOOTSTRAPS} ... ", end='', flush=True)
                try:
                    results = model.val(
                        data=yaml_path, conf=0.5, iou=0.5, batch=16, 
                        verbose=False, plots=False, save_json=False,
                        project=os.path.join(BOOTSTRAP_DIR, 'runs_occ_med'),
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

    print(f"\nFinished evaluating {args.model_type} medium models.")
    print(f"Results saved to {csv_file}")

def run_anova():
    csv_v2 = os.path.join(BOOTSTRAP_DIR, 'bootstrap_results_medium_dcnv2_occ.csv')
    csv_v3 = os.path.join(BOOTSTRAP_DIR, 'bootstrap_results_medium_dcnv3_occ.csv')
    
    if not os.path.exists(csv_v2) or not os.path.exists(csv_v3):
        print("Please run evaluation for both dcnv2 and dcnv3 medium models first.")
        return
        
    df1 = pd.read_csv(csv_v2)
    df2 = pd.read_csv(csv_v3)
    df = pd.concat([df1, df2], ignore_index=True)
    
    df['Type'] = df['Model'].apply(lambda x: 'Vanilla Baseline' if 'Vanilla' in x else 'With DCN Modifications')
    
    output_file = os.path.join(BOOTSTRAP_DIR, 'final_statistical_results_medium_occ.txt')

    with open(output_file, 'w') as f:
        f.write("================================================================================\n")
        f.write(" FINAL OCCLUDED DATASET STATISTICAL ANALYSIS RESULTS (MEDIUM MODELS)\n")
        f.write(" Dataset: OccludedYOLO (800 sampled images per bootstrap, N=5)\n")
        f.write("================================================================================\n\n")

        # PART 1
        f.write("--------------------------------------------------------------------------------\n")
        f.write(" PART 1: VANILLA BASELINE VS. WITH DCN MODIFICATIONS (ONE-WAY ANOVA)\n")
        f.write(" Hypothesis: Does applying DCN modifications generally improve performance under occlusion?\n")
        f.write("--------------------------------------------------------------------------------\n\n")
        
        for metric in ['Recall', 'mAP50_95']:
            f.write(f">>> Dependent Variable: {metric}\n")
            model_t1 = ols(f'{metric} ~ C(Type)', data=df).fit()
            anova_t1 = sm.stats.anova_lm(model_t1, typ=2)
            f.write(anova_t1.to_string() + "\n")
            p_val = anova_t1.loc['C(Type)', 'PR(>F)']
            sig = "Significant" if p_val < 0.05 else "Not Significant"
            f.write(f"Conclusion: p = {p_val:.4f} ({sig})\n\n")

        # PART 2
        f.write("--------------------------------------------------------------------------------\n")
        f.write(" PART 2: TWO-WAY ANOVA (DCN VERSION x PLACEMENT) - Main Effects Only\n")
        f.write(" Note: Interaction term excluded to accommodate Vanilla (lacks Neck placements).\n")
        f.write("--------------------------------------------------------------------------------\n\n")
        
        for metric in ['Recall', 'mAP50_95']:
            f.write(f">>> Dependent Variable: {metric}\n")
            model_t2 = ols(f'{metric} ~ C(Version) + C(Placement)', data=df).fit()
            anova_t2 = sm.stats.anova_lm(model_t2, typ=2)
            f.write(anova_t2.to_string() + "\n")
            f.write(f"Version Effect p-value:   {anova_t2.loc['C(Version)', 'PR(>F)']:.6f}\n")
            f.write(f"Placement Effect p-value: {anova_t2.loc['C(Placement)', 'PR(>F)']:.6f}\n\n")

        # PART 3
        f.write("--------------------------------------------------------------------------------\n")
        f.write(" PART 3: SPECIFIC MODIFICATIONS VS. VANILLA BASELINE (TUKEY HSD POST-HOC)\n")
        f.write(" Hypothesis: Which specific modified models significantly outperform the Vanilla baseline?\n")
        f.write("--------------------------------------------------------------------------------\n\n")
        
        for metric in ['Recall', 'mAP50_95']:
            f.write(f">>> Dependent Variable: {metric}\n")
            tukey = pairwise_tukeyhsd(endog=df[metric], groups=df['Model'], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            
            vanilla_comparisons = tukey_df[(tukey_df['group1'] == 'Vanilla-YOLOv8m') | (tukey_df['group2'] == 'Vanilla-YOLOv8m')]
            
            def format_comparison(row):
                mod = row['group2'] if row['group1'] == 'Vanilla-YOLOv8m' else row['group1']
                diff = row['meandiff'] if row['group1'] == 'Vanilla-YOLOv8m' else -row['meandiff']
                p_adj = row['p-adj']
                reject = row['reject']
                sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
                return f"Vanilla vs {mod:<15} | Mean Diff: {diff:>7.4f} | p-adj: {p_adj:>6.4f} | Significant: {str(reject):<5} {sig}"
                
            for _, row in vanilla_comparisons.iterrows():
                f.write(format_comparison(row) + "\n")
            f.write("\n")

    print(f"Results successfully saved to {output_file}")

if __name__ == "__main__":
    main()
