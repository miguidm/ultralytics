#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Nano Models Evaluation..."
/home/migui/miniconda3/envs/dcnv2/bin/python /media/mydrive/GitHub/ultralytics/tracking/annotated_eval/annotated_bootstrap_anova.py --model-size nano --model-type dcnv2
/home/migui/miniconda3/envs/dcn/bin/python /media/mydrive/GitHub/ultralytics/tracking/annotated_eval/annotated_bootstrap_anova.py --model-size nano --model-type dcnv3
/home/migui/miniconda3/envs/dcn/bin/python /media/mydrive/GitHub/ultralytics/tracking/annotated_eval/annotated_bootstrap_anova.py --model-size nano --model-type anova_only
echo "Nano Models Evaluation Complete!"

echo "Starting Medium Models Evaluation..."
/home/migui/miniconda3/envs/dcnv2/bin/python /media/mydrive/GitHub/ultralytics/tracking/annotated_eval/annotated_bootstrap_anova.py --model-size medium --model-type dcnv2
/home/migui/miniconda3/envs/dcn/bin/python /media/mydrive/GitHub/ultralytics/tracking/annotated_eval/annotated_bootstrap_anova.py --model-size medium --model-type dcnv3
/home/migui/miniconda3/envs/dcn/bin/python /media/mydrive/GitHub/ultralytics/tracking/annotated_eval/annotated_bootstrap_anova.py --model-size medium --model-type anova_only
echo "Medium Models Evaluation Complete!"

echo "All ANOVA statistical analysis finished successfully."
