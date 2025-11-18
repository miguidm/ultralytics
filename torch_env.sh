#!/bin/bash

set -e  # Stop on first error
ENV_NAME="dcnv2-yolov8"
PYTHON_VERSION=3.11

echo "🚀 Starting setup for $ENV_NAME (Python $PYTHON_VERSION)..."

# ------------------------------------------------------------
# 1. Create and activate Conda environment
# ------------------------------------------------------------
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "⚙️ Environment $ENV_NAME already exists. Skipping creation."
else
    echo "📦 Creating Conda environment..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# ------------------------------------------------------------
# 2. Install PyTorch with CUDA 12.4 (for RTX 3060)
# ------------------------------------------------------------
echo "🔥 Installing PyTorch 2.6.0 + cu124..."
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------
# 3. Core dependencies
# ------------------------------------------------------------
echo "📚 Installing dependencies..."
pip install \
  addict==2.4.0 aliyun-python-sdk-core==2.16.0 aliyun-python-sdk-kms==2.16.5 \
  mmengine==0.10.7 openmim==0.3.9 \
  ultralytics==8.3.212 ultralytics-thop==2.0.17 \
  opencv-python==4.12.0.88 pandas==2.3.3 matplotlib==3.10.7 \
  scipy==1.16.2 pyyaml==6.0.3 requests==2.28.2 tqdm==4.65.2 rich==13.4.2

pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
