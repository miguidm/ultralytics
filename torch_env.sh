#!/bin/bash
# ============================================================
# YOLOv8 + DCNv3 Local Setup Script (Ubuntu + RTX 3060)
# Compatible with CUDA 12.x and PyTorch 2.6.0
# Author: James Abarca
# ============================================================

set -e  # Stop on first error
ENV_NAME="torch26_dcnv3"
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
  mmcv==2.2.0 mmengine==0.10.7 openmim==0.3.9 \
  ultralytics==8.3.212 ultralytics-thop==2.0.17 \
  opencv-python==4.12.0.88 pandas==2.3.3 matplotlib==3.10.7 \
  scipy==1.16.2 pyyaml==6.0.3 requests==2.28.2 tqdm==4.65.2 rich==13.4.2

# ------------------------------------------------------------
# 4. Clone and build DCNv3 (InternImage)
# ------------------------------------------------------------
WORKDIR="$HOME/Projects"
REPO_DIR="$WORKDIR/InternImage"

mkdir -p $WORKDIR
cd $WORKDIR

if [ ! -d "$REPO_DIR" ]; then
    echo "📂 Cloning InternImage repository..."
    git clone https://github.com/OpenGVLab/InternImage.git
else
    echo "🔄 Updating existing InternImage repo..."
    cd InternImage && git pull
fi

cd $REPO_DIR/detection/ops_dcnv3

DCN_SRC="$REPO_DIR/detection/ops_dcnv3/src/cuda/dcnv3_cuda.cu"
if grep -q "input.type()" "$DCN_SRC"; then
  echo "⚠️ Applying DCNv3 patch for PyTorch 2.6..."
  sed -i 's/input.type()/input.scalar_type()/g' "$DCN_SRC"
  sed -i 's/offset.type()/offset.scalar_type()/g' "$DCN_SRC"
  sed -i 's/mask.type()/mask.scalar_type()/g' "$DCN_SRC"
fi

echo "🧱 Building DCNv3 CUDA extension..."
if ! python setup.py build install; then
    echo "⚠️ Build failed — applying PyTorch 2.6 compatibility patch..."

    # ---------------- PATCH SECTION ----------------
    cp src/cuda/dcnv3_cuda.cu src/cuda/dcnv3_cuda.cu.bak
    cp src/dcnv3.h src/dcnv3.h.bak

    # Fix .type() → .scalar_type() conversions
    sed -i 's/\.type()/\.scalar_type()/g' src/cuda/dcnv3_cuda.cu
    sed -i 's/\.type()/\.scalar_type()/g' src/dcnv3.h

    # Fix .type().is_cuda() → .is_cuda()
    sed -i 's/\.scalar_type().is_cuda()/\.is_cuda()/g' src/cuda/dcnv3_cuda.cu
    sed -i 's/\.scalar_type().is_cuda()/\.is_cuda()/g' src/dcnv3.h

    echo "🔧 Patch applied. Rebuilding DCNv3..."
    rm -rf build
    export TORCH_CXX11_ABI=1
    export MAX_JOBS=4
    python setup.py build install
    echo "✅ DCNv3 rebuild successful after patch."
fi

# ------------------------------------------------------------
# 5. Verify DCNv3 installation
# ------------------------------------------------------------
echo "🧪 Testing DCNv3 import..."
python - <<'PYCODE'
import torch, sys
sys.path.insert(0, "/home/james/Projects/InternImage/detection")
from ops_dcnv3.modules.dcnv3 import DCNv3
print("✅ DCNv3 import successful!")
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(1, 32, 32, 32, 64, device=device)
print("Input tensor:", x.shape)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("✅ CUDA and DCNv3 are working fine!")
PYCODE

# ------------------------------------------------------------
# 6. Print success message
# ------------------------------------------------------------
echo "🎉 Setup complete! You can now use YOLOv8 with DCNv3 in your environment '$ENV_NAME'."
echo "To activate it next time:  conda activate $ENV_NAME"
