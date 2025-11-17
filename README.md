# YOLOv8 + DCNv3 Setup Guide


#### 2. Install Conda (if not already installed)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

#### 4. Clone Repository
```bash
git clone https://github.com/miguidm/ultralytics.git
cd ultralytics
```



---

## Installation

### Option 1: Automated Setup Script

Create and run the setup script:

```bash
# Create setup script
cat > setup_yolo_dcn.sh << 'EOF'
#!/bin/bash
#============================================================
# YOLOv8 + DCNv2 Setup Script (Ubuntu + CUDA 12.x)
# Compatible with CUDA 12.4 and PyTorch 2.6.0
#============================================================

set -e  # Stop on first error

ENV_NAME="torch26_dcnv2"
PYTHON_VERSION=3.11

echo "🚀 Starting setup for $ENV_NAME (Python $PYTHON_VERSION)..."

#------------------------------------------------------------
# 1. Create and activate Conda environment
#------------------------------------------------------------

if conda info --envs | grep -q "$ENV_NAME"; then
    echo "⚙️ Environment $ENV_NAME already exists. Skipping creation."
else
    echo "📦 Creating Conda environment..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

#------------------------------------------------------------
# 2. Install PyTorch with CUDA 12.4
#------------------------------------------------------------

echo "🔥 Installing PyTorch 2.6.0 + cu124..."
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

#------------------------------------------------------------
# 3. Core dependencies
#------------------------------------------------------------

echo "📚 Installing dependencies..."
pip install \
    addict==2.4.0 \
    aliyun-python-sdk-core==2.16.0 \
    aliyun-python-sdk-kms==2.16.5 \
    mmengine==0.10.7 \
    openmim==0.3.9 \
    ultralytics==8.3.212 \
    ultralytics-thop==2.0.17 \
    opencv-python==4.12.0.88 \
    pandas==2.3.3 \
    matplotlib==3.10.7 \
    scipy==1.16.2 \
    pyyaml==6.0.3 \
    requests==2.28.2 \
    tqdm==4.65.2 \
    rich==13.4.2

# Install MMCV
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.6/index.html

#------------------------------------------------------------
# 4. Install DCNv2/DCNv3
#------------------------------------------------------------

echo "🔧 Installing DCNv2 and DCNv3..."

# DCNv2
git clone https://github.com/lucasjinreal/DCNv2_latest.git
cd DCNv2_latest
python setup.py install
cd ..

echo "✅ Installation complete!"
echo "Activate environment with: conda activate $ENV_NAME"
EOF

# Make script executable and run
chmod +x setup_yolo_dcn.sh
./setup_yolo_dcn.sh
```

### Option 2: Manual Installation

```bash
# 1. Create conda environment
conda create -n torch26_dcnv2 python=3.11 -y
conda activate torch26_dcnv2

# 2. Install PyTorch with CUDA 12.4
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. Install dependencies
pip install -r reqs_full.txt

# 4. Install MMCV
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.6/index.html

# 5. Install DCNv2
git clone https://github.com/lucasjinreal/DCNv2_latest.git
cd DCNv2_latest
python setup.py install
cd ..
```

---

## Dataset Setup

### 1. Prepare Your Dataset

Your dataset should follow the YOLO format:

```
datasets/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt
│       └── img2.txt
└── val/
    ├── images/
    └── labels/
```

### 2. Create Dataset Configuration


```yaml
path: /path/to/datasets  # Dataset root directory
train: train    # Train images (relative to path)
val: val/         # Val images (relative to path)

## Training

### 1. Verify Installation

```bash
# Activate environment
conda activate torch26_dcnv2

# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('Ultralytics imported successfully')"
```

### 2. Start Training


```
python train.py
```



### Resume Training
```
run python checkpoint.py
```


