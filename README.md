# YOLOv8 + DCNv3 Setup Guide

#### 1. Clone Repository
```bash
git clone https://github.com/miguidm/ultralytics.git
cd ultralytics
```
---
#### 2. Run the torch_env.sh script

sh torch_env.sh

---

## 3. Dataset Setup

### Prepare Your Dataset



```
mkdir datasets
cd datasets
```


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
```yaml
path: /datasets  # Dataset root directory
train: /datasets/train/    # Train images (relative to path)
val: /datasets/val/         # Val images (relative to path)

change the details in 100data.yaml based on the relative to path


## Training

### 1. Verify Installation

```bash
# Activate environment
conda activate dcnv2-yolov8

# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('Ultralytics imported successfully')"
```

### 2. Start Training


```
python train.py
```

when asked "Enter name of the model you want to use:"
input DCNv2-Neck-Full then DCNv2-Neck-FPN then DCNv2-Neck-PAN


### Resume Training
```
run python checkpoint.py
```


