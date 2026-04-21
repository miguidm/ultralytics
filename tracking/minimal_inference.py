import argparse
from ultralytics import YOLO
import os

# --- Environment Diagnosis ---
print("\n--- Environment Diagnosis ---")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"ERROR: Could not import torch: {e}")

try:
    # This is the critical import that has been failing
    import DCNv3
    print(f"Successfully imported DCNv3 from: {DCNv3.__file__}")
except ImportError as e:
    print(f"ERROR: Could not import DCNv3: {e}")
    # Print LD_LIBRARY_PATH to see if it's set correctly
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")
print("---------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Minimal DCNv3 Inference Script")
    parser.add_argument('--model', type=str, required=True, help='Path to DCNv3 model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}")
        return
        
    print(f"Running inference on: {args.image}")
    try:
        results = model.predict(args.image)
        print("Inference successful!")
        
        # Print detected boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                c = int(box.cls)
                class_name = model.names[c]
                conf = float(box.conf)
                print(f"  - Found class: {class_name} with confidence {conf:.2f}")

    except Exception as e:
        print(f"ERROR: Inference failed: {e}")

if __name__ == "__main__":
    main()
