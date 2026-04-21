#!/usr/bin/env python3
import os
import sys
import platform

def print_header(title):
    print("\n" + "="*30)
    print(f" {title}")
    print("="*30)

def print_status(name, status, details=""):
    symbol = "✓" if status else "❌"
    print(f"[{symbol}] {name:.<40}{'OK' if status else 'FAIL'}")
    if details:
        print(f"    └── {details}")

def check_env_vars():
    print_header("Environment Variables")
    conda_prefix = os.environ.get('CONDA_PREFIX')
    print_status("CONDA_PREFIX is set", bool(conda_prefix), conda_prefix)
    
    ld_path = os.environ.get('LD_LIBRARY_PATH')
    print_status("LD_LIBRARY_PATH is set", bool(ld_path))
    if ld_path:
        for path in ld_path.split(':'):
            print(f"    - {path}")

def check_python():
    print_header("Python Environment")
    python_version = platform.python_version()
    python_path = sys.executable
    print_status("Python version >= 3.8", True, f"Version: {python_version}")
    print_status("Python executable path", True, python_path)
    
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix and conda_prefix in python_path:
        print_status("Python executable is from CONDA_PREFIX", True)
    else:
        print_status("Python executable is from CONDA_PREFIX", False, "WARNING: Python might not be from the active conda env.")

def check_pytorch():
    print_header("PyTorch and CUDA")
    try:
        import torch
        print_status("PyTorch import", True, f"Version: {torch.__version__}")
        print_status("PyTorch file path", True, torch.__file__)
        
        cuda_available = torch.cuda.is_available()
        print_status("CUDA available via PyTorch", cuda_available)

        if cuda_available:
            print_status("CUDA version", True, f"Version: {torch.version.cuda}")
            print_status("cuDNN version", True, f"Version: {torch.backends.cudnn.version()}")
            try:
                device_name = torch.cuda.get_device_name(0)
                print_status("GPU detected", True, f"Device: {device_name}")
            except Exception as e:
                print_status("GPU detected", False, f"Error: {e}")
        else:
            print("    └── INFO: PyTorch was built without CUDA support or the driver is not accessible.")

    except ImportError as e:
        print_status("PyTorch import", False, f"Error: {e}")
    except Exception as e:
        print_status("PyTorch check", False, f"An unexpected error occurred: {e}")

def check_opencv():
    print_header("OpenCV")
    try:
        import cv2
        print_status("OpenCV import", True, f"Version: {cv2.__version__}")
        print_status("OpenCV file path", True, cv2.__file__)
    except ImportError as e:
        print_status("OpenCV import", False, f"Error: {e}")
    except Exception as e:
        print_status("OpenCV check", False, f"An unexpected error occurred: {e}")

def check_dcnv3():
    print_header("DCNv3 Module")
    
    # Temporarily add local ops_dcnv3 path to simulate inference script environment
    original_sys_path = sys.path[:]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ops_dcnv3_path = os.path.join(project_root, 'ops_dcnv3')
    
    if os.path.isdir(ops_dcnv3_path):
        sys.path.insert(0, ops_dcnv3_path)
        print_status("Local ops_dcnv3 found and added to path", True, ops_dcnv3_path)
    else:
        print_status("Local ops_dcnv3 found", False, f"Path not found: {ops_dcnv3_path}")

    try:
        import DCNv3
        print_status("DCNv3 import", True)
        print_status("DCNv3 file path", True, DCNv3.__file__)
        
        # Check if it's the one from the conda env or a local build
        if 'site-packages' in DCNv3.__file__:
            print("    └── INFO: DCNv3 appears to be installed in the conda environment.")
        else:
            print("    └── INFO: DCNv3 appears to be loaded from a local directory.")
            
    except ImportError as e:
        print_status("DCNv3 import", False, f"Error: {e}")
        print("\n--- FAILED: Key DCNv3 module could not be imported ---")
        print("This is the main reason the inference script is failing.")
        print("The error above indicates that a required library (like PyTorch's libc10.so) could not be found when loading the DCNv3 module.")
        print("Please ensure your environment is correctly configured.")

    finally:
        # Clean up sys.path
        sys.path = original_sys_path

if __name__ == "__main__":
    print("======================================================================")
    print(" DCNv3 Environment Diagnostic Script")
    print("======================================================================")
    check_env_vars()
    check_python()
    check_pytorch()
    check_opencv()
    check_dcnv3()
    print("\n--- Diagnosis Complete ---\n")
