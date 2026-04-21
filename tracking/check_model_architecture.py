#!/usr/bin/env python3
"""
Script to verify if models actually contain DCNv3 layers
"""
import sys
import os
import torch

# Add paths for DCNv3
sys.path.insert(0, "/media/mydrive/GitHub/ultralytics")
dcnv3_path = '/media/mydrive/GitHub/ultralytics/ops_dcnv3'
if os.path.exists(dcnv3_path):
    sys.path.insert(0, dcnv3_path)

from ultralytics import YOLO


def check_model_architecture(model_path):
    """Check if model contains DCNv3 layers"""
    print(f"\n{'='*80}")
    print(f"CHECKING: {os.path.basename(model_path)}")
    print(f"{'='*80}")

    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Get the PyTorch model
    pytorch_model = model.model

    # Count different layer types
    dcnv3_layers = []
    dcnv2_layers = []
    conv_layers = []
    all_layer_types = set()

    def inspect_module(module, prefix=""):
        """Recursively inspect module for layer types"""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            layer_type = type(child).__name__
            all_layer_types.add(layer_type)

            # Check for DCNv3
            if 'DCNv3' in layer_type or 'dcnv3' in layer_type.lower():
                dcnv3_layers.append((full_name, layer_type))

            # Check for DCNv2
            if 'DCNv2' in layer_type or 'dcnv2' in layer_type.lower():
                dcnv2_layers.append((full_name, layer_type))

            # Check for regular Conv
            if 'Conv' in layer_type and 'DCN' not in layer_type:
                conv_layers.append((full_name, layer_type))

            # Recursively check children
            inspect_module(child, full_name)

    print("\nInspecting model architecture...")
    inspect_module(pytorch_model)

    # Print results
    print(f"\n{'='*80}")
    print("LAYER TYPE SUMMARY:")
    print(f"{'='*80}")
    print(f"  - DCNv3 layers found: {len(dcnv3_layers)}")
    print(f"  - DCNv2 layers found: {len(dcnv2_layers)}")
    print(f"  - Regular Conv layers found: {len(conv_layers)}")
    print(f"  - Total unique layer types: {len(all_layer_types)}")

    if dcnv3_layers:
        print(f"\n✓ DCNv3 LAYERS DETECTED:")
        for i, (name, layer_type) in enumerate(dcnv3_layers[:10], 1):  # Show first 10
            print(f"  {i}. {name} ({layer_type})")
        if len(dcnv3_layers) > 10:
            print(f"  ... and {len(dcnv3_layers) - 10} more DCNv3 layers")
    else:
        print(f"\n✗ NO DCNv3 LAYERS FOUND!")

    if dcnv2_layers:
        print(f"\n✓ DCNv2 LAYERS DETECTED:")
        for i, (name, layer_type) in enumerate(dcnv2_layers[:10], 1):
            print(f"  {i}. {name} ({layer_type})")
        if len(dcnv2_layers) > 10:
            print(f"  ... and {len(dcnv2_layers) - 10} more DCNv2 layers")

    print(f"\nAll unique layer types in model:")
    for layer_type in sorted(all_layer_types):
        print(f"  - {layer_type}")

    # Check model metadata
    print(f"\n{'='*80}")
    print("MODEL METADATA:")
    print(f"{'='*80}")

    if hasattr(model, 'ckpt') and model.ckpt is not None:
        ckpt = model.ckpt
        if 'epoch' in ckpt:
            epoch = ckpt['epoch']
            print(f"  - Training epoch: {epoch}")
            if epoch == -1:
                print(f"    ⚠ WARNING: Epoch = -1 indicates this might be a 'last.pt' checkpoint")
                print(f"             from a resumed training, not a 'best.pt' model!")

        # Try to get model file info
        if 'model' in ckpt['train_args']:
            source_model = ckpt['train_args']['model']
            print(f"  - Source model path: {source_model}")
            if 'last.pt' in source_model:
                print(f"    ⚠ Model was resumed from 'last.pt', not 'best.pt'")

    print(f"\n{'='*80}\n")


def main():
    """Main function"""

    models_to_check = [
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-FPN.pt",
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-Full.pt",
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-Pan.pt",
        "/media/mydrive/GitHub/ultralytics/modified_model/DCNv3-Liu.pt",
    ]

    print("\n" + "="*80)
    print("MODEL ARCHITECTURE VERIFICATION TOOL")
    print("="*80)
    print("\nChecking if models actually contain DCNv3 layers...")

    for model_path in models_to_check:
        if os.path.exists(model_path):
            check_model_architecture(model_path)
        else:
            print(f"\n✗ Model not found: {model_path}")

    print("\n" + "="*80)
    print("ARCHITECTURE CHECK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
