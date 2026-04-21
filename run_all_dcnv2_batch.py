#!/usr/bin/env python3
"""
Batch Inference for All DCNv2 Models with Skip-if-Done Logic

Processes all DCNv2 models against all videos, skipping already completed combinations.

Directory Structure:
- Models: /media/mydrive/GitHub/ultralytics/modified_model/DCNv2-*.pt
- Videos: /media/mydrive/GitHub/ultralytics/videos/*.mp4
- Results: /media/mydrive/GitHub/ultralytics/tracking/inference_results_new/{model_name}/{video_name}/

Usage:
    python run_all_dcnv2_batch.py
    python run_all_dcnv2_batch.py --force  # Reprocess all, ignore existing
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

# Paths
MODEL_DIR = Path("/media/mydrive/GitHub/ultralytics/modified_model")
VIDEO_DIR = Path("/media/mydrive/GitHub/ultralytics/videos")
RESULTS_DIR = Path("/media/mydrive/GitHub/ultralytics/tracking/inference_results_new")
INFERENCE_SCRIPT = Path("/media/mydrive/GitHub/ultralytics/tracking/inference_dcnv2.py")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch DCNv2 Inference')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing, ignore existing results')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without running')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml',
                        help='Tracker config (default: bytetrack.yaml)')
    return parser.parse_args()


def get_dcnv2_models():
    """Get all DCNv2 model files"""
    models = sorted(MODEL_DIR.glob("DCNv2-*.pt"))
    # Exclude _fixed versions
    models = [m for m in models if '_fixed' not in m.name]
    return models


def get_videos():
    """Get all video files"""
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    return videos


def is_already_processed(model_name, video_name):
    """
    Check if model-video combination has already been processed

    A combination is considered processed if the metrics file exists
    """
    # Remove .pt extension from model name
    model_stem = model_name.replace('.pt', '')
    # Remove .mp4 extension from video name
    video_stem = video_name.replace('.mp4', '')

    # Check for metrics file
    metrics_file = RESULTS_DIR / model_stem / video_stem / f"{video_stem}_{model_stem}_metrics.txt"

    return metrics_file.exists()


def run_inference(model_path, video_path, model_name, video_name, conf, tracker):
    """Run inference for a single model-video combination"""

    # Prepare output paths
    model_stem = model_name.replace('.pt', '')
    video_stem = video_name.replace('.mp4', '')

    output_dir = RESULTS_DIR / model_stem / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / f"{video_stem}_{model_stem}_output.mp4"

    print(f"\n{'='*80}")
    print(f"Processing: {model_stem} + {video_stem}")
    print(f"{'='*80}")

    # Build command
    cmd = [
        'python', str(INFERENCE_SCRIPT),
        '--model', str(model_path),
        '--source', str(video_path),
        '--output', str(output_video),
        '--conf', str(conf),
        '--tracker', tracker,
        '--no-display'
    ]

    print(f"Command: {' '.join(cmd)}\n")

    # Set up environment with CUDA libraries
    env = os.environ.copy()
    cuda_lib_path = "/home/migui/miniconda3/envs/dcn/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    torch_lib_path = "/home/migui/miniconda3/envs/dcn/lib/python3.10/site-packages/torch/lib"

    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}:{env['LD_LIBRARY_PATH']}"
    else:
        env["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{torch_lib_path}"

    # Run inference
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True, env=env)
        print(f"✓ SUCCESS: {model_stem} + {video_stem}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {model_stem} + {video_stem}")
        print(f"   Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ INTERRUPTED by user")
        raise


def main():
    """Main batch processing function"""
    args = parse_args()

    print("="*80)
    print("DCNv2 Batch Inference with Skip-if-Done Logic")
    print("="*80)

    # Get models and videos
    models = get_dcnv2_models()
    videos = get_videos()

    if not models:
        print(f"❌ No DCNv2 models found in {MODEL_DIR}")
        return

    if not videos:
        print(f"❌ No videos found in {VIDEO_DIR}")
        return

    print(f"\nFound {len(models)} DCNv2 models:")
    for model in models:
        print(f"  - {model.name}")

    print(f"\nFound {len(videos)} videos:")
    for video in videos:
        print(f"  - {video.name}")

    print(f"\nResults directory: {RESULTS_DIR}")
    print(f"Force reprocessing: {args.force}")
    print(f"Dry run mode: {args.dry_run}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Tracker: {args.tracker}")

    # Calculate total combinations
    total_combinations = len(models) * len(videos)
    print(f"\nTotal combinations: {total_combinations}")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No inference will be executed")

    # Track statistics
    stats = {
        'total': total_combinations,
        'skipped': 0,
        'processed': 0,
        'success': 0,
        'failed': 0
    }

    # Process each combination
    current = 0

    try:
        for model in models:
            for video in videos:
                current += 1

                model_name = model.name
                video_name = video.name

                print(f"\n{'='*80}")
                print(f"Combination {current}/{total_combinations}")
                print(f"Model: {model_name}")
                print(f"Video: {video_name}")
                print(f"{'='*80}")

                # Check if already processed
                if not args.force and is_already_processed(model_name, video_name):
                    print(f"✓ SKIPPED: Already processed")
                    stats['skipped'] += 1
                    continue

                # Dry run mode - just report what would be done
                if args.dry_run:
                    print(f"🔄 WOULD PROCESS: {model_name} + {video_name}")
                    stats['processed'] += 1
                    stats['success'] += 1
                    continue

                # Run inference
                stats['processed'] += 1
                success = run_inference(
                    model, video, model_name, video_name,
                    args.conf, args.tracker
                )

                if success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1

    except KeyboardInterrupt:
        print(f"\n\n⚠ Batch processing interrupted by user")

    # Print final summary
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Total combinations: {stats['total']}")
    print(f"  Skipped (already done): {stats['skipped']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")

    if stats['skipped'] > 0:
        print(f"\n💡 Tip: Use --force to reprocess skipped combinations")

    print("="*80)


if __name__ == "__main__":
    main()
