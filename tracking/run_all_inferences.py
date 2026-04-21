#!/usr/bin/env python3
"""
Runner script to perform inference for multiple models on a given video.
"""

import os
import subprocess
from pathlib import Path
import json

def main():
    """Main function"""
    
    # Assuming this script is in the 'tracking' directory
    tracking_dir = Path(__file__).parent.resolve()
    project_root = tracking_dir.parent

    # List of models to test from the new DCNv3 models directory
    modified_model_dir = 'tracking/DCNv3-models'
    models = [str(p.relative_to(project_root)) for p in (project_root / modified_model_dir).glob('*DCNv3*.pt')]

    # List of videos to run inference on from videos directory
    videos_dir = 'videos'
    video_sources = [str(p.relative_to(project_root)) for p in (project_root / videos_dir).glob('*.mp4')]

    # Directory to save the output files
    output_dir = 'tracking/inference_results_new'
    
    # Absolute path to the DCNv3 inference script
    inference_script = tracking_dir / 'inference_dcnv3.py'

    # Get the dcn conda environment's lib path
    try:
        conda_info_process = subprocess.run(['conda', 'info', '--envs', '--json'], capture_output=True, text=True, check=True)
        conda_info = json.loads(conda_info_process.stdout)
        dcn_env_path = None
        for env in conda_info['envs']:
            if env.endswith('/dcn'):
                dcn_env_path = env
                break
        
        if dcn_env_path:
            dcn_lib_path = os.path.join(dcn_env_path, 'lib')
            env = os.environ.copy()
            if 'LD_LIBRARY_PATH' in env:
                env['LD_LIBRARY_PATH'] = f"{dcn_lib_path}:{env['LD_LIBRARY_PATH']}"
            else:
                env['LD_LIBRARY_PATH'] = dcn_lib_path
            print(f"[Info] Setting LD_LIBRARY_PATH to include: {dcn_lib_path}")
        else:
            print("[Warning] 'dcn' conda environment not found. Using default environment.")
            env = os.environ.copy()

    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        print("[Warning] Could not determine conda path, using default environment.")
        env = os.environ.copy()

    print("="*70)
    print("STARTING BATCH DCNv3 INFERENCE")
    print("="*70)
    
    for model_path_str in models:
        model_path = project_root / model_path_str
        
        if not model_path.exists():
            print(f"SKIPPING: Model not found at {model_path}")
            continue

        for video_source_str in video_sources:
            video_source = project_root / video_source_str
            
            if not video_source.exists():
                print(f"SKIPPING: Video not found at {video_source}")
                continue

            print(f"\nProcessing model: {model_path.name} on video: {video_source.name}")
            
            # Create a specific output directory for each model-video pair
            current_output_dir = project_root / output_dir / model_path.stem / video_source.stem
            current_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Construct the full path for the output video
            output_video_path = current_output_dir / f"{video_source.stem}_output.mp4"

            command = [
                'conda', 'run', '-n', 'dcn', 'python', str(inference_script),
                '--model', str(model_path),
                '--source', str(video_source),
                '--output', str(output_video_path),
            ]
            
            try:
                # Using shell=True and passing command as a string can help with complex envs
                subprocess.run(" ".join(command), check=True, env=env, shell=True)
                print(f"✓ Successfully processed model: {model_path.name} on video: {video_source.name}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Error processing model {model_path.name} on video {video_source.name}: {e}")
            except FileNotFoundError:
                print(f"✗ Error: 'python' command not found. Make sure Python is in your PATH.")
                break # Stop if python is not found


    print("\n" + "="*70)
    print("BATCH INFERENCE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
