import sys
import os
import glob

print("Script started, importing modules...", flush=True)

# Add the parent directory to Python path to use local ultralytics
sys.path.insert(0, '/mnt/sda2/GitHub/ultralytics')
print("Path updated", flush=True)

from ultralytics import YOLO
import cv2
import numpy as np
import random
import time
from pathlib import Path

def get_unique_color(obj_id):
    """Generate a unique color for each tracking ID"""
    random.seed(obj_id)
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

def check_inference_complete(output_dir, video_name, model_name):
    """Check if inference is already complete for this video-model combination"""
    metrics_file = output_dir / f"{video_name}_{model_name}_metrics.txt"
    output_video = output_dir / f"{video_name}_{model_name}_output.mp4"

    if metrics_file.exists() and output_video.exists():
        print(f"✓ Already complete: {model_name} - {video_name}")
        return True
    return False

def run_inference(model_path, video_path, output_dir, model_name):
    """Run inference on a video with the specified model"""
    video_name = Path(video_path).stem

    # Check if already done
    if check_inference_complete(output_dir, video_name, model_name):
        return

    print(f"\n{'='*80}")
    print(f"Running inference: {model_name} on {video_name}")
    print(f"{'='*80}\n")

    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Load video
    cap = cv2.VideoCapture(video_path)
    print(f"Loading video: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_width = 250
    total_width = frame_width + panel_width

    # Create output video
    output_video_path = output_dir / f"{video_name}_{model_name}_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, 30, (total_width, frame_height))

    # Tracking variables
    track_classes = {}
    vehicle_counts = {"car": 0, "motorcycle": 0, "tricycle": 0, "van": 0, "bus": 0, "truck": 0}
    counted_objects = set()

    # Trajectory tracking
    trajectories = {}
    max_trail_length = 50
    trajectory_colors = {}

    # Metrics tracking
    frame_count = 0
    total_processing_time = 0
    identity_switches = 0
    track_history = {}
    track_lifespans = {}
    previous_tracks = {}
    start_time = time.time()

    total_detections = 0

    # Counting line configuration
    line_y = int(frame_height * 0.7)
    object_positions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        frame_count += 1

        # Run tracking
        results = model.track(frame, conf=0.5, persist=True, verbose=False)[0]

        # Extract tracked objects
        tracked_objects = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                obj_id = track_ids[i]
                conf = confidences[i]
                class_id = class_ids[i]
                class_name = model.names[class_id]

                tracked_objects.append([x1, y1, x2, y2, obj_id])
                track_classes[obj_id] = class_name

        tracked_objects = np.array(tracked_objects) if tracked_objects else np.array([])

        # Track metrics
        current_track_ids = set()
        if len(tracked_objects) > 0:
            current_track_ids = set([int(obj_id) for obj_id in tracked_objects[:, 4]])

        total_detections += len(tracked_objects)

        # Identity switch detection
        for track_id in current_track_ids:
            if track_id not in previous_tracks:
                track_lifespans[track_id] = 1
                track_history[track_id] = [frame_count]

            if track_id in track_history:
                track_history[track_id].append(frame_count)
                track_lifespans[track_id] = track_lifespans.get(track_id, 0) + 1

        previous_tracks = current_track_ids.copy()

        # Update trajectories and draw
        for x1, y1, x2, y2, obj_id in tracked_objects:
            obj_id = int(obj_id)
            class_name = track_classes.get(obj_id, "unknown")

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_point = (center_x, center_y)

            # Update trajectory
            if obj_id not in trajectories:
                trajectories[obj_id] = []
            trajectories[obj_id].append(center_point)

            if len(trajectories[obj_id]) > max_trail_length:
                trajectories[obj_id].pop(0)

            if obj_id not in trajectory_colors:
                trajectory_colors[obj_id] = get_unique_color(obj_id)

            # Line crossing detection
            if center_y < line_y:
                current_position = 'above'
            else:
                current_position = 'below'

            if obj_id in object_positions:
                previous_position = object_positions[obj_id]

                if previous_position != current_position and obj_id not in counted_objects:
                    if class_name in vehicle_counts:
                        vehicle_counts[class_name] += 1
                        counted_objects.add(obj_id)
                        print(f"[{model_name}] Vehicle {class_name} ID:{obj_id} crossed line! Count: {vehicle_counts[class_name]}")

            object_positions[obj_id] = current_position

            # Class colors
            class_colors = {
                "car": (55, 250, 250),
                "motorcycle": (83, 179, 36),
                "tricycle": (83, 50, 250),
                "bus": (245, 61, 184),
                "van": (255, 221, 51),
                "truck": (49, 147, 245)
            }

            bbox_color = class_colors.get(class_name.lower(), (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 2)
            cv2.putText(frame, f"{class_name} ID:{obj_id}", (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

        # Draw trajectories
        current_ids = set([int(obj_id) for _, _, _, _, obj_id in tracked_objects])

        for obj_id, trajectory in trajectories.items():
            if len(trajectory) > 1:
                color = trajectory_colors.get(obj_id, (255, 255, 255))

                for i in range(1, len(trajectory)):
                    alpha = i / len(trajectory)
                    thickness = max(2, int(8 * alpha))
                    cv2.line(frame, trajectory[i-1], trajectory[i], color, thickness)

                if trajectory:
                    cv2.circle(frame, trajectory[-1], 6, color, -1)

        # Clean up old trajectories
        trajectories_to_remove = []
        for obj_id in trajectories.keys():
            if obj_id not in current_ids:
                if len(trajectories[obj_id]) > 5:
                    trajectories[obj_id] = trajectories[obj_id][5:]
                else:
                    trajectories_to_remove.append(obj_id)

        for obj_id in trajectories_to_remove:
            del trajectories[obj_id]
            if obj_id in trajectory_colors:
                del trajectory_colors[obj_id]
            if obj_id in object_positions:
                del object_positions[obj_id]
            counted_objects.discard(obj_id)

        # Create extended frame with panel
        extended_frame = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
        extended_frame[:, :frame_width] = frame

        panel_color = (40, 40, 40)
        extended_frame[:, frame_width:] = panel_color

        # Add title and counts
        cv2.putText(extended_frame, f"VEHICLE COUNT [{model_name}]", (frame_width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.line(extended_frame, (frame_width + 10, 40), (frame_width + 230, 40), (255, 255, 255), 1)

        y_pos = 80
        colors = {
            "car": (55, 250, 250),
            "motorcycle": (83, 179, 36),
            "tricycle": (83, 50, 250),
            "van": (255, 221, 51),
            "bus": (245, 61, 184),
            "truck": (49, 147, 245)
        }

        for vehicle_type, count in vehicle_counts.items():
            color = colors.get(vehicle_type, (255, 255, 255))
            cv2.putText(extended_frame, f"{vehicle_type.upper()}: {count}",
                       (frame_width + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 40

        total_count = sum(vehicle_counts.values())
        cv2.putText(extended_frame, f"TOTAL: {total_count}",
                   (frame_width + 15, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(extended_frame)

        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        total_processing_time += frame_processing_time

        # Progress update every 100 frames
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")

    cap.release()
    out.release()

    # Calculate metrics
    end_time = time.time()
    total_runtime = end_time - start_time
    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0

    # Identity switches
    identity_switches = 0
    for track_id, history in track_history.items():
        if len(history) > 1:
            for i in range(1, len(history)):
                if history[i] - history[i-1] > 5:
                    identity_switches += 1

    # MT/ML calculation
    total_tracks = len(track_lifespans)
    mostly_tracked = 0
    mostly_lost = 0

    if total_tracks > 0:
        for track_id, lifespan in track_lifespans.items():
            track_ratio = lifespan / frame_count
            if track_ratio >= 0.8:
                mostly_tracked += 1
            elif track_ratio <= 0.2:
                mostly_lost += 1

    mt_ratio = mostly_tracked / total_tracks if total_tracks > 0 else 0
    ml_ratio = mostly_lost / total_tracks if total_tracks > 0 else 0

    # MOTA calculation
    false_negatives = max(0, total_detections - len(current_track_ids) * frame_count)
    false_positives = max(0, len(current_track_ids) * frame_count - total_detections)
    mota = 1 - (false_negatives + false_positives + identity_switches) / max(1, total_detections)
    mota = max(0, min(1, mota))

    # IDF1 calculation
    id_true_positives = sum(track_lifespans.values())
    id_false_positives = identity_switches
    id_false_negatives = max(0, total_detections - id_true_positives)
    idf1 = (2 * id_true_positives) / max(1, 2 * id_true_positives + id_false_positives + id_false_negatives)

    # Create metrics report
    metrics_content = f"""YOLOv8-{model_name} Vehicle Tracking Metrics Report
==========================================
Video: {video_name}
Model: {model_name}

1. IDF1 (Identity F1 Score): {idf1:.4f}
   - Measures identity preservation accuracy
   - Range: 0.0 (worst) to 1.0 (best)

2. MT/ML (Mostly Tracked/Lost Ratios):
   - MT (Mostly Tracked): {mt_ratio:.4f} ({mostly_tracked}/{total_tracks} tracks)
   - ML (Mostly Lost): {ml_ratio:.4f} ({mostly_lost}/{total_tracks} tracks)
   - Tracks tracked ≥80% of time vs ≤20% of time

3. IDSW (Identity Switches): {identity_switches}
   - Number of identity switches detected
   - Lower is better (0 = no switches)

4. MOTA (Multiple Object Tracking Accuracy): {mota:.4f}
   - Overall tracking accuracy measure
   - Range: 0.0 (worst) to 1.0 (best)

5. FPS (Frames Per Second): {avg_fps:.2f}
   - Processing speed measure
   - Higher is better for real-time applications

Additional Statistics:
- Total Frames Processed: {frame_count}
- Total Runtime: {total_runtime:.2f} seconds
- Total Tracks Created: {total_tracks}
- Total Detections: {total_detections}
- Average Track Lifespan: {sum(track_lifespans.values())/max(1,len(track_lifespans)):.1f} frames

Vehicle Counts:
{chr(10).join([f"- {vehicle.upper()}: {count}" for vehicle, count in vehicle_counts.items()])}
- TOTAL VEHICLES: {sum(vehicle_counts.values())}
"""

    # Write metrics
    metrics_file = output_dir / f"{video_name}_{model_name}_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(metrics_content)

    print(f"\n✓ Inference complete!")
    print(f"  Output video: {output_video_path}")
    print(f"  Metrics file: {metrics_file}")
    print(f"  Processed {frame_count} frames in {total_runtime:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  MOTA: {mota:.4f}, IDF1: {idf1:.4f}")
    print(f"  Total vehicles counted: {sum(vehicle_counts.values())}")

def main():
    print("Starting main function...", flush=True)
    # Configuration
    videos_dir = Path("/media/mydrive/GitHub/ultralytics/videos")
    models_dir = Path("/media/mydrive/GitHub/ultralytics/modified_model")
    output_base_dir = Path("/media/mydrive/GitHub/ultralytics/tracking/inference_results_new")

    print("Getting videos...", flush=True)
    # Get all videos
    videos = sorted(videos_dir.glob("*.mp4"))
    print(f"Found {len(videos)} videos", flush=True)

    # DCNv2 models only
    dcnv2_models = [
        "DCNv2-FPN.pt",
        "DCNv2-Full.pt",
        "DCNv2-LIU.pt",
        "DCNv2-Pan.pt"
    ]

    print("\n" + "="*80)
    print("DCNv2 Inference Runner")
    print("="*80)
    print(f"\nFound {len(videos)} videos:")
    for v in videos:
        print(f"  - {v.name}")
    print(f"\nDCNv2 Models to process: {len(dcnv2_models)}")
    for m in dcnv2_models:
        print(f"  - {m}")
    print("\n" + "="*80 + "\n")

    # Process each model-video combination
    total_combinations = len(videos) * len(dcnv2_models)
    completed = 0
    skipped = 0
    processed = 0

    for model_file in dcnv2_models:
        model_name = model_file.replace('.pt', '')
        model_path = models_dir / model_file

        if not model_path.exists():
            print(f"⚠ Warning: Model not found: {model_path}")
            continue

        # Create output directory for this model
        output_dir = output_base_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for video_path in videos:
            video_name = video_path.stem

            # Create subdirectory for this video
            video_output_dir = output_dir / video_name
            video_output_dir.mkdir(parents=True, exist_ok=True)

            # Check if already complete
            if check_inference_complete(video_output_dir, video_name, model_name):
                skipped += 1
                completed += 1
                continue

            # Run inference
            try:
                run_inference(str(model_path), str(video_path), video_output_dir, model_name)
                processed += 1
                completed += 1
            except Exception as e:
                print(f"\n✗ Error processing {model_name} - {video_name}:")
                print(f"  {str(e)}")
                import traceback
                traceback.print_exc()

            print(f"\nProgress: {completed}/{total_combinations} combinations complete")

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total combinations: {total_combinations}")
    print(f"Already complete (skipped): {skipped}")
    print(f"Newly processed: {processed}")
    print(f"Total complete: {completed}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
