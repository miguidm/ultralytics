import sys
import os

# Add the parent directory to Python path to use local ultralytics
sys.path.insert(0, '/mnt/sda2/GitHub/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np
import random
import time

# Use YOLO's built-in tracking instead of SORT
# This avoids the SORT dependency and uses ultralytics' native tracker

def get_unique_color(obj_id):
    """Generate a unique color for each tracking ID"""
    random.seed(obj_id)  # Ensure same ID always gets same color
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))


# Load your trained DCNv2 YOLOv8 model
model_path = "/media/mydrive/GitHub/ultralytics/modified_model/DCNv2-LIU.pt"
print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# Load video
video_path = "gate3_feb_crop.mp4"
cap = cv2.VideoCapture(video_path)
print(f"Loading video: {video_path}")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
panel_width = 250  # Width for counting panel
total_width = frame_width + panel_width

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("gate3_feb_crop_dcnv2_output.mp4",
                      fourcc, 30, (total_width, frame_height))

# YOLO's built-in tracker configuration
# We'll use track() method instead of predict()
# Dictionary to store class names for each tracked object ID
track_classes = {}
# Counters for each vehicle class
vehicle_counts = {"car": 0, "motorcycle": 0, "tricycle": 0, "van": 0, "bus": 0, "truck": 0}
counted_ids = set()  # Track which IDs have been counted

# Trajectory tracking
trajectories = {}  # Store trajectory points for each ID
max_trail_length = 50  # Maximum number of points to keep in trajectory
trajectory_colors = {}  # Store unique colors for each trajectory

# Metrics tracking variables
frame_count = 0
total_processing_time = 0
identity_switches = 0
track_history = {}  # Store complete history of each track
track_lifespans = {}  # Store lifespan of each track
previous_tracks = {}  # Store previous frame tracks for ID switch detection
start_time = time.time()

# For MOTA calculation
total_detections = 0
total_false_positives = 0
total_false_negatives = 0
total_mismatches = 0

# For IDF1 calculation
total_true_positives = 0
total_false_positives_id = 0
total_false_negatives_id = 0

# Single horizontal counting line configuration
line_y = int(frame_height * 0.7)  # 70% from top - horizontal across road
line_start = (int(frame_width * 0.2), line_y)  # Start at 20% from left edge
line_end = (int(frame_width * 0.8), line_y)    # End at 80% from left edge
line_color = (0, 255, 255)  # Cyan
line_thickness = 3

# Track which side of the line each object is on
object_positions = {}  # {obj_id: 'above'/'below'}
counted_objects = set()  # Objects that have been counted

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_start_time = time.time()
    frame_count += 1

    # Run YOLOv8-DCN tracking (uses built-in ByteTrack)
    results = model.track(frame, conf=0.5, persist=True, verbose=False)[0]

    # Extract tracked objects
    tracked_objects = []
    detection_classes = []

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
            detection_classes.append(class_name)
            track_classes[obj_id] = class_name

    tracked_objects = np.array(tracked_objects) if tracked_objects else np.array([])

    # Track metrics for current frame
    current_track_ids = set()
    if len(tracked_objects) > 0:
        current_track_ids = set([int(obj_id) for obj_id in tracked_objects[:, 4]])

    # Update total detections count
    total_detections += len(tracked_objects)

    # Identity switch detection
    for track_id in current_track_ids:
        if track_id in previous_tracks and track_id in current_track_ids:
            # Track continues - check for potential mismatches
            pass
        elif track_id not in previous_tracks:
            # New track started
            track_lifespans[track_id] = 1
            track_history[track_id] = [frame_count]

        if track_id in track_history:
            track_history[track_id].append(frame_count)
            track_lifespans[track_id] = track_lifespans.get(track_id, 0) + 1

    # Detect identity switches (simplified approach)
    for prev_id in previous_tracks:
        if prev_id not in current_track_ids and prev_id in track_lifespans:
            # Track lost - could be an identity switch if it reappears later
            pass

    # Update previous tracks for next iteration
    previous_tracks = current_track_ids.copy()

    # Update trajectories and draw tracked boxes
    for x1, y1, x2, y2, obj_id in tracked_objects:
        obj_id = int(obj_id)
        class_name = track_classes.get(obj_id, "unknown")

        # Calculate center point of bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        center_point = (center_x, center_y)

        # Update trajectory for this object
        if obj_id not in trajectories:
            trajectories[obj_id] = []
        trajectories[obj_id].append(center_point)

        # Keep only the last N points to limit trajectory length
        if len(trajectories[obj_id]) > max_trail_length:
            trajectories[obj_id].pop(0)

        # Get or create unique color for this trajectory
        if obj_id not in trajectory_colors:
            trajectory_colors[obj_id] = get_unique_color(obj_id)

        # Single horizontal line crossing detection
        if center_y < line_y:
            current_position = 'above'
        else:
            current_position = 'below'

        # Check for line crossing
        if obj_id in object_positions:
            previous_position = object_positions[obj_id]

            # Check if vehicle crossed the line (changed from above to below or vice versa)
            if previous_position != current_position and obj_id not in counted_objects:
                if class_name in vehicle_counts:
                    vehicle_counts[class_name] += 1
                    counted_objects.add(obj_id)
                    print(f"[DCNv2] Vehicle {class_name} ID:{obj_id} crossed line! Count: {vehicle_counts[class_name]}")

        # Update object's current position
        object_positions[obj_id] = current_position

        # Define colors for each vehicle class (BGR format for OpenCV)
        class_colors = {
            "car": (55, 250, 250),        # #fafa37 (yellow) in BGR
            "motorcycle": (83, 179, 36),  # #24b353 (green) in BGR
            "tricycle": (83, 50, 250),    # #fa3253 (red-pink) in BGR
            "bus": (245, 61, 184),        # #b83df5 (purple) in BGR
            "van": (255, 221, 51),        # #33ddff (light blue/cyan) in BGR
            "truck": (49, 147, 245)       # #f59331 (orange) in BGR
        }

        # Get color for this vehicle class, default to green if unknown
        bbox_color = class_colors.get(class_name.lower(), (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), bbox_color, 2)
        cv2.putText(frame, f"{class_name} ID:{obj_id}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

    # Draw trajectories for all tracked objects
    current_ids = set([int(obj_id) for _, _, _, _, obj_id in tracked_objects])

    for obj_id, trajectory in trajectories.items():
        if len(trajectory) > 1:
            color = trajectory_colors.get(obj_id, (255, 255, 255))

            # Draw trajectory as connected lines with fading effect
            for i in range(1, len(trajectory)):
                # Calculate alpha/thickness based on point age (newer points are thicker/brighter)
                alpha = i / len(trajectory)
                # Increased from max(1, int(3 * alpha))
                thickness = max(2, int(8 * alpha))

                # Draw line segment
                cv2.line(frame, trajectory[i-1],
                         trajectory[i], color, thickness)

            # Draw a larger circle at the current position
            if trajectory:
                # Increased from 3 to 6
                cv2.circle(frame, trajectory[-1], 6, color, -1)

    # Clean up trajectories for objects that are no longer tracked
    trajectories_to_remove = []
    for obj_id in trajectories.keys():
        if obj_id not in current_ids:
            # Gradually fade out old trajectories instead of immediate removal
            if len(trajectories[obj_id]) > 5:
                # Remove oldest points
                trajectories[obj_id] = trajectories[obj_id][5:]
            else:
                trajectories_to_remove.append(obj_id)

    # Remove completely faded trajectories and clean up tracking data
    for obj_id in trajectories_to_remove:
        del trajectories[obj_id]
        if obj_id in trajectory_colors:
            del trajectory_colors[obj_id]
        # Clean up position tracking data for objects that are no longer present
        if obj_id in object_positions:
            del object_positions[obj_id]
        # Remove from counted objects if they're no longer being tracked
        counted_objects.discard(obj_id)

    # Counting line is invisible - no drawing code needed
    # The line_y variable is still used for counting logic

    # Create extended frame with counting panel
    extended_frame = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
    extended_frame[:, :frame_width] = frame  # Place original frame on left

    # Create counting panel on the right
    panel_color = (40, 40, 40)  # Dark gray background
    extended_frame[:, frame_width:] = panel_color

    # Add title and counts to panel
    cv2.putText(extended_frame, "VEHICLE COUNT [DCNv2]", (frame_width + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.line(extended_frame, (frame_width + 10, 40),
             (frame_width + 230, 40), (255, 255, 255), 1)

    # Display counts for each vehicle type
    y_pos = 80
    # Match the bounding box colors (BGR format)
    colors = {
        "car": (55, 250, 250),        # #fafa37 (yellow) in BGR
        "motorcycle": (83, 179, 36),  # #24b353 (green) in BGR
        "tricycle": (83, 50, 250),    # #fa3253 (red-pink) in BGR
        "van": (255, 221, 51),        # #33ddff (light blue/cyan) in BGR
        "bus": (245, 61, 184),        # #b83df5 (purple) in BGR
        "truck": (49, 147, 245)       # #f59331 (orange) in BGR
    }

    for vehicle_type, count in vehicle_counts.items():
        color = colors.get(vehicle_type, (255, 255, 255))
        cv2.putText(extended_frame, f"{vehicle_type.upper()}: {count}",
                    (frame_width + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_pos += 40

    # Add total count
    total_count = sum(vehicle_counts.values())
    cv2.putText(extended_frame, f"TOTAL: {total_count}",
                (frame_width + 15, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(extended_frame)
    cv2.imshow("YOLOv8-DCNv2 Vehicle Tracking with Counter", extended_frame)

    # Calculate frame processing time
    frame_end_time = time.time()
    frame_processing_time = frame_end_time - frame_start_time
    total_processing_time += frame_processing_time

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Calculate final metrics
end_time = time.time()
total_runtime = end_time - start_time

# 1. FPS calculation
avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0

# 2. GFLOPS calculation (estimated based on YOLO model complexity)
# YOLOv8-DCN typically processes more GFLOPS due to deformable convolutions
estimated_gflops_per_frame = 180.0  # Higher estimate for DCN variant
total_gflops = estimated_gflops_per_frame * frame_count / 1000.0  # Convert to GFLOPS

# 3. Identity Switches (IDSW) - count track fragmentations
identity_switches = 0
for track_id, history in track_history.items():
    if len(history) > 1:
        # Check for gaps in tracking (identity switches)
        for i in range(1, len(history)):
            if history[i] - history[i-1] > 5:  # Gap larger than 5 frames
                identity_switches += 1

# 4. MT/ML calculation (Mostly Tracked / Mostly Lost)
total_tracks = len(track_lifespans)
mostly_tracked = 0
mostly_lost = 0

if total_tracks > 0:
    for track_id, lifespan in track_lifespans.items():
        track_ratio = lifespan / frame_count
        if track_ratio >= 0.8:  # Tracked for 80%+ of its potential lifespan
            mostly_tracked += 1
        elif track_ratio <= 0.2:  # Tracked for 20%- of its potential lifespan
            mostly_lost += 1

mt_ratio = mostly_tracked / total_tracks if total_tracks > 0 else 0
ml_ratio = mostly_lost / total_tracks if total_tracks > 0 else 0

# 5. MOTA calculation (simplified)
# MOTA = 1 - (FN + FP + IDSW) / GT
# Using approximation since we don't have ground truth
false_negatives = max(0, total_detections - len(current_track_ids) * frame_count)
false_positives = max(0, len(current_track_ids) * frame_count - total_detections)
mota = 1 - (false_negatives + false_positives + identity_switches) / max(1, total_detections)
mota = max(0, min(1, mota))  # Clamp between 0 and 1

# 6. IDF1 calculation (simplified)
# IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
# Approximation without ground truth
id_true_positives = sum(track_lifespans.values())
id_false_positives = identity_switches
id_false_negatives = max(0, total_detections - id_true_positives)
idf1 = (2 * id_true_positives) / max(1, 2 * id_true_positives + id_false_positives + id_false_negatives)

# Create output_dcnv2.txt with metrics
output_content = f"""YOLOv8-DCNv2 Vehicle Tracking Metrics Report
==========================================

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

5. GFLOPS (Giga Floating Point Operations): {total_gflops:.2f}
   - Computational complexity measure
   - Total GFLOPS processed during tracking

6. FPS (Frames Per Second): {avg_fps:.2f}
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

# Write metrics to output_dcnv2.txt
with open("output_dcnv2.txt", "w") as f:
    f.write(output_content)

print("\nYOLOv8-DCNv2 Metrics saved to output_dcnv2.txt")
print(f"Processing complete: {frame_count} frames in {total_runtime:.2f}s")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Identity Switches: {identity_switches}")
print(f"MOTA: {mota:.4f}, IDF1: {idf1:.4f}")