from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort
import random
import time


def get_unique_color(obj_id):
    """Generate a unique color for each tracking ID"""
    random.seed(obj_id)
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))


# Load both YOLO models
model_yolov8n = YOLO("last.pt")
model_dcn_yolov8 = YOLO("best-dcn.pt")

print("Both models loaded: yolov8n and dcn-yolov8")

# Load video (same as main_gate3)
cap = cv2.VideoCapture("gate3.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
panel_width = 250  # Width for counting panel (same as main_gate3)

# Create side-by-side output video with panels
single_section_width = frame_width + panel_width
total_width = single_section_width * 2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("model_comparison_output_gate3.mp4",
                      fourcc, 30, (total_width, frame_height))

# Initialize separate SORT trackers for each model (same parameters as main_gate3)
tracker_yolov8n = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
tracker_dcn_yolov8 = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

# Separate tracking data for each model
track_classes_yolov8n = {}
track_classes_dcn_yolov8 = {}
track_confidences_yolov8n = {}
track_confidences_dcn_yolov8 = {}
trajectories_yolov8n = {}
trajectories_dcn_yolov8 = {}
trajectory_colors_yolov8n = {}
trajectory_colors_dcn_yolov8 = {}
max_trail_length = 50  # Same as main_gate3

# Vehicle counting for both models (same as main_gate3)
vehicle_counts_yolov8n = {"car": 0, "motorcycle": 0, "tricycle": 0, "van": 0}
vehicle_counts_dcn_yolov8 = {
    "car": 0, "motorcycle": 0, "tricycle": 0, "van": 0}
counted_ids_yolov8n = set()
counted_ids_dcn_yolov8 = set()

# Curved counting line configuration
line_color = (147, 20, 255)  # Pink
line_thickness = 3

# Define curved line points


def get_curved_line_points(frame_width, frame_height):
    # Diagonal line from top-left to bottom-right for better perspective matching
    points = np.array([
        [0, int(frame_height * 0.45)],           # Top-left - higher up
        [frame_width, int(frame_height * 0.75)]  # Bottom-right - lower down
    ], np.int32)
    return points.reshape((-1, 1, 2))


# Track which side of the line each object is on
object_positions_yolov8n = {}  # {obj_id: 'above'/'below'}
counted_objects_yolov8n = set()  # Objects that have been counted
object_positions_dcn_yolov8 = {}
counted_objects_dcn_yolov8 = set()

# Metrics tracking variables (same as main_gate3)
frame_count = 0
total_processing_time = 0
start_time = time.time()


def process_model_detections(frame, results, model, tracker, track_classes, track_confidences, trajectories, trajectory_colors,
                             vehicle_counts, object_positions, counted_objects, model_name):
    """Process detections for one model - exactly like main_gate3"""

    # Convert YOLO detections (exactly like main_gate3)
    detections = []
    detection_classes = []
    detection_confidences = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        detections.append([x1, y1, x2, y2, conf])
        detection_classes.append(class_name)
        detection_confidences.append(conf)
    detections = np.array(detections)

    # Update tracker
    tracked_objects = tracker.update(detections)

    # Map detections to tracked objects (exactly like main_gate3)
    if len(tracked_objects) > 0 and len(detection_classes) > 0:
        for x1, y1, x2, y2, obj_id in tracked_objects:
            track_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            min_dist = float('inf')
            best_class = "unknown"
            best_confidence = 0.0

            for i, class_name in enumerate(detection_classes):
                if i < len(detections):
                    det_center = np.array([(detections[i][0] + detections[i][2]) / 2,
                                           (detections[i][1] + detections[i][3]) / 2])
                    dist = np.linalg.norm(track_center - det_center)
                    if dist < min_dist:
                        min_dist = dist
                        best_class = class_name
                        best_confidence = detection_confidences[i]

            track_classes[int(obj_id)] = best_class
            track_confidences[int(obj_id)] = best_confidence

    # Update trajectories and draw tracked boxes (exactly like main_gate3)
    for x1, y1, x2, y2, obj_id in tracked_objects:
        obj_id = int(obj_id)
        class_name = track_classes.get(obj_id, "unknown")
        confidence = track_confidences.get(obj_id, 0.0)

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

        # Diagonal line crossing detection - use interpolation to find Y value at object's X position
        # Get frame dimensions from the frame shape
        current_frame_height, current_frame_width = frame.shape[:2]

        # Simple diagonal line interpolation (2 points)
        x_positions = [0, current_frame_width]
        y_positions = [int(current_frame_height * 0.45), int(current_frame_height * 0.75)]

        # Find the interpolated Y value for the object's X position
        line_y_at_x = np.interp(center_x, x_positions, y_positions)

        if center_y < line_y_at_x:
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
                    print(
                        f"{model_name} - Vehicle {class_name} ID:{obj_id} crossed line! Count: {vehicle_counts[class_name]}")

        # Update object's current position
        object_positions[obj_id] = current_position

        # Define colors for each vehicle class (exactly like main_gate3)
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
        cv2.putText(frame, f"{class_name} ID:{obj_id} ({confidence:.2f})", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

    # Draw the curved counting line
    current_frame_height, current_frame_width = frame.shape[:2]
    curved_points = get_curved_line_points(
        current_frame_width, current_frame_height)
    cv2.polylines(frame, [curved_points], isClosed=False,
                  color=line_color, thickness=line_thickness)

    # Draw trajectories for all tracked objects (exactly like main_gate3)
    current_ids = set([int(obj_id) for _, _, _, _, obj_id in tracked_objects])

    for obj_id, trajectory in trajectories.items():
        if len(trajectory) > 1:
            color = trajectory_colors.get(obj_id, (255, 255, 255))

            # Draw trajectory as connected lines with fading effect
            for i in range(1, len(trajectory)):
                # Calculate alpha/thickness based on point age (newer points are thicker/brighter)
                alpha = i / len(trajectory)
                thickness = max(2, int(8 * alpha))

                # Draw line segment
                cv2.line(frame, trajectory[i-1],
                         trajectory[i], color, thickness)

            # Draw a larger circle at the current position
            if trajectory:
                cv2.circle(frame, trajectory[-1], 6, color, -1)

    # Clean up trajectories for objects that are no longer tracked (exactly like main_gate3)
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

    # Create extended frame with counting panel (exactly like main_gate3)
    extended_frame = np.zeros(
        (frame_height, frame_width + panel_width, 3), dtype=np.uint8)
    extended_frame[:, :frame_width] = frame  # Place original frame on left

    # Create counting panel on the right
    panel_color = (40, 40, 40)  # Dark gray background
    extended_frame[:, frame_width:] = panel_color

    # Add title and counts to panel
    cv2.putText(extended_frame, model_name, (frame_width + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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

    return extended_frame


print("Starting Model Comparison (identical to main_gate3.py functionality):")
print("Model 1: YOLOv8n")
print("Model 2: DCN-YOLOv8")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_start_time = time.time()
    frame_count += 1

    # Create copies for each model
    frame_yolov8n = frame.copy()
    frame_dcn_yolov8 = frame.copy()

    # Run both models with same confidence (exactly like main_gate3)
    results_yolov8n = model_yolov8n(frame, conf=0.5)[0]
    results_dcn_yolov8 = model_dcn_yolov8(frame, conf=0.5)[0]

    # Process each model (exactly like main_gate3)
    extended_frame_yolov8n = process_model_detections(
        frame_yolov8n, results_yolov8n, model_yolov8n, tracker_yolov8n, track_classes_yolov8n, track_confidences_yolov8n,
        trajectories_yolov8n, trajectory_colors_yolov8n, vehicle_counts_yolov8n,
        object_positions_yolov8n, counted_objects_yolov8n, "YOLOv8n MODEL"
    )

    extended_frame_dcn_yolov8 = process_model_detections(
        frame_dcn_yolov8, results_dcn_yolov8, model_dcn_yolov8, tracker_dcn_yolov8, track_classes_dcn_yolov8, track_confidences_dcn_yolov8,
        trajectories_dcn_yolov8, trajectory_colors_dcn_yolov8, vehicle_counts_dcn_yolov8,
        object_positions_dcn_yolov8, counted_objects_dcn_yolov8, "DCN-YOLOv8 MODEL"
    )

    # Combine frames side by side
    combined_frame = np.hstack(
        (extended_frame_yolov8n, extended_frame_dcn_yolov8))

    # Write to output video
    out.write(combined_frame)
    cv2.imshow("Model Comparison: YOLOv8n vs DCN-YOLOv8", combined_frame)

    # Calculate frame processing time (same as main_gate3)
    frame_end_time = time.time()
    frame_processing_time = frame_end_time - frame_start_time
    total_processing_time += frame_processing_time

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Calculate final metrics (same as main_gate3)
end_time = time.time()
total_runtime = end_time - start_time

print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)
print(f"Total frames processed: {frame_count}")
print(f"Total runtime: {total_runtime:.2f} seconds")
print(f"Average FPS: {frame_count/total_runtime:.2f}")

print(f"\nYOLOv8n MODEL Results:")
print("Vehicle counts:")
for vehicle_type, count in vehicle_counts_yolov8n.items():
    if count > 0:
        print(f"  {vehicle_type.upper()}: {count}")
print(f"  TOTAL: {sum(vehicle_counts_yolov8n.values())}")

print(f"\nDCN-YOLOv8 MODEL Results:")
print("Vehicle counts:")
for vehicle_type, count in vehicle_counts_dcn_yolov8.items():
    if count > 0:
        print(f"  {vehicle_type.upper()}: {count}")
print(f"  TOTAL: {sum(vehicle_counts_dcn_yolov8.values())}")

difference = sum(vehicle_counts_dcn_yolov8.values()) - \
    sum(vehicle_counts_yolov8n.values())
if difference > 0:
    print(
        f"\nDifference: DCN-YOLOv8 counted {difference} more vehicles than YOLOv8n")
elif difference < 0:
    print(
        f"\nDifference: YOLOv8n counted {abs(difference)} more vehicles than DCN-YOLOv8")
else:
    print(
        f"\nBoth models counted the same number of vehicles: {sum(vehicle_counts_yolov8n.values())}")

print(f"\nOutput video saved as: model_comparison_output.mp4")
