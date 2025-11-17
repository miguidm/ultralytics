import os
from ultralytics import YOLO

# Available models and their corresponding YAML paths
MODEL_MAP = {
    "DCNv2-Neck-Pan": "ultralytics/cfg/models/v8/dcnv2-yolov8-neck-pan.yaml",
    "DCNv2-Neck-FPN": "ultralytics/cfg/models/v8/dcnv2-yolov8-neck-fpn.yaml",
    "DCNv2-Liu": "ultralytics/cfg/models/v8/dcnv2-yolov8-liu.yaml",
    "DCNv2-Neck-Full": "ultralytics/cfg/models/v8/dcnv2-yolov8-neck-full.yaml",
}


def choose_model():
    print("\nAvailable YOLO models:\n")
    for i, model_name in enumerate(MODEL_MAP.keys(), 1):
        print(f"{i}. {model_name}")

    while True:
        choice = input(
            "\nEnter the number or name of the model you want to use: ").strip()

        # Allow either numeric or string input
        if choice.isdigit() and 1 <= int(choice) <= len(MODEL_MAP):
            model_name = list(MODEL_MAP.keys())[int(choice) - 1]
            return model_name
        elif choice in MODEL_MAP:
            return choice
        else:
            print("Invalid input. Please try again.")


if __name__ == "__main__":
    # Ask user for model
    selected_model = choose_model()
    model_path = MODEL_MAP[selected_model]
    print(f"\n Selected model: {selected_model}")
    print(f"Using config: {model_path}")

    # Pick up environment variable or default
    project_dir = os.getenv("YOLO_OUTPUT", "./YOLO_outputs")

    # Initialize YOLO model
    model = YOLO(model_path)

    # Train the model
    results = model.train(
        data="100data_linux.yaml",  # dataset config
        project=project_dir,        # output directory
        name=selected_model.lower().replace("-", "_"),  # clean experiment name
        epochs=300,
        patience=50,
        imgsz=640,
        batch=16,
    )

    print(
        f"\n🎯 Training completed. Check {project_dir}/{selected_model.lower().replace('-', '_')} for results.")
