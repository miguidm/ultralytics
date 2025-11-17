from ultralytics import YOLO

model = YOLO('YOLO_outputs/100_dcnv2_yolov8m/weights/last.pt')
results = model.train(resume=True)
