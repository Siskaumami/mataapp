from ultralytics import YOLO


dataset_yaml = "dataset/Pupil Tracking.v3i.yolov8-obb/data.yaml"

model = YOLO("yolov8n.pt")   

# Mulai training
model.train(
    data=dataset_yaml,
    epochs=30,   
    imgsz=640,
    batch=4,
    device="cpu"  
)

print("Training selesai! Cek folder runs/detect/train/")
