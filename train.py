from clearml import Task
from ultralytics import YOLO
import torch

print(f"GPU: {torch.cuda.get_device_name(0)}")

task = Task.init(project_name="Test_for_CS2", task_name="yolo11s-wildlife-v1")

model = YOLO("yolo11s.pt")

model.train(
    data="E:\\dataset\\dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    patience=15,
    augment=True,
    workers=8,
    name="wildlife_v1"
)
