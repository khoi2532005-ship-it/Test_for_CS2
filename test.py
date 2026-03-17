from ultralytics import YOLO

model = YOLO(r"D:\Test_for_CS2\runs\detect\wildlife_v15\weights\best.pt")
print(model.names)
