from ultralytics import YOLO
from PIL import Image

model = YOLO(r"D:\Test_for_CS2\runs\detect\wildlife_v15\weights\best.pt")

# Point to your downloaded image
results = model.predict(
    source=r"C:\Users\Khoi\Downloads\kangaroojpg.jpg",
    conf=0.05,   # very low confidence to see ANYTHING it detects
    save=True,   # saves annotated image
    show=True    # pops up a window
)

# Print what it found
for r in results:
    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        print(f"{model.names[cls]}: {conf:.2f}")
