from ultralytics import YOLO
import cv2, numpy as np, base64

class Detector:
    def __init__(self, model_path="model/yolo11n.pt"):
        self.model = YOLO(model_path)

    def process_frame(self, data_url, classes=[0, 2], conf=0.4):
        img_bytes = base64.b64decode(data_url.split(",")[1])
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        results = self.model(frame, classes=classes, conf=conf)
        annotated = results[0].plot()
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
