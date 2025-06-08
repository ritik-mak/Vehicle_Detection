from ultralytics import YOLO
import cv2

# Use correct slashes or raw string
yolo_model = YOLO("PROJECT/models/Detection.pt")

def process_yolo(frame):
    results = yolo_model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        label = yolo_model.names[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame
