import os
import uuid
from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
import cv2
import torch
from ultralytics import YOLO
from lane_utils import LaneNet, segment_lane  # Your lane model and segmentation function
import numpy as np
# Ensure the necessary directories exist

# Flask app setup
app = Flask(__name__)

# Define base, input, and output folders
BASE_STATIC_FOLDER = os.path.join(app.root_path, 'static')
INPUT_FOLDER = os.path.join(BASE_STATIC_FOLDER, 'uploads', 'input')
OUTPUT_FOLDER = os.path.join(BASE_STATIC_FOLDER, 'uploads', 'output')

# Create folders if they don't exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load models
detection_model = YOLO("PROJECT/models/Detection.pt")
lane_model = LaneNet()
lane_model.load_state_dict(torch.load("PROJECT/models/LANE_MODEL.pth", map_location='cpu'))
lane_model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        task = request.form.get('task')

        if not file or not task:
            return "Missing file or task selection", 400

        # Validate file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']:
            return "Only image/video files (.jpg, .jpeg, .png, .mp4, .avi, .mov) are supported.", 400

        # Save input file
        filename = f"{uuid.uuid4()}{ext}"
        input_path = os.path.join(INPUT_FOLDER, filename)
        file.save(input_path)

        # Prepare output file path
        output_filename = f"out_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Run the selected task
        is_video = ext in ['.mp4', '.avi', '.mov']
        if ext in ['.jpg', '.jpeg', '.png']:
            if task == 'detect_car':
                result_img = detect_cars(input_path)
                count = None
                cv2.imwrite(output_path, result_img)
            elif task == 'count_car':
                result_img, count = count_cars(input_path)
                cv2.imwrite(output_path, result_img)
            elif task == 'lane_segmentation':
                result_img = segment_lanes(input_path)
                count = None
                cv2.imwrite(output_path, result_img)
            else:
                return "Invalid task", 400
            output_media_url = url_for('static', filename=f"uploads/output/{output_filename}")
        else:  # Video file
            if task == 'detect_car':
                count = None
                process_video(input_path, output_path, mode='detect')
            elif task == 'count_car':
                count = process_video(input_path, output_path, mode='count')
            elif task == 'lane_segmentation':
                count = None
                process_video(input_path, output_path, mode='lane')
            else:
                return "Invalid task", 400
            output_media_url = url_for('static', filename=f"uploads/output/{output_filename}")

        return render_template('index.html', output_image=output_media_url, count=count, is_video=is_video)

    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

# Helper functions
def detect_cars(image_path):
    img = cv2.imread(image_path)
    results = detection_model(img)
    return results[0].plot()

def count_cars(image_path):
    img = cv2.imread(image_path)
    results = detection_model(img)
    count = len(results[0].boxes)
    img_out = results[0].plot()
    return img_out, count

def segment_lanes(image_path):
    img = cv2.imread(image_path)
    return segment_lane(img, lane_model)

def process_video(input_path, output_path, mode='detect'):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if mode == 'count':
        from sort.Sort import Sort
        tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)
        counted_ids = set()
        track_memory = {}
        line_y = int(frame_height * 2 / 3)
        offset = 10  # Tolerance for crossing

    total_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if mode == 'detect':
            results = detection_model(frame)
            frame_out = results[0].plot()

        elif mode == 'count':
            results = detection_model(frame)
            boxes = results[0].boxes
            car_detections = []
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                for box, score, class_id in zip(xyxy, conf, cls):
                    if int(class_id) == 2:  # Class 2 is 'car' in COCO
                        x1, y1, x2, y2 = box
                        car_detections.append([x1, y1, x2, y2, score])
            car_detections = np.array(car_detections)
            if len(car_detections) == 0:
                car_detections = np.empty((0, 5))
            tracked_objects = tracker.update(car_detections)

            # Draw counting line
            cv2.line(frame, (0, line_y), (frame_width, line_y), (255, 0, 0), 2)

            for x1, y1, x2, y2, track_id in tracked_objects:
                track_id = int(track_id)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Draw bounding box and ID
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # Track the previous position of each ID
                prev_cy = track_memory.get(track_id, None)
                track_memory[track_id] = cy

                # Count only if the car crosses the line from above to below
                if prev_cy is not None and prev_cy < line_y <= cy and track_id not in counted_ids:
                    counted_ids.add(track_id)

            total_count = len(counted_ids)
            cv2.putText(frame, f"Cars Counted: {total_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame_out = frame

        elif mode == 'lane':
            frame_out = segment_lane(frame, lane_model)
        else:
            frame_out = frame

        out.write(frame_out)

    cap.release()
    out.release()
    if mode == 'count':
        return total_count
    return None

if __name__ == '__main__':
    app.run(debug=True)