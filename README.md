# Vehicle & Lane Detection Web App

A complete solution for vehicle detection, car counting, and lane segmentation using deep learning. This project provides:
- **YOLO-based vehicle detection and car counting** (with line crossing logic)
- **UNet-based lane segmentation**
- **A Flask web app** for uploading images/videos and visualizing results

---

## 🚀 Features

- **Detect Cars:** Detect vehicles in images and videos using YOLO.
- **Count Cars:** Count cars as they cross a virtual line in videos (using SORT tracker).
- **Lane Segmentation:** Segment road lanes using a UNet model.
- **Web Interface:** Upload images/videos, select tasks, and view/download results.
- **Supports both images and videos** (mp4, avi, mov, jpg, png, etc.)

---

## 🗂️ Project Structure

```
PROJECT/
    Vehicle_Detection(TPP).ipynb      # YOLO vehicle detection notebook
    UNet_Lane_Detection.ipynb         # UNet lane segmentation notebook
    car_conuting.ipynb                # Car counting notebook (YOLO + SORT)
    dataset_custom_TPP.yaml           # YOLO dataset config for TPP
    dataset_custom_DASH.yaml          # (example) dataset config
    yolov11x.pt                       # YOLO model weights
UI/
    app.py                            # Flask web app
    Detection.py                      # Detection utilities
    lane_utils.py                     # LaneNet model & segmentation utils
    yolo_utils.py                     # YOLO utilities
    sort/
        Sort.py                       # SORT tracker for multi-object tracking
        requirements.txt              # SORT dependencies
    static/
        uploads/
            input/                    # Uploaded files
            output/                   # Processed results
    templates/
        index.html                    # Web app frontend
```

---

## 🏁 Quick Start

### 1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/vehicle-lane-detection.git
cd vehicle-lane-detection
```

### 2. **Install Requirements**
```bash
pip install -r UI/sort/requirements.txt
pip install flask torch opencv-python ultralytics numpy
```

### 3. **Download/Place Model Weights**
- Place your YOLO weights (`Detection.pt` or `yolov11x.pt`) in `PROJECT/models/`
- Place your UNet lane model (`LANE_MODEL.pth`) in `PROJECT/models/`

### 4. **Run the Web App**
```bash
cd UI
python app.py
```
- Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🖼️ Web App Usage

1. **Upload** an image or video.
2. **Select a task:**  
   - Detect Cars  
   - Count Cars  
   - Lane Segmentation
3. **Run** and view results directly in the browser.
4. **Download** the processed output.

---

## 📁 Notebooks

- **Vehicle_Detection(TPP).ipynb:** Train and evaluate YOLO for vehicle detection.
- **UNet_Lane_Detection.ipynb:** Train and evaluate UNet for lane segmentation.
- **car_conuting.ipynb:** Car counting in videos using YOLO + SORT.

---

## 🛠️ Dependencies

- Python 3.8+
- Flask
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- SORT (included in `UI/sort/`)

---

## 📄 Dataset

- **TPP Dataset:** Used for vehicle detection and counting.
- **Lane Dataset:** Used for lane segmentation.
- Dataset configs: `dataset_custom_TPP.yaml`, etc.

---

## 📦 Credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [SORT Tracker](https://github.com/abewley/sort)
- [UNet Paper](https://arxiv.org/abs/1505.04597)

---


## 🤝 Contributing

Pull requests and issues are welcome!

---

## ✨ Demo

![Demo Screenshot](demo_screenshot.png)

---

**Made with ❤️ for Computer Vision Projects**
