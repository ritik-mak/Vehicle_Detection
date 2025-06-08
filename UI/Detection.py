from ultralytics import YOLO
from pathlib import Path
import cv2

# Load the custom-trained YOLO model
model_path = "C:/Users/ritik/Desktop/New folder/PROJECT/models/Detection.pt"
model = YOLO(model_path)

# Directory of input images
image_dir = Path("C:/Users/ritik/Desktop/New folder/PROJECT/TPP/TPP_val/images")  # Update if your image folder is elsewhere
output_dir = Path("runs/inference_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Get all image files
image_extensions = ['.jpg', '.jpeg', '.png']
image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in image_extensions]

# Run inference and save results
for image_path in image_paths:
    results = model(image_path)
    results[0].save(filename=output_dir / image_path.name)
    print(f"Processed: {image_path.name}")

print(f"Inference complete. Results saved to: {output_dir}")
