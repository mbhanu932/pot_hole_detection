from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")  # Nano model (smallest & fastest)

    model.train(
        data="pothole_dataset/data.yaml",
        epochs=15,       # Reduced from 50 to 15 (fast testing)
        imgsz=320,       # Reduced image size from 640 to 320 for 4x faster processing
        batch=8,         # Lower batch size for CPU
        workers=2,       # Prevents CPU thread locking
        name="pothole_detector"
    )

if __name__ == "__main__":
    train_model()