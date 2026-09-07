import cv2
from ultralytics import YOLO

print("Loading model...")
model = YOLO("yolov8n.pt")
print("Model loaded.")

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

for i in range(30):

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    results = model(
        frame,
        imgsz=640,
        conf=0.10,
        verbose=False
    )

    boxes = results[0].boxes

    count = 0

    if boxes is not None:
        count = len(boxes)

    print(f"Frame {i + 1}: detections = {count}")

cap.release()

print("Camera test finished.")