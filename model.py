from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.fuse()

MAX_PEOPLE = 10  # adjust based on your use case

def predict_risk(frame):
    results = model(frame, verbose=False)

    person_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                person_count += 1

    # ✅ Crowd percentage
    crowd_percent = int((person_count / MAX_PEOPLE) * 100)

    # Risk logic
    if crowd_percent >= 70:
        risk = "HIGH"
    elif crowd_percent >= 30:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return risk, crowd_percent, person_count