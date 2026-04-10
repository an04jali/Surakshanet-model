import cv2
from model import predict_risk

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    risk, conf = predict_risk(frame)

    color = (0, 255, 0)
    if risk == "MEDIUM":
        color = (0, 165, 255)
    elif risk == "HIGH":
        color = (0, 0, 255)

    cv2.putText(frame, f"{risk} ({conf}%)",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2)

    cv2.imshow("SurakshaNet Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()