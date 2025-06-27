from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # Model đã huấn luyện sẵn

cap = cv2.VideoCapture(0)  # Dùng webcam (hoặc thay bằng "video.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
