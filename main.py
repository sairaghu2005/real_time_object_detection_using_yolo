import cv2
from ultralytics import YOLO

# Load the trained model
model_path = "yolov8m.pt"  # Change this to your trained model path
model = YOLO(model_path)
# Open webcam (0 is the default webcam, change if needed)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
while cap.isOpened():
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        break

    # Run YOLO model on the frame
    results = model(frame)

    # Show results
    for result in results:
        frame = result.plot()  # Draw detections on the frame

    cv2.imshow("YOLOv8 Webcam Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
