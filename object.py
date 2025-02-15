from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import csv
from playsound import playsound

# Initialize webcam and model
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height
model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", 
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", 
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
              "teddy bear", "hair drier", "toothbrush", "earbuds"]

# Initialize FPS calculation and CSV file
prev_time = time.time()
file = open('detections.csv', 'w', newline='')
writer = csv.writer(file)
writer.writerow(["Class", "Confidence", "x1", "y1", "x2", "y2"])

# Initialize video writer
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1280, 720))

while True:
    success, img = cap.read()
    if not success:
        break

    # FPS calculation
    current_time = time.time()
    fps = int(1 / (current_time - prev_time))
    prev_time = current_time

    # Object detection
    results = model(img, stream=True)
    object_count = 0  # Counter for objects in each frame

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            w, h = x2 - x1, y2 - y1
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index
            object_count += 1

            # Filter detections by confidence and specific classes (optional)
            if conf > 0.5 and classNames[cls] in ["person", "car"]:  
                cvzone.cornerRect(img, (x1, y1, w, h))  # Draw bounding box
                cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}', (x1, max(35, y1)))

                # Save detections to CSV
                writer.writerow([classNames[cls], conf, x1, y1, x2, y2])

                # Play sound alert for specific class
                if classNames[cls] == "person":
                    playsound('alert.wav')

    # Display FPS and object count
    cv2.putText(img, f'FPS: {fps}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f'Objects detected: {object_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image with detections
    cv2.imshow("Image", img)

    # Save frame to video
    out.write(img)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
file.close()
cv2.destroyAllWindows()
