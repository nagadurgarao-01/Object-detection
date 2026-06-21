from ultralytics import YOLO
import cv2
import cvzone
import time
import csv
from sound import play_sound_async

# --- Constants and Configuration ---
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
MODEL_PATH = "yolov8l.pt"
CLASS_NAMES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
               "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
               "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush", "earbuds"]
TARGET_CLASSES = ["person", "car"]
CONFIDENCE_THRESHOLD = 0.5
CSV_FILE_PATH = 'detections.csv'
OUTPUT_VIDEO_PATH = 'output.avi'
ALERT_SOUND_PATH = 'alert.wav'


def initialize_resources():
    """Initializes and returns the webcam capture, YOLO model, CSV writer, and video writer."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None, None, None, None, None

    cap.set(3, WEBCAM_WIDTH)
    cap.set(4, WEBCAM_HEIGHT)

    model = YOLO(MODEL_PATH)

    file = open(CSV_FILE_PATH, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(["Class", "Confidence", "x1", "y1", "x2", "y2"])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, (WEBCAM_WIDTH, WEBCAM_HEIGHT))

    return cap, model, writer, file, out


def process_frame(img, model, csv_writer):
    """Processes a single frame for object detection and returns the annotated image and object count."""
    results = model(img, stream=True)
    object_count = 0

    for r in results:
        for box in r.boxes:
            object_count += 1
            conf = box.conf[0]
            cls_index = int(box.cls[0])
            class_name = CLASS_NAMES[cls_index]

            if conf > CONFIDENCE_THRESHOLD and class_name in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                # Draw bounding box and label
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{class_name} {conf:.2f}', (x1, max(35, y1)))

                # Save detection to CSV
                csv_writer.writerow([class_name, f'{conf:.4f}', x1, y1, x2, y2])

                # Play alert sound
                if class_name == "person":
                    play_sound_async(ALERT_SOUND_PATH)

    return img, object_count


def main():
    """Main function to run the object detection application."""
    resources = initialize_resources()
    if not all(resources):
        return  # Exit if initialization failed

    cap, model, csv_writer, csv_file, video_writer = resources
    prev_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame from webcam. Exiting...")
            break

        # Process the frame for object detection
        processed_img, object_count = process_frame(img, model, csv_writer)

        # Calculate and display FPS
        current_time = time.time()
        fps = int(1 / (current_time - prev_time))
        prev_time = current_time
        cv2.putText(processed_img, f'FPS: {fps}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(processed_img, f'Objects Detected: {object_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show image and save frame
        cv2.imshow("Object Detection", processed_img)
        video_writer.write(processed_img)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    video_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("Resources released and windows closed.")


if __name__ == "__main__":
    main()
