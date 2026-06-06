# Real-Time Object Detection System using YOLOv8

## Overview

This project implements a real-time object detection system using the YOLOv8 deep learning model, OpenCV, and Python. The application captures live video from a webcam, detects objects in real-time, displays bounding boxes with confidence scores, logs detection data to a CSV file, records the processed video, and generates audio alerts when specific objects are detected.

---

## Features

* Real-time object detection using YOLOv8
* Webcam video stream processing
* Bounding box visualization
* Confidence score display
* FPS (Frames Per Second) monitoring
* Object counting per frame
* Detection logging to CSV file
* Video recording of processed output
* Audio alerts for detected persons
* Detection filtering based on confidence threshold

---

## Technologies Used

* Python 3.x
* OpenCV
* YOLOv8 (Ultralytics)
* CVZone
* NumPy
* CSV Module
* Playsound

---

## Project Structure

```
project/
│
├── main.py
├── detections.csv
├── output.avi
├── alert.wav
├── README.md
│
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yolov8-object-detection.git
cd yolov8-object-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install cvzone
pip install playsound
```

Or install using:

```bash
pip install -r requirements.txt
```

---

## Download YOLOv8 Model

Download the YOLOv8 Large model:

```bash
yolo predict model=yolov8l.pt
```

Or download directly from Ultralytics and place it inside:

```
Yolo-Weights/
```

---

## Running the Project

Execute:

```bash
python main.py
```

Press:

```text
Q
```

to quit the application.

---

## Detection Workflow

1. Webcam captures live video.
2. YOLOv8 processes each frame.
3. Objects are detected and classified.
4. Bounding boxes are drawn.
5. Confidence scores are displayed.
6. Detection data is stored in CSV.
7. Processed video is recorded.
8. Alert sound plays when a person is detected.

---

## Output Files

### detections.csv

Stores:

| Column     |
| ---------- |
| Class      |
| Confidence |
| x1         |
| y1         |
| x2         |
| y2         |

Example:

```csv
person,0.91,120,50,340,560
car,0.84,400,210,700,520
```

---

### output.avi

Contains recorded video with:

* Bounding boxes
* Confidence labels
* FPS information
* Object count

---

## Classes Detected

The model supports 80 COCO dataset classes including:

* Person
* Car
* Bus
* Truck
* Bicycle
* Motorcycle
* Dog
* Cat
* Laptop
* Mobile Phone
* Chair
* Bottle
* TV
* And many more

---

## Configuration

### Confidence Threshold

Current threshold:

```python
if conf > 0.5:
```

Modify as needed:

```python
if conf > 0.7:
```

---

### Detect Specific Classes

Current configuration:

```python
["person", "car"]
```

Example:

```python
["person", "car", "dog", "bus"]
```

---

## Performance Considerations

* YOLOv8l provides high accuracy but requires more computational resources.
* For faster inference on low-end systems:

```python
yolov8n.pt
```

or

```python
yolov8s.pt
```

can be used.

---

## Future Enhancements

* GPU acceleration using CUDA
* Object tracking
* Face recognition integration
* Automatic screenshot capture
* Email notifications
* Telegram alerts
* Web dashboard
* Database storage
* Multi-camera support
* Vehicle number plate recognition

---

## Applications

* Smart Surveillance
* Security Monitoring
* Traffic Analysis
* Crowd Monitoring
* Vehicle Detection
* Smart City Solutions
* Industrial Safety Systems

---

## Author

**KOTHEM NAGA DURGARAO**

Data Science Student

Project: Real-Time Object Detection using YOLOv8

---

## License

This project is intended for educational and research purposes.
