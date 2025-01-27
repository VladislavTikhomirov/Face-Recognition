import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import ObjectDetectorOptions, ObjectDetector

# Load the TFLite model
model_path = "objectdect_mp/efficientdet_lite0.tflite"
max_objects = 3  # Max on-screen objects
BaseOptions = mp.tasks.BaseOptions

# Initialize MediaPipe Object Detector
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=max_objects,
    score_threshold=0.1
)

# Corrected the creation of the ObjectDetector object. The previous code had `with ObjectDetector.create_from_options(options) as detector` that was missing a proper structure
detector = ObjectDetector.create_from_options(options)

# Function to map certainty to color gradient
def certainty_to_color(score):
    green = int(score * 255)
    red = int((1 - score) * 255)
    return (0, green, red)

# Start webcam feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)  # Use the correct detector instance

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        score = detection.categories[0].score
        label = detection.categories[0].category_name

        # Draw bounding box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
        color = (255, 255, 255)  # White for bounding box
        cv2.rectangle(frame, start_point, end_point, color, 2)

        # Draw label with gradient color
        gradient_color = certainty_to_color(score)
        label_position = (start_point[0], start_point[1] - 10)
        label_text = f"{label}: {int(score * 100)}%"
        cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, gradient_color, 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
