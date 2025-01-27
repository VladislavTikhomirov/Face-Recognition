import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
import mediapipe as mp

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)


# Constants for face recognition
IMG_SIZE = 128
DATASET_DIR = "face_data"
MODEL_PATH = "face_id_model.h5"

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize Pose
pose = mp_pose.Pose(
    model_complexity=2,  # Use heavy model
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Excluded landmarks (face landmarks for pose detection)
EXCLUDED_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.LEFT_EYE_INNER,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER,
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.MOUTH_LEFT,
    mp_pose.PoseLandmark.MOUTH_RIGHT
]

# Function to collect face data
def collect_data(label_name):
    label_dir = os.path.join(DATASET_DIR, label_name)
    os.makedirs(label_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    print(f"Collecting data for label: {label_name}. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(label_dir, f"{label_name}_{len(os.listdir(label_dir))}.jpg"), face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to load the face recognition dataset
def load_data():
    images, labels = [], []
    label_map = {label: idx for idx, label in enumerate(os.listdir(DATASET_DIR))}
    for label, idx in label_map.items():
        label_dir = os.path.join(DATASET_DIR, label)
        for image_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            images.append(img)
            labels.append(idx)

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = to_categorical(labels, num_classes=len(label_map))
    return images, labels, label_map

# Function to train the face recognition model
def train_model():
    images, labels, _ = load_data()
    model = Sequential([
        Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(labels[0]), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, epochs=10, batch_size=16)
    model.save(MODEL_PATH)

# Function for face recognition prediction
def predict():
    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, label_map = load_data()
    label_map = {v: k for k, v in label_map.items()}

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        pose_results = pose.process(rgb_frame)

        # Process the frame to find face landmarks
        face_results = face_mesh.process(rgb_frame)

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            for connection in mp_pose.POSE_CONNECTIONS:
                # Skip connections that involve excluded landmarks
                if (connection[0] in EXCLUDED_LANDMARKS) or (connection[1] in EXCLUDED_LANDMARKS):
                    continue

                # Get the landmarks
                landmark_1 = pose_results.pose_landmarks.landmark[connection[0]]
                landmark_2 = pose_results.pose_landmarks.landmark[connection[1]]

                # Draw the connection if both landmarks are visible
                if (landmark_1.visibility > 0.5) and (landmark_2.visibility > 0.5):
                    x1, y1 = int(landmark_1.x * frame.shape[1]), int(landmark_1.y * frame.shape[0])
                    x2, y2 = int(landmark_2.x * frame.shape[1]), int(landmark_2.y * frame.shape[0])

                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE)) / 255.0
            face_resized = face_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            # Predict face identity
            prediction = model.predict(face_resized)
            confidence = np.max(prediction)
            label_idx = np.argmax(prediction)

            # Determine color based on confidence
            if confidence < 0.5:
                label = "Unknown"
                color = (0, 0, 255)  # Red for unknown
            else:
                label = label_map[label_idx]
                green = int(255 * confidence)
                red = int(255 * (1 - confidence))
                color = (0, green, red)  # Gradient from green to red

            # Draw face bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display the frame
        cv2.imshow("Pose, Face Mesh, and Face Recognition", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Menu for the user to choose options
print("Options:")
print("1. Collect Data")
print("2. Train Model")
print("3. Predict")
choice = input("Enter choice: ")

if choice == '1':
    label_name = input("Enter label name: ")
    collect_data(label_name)
elif choice == '2':
    train_model()
elif choice == '3':
    predict()
else:
    print("Invalid choice!")
