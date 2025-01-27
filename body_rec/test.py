import cv2
import mediapipe as mp

# Initialize MediaPipe Pose with heavy task model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=2,  # Use heavy model
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Excluded landmarks (face landmarks)
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

# Open a video feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Draw landmarks
    if results.pose_landmarks:
        annotated_frame = frame.copy()
        
        for connection in mp_pose.POSE_CONNECTIONS:
            # Skip connections that involve excluded landmarks
            if (connection[0] in EXCLUDED_LANDMARKS) or (connection[1] in EXCLUDED_LANDMARKS):
                continue

            # Get the landmarks
            landmark_1 = results.pose_landmarks.landmark[connection[0]]
            landmark_2 = results.pose_landmarks.landmark[connection[1]]

            # Draw the connection if both landmarks are visible
            if (landmark_1.visibility > 0.5) and (landmark_2.visibility > 0.5):
                x1, y1 = int(landmark_1.x * frame.shape[1]), int(landmark_1.y * frame.shape[0])
                x2, y2 = int(landmark_2.x * frame.shape[1]), int(landmark_2.y * frame.shape[0])

                cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow('Pose Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
