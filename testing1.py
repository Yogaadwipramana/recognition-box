import cv2
import mediapipe as mp
import time
    
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils 

# Function to detect and draw poses
def detect_pose(frame, action_status):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get bounding box coordinates
        bbox_coords = get_bbox_coords(results.pose_landmarks, frame)

        # Draw bounding box
        cv2.rectangle(frame, bbox_coords[0], bbox_coords[1], (0, 255, 0), 2)

        # Human Action Recognition
        action = recognize_action(results.pose_landmarks, action_status)

        # Get nose coordinates
        nose_coords = (
            int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * frame.shape[1]),
            int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * frame.shape[0])
        )

        # Display the label next to the person
        cv2.putText(frame, f"Person 1 - Action: {action}", (nose_coords[0] + 10, nose_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Print action to the terminal
        print(f"Person 1 - Action: {action}")

def get_bbox_coords(landmarks, frame, margin=210):
    min_x = max(0, min(landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * frame.shape[1], landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]) - margin)
    max_x = min(frame.shape[1], max(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * frame.shape[1], landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]) + margin)
    min_y = max(0, min(landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0], landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]) - margin)
    max_y = min(frame.shape[0], max(landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * frame.shape[0], landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * frame.shape[0]) + margin)

    return ((int(min_x), int(min_y)), (int(max_x), int(max_y)))


def recognize_action(landmarks, action_status):
    # Get relevant joint positions
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
    right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y

    # Threshold to identify prayer movements
    threshold = 0.05

    # Rukuk detection (if wrists are lower than elbows)
    if left_wrist > left_elbow + threshold and right_wrist > right_elbow + threshold:
        return "rukuk"

    # Takbir detection (if hands are higher than shoulders)
    if left_shoulder < left_hip - threshold and right_shoulder < right_hip - threshold:
        return "takbir"

    # Sujud detection (if knees are higher than hips)
    if left_knee > left_hip + threshold and right_knee > right_hip + threshold:
        return "sujud"

    # If no prayer movement is detected
    return "unknown"

def main():
    cap = cv2.VideoCapture(0)

    # Initialize activity status
    action_status = {activity: "unknown" for activity in ["takbir", "rukuk", "sujud"]}

    # Initialize for FPS calculation
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame, exiting.")
            break

        detect_pose(frame, action_status)

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Pose Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
