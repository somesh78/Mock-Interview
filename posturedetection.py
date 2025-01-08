import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to assess head orientation
def check_head_orientation(landmarks):
    facing_forward = False

    if landmarks:
        # Extract nose and eye landmarks for head orientation
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

        if (nose.visibility > 0.5 and left_eye.visibility > 0.5 and right_eye.visibility > 0.5):
            eye_distance_x = abs(left_eye.x - right_eye.x)
            nose_centered_x = abs(nose.x - (left_eye.x + right_eye.x) / 2)

            # Check if the nose is aligned between the eyes (facing forward)
            if nose_centered_x < 0.05 * eye_distance_x:  # Relative threshold based on eye distance
                facing_forward = True
                print("Good: User is looking at the screen.")
            else:
                print("Warning: User is not looking directly at the screen.")

    return facing_forward

# Function to assess overall posture
def assess_overall_posture(landmarks):
    neck_good = False
    sitting_good = False

    if landmarks:
        # Extract keypoints for posture analysis
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Calculate the midpoint between shoulders and hips
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        mid_hip_x = (left_hip.x + right_hip.x) / 2
        mid_hip_y = (left_hip.y + right_hip.y) / 2

        # Check neck alignment (front view)
        if nose.visibility > 0.5 and left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
            neck_offset_x = abs(nose.x - mid_shoulder_x)
            if neck_offset_x < 0.05:  # Threshold for good neck alignment
                neck_good = True
                print("Good neck posture detected: head is aligned.")
            else:
                print("Warning: Neck misalignment detected. Adjust to align the head.")

        # Check for upright sitting posture
        if (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and
            left_hip.visibility > 0.5 and right_hip.visibility > 0.5):
            shoulder_hip_diff_y = abs(mid_shoulder_y - mid_hip_y)
            if shoulder_hip_diff_y > 0.2:  # Threshold for upright sitting
                sitting_good = True
                print("Good sitting posture detected: upright.")
            elif mid_shoulder_y < mid_hip_y - 0.05:  # Leaning forward check (front view)
                print("Leaning forward detected: adjust to sit upright.")
            elif shoulder_hip_diff_y < 0.1:  # Slouching check
                print("Slouching detected: straighten your back.")

        # Overall posture check with head orientation
        head_facing = check_head_orientation(landmarks)
        if neck_good and sitting_good and head_facing:
            print("Overall posture and head orientation are good.")
        else:
            print("Overall posture needs adjustment.")

def assess_posture(landmarks):
    posture_feedback = assess_overall_posture(landmarks)
    eye_movement = check_head_orientation(landmarks)
    return posture_feedback, eye_movement

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB as MediaPipe uses RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Draw landmarks and connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Assess overall posture
        assess_overall_posture(results.pose_landmarks.landmark)

    # Display the video feed with landmarks
    cv2.imshow('Mock Interview Posture and Head Orientation Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
