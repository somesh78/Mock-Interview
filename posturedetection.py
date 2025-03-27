import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def check_head_orientation(landmarks):
    facing_forward = False

    if landmarks:
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

        if (nose.visibility > 0.5 and left_eye.visibility > 0.5 and right_eye.visibility > 0.5):
            eye_distance_x = abs(left_eye.x - right_eye.x)
            nose_centered_x = abs(nose.x - (left_eye.x + right_eye.x) / 2)

            if nose_centered_x < 0.05 * eye_distance_x:
                facing_forward = True
                print("Good: User is looking at the screen.")
            else:
                print("Warning: User is not looking directly at the screen.")

    return facing_forward

def assess_overall_posture(landmarks):
    neck_good = False
    sitting_good = False

    if landmarks:
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        mid_hip_x = (left_hip.x + right_hip.x) / 2
        mid_hip_y = (left_hip.y + right_hip.y) / 2

        if nose.visibility > 0.5 and left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
            neck_offset_x = abs(nose.x - mid_shoulder_x)
            if neck_offset_x < 0.05:
                neck_good = True
                print("Good neck posture detected: head is aligned.")
            else:
                print("Warning: Neck misalignment detected. Adjust to align the head.")

        if (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and
            left_hip.visibility > 0.5 and right_hip.visibility > 0.5):
            shoulder_hip_diff_y = abs(mid_shoulder_y - mid_hip_y)
            if shoulder_hip_diff_y > 0.2:
                sitting_good = True
                print("Good sitting posture detected: upright.")
            elif mid_shoulder_y < mid_hip_y - 0.05:
                print("Leaning forward detected: adjust to sit upright.")
            elif shoulder_hip_diff_y < 0.1:
                print("Slouching detected: straighten your back.")

        head_facing = check_head_orientation(landmarks)
        if neck_good and sitting_good and head_facing:
            print("Overall posture and head orientation are good.")
        else:
            print("Overall posture needs adjustment.")

def assess_posture(landmarks):
    posture_feedback = assess_overall_posture(landmarks)
    eye_movement = check_head_orientation(landmarks)
    return posture_feedback, eye_movement

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        assess_overall_posture(results.pose_landmarks.landmark)

    cv2.imshow('Mock Interview Posture and Head Orientation Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()