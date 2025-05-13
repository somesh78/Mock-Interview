import cv2
import mediapipe as mp
import numpy as np

class PostureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # Configure pose to focus on upper body
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,  # Using medium complexity for better detection
            smooth_landmarks=True  # Enable landmark smoothing
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.feedback_messages = []
        
    def check_head_orientation(self, landmarks):
        """Check if user is looking directly at the screen"""
        facing_forward = False
        feedback = ""

        if landmarks:
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE.value]

            if (nose.visibility > 0.5 and left_eye.visibility > 0.5 and right_eye.visibility > 0.5):
                eye_distance_x = abs(left_eye.x - right_eye.x)
                nose_centered_x = abs(nose.x - (left_eye.x + right_eye.x) / 2)

                if nose_centered_x < 0.05 * eye_distance_x:
                    facing_forward = True
                    feedback = "Good: User is looking at the screen."
                else:
                    feedback = "Warning: User is not looking directly at the screen."

        return facing_forward, feedback

    def assess_overall_posture(self, landmarks):
        """Assess the upper body posture focusing on head, neck and shoulder alignment"""
        neck_good = False
        shoulders_good = False
        feedback_list = []

        if landmarks:
            # Key landmarks for upper body assessment
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
            
            # Calculate midpoints
            mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            mid_ear_x = (left_ear.x + right_ear.x) / 2

            # Check neck alignment (vertical)
            if (nose.visibility > 0.5 and left_ear.visibility > 0.5 and 
                right_ear.visibility > 0.5 and left_shoulder.visibility > 0.5 and 
                right_shoulder.visibility > 0.5):
                
                # Check if head is centered above shoulders horizontally
                neck_offset_x = abs(mid_ear_x - mid_shoulder_x)
                if neck_offset_x < 0.05:
                    neck_good = True
                    feedback_list.append("Good neck posture: head is aligned with shoulders.")
                else:
                    feedback_list.append("Warning: Neck misalignment detected. Center your head above your shoulders.")
                
                # Check if head is tilted (ear height difference)
                ear_height_diff = abs(left_ear.y - right_ear.y)
                if ear_height_diff > 0.03:
                    feedback_list.append("Head tilt detected: Try to keep your head level.")
                
                # Check forward head posture (ears should be aligned with shoulders)
                ear_forward_position = mid_ear_x - mid_shoulder_x
                if ear_forward_position > 0.05:
                    feedback_list.append("Forward head posture detected: Pull your head back slightly.")
                
            # Check shoulder alignment (are shoulders level?)
            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
                if shoulder_height_diff < 0.03:
                    shoulders_good = True
                    feedback_list.append("Good shoulder alignment: shoulders are level.")
                else:
                    feedback_list.append("Uneven shoulders detected: Try to level your shoulders.")
            
            # Check for slouching (using shoulder and head position)
            if (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and 
                nose.visibility > 0.5):
                # In a slouched position, shoulders tend to roll forward and head drops
                # We use shoulder z-depth (if available) or head position relative to frame
                
                # Head height in frame (lower values = higher in frame)
                head_height_in_frame = nose.y
                if head_height_in_frame > 0.4:  # Head is too low in frame
                    feedback_list.append("Possible slouching detected: Sit up straight, pull shoulders back.")
                
            # Check head orientation
            head_facing, head_feedback = self.check_head_orientation(landmarks)
            if head_feedback:
                feedback_list.append(head_feedback)
            
            # Overall assessment
            if neck_good and shoulders_good and head_facing:
                feedback_list.append("Overall upper body posture is good.")
            else:
                feedback_list.append("Your posture needs some adjustment for a professional appearance.")

        return feedback_list

    def process_frame(self, frame):
        """Process a single frame and return the annotated frame and feedback"""
        self.feedback_messages = []
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get frame dimensions for visualization
        h, w, _ = frame.shape
        
        # Process with MediaPipe Pose
        results = self.pose.process(frame_rgb)

        annotated_frame = frame.copy()
        
        if results.pose_landmarks:
            # Customize drawing style to focus on upper body
            drawing_spec = self.mp_drawing.DrawingSpec(
                color=(0, 255, 0),  # Green color for visibility
                thickness=2,
                circle_radius=2
            )
            
            # Draw only the landmarks we can see (focusing on upper body)
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec
            )
            
            # Draw a rectangle around the upper body area for better visual feedback
            landmarks = results.pose_landmarks.landmark
            if (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.5 and 
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.5):
                left_x = int(min(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                              landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x) * w) - 20
                right_x = int(max(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x) * w) + 20
                top_y = int(landmarks[self.mp_pose.PoseLandmark.NOSE.value].y * h) - 50
                bottom_y = int(max(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, 
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) * h) + 50
                
                # Ensure coordinates are within frame boundaries
                left_x = max(0, left_x)
                right_x = min(w, right_x)
                top_y = max(0, top_y)
                bottom_y = min(h, bottom_y)
                
                # Draw upper body region of interest
                cv2.rectangle(annotated_frame, (left_x, top_y), (right_x, bottom_y), (0, 165, 255), 2)
            
            # Analyze posture and get feedback
            self.feedback_messages = self.assess_overall_posture(results.pose_landmarks.landmark)
        else:
            # If no landmarks detected
            self.feedback_messages = ["No body detected or not fully visible. Please adjust your position."]
        
        return annotated_frame, self.feedback_messages
    
    def __del__(self):
        """Clean up resources"""
        self.pose.close()
