"""
BioVerify: Multi-Modal Fake Account Detector
Liveness Detection Module - Eye Blink Detection using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Eye landmark indices for MediaPipe Face Mesh
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR)
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # Vertical distances
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal distance
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

def extract_eye_landmarks(face_landmarks, eye_indices, image_width, image_height):
    """
    Extract eye landmarks from face mesh
    """
    landmarks = []
    for idx in eye_indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        landmarks.append((x, y))
    return landmarks

def detect_blink(ear, threshold=0.25):
    """
    Detect if a blink occurred based on EAR threshold
    """
    return ear < threshold

class LivenessDetector:
    """
    Real-time liveness detection using eye blink detection
    """
    
    def __init__(self, required_blinks=3, ear_threshold=0.25):
        self.required_blinks = required_blinks
        self.ear_threshold = ear_threshold
        self.blink_count = 0
        self.blink_in_progress = False
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def reset(self):
        """Reset blink counter"""
        self.blink_count = 0
        self.blink_in_progress = False
    
    def process_frame(self, frame):
        """
        Process a single frame for blink detection
        Returns: (processed_frame, ear, blink_detected, face_detected)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        face_detected = False
        ear = 0.0
        blink_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            
            h, w, _ = frame.shape
            
            # Extract eye landmarks
            left_eye = extract_eye_landmarks(face_landmarks, LEFT_EYE_LANDMARKS, w, h)
            right_eye = extract_eye_landmarks(face_landmarks, RIGHT_EYE_LANDMARKS, w, h)
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Detect blink
            if detect_blink(ear, self.ear_threshold):
                if not self.blink_in_progress:
                    self.blink_in_progress = True
            else:
                if self.blink_in_progress:
                    self.blink_count += 1
                    blink_detected = True
                    self.blink_in_progress = False
            
            # Draw face mesh (optional - can be removed for performance)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Highlight eyes
            for landmark in left_eye + right_eye:
                cv2.circle(frame, landmark, 2, (0, 255, 0), -1)
        
        return frame, ear, blink_detected, face_detected
    
    def get_liveness_score(self):
        """
        Calculate liveness score based on blink count
        Returns: score (0-100)
        """
        if self.blink_count >= self.required_blinks:
            return 100
        else:
            return (self.blink_count / self.required_blinks) * 100
    
    def is_liveness_passed(self):
        """Check if liveness test passed"""
        return self.blink_count >= self.required_blinks
    
    def cleanup(self):
        """Release resources"""
        self.face_mesh.close()

def run_liveness_test(required_blinks=3):
    """
    Run standalone liveness test (for testing purposes)
    Returns: (passed, blink_count, liveness_score)
    """
    detector = LivenessDetector(required_blinks=required_blinks)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access webcam")
        return False, 0, 0
    
    print("👁️ Liveness Test Started")
    print(f"Please blink {required_blinks} times naturally")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, ear, blink_detected, face_detected = detector.process_frame(frame)
            
            # Display info
            if face_detected:
                cv2.putText(processed_frame, f"EAR: {ear:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Blinks: {detector.blink_count}/{required_blinks}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(processed_frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Liveness Test - Press Q to quit', processed_frame)
            
            # Check if test passed
            if detector.is_liveness_passed():
                cv2.putText(processed_frame, "LIVENESS TEST PASSED!", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Liveness Test - Press Q to quit', processed_frame)
                cv2.waitKey(2000)  # Show success message for 2 seconds
                break
            
            # Quit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
    
    liveness_score = detector.get_liveness_score()
    passed = detector.is_liveness_passed()
    
    return passed, detector.blink_count, liveness_score

if __name__ == "__main__":
    # Test liveness detection
    passed, blinks, score = run_liveness_test(required_blinks=3)
    print(f"\nTest Result:")
    print(f"Passed: {passed}")
    print(f"Blinks Detected: {blinks}")
    print(f"Liveness Score: {score}%")