"""
BioVerify: Multi-Modal Fake Account Detector
Liveness Detection Module - Face Movement + Blink Detection
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from scipy.spatial import distance

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Facial landmarks
NOSE_TIP = 4
CHIN = 152
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33

# Eye landmarks for blink detection
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio for blink detection"""
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_eye_landmarks(face_landmarks, eye_indices, w, h):
    """Extract eye landmarks"""
    landmarks = []
    for idx in eye_indices:
        lm = face_landmarks.landmark[idx]
        landmarks.append((int(lm.x * w), int(lm.y * h)))
    return landmarks


# ========== ANTI-SPOOFING DETECTION ==========

def compute_lbp(image, P=8, R=1):
    """Compute Local Binary Pattern for texture analysis"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    rows, cols = gray.shape
    lbp = np.zeros_like(gray)
    
    for i in range(R, rows - R):
        for j in range(R, cols - R):
            center = gray[i, j]
            binary = 0
            for p in range(P):
                angle = 2 * np.pi * p / P
                x = int(round(j + R * np.cos(angle)))
                y = int(round(i - R * np.sin(angle)))
                if gray[y, x] >= center:
                    binary |= (1 << p)
            lbp[i, j] = binary
    
    return lbp


def compute_lbp_histogram(image):
    """Compute LBP histogram for texture classification"""
    # Use simplified LBP via Laplacian variance (faster)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Laplacian variance - real faces have more texture detail
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return laplacian_var


class AntiSpoofingDetector:
    """
    Detects spoofing attempts (photos, screens, printed faces)
    using texture analysis and color variance.
    """
    
    def __init__(self):
        self.texture_threshold = 100  # Laplacian variance threshold
        self.color_var_threshold = 15  # Color variance threshold
        self.history = deque(maxlen=30)  # Store recent scores
        self.spoof_score = 0
        self.is_real = True
        
    def analyze_texture(self, face_roi):
        """Analyze face texture using Laplacian variance"""
        if face_roi is None or face_roi.size == 0:
            return 0
        
        texture_score = compute_lbp_histogram(face_roi)
        return texture_score
    
    def analyze_color_variance(self, face_roi):
        """Real faces have more natural color variations"""
        if face_roi is None or face_roi.size == 0:
            return 0
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        # Check saturation variance (screens often have lower)
        s_channel = hsv[:, :, 1]
        s_var = np.var(s_channel)
        
        return s_var
    
    def detect_screen_reflection(self, face_roi):
        """Detect bright spots that may indicate screen reflection"""
        if face_roi is None or face_roi.size == 0:
            return 0
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        
        # Find very bright pixels (potential screen glare)
        bright_pixels = np.sum(gray > 240)
        total_pixels = gray.size
        bright_ratio = bright_pixels / total_pixels if total_pixels > 0 else 0
        
        return bright_ratio
    
    def analyze_frame(self, frame, face_bbox):
        """
        Analyze frame for spoofing indicators.
        Returns: (is_real, spoof_score, details)
        """
        x, y, w, h = face_bbox
        
        # Ensure valid ROI
        x = max(0, x)
        y = max(0, y)
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return True, 0, {}
        
        # Texture analysis
        texture_score = self.analyze_texture(face_roi)
        
        # Color variance
        color_var = self.analyze_color_variance(face_roi)
        
        # Reflection detection
        reflection_ratio = self.detect_screen_reflection(face_roi)
        
        # Calculate spoof probability
        spoof_indicators = 0
        
        if texture_score < self.texture_threshold:
            spoof_indicators += 1
        if color_var < self.color_var_threshold:
            spoof_indicators += 1
        if reflection_ratio > 0.05:
            spoof_indicators += 1
        
        self.spoof_score = (spoof_indicators / 3) * 100
        self.is_real = spoof_indicators < 2
        
        self.history.append(self.is_real)
        
        details = {
            'texture': texture_score,
            'color_var': color_var,
            'reflection': reflection_ratio,
            'is_real': self.is_real
        }
        
        return self.is_real, self.spoof_score, details
    
    def get_final_verdict(self):
        """Get final anti-spoofing verdict based on history"""
        if len(self.history) == 0:
            return True, 100
        
        real_count = sum(self.history)
        real_percentage = (real_count / len(self.history)) * 100
        
        return real_percentage > 70, real_percentage


class LivenessDetector:
    """
    Liveness detection using BOTH face movement AND blink detection.
    """
    
    def __init__(self, required_moments=3, movement_threshold=35.0, cooldown_frames=20, num_challenges=3):
        self.required_moments = required_moments
        self.movement_threshold = movement_threshold  # Increased from 15 to 35 for less sensitivity
        self.cooldown_frames = cooldown_frames  # Increased cooldown between detections
        
        # Movement tracking
        self.moment_count = 0
        self.frames_since_last_moment = 0
        self.position_history = deque(maxlen=5)
        self.last_stable_position = None
        
        # Blink tracking
        self.blink_count = 0
        self.blink_in_progress = False
        self.ear_threshold = 0.20  # Lowered - requires more closed eyes to detect
        self.frames_since_last_blink = 0
        self.blink_cooldown = 15  # Frames to wait between blinks
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    @property
    def challenges_completed(self):
        return self.moment_count + self.blink_count
    
    def get_current_challenge(self):
        total = self.moment_count + self.blink_count
        if total >= self.required_moments:
            return "✅ COMPLETE"
        return "🔄 MOVE HEAD or 👁️ BLINK"
    
    def reset(self):
        self.moment_count = 0
        self.blink_count = 0
        self.frames_since_last_moment = 0
        self.position_history.clear()
        self.last_stable_position = None
        self.blink_in_progress = False
    
    def _get_head_center(self, face_landmarks, w, h):
        nose = face_landmarks.landmark[NOSE_TIP]
        chin = face_landmarks.landmark[CHIN]
        left_eye = face_landmarks.landmark[LEFT_EYE_OUTER]
        right_eye = face_landmarks.landmark[RIGHT_EYE_OUTER]
        
        center_x = (nose.x + chin.x + left_eye.x + right_eye.x) / 4 * w
        center_y = (nose.y + chin.y + left_eye.y + right_eye.y) / 4 * h
        
        return np.array([center_x, center_y])
    
    def process_frame(self, frame):
        """Process frame for both face movement and blink detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        face_detected = False
        movement_amount = 0.0
        moment_detected = False
        current_ear = 0.0
        
        self.frames_since_last_moment += 1
        
        if results.multi_face_landmarks:
            face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # === FACE MOVEMENT DETECTION ===
            current_position = self._get_head_center(face_landmarks, w, h)
            self.position_history.append(current_position)
            
            if self.last_stable_position is not None:
                movement_amount = np.linalg.norm(current_position - self.last_stable_position)
            else:
                self.last_stable_position = current_position
            
            if movement_amount > self.movement_threshold and self.frames_since_last_moment > self.cooldown_frames:
                self.moment_count += 1
                moment_detected = True
                self.frames_since_last_moment = 0
                self.last_stable_position = current_position
            elif movement_amount < self.movement_threshold * 0.3:
                self.last_stable_position = current_position
            
            # === BLINK DETECTION ===
            left_eye_lm = get_eye_landmarks(face_landmarks, LEFT_EYE, w, h)
            right_eye_lm = get_eye_landmarks(face_landmarks, RIGHT_EYE, w, h)
            
            left_ear = calculate_ear(left_eye_lm)
            right_ear = calculate_ear(right_eye_lm)
            current_ear = (left_ear + right_ear) / 2.0
            
            if current_ear < self.ear_threshold:
                if not self.blink_in_progress:
                    self.blink_in_progress = True
            else:
                if self.blink_in_progress and self.frames_since_last_blink > self.blink_cooldown:
                    self.blink_count += 1
                    self.frames_since_last_blink = 0
                self.blink_in_progress = False
            
            self.frames_since_last_blink += 1
            
            # Draw face mesh
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Draw eye landmarks
            for lm in left_eye_lm + right_eye_lm:
                cv2.circle(frame, lm, 2, (0, 255, 0), -1)
            
            # Draw nose indicator
            nose = face_landmarks.landmark[NOSE_TIP]
            nose_x, nose_y = int(nose.x * w), int(nose.y * h)
            pulse_size = 12 + int(5 * np.sin(self.frames_since_last_moment * 0.3))
            color = (0, 255, 0) if moment_detected else (255, 255, 0)
            cv2.circle(frame, (nose_x, nose_y), pulse_size, color, -1)
        
        # === DRAW UI OVERLAY ===
        h, w = frame.shape[:2]
        
        # Corner brackets
        bracket_len = 60
        bracket_color = (0, 255, 200) if face_detected else (0, 100, 255)
        cv2.line(frame, (20, 20), (20 + bracket_len, 20), bracket_color, 3)
        cv2.line(frame, (20, 20), (20, 20 + bracket_len), bracket_color, 3)
        cv2.line(frame, (w-20, 20), (w-20-bracket_len, 20), bracket_color, 3)
        cv2.line(frame, (w-20, 20), (w-20, 20+bracket_len), bracket_color, 3)
        cv2.line(frame, (20, h-20), (20+bracket_len, h-20), bracket_color, 3)
        cv2.line(frame, (20, h-20), (20, h-20-bracket_len), bracket_color, 3)
        cv2.line(frame, (w-20, h-20), (w-20-bracket_len, h-20), bracket_color, 3)
        cv2.line(frame, (w-20, h-20), (w-20, h-20-bracket_len), bracket_color, 3)
        
        # Movement bar
        bar_x, bar_y = 30, h - 60
        bar_width = min(int(movement_amount * 3), 150)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 150, bar_y + 15), (50, 50, 50), -1)
        bar_color = (0, 255, 100) if movement_amount > self.movement_threshold else (255, 150, 50)
        if bar_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), bar_color, -1)
        cv2.putText(frame, "MOVEMENT", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Stats display
        cv2.putText(frame, f"MOVES: {self.moment_count}", (w - 130, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        cv2.putText(frame, f"BLINKS: {self.blink_count}", (w - 130, h - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        
        return frame, current_ear, moment_detected, face_detected
    
    def get_liveness_score(self):
        """Calculate score from both movements and blinks"""
        total = self.moment_count + self.blink_count
        if total >= self.required_moments:
            return 100
        return (total / self.required_moments) * 100
    
    def is_liveness_passed(self):
        return (self.moment_count + self.blink_count) >= self.required_moments
    
    def cleanup(self):
        self.face_mesh.close()


def run_liveness_test(required_moments=3):
    detector = LivenessDetector(required_moments=required_moments)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access webcam")
        return False, 0, 0
    
    print("🎭 Liveness Test: Move head OR blink naturally")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame, ear, detected, face = detector.process_frame(frame)
            cv2.imshow('Liveness Test', frame)
            
            if detector.is_liveness_passed():
                cv2.waitKey(2000)
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
    
    return detector.is_liveness_passed(), detector.moment_count + detector.blink_count, detector.get_liveness_score()


if __name__ == "__main__":
    passed, total, score = run_liveness_test(3)
    print(f"Passed: {passed}, Total: {total}, Score: {score}%")