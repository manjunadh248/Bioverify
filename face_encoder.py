"""
BioVerify: Face Encoding Module
Uses MediaPipe Face Mesh landmarks to generate face encodings for duplicate detection.

This module creates a unique face signature from facial landmarks that can be:
1. Stored in the database
2. Compared against other face encodings to detect duplicates
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Key landmark indices for face encoding (subset of 468 landmarks)
# These are stable, distinctive points that remain consistent across expressions
KEY_LANDMARK_INDICES = [
    # Face contour
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    # Eyebrows
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
    # Eyes
    33, 160, 158, 133, 153, 144, 163, 7,
    362, 385, 387, 263, 373, 380, 390, 249,
    # Nose
    1, 2, 98, 327, 4, 5, 195, 197,
    # Mouth
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    # Additional key points
    6, 122, 196, 3, 51, 281, 248, 419, 456, 236
]

# Total encoding length
ENCODING_LENGTH = len(KEY_LANDMARK_INDICES) * 3  # x, y, z for each landmark


@dataclass
class FaceEncodingResult:
    """Result of face encoding operation."""
    success: bool
    encoding: Optional[np.ndarray] = None
    error_message: str = ""
    confidence: float = 0.0


class FaceEncoder:
    """
    Encode faces using MediaPipe Face Mesh landmarks.
    
    This creates a normalized face descriptor that can be compared
    across different lighting conditions and slight pose variations.
    """
    
    # Similarity threshold for considering two faces as the same person
    SIMILARITY_THRESHOLD = 0.85
    
    def __init__(self, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        """Initialize the face encoder with MediaPipe Face Mesh."""
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self._encoding_history = []  # Store multiple encodings for averaging
    
    def encode_face(self, frame: np.ndarray) -> FaceEncodingResult:
        """
        Extract face encoding from a single frame.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            FaceEncodingResult with encoding or error
        """
        if frame is None or frame.size == 0:
            return FaceEncodingResult(
                success=False,
                error_message="Invalid frame provided"
            )
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return FaceEncodingResult(
                success=False,
                error_message="No face detected in frame"
            )
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract key landmarks
        landmarks = []
        for idx in KEY_LANDMARK_INDICES:
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0])
        
        encoding = np.array(landmarks, dtype=np.float32)
        
        # Normalize the encoding
        encoding = self._normalize_encoding(encoding)
        
        return FaceEncodingResult(
            success=True,
            encoding=encoding,
            confidence=0.95
        )
    
    def _normalize_encoding(self, encoding: np.ndarray) -> np.ndarray:
        """
        Normalize encoding for consistent comparison.
        
        This makes the encoding invariant to:
        - Face position in frame
        - Face size/distance from camera
        """
        if len(encoding) == 0:
            return encoding
        
        # Reshape to (N, 3) for x, y, z coordinates
        points = encoding.reshape(-1, 3)
        
        # Center the points (translation invariance)
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # Scale to unit variance (scale invariance)
        scale = np.std(centered)
        if scale > 1e-6:
            normalized = centered / scale
        else:
            normalized = centered
        
        return normalized.flatten()
    
    def add_encoding_sample(self, frame: np.ndarray) -> bool:
        """
        Add a face encoding sample to the history for averaging.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            True if encoding was added successfully
        """
        result = self.encode_face(frame)
        if result.success and result.encoding is not None:
            self._encoding_history.append(result.encoding)
            return True
        return False
    
    def get_average_encoding(self) -> Optional[np.ndarray]:
        """
        Get the average of all collected encoding samples.
        
        This provides a more stable representation by averaging
        multiple samples taken at different times/poses.
        
        Returns:
            Average encoding, or None if no samples collected
        """
        if not self._encoding_history:
            return None
        
        # Stack all encodings and compute mean
        stacked = np.stack(self._encoding_history)
        average = np.mean(stacked, axis=0)
        
        # Re-normalize after averaging
        return self._normalize_encoding(average)
    
    def clear_history(self):
        """Clear the encoding history."""
        self._encoding_history = []
    
    def get_sample_count(self) -> int:
        """Get the number of encoding samples collected."""
        return len(self._encoding_history)
    
    @staticmethod
    def compare_faces(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Compare two face encodings and return similarity score.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            Similarity score between 0 (different) and 1 (same person)
        """
        if encoding1 is None or encoding2 is None:
            return 0.0
        
        if len(encoding1) != len(encoding2):
            return 0.0
        
        # Use cosine similarity
        dot_product = np.dot(encoding1, encoding2)
        norm1 = np.linalg.norm(encoding1)
        norm2 = np.linalg.norm(encoding2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Convert from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    @staticmethod
    def is_same_person(encoding1: np.ndarray, encoding2: np.ndarray,
                       threshold: float = None) -> Tuple[bool, float]:
        """
        Determine if two face encodings belong to the same person.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            threshold: Similarity threshold (default: 0.85)
            
        Returns:
            Tuple of (is_same_person, similarity_score)
        """
        if threshold is None:
            threshold = FaceEncoder.SIMILARITY_THRESHOLD
        
        similarity = FaceEncoder.compare_faces(encoding1, encoding2)
        is_same = similarity >= threshold
        
        return is_same, similarity
    
    @staticmethod
    def serialize_encoding(encoding: np.ndarray) -> str:
        """
        Serialize face encoding to a string for database storage.
        
        Args:
            encoding: Face encoding numpy array
            
        Returns:
            JSON string representation
        """
        if encoding is None:
            return ""
        return json.dumps(encoding.tolist())
    
    @staticmethod
    def deserialize_encoding(encoding_str: str) -> Optional[np.ndarray]:
        """
        Deserialize face encoding from database string.
        
        Args:
            encoding_str: JSON string from database
            
        Returns:
            Face encoding as numpy array, or None if invalid
        """
        if not encoding_str:
            return None
        try:
            data = json.loads(encoding_str)
            return np.array(data, dtype=np.float32)
        except (json.JSONDecodeError, ValueError):
            return None
    
    def cleanup(self):
        """Release resources."""
        self.face_mesh.close()


def find_matching_face(new_encoding: np.ndarray, 
                       stored_encodings: List[Tuple[str, np.ndarray]],
                       threshold: float = 0.85) -> Optional[Tuple[str, float]]:
    """
    Find if a face encoding matches any stored encodings.
    
    Args:
        new_encoding: Face encoding to check
        stored_encodings: List of (username, encoding) tuples
        threshold: Similarity threshold
        
    Returns:
        Tuple of (matching_username, similarity) if match found, else None
    """
    if new_encoding is None or not stored_encodings:
        return None
    
    best_match = None
    best_similarity = 0.0
    
    for username, stored_encoding in stored_encodings:
        if stored_encoding is None:
            continue
        
        is_same, similarity = FaceEncoder.is_same_person(
            new_encoding, stored_encoding, threshold
        )
        
        if is_same and similarity > best_similarity:
            best_match = username
            best_similarity = similarity
    
    if best_match:
        return (best_match, best_similarity)
    
    return None


# Singleton encoder instance
_encoder_instance = None

def get_encoder() -> FaceEncoder:
    """Get singleton face encoder instance."""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = FaceEncoder()
    return _encoder_instance
