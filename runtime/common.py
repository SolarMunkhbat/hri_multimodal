import numpy as np
from typing import Tuple, Optional
import mediapipe as mp

SEQ_LEN = 40
FEATURE_SIZE = 63  # 21 landmarks * 3 (x, y, z)


def extract_hand_features(hand_landmarks) -> np.ndarray:
    """
    Wrist-relative coordinates буцаана.
    21 landmark * 3 (x, y, z) = 63 feature
    """
    wrist = hand_landmarks.landmark[0]
    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x - wrist.x)
        features.append(lm.y - wrist.y)
        features.append(lm.z - wrist.z)
    return np.array(features, dtype=np.float32)


def extract_left_and_right(results) -> Tuple[np.ndarray, np.ndarray, Optional[object], Optional[object]]:
    """
    MediaPipe results-с зүүн/баруун гарын feature болон landmark буцаана.
    Гар олдохгүй бол zero array буцаана.
    """
    left_features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    right_features = np.zeros(FEATURE_SIZE, dtype=np.float32)

    left_landmarks = None
    right_landmarks = None

    if not results.multi_hand_landmarks:
        return left_features, right_features, left_landmarks, right_landmarks

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness.classification[0].label
        feat = extract_hand_features(hand_landmarks)

        if label == "Left":
            left_features = feat
            left_landmarks = hand_landmarks
        elif label == "Right":
            right_features = feat
            right_landmarks = hand_landmarks

    return left_features, right_features, left_landmarks, right_landmarks


def get_pinch_distance(hand_landmarks) -> float:
    """
    Эрхий болон долоовор хуруун дээрх зайг буцаана.
    Гар байхгүй бол 0.0.
    """
    if hand_landmarks is None:
        return 0.0

    thumb = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    p1 = np.array([thumb.x, thumb.y, thumb.z])
    p2 = np.array([index_tip.x, index_tip.y, index_tip.z])

    return float(np.linalg.norm(p1 - p2))


def normalize_speed(dist: float, min_d: float = 0.03, max_d: float = 0.18) -> float:
    """
    Pinch зайг [0, 1] хооронд normalize хийнэ.
    """
    val = (dist - min_d) / (max_d - min_d)
    return float(max(0.0, min(1.0, val)))