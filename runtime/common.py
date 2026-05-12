import numpy as np
from typing import Tuple, Optional
import mediapipe as mp

SEQ_LEN = 20
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

    Нэг гар байвал MediaPipe-ийн label-г ашиглана.
    Хоёр гар байвал wrist x-байрлалаар ялгана — frame flip хийсний дараа
    зүүн гар нь image-ийн зүүн талд (x < 0.5) байдаг тул энэ нь MediaPipe-ийн
    label-ийн хооронд frame-ийн handedness swap-с илүү найдвартай.
    """
    left_features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    right_features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    left_landmarks = None
    right_landmarks = None

    if not results.multi_hand_landmarks:
        return left_features, right_features, left_landmarks, right_landmarks

    detected = results.multi_hand_landmarks

    if len(detected) == 1:
        # Single hand: MediaPipe label is reliable
        lm = detected[0]
        label = results.multi_handedness[0].classification[0].label
        feat = extract_hand_features(lm)
        if label == "Left":
            left_features, left_landmarks = feat, lm
        else:
            right_features, right_landmarks = feat, lm
    else:
        # Two hands: sort by wrist x-position.
        # After cv2.flip(frame,1) the user's left hand is on the left side of
        # the image (smaller x) regardless of MediaPipe's handedness label,
        # which can swap between frames when both hands are close together.
        by_x = sorted(detected, key=lambda lm: lm.landmark[0].x)
        left_landmarks  = by_x[0]   # leftmost  = user's left hand
        right_landmarks = by_x[-1]  # rightmost = user's right hand
        left_features  = extract_hand_features(left_landmarks)
        right_features = extract_hand_features(right_landmarks)

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