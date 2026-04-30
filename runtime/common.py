import numpy as np

SEQ_LEN = 40

def extract_hand_features(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x - wrist.x)
        features.append(lm.y - wrist.y)
        features.append(lm.z - wrist.z)
    return np.array(features, dtype=np.float32)


def extract_left_and_right(results):
    left_features = np.zeros(63, dtype=np.float32)
    right_features = np.zeros(63, dtype=np.float32)

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


def get_pinch_distance(hand_landmarks):
    if hand_landmarks is None:
        return 0.0

    thumb = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    p1 = np.array([thumb.x, thumb.y, thumb.z])
    p2 = np.array([index_tip.x, index_tip.y, index_tip.z])

    return np.linalg.norm(p1 - p2)


def normalize_speed(dist, min_d=0.03, max_d=0.18):
    val = (dist - min_d) / (max_d - min_d)
    return max(0.0, min(1.0, val))