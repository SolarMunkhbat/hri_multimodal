import os
import time
import cv2
import numpy as np
import mediapipe as mp

from runtime.common import SEQ_LEN, extract_left_and_right

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "data", "right_dynamic")

CLASSES = ["left", "right", "idle"]
NUM_SEQUENCES_PER_CLASS = 40
CAMERA_INDEX = 0
FRAME_DELAY = 0.06
WINDOW_NAME = "Collect Right Hand Sequences"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


def ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)
    for class_name in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, class_name), exist_ok=True)


def get_next_index(class_dir):
    files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
    if not files:
        return 0

    arr = []
    for f in files:
        try:
            idx = int(f.split("_")[-1].split(".")[0])
            arr.append(idx)
        except:
            pass

    if not arr:
        return 0
    return max(arr) + 1


def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.flip(frame, 1)


def process_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return hands.process(rgb)


def draw_landmarks(frame, results):
    if not results.multi_hand_landmarks:
        return

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


def draw_info(frame, class_name, saved):
    cv2.putText(frame, f"Class: {class_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Saved: {saved}/{NUM_SEQUENCES_PER_CLASS}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, "RIGHT hand only", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.putText(frame, "s=record  n=next  x=quit", (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)


def record_sequence(cap, class_name, seq_id):
    sequence = []

    while len(sequence) < SEQ_LEN:
        frame = read_frame(cap)
        if frame is None:
            continue

        results = process_frame(frame)
        _, right_features, _, _ = extract_left_and_right(results)

        sequence.append(right_features)

        draw_landmarks(frame, results)

        cv2.putText(frame, f"Recording: {class_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Seq: {seq_id}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"Frame: {len(sequence)}/{SEQ_LEN}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, "RIGHT hand only", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None

        time.sleep(FRAME_DELAY)

    return np.array(sequence, dtype=np.float32)


def save_sequence(class_dir, seq_id, seq):
    path = os.path.join(class_dir, f"seq_{seq_id:04d}.npy")
    np.save(path, seq)
    print(f"[SAVED] {path} shape={seq.shape}")


def collect_class(cap, class_name):
    class_dir = os.path.join(DATASET_DIR, class_name)
    start_idx = get_next_index(class_dir)
    saved = 0

    while saved < NUM_SEQUENCES_PER_CLASS:
        frame = read_frame(cap)
        if frame is None:
            continue

        results = process_frame(frame)
        draw_landmarks(frame, results)
        draw_info(frame, class_name, saved)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("x"):
            return "quit"

        if key == ord("n"):
            return "next"

        if key == ord("s"):
            seq_id = start_idx + saved
            print(f"[INFO] Recording {class_name} seq {seq_id}")

            seq = record_sequence(cap, class_name, seq_id)
            if seq is None:
                continue

            save_sequence(class_dir, seq_id, seq)
            saved += 1

    return "done"


def main():
    ensure_dirs()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Camera not opened")
        return

    for class_name in CLASSES:
        result = collect_class(cap, class_name)
        if result == "quit":
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("[INFO] Right-hand collection done")


if __name__ == "__main__":
    main()