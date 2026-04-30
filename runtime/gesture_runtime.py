import cv2
import numpy as np
import mediapipe as mp
import threading
from collections import deque
from typing import Tuple, Optional
from tensorflow.keras.models import load_model

from runtime.common import extract_left_and_right, get_pinch_distance, normalize_speed, SEQ_LEN

CONFIDENCE_THRESHOLD = 0.7


class Gesture:
    """
    MediaPipe + LSTM ашиглан гарын gesture таньж команд буцаана.

    Thread-safe: run() method нь өөр thread-с дуудагдаж болно.
    """

    def __init__(
        self,
        left_model_path: str = "models/left_lstm.keras",
        right_model_path: str = "models/right_lstm.keras",
        left_labels_path: str = "models/left_label_map.npy",
        right_labels_path: str = "models/right_label_map.npy",
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_num_hands: int = 2,
    ):
        self.left_model = load_model(left_model_path)
        self.right_model = load_model(right_model_path)

        self.left_labels: np.ndarray = np.load(left_labels_path, allow_pickle=True)
        self.right_labels: np.ndarray = np.load(right_labels_path, allow_pickle=True)

        self.confidence_threshold = confidence_threshold

        self.lseq: deque = deque(maxlen=SEQ_LEN)
        self.rseq: deque = deque(maxlen=SEQ_LEN)

        self._lock = threading.Lock()

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    def _predict(self, model, labels: np.ndarray, seq: deque) -> str:
        """Sequence дүүрэн байвал prediction хийнэ."""
        if len(seq) < SEQ_LEN:
            return "idle"

        inp = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)
        pred = model.predict(inp, verbose=0)[0]
        conf = float(np.max(pred))

        if conf >= self.confidence_threshold:
            return str(labels[int(np.argmax(pred))])
        return "idle"

    def run(self, frame: np.ndarray) -> Tuple[str, str, float]:
        """
        BGR frame авч (left_cmd, right_cmd, speed) буцаана.
        Thread-safe.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        lf, rf, ll, rl = extract_left_and_right(res)

        with self._lock:
            # --- Left hand ---
            if ll is not None:
                self.lseq.append(lf)
                lcmd = self._predict(self.left_model, self.left_labels, self.lseq)
                speed = normalize_speed(get_pinch_distance(ll))
            else:
                self.lseq.clear()
                lcmd = "idle"
                speed = 0.0

            # --- Right hand ---
            if rl is not None:
                self.rseq.append(rf)
                rcmd = self._predict(self.right_model, self.right_labels, self.rseq)
            else:
                self.rseq.clear()
                rcmd = "idle"

        return lcmd, rcmd, speed

    def close(self) -> None:
        """MediaPipe resource-г чөлөөлнэ."""
        self.hands.close()