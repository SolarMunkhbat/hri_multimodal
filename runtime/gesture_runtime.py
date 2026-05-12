import cv2
import json
import os
import tempfile
import zipfile
import numpy as np
import mediapipe as mp
import threading
from collections import Counter, deque
from typing import Tuple, Optional
from keras.models import load_model as _keras_load_model

from runtime.common import extract_left_and_right, get_pinch_distance, normalize_speed, SEQ_LEN, FEATURE_SIZE

CONFIDENCE_THRESHOLD = 0.7
_SMOOTH_WINDOW = 3   # majority-vote window (frames)
_CLEAR_GRACE  = 5    # absent frames before sequence is reset


def _load_model_compat(path: str):
    """Load a .keras file saved with an older Keras 3.x by stripping unknown config keys."""
    def _strip(obj):
        if isinstance(obj, dict):
            if obj.get("class_name") == "Dense":
                obj.get("config", {}).pop("quantization_config", None)
            for v in obj.values():
                _strip(v)
        elif isinstance(obj, list):
            for v in obj:
                _strip(v)

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with zipfile.ZipFile(path, "r") as zin:
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == "config.json":
                        cfg = json.loads(data.decode())
                        _strip(cfg)
                        data = json.dumps(cfg).encode()
                    zout.writestr(item, data)
        return _keras_load_model(tmp_path, compile=False)
    finally:
        os.unlink(tmp_path)


class _Smoother:
    """Majority-vote smoothing over the last N predictions."""

    def __init__(self, window: int = _SMOOTH_WINDOW):
        self._buf: deque = deque(maxlen=window)

    def update(self, cmd: str, conf: float) -> Tuple[str, float]:
        self._buf.append((cmd, conf))
        counts = Counter(c for c, _ in self._buf)
        best = counts.most_common(1)[0][0]
        best_conf = max((f for c, f in self._buf if c == best), default=conf)
        return best, best_conf

    def reset(self) -> None:
        self._buf.clear()


class Gesture:
    """
    MediaPipe + LSTM ашиглан гарын gesture таньж команд буцаана.

    run() нь тухайн frame-г background thread-д дамжуулж, сүүлийн
    бэлэн үр дүнг шууд буцаана — main loop блокддоггүй.
    """

    def __init__(
        self,
        left_model_path: str = "models/left_lstm.keras",
        right_model_path: str = "models/right_lstm.keras",
        left_labels_path: str = "models/left_label_map.npy",
        right_labels_path: str = "models/right_label_map.npy",
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        speed_decay: float = 0.85,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ):
        self.left_model = _load_model_compat(left_model_path)
        self.right_model = _load_model_compat(right_model_path)

        self.left_labels: np.ndarray = np.load(left_labels_path, allow_pickle=True)
        self.right_labels: np.ndarray = np.load(right_labels_path, allow_pickle=True)

        self.confidence_threshold = confidence_threshold
        self.speed_decay = speed_decay

        # Sequence buffers and speed — accessed only from bg thread
        self._lseq: deque = deque(maxlen=SEQ_LEN)
        self._rseq: deque = deque(maxlen=SEQ_LEN)
        self._last_speed: float = 0.0
        self._lsmoother = _Smoother()
        self._rsmoother = _Smoother()
        self._l_absent: int = 0
        self._r_absent: int = 0

        # Latest result returned to the main loop
        self._result: Tuple = ("idle", "idle", 0.0, 0.0, 0.0)
        self._result_lock = threading.Lock()

        # Pending frame queue (keep only the newest)
        self._pending: Optional[np.ndarray] = None
        self._pending_lock = threading.Lock()
        self._frame_ready = threading.Event()

        # MediaPipe (only used inside bg thread)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._running = True
        self._worker = threading.Thread(target=self._bg_loop, daemon=True, name="gesture-worker")
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, frame: np.ndarray) -> Tuple[str, str, float, float, float]:
        """Submit frame for async inference; return latest cached result immediately."""
        with self._pending_lock:
            self._pending = frame.copy()
        self._frame_ready.set()
        with self._result_lock:
            return self._result

    def close(self) -> None:
        self._running = False
        self._frame_ready.set()
        self._worker.join(timeout=2.0)
        self.hands.close()

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _bg_loop(self) -> None:
        while self._running:
            self._frame_ready.wait(timeout=0.1)
            self._frame_ready.clear()
            with self._pending_lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                continue
            result = self._compute(frame)
            with self._result_lock:
                self._result = result

    def _compute(self, frame: np.ndarray) -> Tuple[str, str, float, float, float]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        lf, rf, ll, rl = extract_left_and_right(res)

        _zeros = np.zeros(FEATURE_SIZE, dtype=np.float32)

        # --- Left hand ---
        if ll is not None:
            self._l_absent = 0
            self._lseq.append(lf)
            speed = normalize_speed(get_pinch_distance(ll))
            self._last_speed = speed
        else:
            self._l_absent += 1
            if self._l_absent >= _CLEAR_GRACE:
                # Hand truly gone — reset everything
                self._lseq.clear()
                self._lsmoother.reset()
            else:
                # Brief absence — append zeros (matches training data collection)
                self._lseq.append(_zeros)
            self._last_speed *= self.speed_decay
            speed = self._last_speed

        lcmd, lconf = self._predict(self.left_model, self.left_labels, self._lseq)
        lcmd, lconf = self._lsmoother.update(lcmd, lconf)

        # --- Right hand ---
        if rl is not None:
            self._r_absent = 0
            self._rseq.append(rf)
        else:
            self._r_absent += 1
            if self._r_absent >= _CLEAR_GRACE:
                self._rseq.clear()
                self._rsmoother.reset()
            else:
                self._rseq.append(_zeros)

        rcmd, rconf = self._predict(self.right_model, self.right_labels, self._rseq)
        rcmd, rconf = self._rsmoother.update(rcmd, rconf)

        return lcmd, rcmd, speed, lconf, rconf

    def _predict(self, model, labels: np.ndarray, seq: deque) -> Tuple[str, float]:
        if len(seq) < SEQ_LEN:
            return "idle", 0.0
        inp = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)
        pred = model(inp, training=False).numpy()[0]
        conf = float(np.max(pred))
        if conf >= self.confidence_threshold:
            return str(labels[int(np.argmax(pred))]), conf
        return "idle", conf
