import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
from runtime.common import extract_left_and_right, get_pinch_distance, normalize_speed

class Gesture:
    def __init__(self):
        self.left_model = load_model("models/left_lstm.keras")
        self.right_model = load_model("models/right_lstm.keras")

        self.left_labels = np.load("models/left_label_map.npy", allow_pickle=True)
        self.right_labels = np.load("models/right_label_map.npy", allow_pickle=True)

        self.lseq = deque(maxlen=40)
        self.rseq = deque(maxlen=40)

        self.hands = mp.solutions.hands.Hands(max_num_hands=2)

    def run(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        lf, rf, ll, rl = extract_left_and_right(res)

        lcmd = "idle"
        rcmd = "idle"
        speed = 0.0

        # =========================
        # LEFT HAND
        # =========================
        if ll is not None:
            self.lseq.append(lf)

            if len(self.lseq) == 40:
                pred = self.left_model.predict(np.expand_dims(self.lseq, 0), verbose=0)[0]
                conf = float(np.max(pred))

                if conf > 0.7:
                    lcmd = self.left_labels[np.argmax(pred)]

            # speed зөвхөн зүүн гараас
            speed = normalize_speed(get_pinch_distance(ll))

        else:
            # 🚨 хамгийн чухал: гар алга болвол reset
            self.lseq.clear()
            lcmd = "idle"
            speed = 0.0

        # =========================
        # RIGHT HAND
        # =========================
        if rl is not None:
            self.rseq.append(rf)

            if len(self.rseq) == 40:
                pred = self.right_model.predict(np.expand_dims(self.rseq, 0), verbose=0)[0]
                conf = float(np.max(pred))

                if conf > 0.7:
                    rcmd = self.right_labels[np.argmax(pred)]

        else:
            # 🚨 хамгийн чухал: баруун гар алга → эргэлт STOP
            self.rseq.clear()
            rcmd = "idle"

        return lcmd, rcmd, speed