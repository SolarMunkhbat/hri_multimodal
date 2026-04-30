import threading
import queue
import tempfile
import os
import re

import numpy as np
import sounddevice as sd
import soundfile as sf
from gradio_client import Client, handle_file


class Voice:
    def __init__(self):
        self.q = queue.Queue()
        self.running = False

        self.client = Client("https://stt.rookies.mn/")

        self.sample_rate = 16000
        self.channels = 1
        self.blocksize = 800   # ~50ms at 16kHz

        # voice activity detection
        self.threshold = 0.015
        self.silence_limit = 0.8
        self.min_voice_len = 0.3
        self.max_voice_len = 3.0

        self.audio_q = queue.Queue()

    def start(self):
        self.running = True
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self.running = False

    def normalize(self, text):
        if text is None:
            return ""
        text = str(text).lower().strip()
        text = re.sub(r"[^a-zA-Z0-9а-яөүё\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def has_any(self, text, keywords):
        return any(k in text for k in keywords)

    def map_command(self, text):
        t = self.normalize(text)
        if not t:
            return None

        # full stop / resume
        if self.has_any(t, [
            "зогс", "зогсоо", "зогсо", "stop", "halt", "pause"
        ]):
            return "stop"

        if self.has_any(t, [
            "үргэлжлүүл", "цааш", "resume", "go on"
        ]):
            return "resume"

        # angular stop only
        if self.has_any(t, [
            "эргэхээ боль", "эргэлтээ зогсоо", "turn stop",
            "straight ahead", "straight only", "чигээрээ"
        ]):
            return "turn_stop"

        # linear stop only
        if self.has_any(t, [
            "явахаа боль", "хөдөлгөөнөө зогсоо", "move stop", "linear stop"
        ]):
            return "move_stop"

        # angle turns
        if (("90" in t or "ерэн" in t) and
            ("баруун" in t or "baruun" in t or "right" in t)):
            return "turn_right_90"

        if (("90" in t or "ерэн" in t) and
            ("зүүн" in t or "zuun" in t or "left" in t)):
            return "turn_left_90"

        if (("45" in t or "дөчин тав" in t) and
            ("баруун" in t or "baruun" in t or "right" in t)):
            return "turn_right_45"

        if (("45" in t or "дөчин тав" in t) and
            ("зүүн" in t or "zuun" in t or "left" in t)):
            return "turn_left_45"

        # speed
        if self.has_any(t, [
            "хурдан", "түргэн", "хурдас", "speed up", "faster"
        ]):
            return "faster"

        if self.has_any(t, [
            "удаан", "зөөлөн", "саар", "slow", "slower"
        ]):
            return "slower"

        # linear
        if self.has_any(t, [
            "урагш", "урагшаа", "forward", "go forward"
        ]):
            return "forward"

        if self.has_any(t, [
            "ухар", "ухраа", "арагш", "back", "backward"
        ]):
            return "backward"

        # angular
        if self.has_any(t, [
            "зүүн", "зүүн тийш", "zuun", "left"
        ]):
            return "left"

        if self.has_any(t, [
            "баруун", "баруун тийш", "baruun", "right"
        ]):
            return "right"

        # optional mode commands
        if self.has_any(t, ["gesture mode", "gesture", "gesture горим"]):
            return "gesture"

        if self.has_any(t, ["voice mode", "voice", "voice горим"]):
            return "voice"

        if self.has_any(t, ["multi", "multimodal", "хоёул", "хамт"]):
            return "multi"

        return None

    def save_temp(self, audio):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        path = tmp.name
        tmp.close()
        sf.write(path, audio, self.sample_rate)
        return path

    def transcribe_and_queue(self, audio):
        path = self.save_temp(audio)
        try:
            text = self.client.predict(
                audio=handle_file(path),
                backend="faster-whisper",
                api_name="/transcribe"
            )

            cmd = self.map_command(text)

            print("[VOICE RAW]:", text)
            print("[VOICE NORM]:", self.normalize(text))
            print("[VOICE CMD]:", cmd)

            if cmd:
                self.q.put(cmd)

        except Exception as e:
            print("Voice error:", e)

        finally:
            if os.path.exists(path):
                os.remove(path)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        self.audio_q.put(indata.copy().flatten())

    def loop(self):
        voiced_frames = []
        speaking = False
        silence_time = 0.0
        voice_time = 0.0
        block_duration = self.blocksize / self.sample_rate

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self.audio_callback
        ):
            print("[VOICE] realtime listener started")

            while self.running:
                chunk = self.audio_q.get()
                rms = np.sqrt(np.mean(chunk ** 2))

                if rms > self.threshold:
                    if not speaking:
                        print("[VOICE] speech started")
                        speaking = True
                        voiced_frames = []
                        silence_time = 0.0
                        voice_time = 0.0

                    voiced_frames.append(chunk)
                    voice_time += block_duration
                    silence_time = 0.0

                    if voice_time >= self.max_voice_len:
                        print("[VOICE] max voice length reached")
                        audio = np.concatenate(voiced_frames)
                        speaking = False

                        if len(audio) / self.sample_rate >= self.min_voice_len:
                            self.transcribe_and_queue(audio)

                else:
                    if speaking:
                        voiced_frames.append(chunk)
                        silence_time += block_duration
                        voice_time += block_duration

                        if silence_time >= self.silence_limit:
                            print("[VOICE] speech ended")
                            audio = np.concatenate(voiced_frames)
                            speaking = False

                            if len(audio) / self.sample_rate >= self.min_voice_len:
                                self.transcribe_and_queue(audio)

    def get(self):
        if not self.q.empty():
            return self.q.get()
        return None