import threading
import queue
import tempfile
import os
import re
import logging
from typing import Optional, List

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chimege STT client
# ---------------------------------------------------------------------------

class ChimegeSTT:
    """
    Chimege.mn монгол дуу таних API.
    POST https://api.chimege.com/v1.2/transcribe
    Header: token: <api_token>
    Body:   multipart/form-data  file=<wav>
    """

    def __init__(self, token: str, base_url: str = "https://api.chimege.com/v1.2"):
        self.token = token
        self.url = f"{base_url}/transcribe"

    def transcribe(self, audio_path: str) -> Optional[str]:
        """WAV файлыг Chimege API-д илгээж текст буцаана."""
        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            resp = requests.post(
                self.url,
                data=audio_data,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Token": self.token,
                    "Punctuate": "false",
                },
                timeout=10.0,
            )
            resp.raise_for_status()

            # Chimege хариу: plain text (JSON биш)
            text = resp.content.decode("utf-8").strip()
            logger.info(f"[CHIMEGE] → {text!r}")
            return text if text else None

        except requests.Timeout:
            logger.warning("[CHIMEGE] Timeout")
        except requests.HTTPError as e:
            logger.error(f"[CHIMEGE] HTTP алдаа: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            logger.error(f"[CHIMEGE] Алдаа: {e}")
        return None


# ---------------------------------------------------------------------------
# Voice class
# ---------------------------------------------------------------------------

class Voice:

    DEFAULT_TOKEN = "ee4859755ab14aae166bd9e0bd8755612c4894b3a004a423a543df24f56d5dec"

    def __init__(
        self,
        chimege_token: str = DEFAULT_TOKEN,
        sample_rate: int = 16000,
        channels: int = 1,
        blocksize: int = 800,
        vad_threshold: float = 0.015,
        silence_limit: float = 0.8,
        min_voice_len: float = 0.3,
        max_voice_len: float = 3.0,
        cmd_queue_maxsize: int = 10,
    ):
        self._cmd_q: queue.Queue = queue.Queue(maxsize=cmd_queue_maxsize)
        self._audio_q: queue.Queue = queue.Queue(maxsize=200)
        self.running: bool = False

        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.threshold = vad_threshold
        self.silence_limit = silence_limit
        self.min_voice_len = min_voice_len
        self.max_voice_len = max_voice_len

        self._stt = ChimegeSTT(token=chimege_token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.running = True
        threading.Thread(target=self._vad_loop, daemon=True, name="vad-loop").start()
        threading.Thread(target=self._transcribe_worker, daemon=True, name="transcribe-worker").start()
        logger.info("[VOICE] threads started (Chimege STT)")

    def stop(self) -> None:
        self.running = False

    def get(self) -> Optional[str]:
        try:
            return self._cmd_q.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        if not text:
            return ""
        text = str(text).lower().strip()
        text = re.sub(r"[^a-zA-Z0-9а-яөүё\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _has_any(text: str, keywords: List[str]) -> bool:
        return any(k in text for k in keywords)

    def _map_command(self, text: str) -> Optional[str]:
        t = self._normalize(text)
        if not t:
            return None

        if self._has_any(t, ["зогс", "зогсоо", "зогсо", "stop", "halt", "pause"]):
            return "stop"
        if self._has_any(t, ["үргэлжлүүл", "цааш", "resume", "go on"]):
            return "resume"
        if self._has_any(t, ["эргэхээ боль", "эргэлтээ зогсоо", "turn stop", "чигээрээ"]):
            return "turn_stop"
        if self._has_any(t, ["явахаа боль", "хөдөлгөөнөө зогсоо", "move stop"]):
            return "move_stop"

        if ("90" in t or "ерэн" in t) and self._has_any(t, ["баруун", "baruun", "right"]):
            return "turn_right_90"
        if ("90" in t or "ерэн" in t) and self._has_any(t, ["зүүн", "zuun", "left"]):
            return "turn_left_90"
        if ("45" in t or "дөчин тав" in t) and self._has_any(t, ["баруун", "baruun", "right"]):
            return "turn_right_45"
        if ("45" in t or "дөчин тав" in t) and self._has_any(t, ["зүүн", "zuun", "left"]):
            return "turn_left_45"

        if self._has_any(t, ["хурдан", "түргэн", "хурдаа", "хурд нэм", "speed up", "faster"]):
            return "faster"
        if self._has_any(t, ["удаан", "зөөлөн", "саар", "slow", "slower"]):
            return "slower"

        if self._has_any(t, ["урагш", "урагшаа", "forward", "go forward"]):
            return "forward"
        if self._has_any(t, ["ухар", "ухраа", "арагш", "back", "backward"]):
            return "backward"

        if self._has_any(t, ["зүүн", "зүүн тийш", "zuun", "left"]):
            return "left"
        if self._has_any(t, ["баруун", "баруун тийш", "baruun", "right"]):
            return "right"

        return None

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _save_temp(self, audio: np.ndarray) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        path = tmp.name
        tmp.close()
        sf.write(path, audio, self.sample_rate)
        return path

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning(f"Audio status: {status}")
        try:
            self._audio_q.put_nowait(indata.copy().flatten())
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Worker threads
    # ------------------------------------------------------------------

    def _transcribe_worker(self) -> None:
        audio_buffer_q: queue.Queue = queue.Queue()
        self._audio_ready_q = audio_buffer_q

        while self.running:
            try:
                audio = audio_buffer_q.get(timeout=1.0)
            except queue.Empty:
                continue

            path = self._save_temp(audio)
            try:
                text = self._stt.transcribe(path)
                if not text:
                    continue

                cmd = self._map_command(text)
                logger.info(f"[STT] {text!r} → {cmd!r}")

                if cmd:
                    try:
                        self._cmd_q.put_nowait(cmd)
                    except queue.Full:
                        logger.warning("Command queue дүүрсэн.")

            finally:
                if os.path.exists(path):
                    os.remove(path)

    def _vad_loop(self) -> None:
        voiced_frames: list = []
        speaking = False
        silence_time = 0.0
        voice_time = 0.0
        block_dur = self.blocksize / self.sample_rate

        import time
        deadline = time.time() + 5.0
        while not hasattr(self, "_audio_ready_q") and time.time() < deadline:
            time.sleep(0.05)

        if not hasattr(self, "_audio_ready_q"):
            logger.error("Transcribe worker эхлээгүй.")
            return

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._audio_callback,
        ):
            logger.info("[VOICE] VAD эхэллээ")

            while self.running:
                try:
                    chunk = self._audio_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms > self.threshold:
                    if not speaking:
                        speaking = True
                        voiced_frames = []
                        silence_time = 0.0
                        voice_time = 0.0

                    voiced_frames.append(chunk)
                    voice_time += block_dur
                    silence_time = 0.0

                    if voice_time >= self.max_voice_len:
                        self._flush(voiced_frames)
                        speaking = False
                else:
                    if speaking:
                        voiced_frames.append(chunk)
                        silence_time += block_dur
                        voice_time += block_dur

                        if silence_time >= self.silence_limit:
                            self._flush(voiced_frames)
                            speaking = False

    def _flush(self, frames: list) -> None:
        audio = np.concatenate(frames)
        if len(audio) / self.sample_rate >= self.min_voice_len:
            try:
                self._audio_ready_q.put_nowait(audio)
            except queue.Full:
                logger.warning("Transcribe queue дүүрсэн.")