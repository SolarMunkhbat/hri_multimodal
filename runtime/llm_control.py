"""
llm_control.py — Local Ollama LLM-р байгалийн хэлний тушаалыг robot командад хөрвүүлнэ.

Pipeline:
    STT (raw text)
        │
        ▼
    OllamaCommandParser.parse(text)   ← LLM горимд ЗӨВХӨН ЭНЭ ажиллана
        │
        ▼
    robot команд

LLM горимд keyword pipeline УНТАРНА — давхардал байхгүй.
Ollama байхгүй үед keyword fallback ашиглана.
"""

import json
import logging
import threading
import queue
from typing import Optional, Dict, Any

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — монгол жишээнүүдийг нэмсэн
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a robot command parser for a Mongolian human-robot interaction system.
Map the user's utterance to EXACTLY ONE of these commands:
  stop, resume, forward, backward, left, right,
  faster, slower, move_stop, turn_stop,
  turn_left_90, turn_right_90, turn_left_45, turn_right_45

Rules:
- Return ONLY valid JSON: {"command": "..."} or {"command": null}
- EXACTLY ONE command — never two words like "slower forward"
- No explanation, no markdown, no extra text
- If utterance has multiple actions, pick the MOST IMPORTANT one (direction > speed)
- "удаан урагшаа яв" → forward (direction wins over speed)
- "хурдан баруун эрэг" → right (direction wins over speed)
- "баруун" / "баруун тийш" / "баруун эрэг" = right (NEVER backward)
- "зүүн" / "зүүн тийш" / "зүүн эрэг" = left
- "урагш" / "урагшаа" / "урагшаа яв" = forward
- "ухар" / "ухраа" / "арагш" = backward
- "зогс" / "зогсоо" / "stop" = stop
- "үргэлжлүүл" / "цааш яв" / "resume" = resume
- "хурдаа нэм" / "хурдан" / "faster" = faster (alone, no direction)
- "удаан" / "саар" / "slower" = slower (alone, no direction)
- "чигээрээ" / "чигээрээ яваарай" = turn_stop (NOT stop)
- "эргэхээ боль" / "ирэхээ боль" = turn_stop
- "явахаа боль" = move_stop
- "ерэн градус баруун" / "90 баруун" = turn_right_90
- "ерэн градус зүүн" / "90 зүүн" = turn_left_90
- "45 баруун" = turn_right_45
- "45 зүүн" = turn_left_45

Examples:
  "баруун эрэг"              → {"command": "right"}
  "удаан урагшаа яв"         → {"command": "forward"}
  "хурдан зүүн эрэг"         → {"command": "left"}
  "зогс"                     → {"command": "stop"}
  "чигээрээ яваарай"         → {"command": "turn_stop"}
  "ерэн градус баруун эрэг"  → {"command": "turn_right_90"}
  "сайн байна уу"            → {"command": null}
  "тэгээрэй"                 → {"command": null}
"""


class OllamaCommandParser:
    """
    Ollama REST API-г ашиглан байгалийн хэлний текстийг robot командад хөрвүүлнэ.
    Ollama байхгүй үед keyword fallback ажиллана.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 15.0,
        max_retries: int = 1,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._available: Optional[bool] = None
        self._fallback = _KeywordFallback()

    def parse(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        if self._is_available():
            cmd = self._llm_parse(text)
            if cmd is not None:
                return cmd
            logger.debug(f"LLM null буцаав, fallback: {text!r}")
        return self._fallback.parse(text)

    def health_check(self) -> bool:
        if not _REQUESTS_AVAILABLE:
            return False
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def _is_available(self) -> bool:
        if not _REQUESTS_AVAILABLE:
            return False
        if self._available is None:
            self._available = self.health_check()
            status = "олдлоо" if self._available else "олдсонгүй — keyword fallback"
            logger.info(f"[LLM] Ollama {status}: {self.base_url} (model={self.model})")
        return self._available

    def _llm_parse(self, text: str) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 32},
        }

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                content = resp.json()["message"]["content"].strip()
                logger.debug(f"[LLM] raw: {content!r}")

                clean = content.replace("```json", "").replace("```", "").strip()
                data: Dict[str, Any] = json.loads(clean)
                cmd = data.get("command")
                if cmd and isinstance(cmd, str):
                    logger.info(f"[LLM] {text!r} → {cmd!r}")
                    return cmd
                return None  # null буцаасан — caller fallback шийднэ

            except requests.Timeout:
                logger.warning(f"[LLM] Timeout ({attempt+1}/{self.max_retries+1})")
                self._available = False
            except requests.ConnectionError:
                logger.warning("[LLM] Холбогдсонгүй → fallback")
                self._available = False
                break
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[LLM] JSON parse алдаа: {e}")
                return None
            except Exception as e:
                logger.error(f"[LLM] Алдаа: {e}")
                break
        return None


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------

class _KeywordFallback:
    _RULES = [
        (["зогс", "зогсоо", "зогсооч","stop", "halt", "pause"],               "stop"),
        (["үргэлжлүүл", "resume", "go on"],                                   "resume"),
        (["эргэхээ боль", "turn stop"],                                      "turn_stop"),
        (["явахаа боль", "move stop"],                                         "move_stop"),
        (["хурдан", "түргэн", "хурдаа", "хурд нэм", "faster", "speed up"],   "faster"),
        (["удаан", "зөөлөн", "саар", "slow", "slower"],                       "slower"),
        (["урагш", "урагшаа", "forward", "go forward"],                       "forward"),
        (["ухар", "ухраа", "арагш", "арагшаа", "backward", "back"],           "backward"),
    ]
    _ANGLE_RULES = [
        (["90", "ерэн"], ["баруун", "right"],      "turn_right_90"),
        (["90", "ерэн"], ["зүүн", "left"],         "turn_left_90"),
        (["45", "дөчин тав"], ["баруун", "right"], "turn_right_45"),
        (["45", "дөчин тав"], ["зүүн", "left"],    "turn_left_45"),
    ]

    def parse(self, text: str) -> Optional[str]:
        t = text.lower().strip()
        for angle_kws, dir_kws, cmd in self._ANGLE_RULES:
            if any(a in t for a in angle_kws) and any(d in t for d in dir_kws):
                return cmd
        for keywords, cmd in self._RULES:
            if any(k in t for k in keywords):
                return cmd
        if any(k in t for k in ["зүүн", "left"]):
            return "left"
        if any(k in t for k in ["баруун", "right"]):
            return "right"
        return None


# ---------------------------------------------------------------------------
# LLMVoice — Voice + LLM, keyword pipeline УНТАРСАН
# ---------------------------------------------------------------------------

class LLMVoice:
    """
    LLM горимд Voice-н keyword pipeline-г унтрааж
    зөвхөн LLM (+ fallback)-р parse хийнэ.
    Ингэснээр давхардсан буруу команд queue-д орохгүй.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        chimege_token: str = "",
        **voice_kwargs,
    ):
        from runtime.voice_control import Voice
        if chimege_token:
            voice_kwargs["chimege_token"] = chimege_token

        # keyword pipeline-г унтраахын тулд _map_command-г override хийнэ
        # Voice зөвхөн raw text-г raw_text_q-д бичнэ — команд map хийхгүй
        self._voice = Voice(**voice_kwargs)
        self._parser = OllamaCommandParser(base_url=ollama_url, model=model)
        self._cmd_q: queue.Queue = queue.Queue(maxsize=10)
        self._raw_text_q: queue.Queue = queue.Queue(maxsize=20)
        self._running = False

        # Voice-н keyword map-г disable хийж raw text-г шууд авна
        self._voice._map_command = self._intercept_text

    def _intercept_text(self, text: str) -> None:
        """Voice-н _map_command-г replace хийнэ — keyword map хийхгүй, raw text авна."""
        if text:
            try:
                self._raw_text_q.put_nowait(text)
            except queue.Full:
                pass
        return None  # Voice cmd queue-д юу ч орохгүй

    def start(self) -> None:
        self._voice.start()
        self._running = True
        threading.Thread(target=self._llm_loop, daemon=True, name="llm-parse").start()
        logger.info(f"[LLMVoice] started — model:{self._parser.model}, keyword pipeline: OFF")

    def stop(self) -> None:
        self._running = False
        self._voice.stop()

    def get(self) -> Optional[str]:
        try:
            return self._cmd_q.get_nowait()
        except queue.Empty:
            return None

    def _llm_loop(self) -> None:
        while self._running:
            try:
                raw_text = self._raw_text_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if not raw_text:
                continue

            cmd = self._parser.parse(raw_text)
            logger.info(f"[LLMVoice] {raw_text!r} → {cmd!r}")

            if cmd:
                try:
                    self._cmd_q.put_nowait(cmd)
                except queue.Full:
                    logger.warning("[LLMVoice] command queue дүүрсэн")