import cv2
import os
import logging
from runtime.gesture_runtime import Gesture
from runtime.fusion import FusionState, fuse
from runtime.udp_sender import UdpSender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Тохиргоо
# ---------------------------------------------------------------------------
USE_LLM      = True
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

# Chimege token — .env файлаас уншина, эсвэл доор шууд оруулна
CHIMEGE_TOKEN = "ee4859755ab14aae166bd9e0bd8755612c4894b3a004a423a543df24f56d5dec"

# HUD өнгөнүүд
COLOR_GREEN   = (0, 255, 0)
COLOR_YELLOW  = (255, 255, 0)
COLOR_CYAN    = (255, 200, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_ORANGE  = (0, 165, 255)
COLOR_RED     = (0, 0, 255)


def draw_hud(frame, left_cmd, right_cmd, speed, state, lin, ang, last_voice, llm_mode):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.65, 2
    mode_label = f"[LLM:{OLLAMA_MODEL}]" if llm_mode else "[Keyword]"
    lines = [
        (f"L:{left_cmd}  R:{right_cmd}  {mode_label}",              COLOR_GREEN),
        (f"Speed:{speed:.2f}  Scale:{state.speed_scale:.1f}",        COLOR_YELLOW),
        (f"Voice: {last_voice}",                                      COLOR_CYAN),
        (f"VoiceLin:{state.voice_linear} VoiceAng:{state.voice_angular}", COLOR_ORANGE),
        (f"OUT  lin:{lin:.2f}  ang:{ang:.2f}",                        COLOR_MAGENTA),
    ]
    if state.is_emergency_stopped:
        lines.insert(0, ("!! EMERGENCY STOP !!", COLOR_RED))
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (10, 30 + i * 30), font, fs, color, th)


def main():
    # --- Voice backend ---
    if USE_LLM:
        try:
            from runtime.llm_control import LLMVoice
            voice = LLMVoice(
                ollama_url=OLLAMA_URL,
                model=OLLAMA_MODEL,
                chimege_token=CHIMEGE_TOKEN,
            )
            llm_mode = True
            logger.info(f"[MODE] LLM+Chimege — model:{OLLAMA_MODEL}")
        except Exception as e:
            logger.warning(f"LLM горим алдаа ({e}), keyword fallback")
            from runtime.voice_control import Voice
            voice = Voice(chimege_token=CHIMEGE_TOKEN)
            llm_mode = False
    else:
        from runtime.voice_control import Voice
        voice = Voice(chimege_token=CHIMEGE_TOKEN)
        llm_mode = False
        logger.info("[MODE] Keyword+Chimege")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Camera нээгдсэнгүй.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    gesture = Gesture()
    state   = FusionState()
    udp     = UdpSender()

    voice.start()
    logger.info("HRI эхэллээ. Гарахын тулд 'q' дарна уу.")

    last_voice_text = "None"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            left_cmd, right_cmd, speed = gesture.run(frame)
            voice_cmd = voice.get()

            if voice_cmd:
                last_voice_text = voice_cmd
                logger.info(f"CMD: {voice_cmd}")

            lin, ang = fuse(state, left_cmd, right_cmd, speed, voice_cmd)
            udp.send({"linear_x": lin, "angular_z": ang})

            draw_hud(frame, left_cmd, right_cmd, speed, state, lin, ang, last_voice_text, llm_mode)
            cv2.imshow("HRI Multimodal Control", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        voice.stop()
        gesture.close()
        udp.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Систем хаагдлаа.")


if __name__ == "__main__":
    main()