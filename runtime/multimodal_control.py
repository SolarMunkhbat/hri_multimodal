import cv2
import logging
from runtime.config_loader import load as load_config
from runtime.gesture_runtime import Gesture
from runtime.fusion import FusionState, fuse
from runtime.udp_sender import UdpSender
from runtime.latency_logger import LatencyLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

COLOR_GREEN   = (0, 255, 0)
COLOR_YELLOW  = (255, 255, 0)
COLOR_CYAN    = (255, 200, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_ORANGE  = (0, 165, 255)
COLOR_RED     = (0, 0, 255)
COLOR_WHITE   = (255, 255, 255)


def draw_hud(frame, left_cmd, right_cmd, left_conf, right_conf,
             speed, state, lin, ang, last_voice, llm_mode,
             gesture_ms=0.0, model_name="", nav_target=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.62, 2
    mode_label = f"[LLM:{model_name}]" if llm_mode else "[Keyword]"
    lines = [
        (f"L:{left_cmd}({left_conf:.0%})  R:{right_cmd}({right_conf:.0%})  {mode_label}", COLOR_GREEN),
        (f"Speed:{speed:.2f}  Scale:{state.speed_scale:.1f}",                             COLOR_YELLOW),
        (f"Voice: {last_voice}",                                                           COLOR_CYAN),
        (f"VoiceLin:{state.voice_linear}  VoiceAng:{state.voice_angular}",                COLOR_ORANGE),
        (f"OUT  lin:{lin:.2f}  ang:{ang:.2f}",                                            COLOR_MAGENTA),
        (f"Gesture latency: {gesture_ms:.1f}ms",                                          (200, 200, 200)),
    ]
    if nav_target:
        lines.insert(0, (f">> NAVIGATE TO: {nav_target.upper()} <<", COLOR_YELLOW))
    if state.is_emergency_stopped:
        lines.insert(0, ("!! EMERGENCY STOP !!", COLOR_RED))

    # Confidence bars — зүүн/баруун гарын итгэл хүчийг визуалаар харуулна
    bar_x, bar_y = 10, frame.shape[0] - 50
    for i, (conf, label, color) in enumerate([
        (left_conf, "L", COLOR_GREEN), (right_conf, "R", COLOR_CYAN)
    ]):
        bx = bar_x + i * 160
        cv2.rectangle(frame, (bx, bar_y), (bx + 140, bar_y + 16), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, bar_y), (bx + int(140 * conf), bar_y + 16), color, -1)
        cv2.putText(frame, f"{label}:{conf:.0%}", (bx + 2, bar_y + 13),
                    font, 0.45, COLOR_WHITE, 1)

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (10, 28 + i * 28), font, fs, color, th)


def main():
    cfg = load_config()
    g_cfg  = cfg["gesture"]
    v_cfg  = cfg["voice"]
    f_cfg  = cfg["fusion"]
    c_cfg  = cfg["camera"]
    o_cfg  = cfg["ollama"]
    r_cfg  = cfg["robot"]
    m_cfg  = cfg["metrics"]

    ll = LatencyLogger(m_cfg["log_path"])

    # --- Voice backend ---
    llm_mode = False
    if o_cfg["use_llm"]:
        try:
            from runtime.llm_control import LLMVoice
            voice = LLMVoice(
                ollama_url=o_cfg["url"],
                model=o_cfg["model"],
                chimege_token=cfg["chimege_token"],
                **{k: v_cfg[k] for k in
                   ("sample_rate", "blocksize", "vad_threshold",
                    "silence_limit", "min_voice_len", "max_voice_len")},
            )
            llm_mode = True
            logger.info(f"[MODE] LLM+Chimege — model:{o_cfg['model']}")
        except Exception as e:
            logger.warning(f"LLM горим алдаа ({e}), keyword fallback")

    if not llm_mode:
        from runtime.voice_control import Voice
        voice = Voice(
            chimege_token=cfg["chimege_token"],
            **{k: v_cfg[k] for k in
               ("sample_rate", "blocksize", "vad_threshold",
                "silence_limit", "min_voice_len", "max_voice_len")},
        )
        logger.info("[MODE] Keyword+Chimege")

    cap = cv2.VideoCapture(c_cfg["index"])
    if not cap.isOpened():
        logger.error("Camera нээгдсэнгүй.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  c_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS,          c_cfg["fps"])

    gesture = Gesture(
        left_model_path=g_cfg["left_model"],
        right_model_path=g_cfg["right_model"],
        left_labels_path=g_cfg["left_labels"],
        right_labels_path=g_cfg["right_labels"],
        confidence_threshold=g_cfg["confidence_threshold"],
        speed_decay=g_cfg["speed_decay"],
        max_num_hands=g_cfg["max_num_hands"],
        min_detection_confidence=g_cfg["min_detection_confidence"],
        min_tracking_confidence=g_cfg["min_tracking_confidence"],
    )
    state = FusionState(
        turn_scale=f_cfg["turn_scale"],
        max_speed_scale=f_cfg["max_speed_scale"],
        min_speed_scale=f_cfg["min_speed_scale"],
        speed_step=f_cfg["speed_step"],
        precise_turn_speed=f_cfg["precise_turn_speed"],
    )
    udp = UdpSender(host=r_cfg["host"], port=r_cfg["port"])

    voice.start()
    logger.info("HRI эхэллээ. Гарахын тулд 'q' дарна уу.")

    last_voice_text = "None"
    gesture_ms = 0.0
    nav_target: str | None = None   # Одоогийн навигаци зорилт

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            t_g = ll.start("gesture")
            left_cmd, right_cmd, speed, left_conf, right_conf = gesture.run(frame)
            gesture_ms = ll.end("gesture", t_g, command=left_cmd)

            t_v = ll.start("fusion")
            voice_cmd = voice.get()
            if voice_cmd:
                last_voice_text = voice_cmd
                logger.info(f"CMD: {voice_cmd}")

            if voice_cmd and voice_cmd.startswith("nav_behind:"):
                _t = voice_cmd[11:]
                nav_target = f"BEHIND:{_t}"
                udp.send({"navigate_to": _t, "maneuver": "behind"})
                logger.info(f"[NAV] Behind: {_t}")
                voice_cmd = None
            elif voice_cmd and voice_cmd.startswith("nav_circle:"):
                _t = voice_cmd[11:]
                nav_target = f"CIRCLE:{_t}"
                udp.send({"navigate_to": _t, "maneuver": "circle"})
                logger.info(f"[NAV] Circle: {_t}")
                voice_cmd = None
            elif voice_cmd and voice_cmd.startswith("nav:"):
                _t = voice_cmd[4:]   # "nav:car_green" → "car_green"
                nav_target = _t
                udp.send({"navigate_to": _t, "maneuver": "approach"})
                logger.info(f"[NAV] Approach: {_t}")
                voice_cmd = None
            elif voice_cmd == "nav_cancel":
                nav_target = None
                udp.send({"navigate_to": None})
                logger.info("[NAV] Navigation cancelled")
                voice_cmd = None
            elif voice_cmd == "stop" and nav_target is not None:
                # "зогс" while navigating → cancel nav, return to manual mode.
                # Don't set emergency stop so gesture/voice work immediately after.
                udp.send({"navigate_to": None})
                udp.send({"linear_x": 0.0, "angular_z": 0.0})
                logger.info(f"[NAV] Stop — '{nav_target}' цуцлагдлаа → manual горим")
                nav_target = None
                voice_cmd = None  # skip emergency-stop so gestures resume right away

            lin, ang = fuse(state, left_cmd, right_cmd, speed, voice_cmd)
            ll.end("fusion", t_v, command=f"{lin:.2f},{ang:.2f}")

            # Навигаци горимд байхгүй үед л manual команд илгээнэ
            if nav_target is None:
                udp.send({"linear_x": lin, "angular_z": ang})

            draw_hud(frame, left_cmd, right_cmd, left_conf, right_conf,
                     speed, state, lin, ang, last_voice_text,
                     llm_mode, gesture_ms, o_cfg["model"], nav_target)
            cv2.imshow("HRI Multimodal Control", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        ll.close()
        voice.stop()
        gesture.close()
        udp.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Систем хаагдлаа.")


if __name__ == "__main__":
    main()
