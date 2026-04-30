import cv2
from runtime.gesture_runtime import Gesture
from runtime.voice_control import Voice
from runtime.fusion import FusionState, fuse
from runtime.udp_sender import UdpSender


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera neegdsenгүй.")
        return

    gesture = Gesture()
    voice = Voice()
    state = FusionState()
    udp = UdpSender()

    voice.start()

    last_voice_text = "None"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera-s frame авч чадсангүй.")
                break

            frame = cv2.flip(frame, 1)

            left_cmd, right_cmd, speed = gesture.run(frame)
            voice_cmd = voice.get()

            if voice_cmd:
                last_voice_text = voice_cmd
                print("VOICE CMD:", voice_cmd)

            lin, ang = fuse(state, left_cmd, right_cmd, speed, voice_cmd)

            udp.send({
                "linear_x": lin,
                "angular_z": ang
            })

            print(
                f"GESTURE -> L:{left_cmd} R:{right_cmd} SPEED:{speed:.2f} | "
                f"VOICE_LINEAR:{state.voice_linear} | "
                f"VOICE_ANGULAR:{state.voice_angular} | "
                f"OUT -> lin:{lin:.2f}, ang:{ang:.2f}"
            )

            cv2.putText(frame, f"L:{left_cmd} R:{right_cmd}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Speed:{speed:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"Last Voice:{last_voice_text}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.putText(frame, f"VoiceLin:{state.voice_linear}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

            cv2.putText(frame, f"VoiceAng:{state.voice_angular}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)

            cv2.putText(frame, f"lin:{lin:.2f} ang:{ang:.2f}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            cv2.imshow("HRI", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        if hasattr(voice, "stop"):
            voice.stop()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()