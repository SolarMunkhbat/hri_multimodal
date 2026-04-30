import time


class FusionState:
    def __init__(self):
        self.speed_scale = 1.0
        self.turn_scale = 0.35

        self.voice_linear = None    # "forward" | "backward" | None
        self.voice_angular = None   # "left" | "right" | None

        self.temp_turn_until = 0.0
        self.temp_turn_direction = 0.0

    def update_voice(self, cmd):
        if not cmd:
            return

        now = time.time()

        if cmd == "stop":
            # voice override-uud clear hiine
            # daraa ni gesture baival gesture-r ajillana
            self.voice_linear = None
            self.voice_angular = None
            self.temp_turn_until = 0.0
            self.temp_turn_direction = 0.0

        elif cmd == "resume":
            # odoogiin architecture deer tusгай emergency lock baihgui
            # tegeheer no-op bolgoj uldeene
            pass

        elif cmd == "faster":
            self.speed_scale = min(2.0, self.speed_scale + 0.1)

        elif cmd == "slower":
            self.speed_scale = max(0.2, self.speed_scale - 0.1)

        elif cmd == "forward":
            self.voice_linear = "forward"

        elif cmd == "backward":
            self.voice_linear = "backward"

        elif cmd == "move_stop":
            self.voice_linear = None

        elif cmd == "left":
            self.voice_angular = "left"
            self.temp_turn_until = 0.0

        elif cmd == "right":
            self.voice_angular = "right"
            self.temp_turn_until = 0.0

        elif cmd == "turn_stop":
            self.voice_angular = None
            self.temp_turn_until = 0.0
            self.temp_turn_direction = 0.0

        elif cmd == "turn_left_90":
            self.temp_turn_direction = self.turn_scale
            self.temp_turn_until = now + 1.0

        elif cmd == "turn_right_90":
            self.temp_turn_direction = -self.turn_scale
            self.temp_turn_until = now + 1.0

        elif cmd == "turn_left_45":
            self.temp_turn_direction = self.turn_scale
            self.temp_turn_until = now + 0.5

        elif cmd == "turn_right_45":
            self.temp_turn_direction = -self.turn_scale
            self.temp_turn_until = now + 0.5


def fuse(state, left, right, speed, voice_cmd):
    state.update_voice(voice_cmd)

    now = time.time()

    lin = 0.0
    ang = 0.0

    # 1. base motion from gesture
    if left == "forward":
        lin = speed
    elif left == "backward":
        lin = -speed

    if right == "left":
        ang = speed * state.turn_scale
    elif right == "right":
        ang = -speed * state.turn_scale

    # 2. voice linear override
    if state.voice_linear == "forward":
        lin = speed
    elif state.voice_linear == "backward":
        lin = -speed

    # 3. temporary precise turns have highest angular priority
    if now < state.temp_turn_until:
        ang = state.temp_turn_direction

    # 4. persistent voice angular override
    elif state.voice_angular == "left":
        ang = speed * state.turn_scale
    elif state.voice_angular == "right":
        ang = -speed * state.turn_scale

    return lin * state.speed_scale, ang