import time
from typing import Optional, Tuple


class FusionState:
    """
    Voice болон gesture командуудыг нэгтгэн robot-н хөдөлгөөнийг тодорхойлно.

    Priority order:
      1. Temp precise turns (turn_left_90 гэх мэт) — хамгийн өндөр priority
      2. Voice angular override (зүүн/баруун)
      3. Gesture-с angular
      4. Voice linear override (урагш/арагш)
      5. Gesture-с linear
    """

    def __init__(
        self,
        speed_scale: float = 1.0,
        turn_scale: float = 0.35,
        max_speed_scale: float = 2.0,
        min_speed_scale: float = 0.2,
        speed_step: float = 0.1,
    ):
        self.speed_scale = speed_scale
        self.turn_scale = turn_scale
        self.max_speed_scale = max_speed_scale
        self.min_speed_scale = min_speed_scale
        self.speed_step = speed_step

        self.voice_linear: Optional[str] = None    # "forward" | "backward" | None
        self.voice_angular: Optional[str] = None   # "left" | "right" | None

        self.temp_turn_until: float = 0.0
        self.temp_turn_direction: float = 0.0

        # Emergency stop flag — voice "stop" командаар идэвхждэг
        self._emergency_stopped: bool = False

    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped

    def update_voice(self, cmd: Optional[str]) -> None:
        """Voice командыг state-д тусгана."""
        if not cmd:
            return

        now = time.time()

        if cmd == "stop":
            self.voice_linear = None
            self.voice_angular = None
            self.temp_turn_until = 0.0
            self.temp_turn_direction = 0.0
            self._emergency_stopped = True

        elif cmd == "resume":
            # Emergency stop-г цуцалж gesture-г дахин идэвхжүүлнэ
            self._emergency_stopped = False

        elif cmd == "faster":
            self.speed_scale = min(self.max_speed_scale, self.speed_scale + self.speed_step)

        elif cmd == "slower":
            self.speed_scale = max(self.min_speed_scale, self.speed_scale - self.speed_step)

        elif cmd == "forward":
            self._emergency_stopped = False
            self.voice_linear = "forward"

        elif cmd == "backward":
            self._emergency_stopped = False
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


def fuse(
    state: FusionState,
    left: str,
    right: str,
    speed: float,
    voice_cmd: Optional[str],
) -> Tuple[float, float]:
    """
    Gesture болон voice командуудыг нэгтгэн (linear_x, angular_z) буцаана.

    Emergency stop идэвхтэй үед бүх хөдөлгөөн тэг байна.
    """
    state.update_voice(voice_cmd)

    # Emergency stop — аль ч эх сурвалж гарч ирсэн тэгийг буцаана
    if state.is_emergency_stopped:
        return 0.0, 0.0

    now = time.time()
    lin = 0.0
    ang = 0.0

    # --- Linear: gesture (base) ---
    if left == "forward":
        lin = speed
    elif left == "backward":
        lin = -speed

    # --- Angular: gesture (base) ---
    if right == "left":
        ang = speed * state.turn_scale
    elif right == "right":
        ang = -speed * state.turn_scale

    # --- Linear: voice override ---
    if state.voice_linear == "forward":
        lin = speed if speed > 0.05 else 0.3   # гар байхгүй үед default хурд
    elif state.voice_linear == "backward":
        lin = -(speed if speed > 0.05 else 0.3)

    # --- Angular: temp precise turns (highest priority) ---
    if now < state.temp_turn_until:
        ang = state.temp_turn_direction
    elif state.voice_angular == "left":
        ang = speed * state.turn_scale
    elif state.voice_angular == "right":
        ang = -speed * state.turn_scale

    return lin * state.speed_scale, ang * state.speed_scale