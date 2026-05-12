"""
object_navigator.py — Visual servoing + Gazebo pose navigation + UDP command receiver.

UDP протокол (Windows→WSL):
  Хэвийн:    {"linear_x": 0.5, "angular_z": 0.1}
  Навигаци:  {"navigate_to": "car_green", "maneuver": "approach|behind|circle"}
  Цуцлах:    {"navigate_to": null}

Суулгалт (WSL):
    sudo apt install ros-jazzy-vision-msgs
"""

import json
import math
import os
import socket
import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray

# ── Label aliases ─────────────────────────────────────────────────────
LABEL_ALIASES: dict[str, str] = {
    "хүн":      "person",
    "сандал":   "chair",
    "ширээ":    "dining table",
    "лонх":     "bottle",
    "аяга":     "cup",
    "машин":    "car",
    "нохой":    "dog",
    "муур":     "cat",
    "ном":      "book",
    "утас":     "cell phone",
    "уут":      "backpack",
    "ногоон машин":  "car_green",
    "цэнхэр машин":  "car_blue",
    "улаан машин":   "car_red",
    "шар машин":     "car_yellow",
    "цагаан машин":  "car_white",
    "саарал машин":  "car_gray",
    "person":       "person",
    "chair":        "chair",
    "table":        "dining table",
    "bottle":       "bottle",
    "cup":          "cup",
    "car":          "car",
    "dog":          "dog",
    "cat":          "cat",
    "book":         "book",
    "phone":        "cell phone",
    "bag":          "backpack",
    "green car":    "car_green",
    "blue car":     "car_blue",
    "red car":      "car_red",
    "yellow car":   "car_yellow",
    "car_green":    "car_green",
    "car_blue":     "car_blue",
    "car_red":      "car_red",
    "car_yellow":   "car_yellow",
}

# ── Gazebo model name mapping (nav_target → GZ model name) ───────────
# These targets use exact Gazebo pose instead of camera detection
_GZ_MODEL: dict[str, str] = {
    "car_green": "vehicle_green",
    "car_blue":  "vehicle_blue",
}

# ── Pose-based nav constants ──────────────────────────────────────────
_APPROACH_STOP  = 1.5    # m — approach stop distance (safe gap from target)
_BEHIND_OFFSET  = 2.0    # m — distance behind target for "behind" maneuver
_ORBIT_RADIUS   = 2.5    # m — circle orbit radius
_V_APPROACH     = 0.35   # m/s max approach speed
_V_ORBIT        = 0.28   # m/s orbit forward speed
_KP_ANG         = 1.2    # angular proportional gain (approach)
_KP_ORBIT_HEAD  = 1.4    # heading gain during orbit
_KD_ORBIT_DIST  = 0.30   # distance correction gain during orbit
_CIRCLE_DURATION = 20.0  # seconds — one full orbit (2π/ω + buffer)

# ── Camera-based nav constants (fallback for non-Gazebo targets) ──────
_BEHIND_TRIGGER_AREA = 0.07
_ORBIT_TRIGGER_AREA  = 0.025
_ARC_DURATION        = 5.0
_ARC_LINEAR          = 0.28
_ARC_ANGULAR         = -0.62
_ORBIT_LINEAR        = 0.30
_ORBIT_ANGULAR_BASE  = 0.35
_KP_CAM_ORBIT        = 0.35
_ORBIT_SETPOINT      = 0.72

# ── Phase names ───────────────────────────────────────────────────────
_PHASE_SERVO = "servo"
_PHASE_ARC   = "arc"
_PHASE_ORBIT = "orbit"
_PHASE_DONE  = "done"


class ObjectNavigatorNode(Node):

    KP_ANGULAR  = 1.2
    KP_LINEAR   = 0.4
    MAX_LINEAR  = 0.4
    MAX_ANGULAR = 0.8
    STOP_AREA   = 0.12
    LOST_TIMEOUT = 2.0

    def __init__(self):
        super().__init__("object_navigator")

        self.declare_parameter("udp_host",    "0.0.0.0")
        self.declare_parameter("udp_port",    5005)
        self.declare_parameter("timeout_sec", 1.0)

        host         = self.get_parameter("udp_host").value
        port         = self.get_parameter("udp_port").value
        self.timeout = self.get_parameter("timeout_sec").value

        self.pub = self.create_publisher(Twist, "/model/vehicle_blue/cmd_vel", 10)

        self.sub_det = self.create_subscription(
            Detection2DArray, "/yolo/detections", self._det_cb, 10)

        # Gazebo odometry subscribers — one per model
        self.sub_odom_blue = self.create_subscription(
            Odometry, "/model/vehicle_blue/odometry", self._odom_blue_cb, 10)
        self.sub_odom_green = self.create_subscription(
            Odometry, "/model/vehicle_green/odometry", self._odom_green_cb, 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

        # Navigation state
        self.nav_target:   str | None = None
        self.nav_maneuver: str        = "approach"
        self._phase:       str        = _PHASE_SERVO
        self._phase_start: float      = 0.0

        # Camera detection state (fallback)
        self.last_detection: dict | None = None
        self.last_det_time  = 0.0

        # SDF initial world positions — odometry is relative to each vehicle's
        # own start, so we need these offsets to convert to absolute world coords.
        self._sdf_poses   = self._parse_sdf_positions()
        self._own_pose:   tuple | None = None     # (x, y, yaw) world frame
        self._model_poses: dict        = dict(self._sdf_poses)  # pre-fill from SDF
        self._pose_logged = False

        self.last_udp_time = time.time()
        self.manual_lin    = 0.0
        self.manual_ang    = 0.0

        self.timer = self.create_timer(0.05, self._loop)
        self.get_logger().info(f"ObjectNavigatorNode эхэллээ  UDP:{host}:{port}")

    # ------------------------------------------------------------------
    # SDF world-position parser
    # ------------------------------------------------------------------

    def _parse_sdf_positions(self) -> dict:
        """Read diff_drive.sdf and return {model_name: (x, y, yaw)} world poses."""
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "diff_drive.sdf"),
            os.path.expanduser("~/diff_drive.sdf"),
            "/opt/ros/jazzy/opt/gz_sim_vendor/share/gz/gz-sim8/worlds/diff_drive.sdf",
            "/usr/share/gz/gz-sim8/worlds/diff_drive.sdf",
        ]
        sdf_path = next((c for c in candidates if os.path.isfile(c)), None)
        _defaults = {"vehicle_blue": (0.0, 0.0, 0.0), "vehicle_green": (0.0, 2.0, 0.0)}

        if sdf_path is None:
            self.get_logger().warn(
                "[SDF] diff_drive.sdf олдсонгүй — default: blue(0,0) green(0,2)")
            return _defaults

        poses: dict = {}
        try:
            root = ET.parse(sdf_path).getroot()
            for model in root.iter("model"):
                name = model.get("name", "")
                el   = model.find("pose")
                if el is not None and el.text:
                    v = list(map(float, el.text.strip().split()))
                    if len(v) >= 6:
                        poses[name] = (v[0], v[1], v[5])   # x, y, yaw
            self.get_logger().info(f"[SDF] {sdf_path} уншлаа → {poses}")
        except Exception as e:
            self.get_logger().warn(f"[SDF] parse алдаа: {e} → default байрлал")
            poses = {}

        poses.setdefault("vehicle_blue",  _defaults["vehicle_blue"])
        poses.setdefault("vehicle_green", _defaults["vehicle_green"])
        return poses

    # ------------------------------------------------------------------
    # Odometry callbacks — convert odom frame → world frame using SDF offsets
    # ------------------------------------------------------------------
    # Gazebo DiffDrive reports odometry relative to the vehicle's own starting
    # position, NOT in absolute world coordinates.  We apply the SDF-derived
    # initial pose to get true world positions so both vehicles share one frame.

    @staticmethod
    def _odom_to_pose(msg: Odometry) -> tuple:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (p.x, p.y, yaw)

    def _odom_blue_cb(self, msg: Odometry) -> None:
        ox, oy, oyaw = self._odom_to_pose(msg)
        bsx, bsy, bsyaw = self._sdf_poses["vehicle_blue"]
        c, s = math.cos(bsyaw), math.sin(bsyaw)
        self._own_pose = (bsx + c * ox - s * oy,
                          bsy + s * ox + c * oy,
                          bsyaw + oyaw)
        self._model_poses["vehicle_blue"] = self._own_pose

    def _odom_green_cb(self, msg: Odometry) -> None:
        # vehicle_green is stationary — its world position equals its SDF pose.
        self._model_poses["vehicle_green"] = self._sdf_poses["vehicle_green"]
        if not self._pose_logged and self._own_pose is not None:
            ox, oy, _ = self._own_pose
            tx, ty, _ = self._sdf_poses["vehicle_green"]
            self.get_logger().info(
                f"[POSE] vehicle_green world: ({tx:.2f}, {ty:.2f})  "
                f"vehicle_blue world: ({ox:.2f}, {oy:.2f})  "
                f"dist: {math.hypot(tx-ox, ty-oy):.2f}m")
            self._pose_logged = True

    # ------------------------------------------------------------------
    # YOLO / HSV camera detection callback (fallback)
    # ------------------------------------------------------------------

    def _label_matches(self, det_label: str) -> bool:
        if det_label == self.nav_target:
            return True
        if "_" not in self.nav_target:
            return det_label.split("_")[0] == self.nav_target
        return False

    def _det_cb(self, msg: Detection2DArray) -> None:
        if self.nav_target is None or self._phase not in (_PHASE_SERVO, _PHASE_ORBIT):
            return
        # Skip camera detection if we have Gazebo poses for this target
        if self.nav_target in _GZ_MODEL and self._own_pose is not None:
            return

        best = None
        best_area = 0.0
        for det in msg.detections:
            for hyp in det.results:
                if not self._label_matches(hyp.hypothesis.class_id):
                    continue
                bw   = det.bbox.size_x
                bh   = det.bbox.size_y
                area = bw * bh
                if area > best_area:
                    best_area = area
                    best = {"cx": det.bbox.center.position.x,
                            "cy": det.bbox.center.position.y,
                            "bw": bw, "bh": bh}
        if best:
            self.last_detection = best
            self.last_det_time  = time.time()

    # ------------------------------------------------------------------
    # UDP receive
    # ------------------------------------------------------------------

    def _recv_udp(self) -> None:
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
                self.last_udp_time = time.time()
                d = json.loads(data.decode("utf-8"))

                if "navigate_to" in d:
                    raw_target = d["navigate_to"]
                    if raw_target is None:
                        if self.nav_target:
                            self.get_logger().info("Навигаци цуцлагдлаа → manual горим")
                        self.nav_target     = None
                        self.last_detection = None
                        self._phase         = _PHASE_SERVO
                        self._pose_logged   = False
                    else:
                        key      = str(raw_target).lower()
                        label    = LABEL_ALIASES.get(key, key)
                        maneuver = str(d.get("maneuver", "approach")).lower()
                        if maneuver not in ("approach", "behind", "circle"):
                            maneuver = "approach"
                        self.nav_target     = label
                        self.nav_maneuver   = maneuver
                        self._phase         = _PHASE_SERVO
                        self._phase_start   = 0.0
                        self.last_detection = None
                        self._pose_logged   = False
                        self.get_logger().info(
                            f"Навигаци: '{raw_target}'→'{label}'  maneuver={maneuver}")
                    return

                self.manual_lin = float(d.get("linear_x",  0.0))
                self.manual_ang = float(d.get("angular_z", 0.0))

            except BlockingIOError:
                break
            except json.JSONDecodeError:
                self.get_logger().error("JSON алдаа")
            except Exception as e:
                self.get_logger().error(f"UDP алдаа: {e}")
                break

    # ------------------------------------------------------------------
    # Pose-based navigation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _norm(angle: float) -> float:
        while angle >  math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def _goto_step(self, ox, oy, oyaw, tx, ty, stop_dist) -> tuple[float, float]:
        """P-controller approach toward (tx, ty)."""
        dx, dy = tx - ox, ty - oy
        dist   = math.hypot(dx, dy)
        if dist <= stop_dist:
            return None, None   # signal: arrived

        target_angle = math.atan2(dy, dx)
        ang_err = self._norm(target_angle - oyaw)
        angular  = float(max(-self.MAX_ANGULAR,
                             min(self.MAX_ANGULAR, _KP_ANG * ang_err)))
        align    = max(0.0, 1.0 - abs(ang_err) / math.pi * 2.0)
        linear   = float(min(_V_APPROACH, _V_APPROACH * align * min(1.0, dist / 1.5)))
        return linear, angular

    def _orbit_step(self, ox, oy, oyaw, tx, ty) -> tuple[float, float]:
        """P-controller orbit around (tx, ty) at _ORBIT_RADIUS, CCW."""
        dx, dy = tx - ox, ty - oy
        dist   = math.hypot(dx, dy)

        radial_angle  = math.atan2(dy, dx)          # own → target
        tangent_angle = radial_angle - math.pi / 2  # CCW tangent

        head_err = self._norm(tangent_angle - oyaw)
        dist_err = dist - _ORBIT_RADIUS

        angular = float(max(-self.MAX_ANGULAR,
                            min(self.MAX_ANGULAR, _KP_ORBIT_HEAD * head_err)))
        linear  = float(max(0.05,
                            min(self.MAX_LINEAR, _V_ORBIT + _KD_ORBIT_DIST * dist_err)))
        return linear, angular

    # ------------------------------------------------------------------
    # Navigation — pose-based (Gazebo models)
    # ------------------------------------------------------------------

    def _navigate_pose(self, model_name: str) -> tuple[float, float]:
        target = self._model_poses.get(model_name)
        own    = self._own_pose
        if target is None or own is None:
            return 0.0, 0.2   # waiting for pose data

        ox, oy, oyaw = own
        tx, ty, tyaw = target

        # DONE
        if self._phase == _PHASE_DONE:
            self.get_logger().info(f"[POSE] '{self.nav_target}' манёвр дууслаа → зогсоно")
            self.nav_target = None
            return 0.0, 0.0

        # ORBIT phase (circle maneuver)
        if self._phase == _PHASE_ORBIT:
            if (time.time() - self._phase_start) >= _CIRCLE_DURATION:
                self._phase = _PHASE_DONE
                return 0.0, 0.0
            return self._orbit_step(ox, oy, oyaw, tx, ty)

        # SERVO phase — approach / behind / circle
        if self.nav_maneuver == "circle":
            dist = math.hypot(tx - ox, ty - oy)
            if dist <= _ORBIT_RADIUS + 0.6:
                self._phase       = _PHASE_ORBIT
                self._phase_start = time.time()
                self.get_logger().info(
                    f"[POSE] CIRCLE: orbit phase эхэллээ dist={dist:.2f}")
                return self._orbit_step(ox, oy, oyaw, tx, ty)
            lin, ang = self._goto_step(ox, oy, oyaw, tx, ty, _ORBIT_RADIUS)
            return (0.0, 0.0) if lin is None else (lin, ang)

        elif self.nav_maneuver == "behind":
            # Target point: directly behind vehicle_green (opposite its heading)
            bx = tx - _BEHIND_OFFSET * math.cos(tyaw)
            by = ty - _BEHIND_OFFSET * math.sin(tyaw)
            lin, ang = self._goto_step(ox, oy, oyaw, bx, by, 0.35)
            if lin is None:
                self.get_logger().info("[POSE] BEHIND: arrived → зогсоно")
                self.nav_target = None
                return 0.0, 0.0
            return lin, ang

        else:   # approach
            lin, ang = self._goto_step(ox, oy, oyaw, tx, ty, _APPROACH_STOP)
            if lin is None:
                self.get_logger().info(f"[POSE] APPROACH: '{self.nav_target}' → зогсоно")
                self.nav_target = None
                return 0.0, 0.0
            return lin, ang

    # ------------------------------------------------------------------
    # Navigation — camera-based (fallback)
    # ------------------------------------------------------------------

    def _navigate_camera(self) -> tuple[float, float]:
        if self._phase == _PHASE_DONE:
            self.nav_target = None
            self._phase     = _PHASE_SERVO
            return 0.0, 0.0

        if self._phase == _PHASE_ARC:
            if (time.time() - self._phase_start) >= _ARC_DURATION:
                self._phase = _PHASE_DONE
                return 0.0, 0.0
            return _ARC_LINEAR, _ARC_ANGULAR

        if self._phase == _PHASE_ORBIT:
            if (time.time() - self._phase_start) >= _CIRCLE_DURATION:
                self._phase = _PHASE_DONE
                return 0.0, 0.0
            det = self.last_detection
            if det is not None and (time.time() - self.last_det_time) < self.LOST_TIMEOUT * 2:
                cx_norm   = det["cx"] / 640.0
                orbit_ang = _ORBIT_ANGULAR_BASE + _KP_CAM_ORBIT * (cx_norm - _ORBIT_SETPOINT)
                orbit_ang = float(max(0.10, min(self.MAX_ANGULAR, orbit_ang)))
            else:
                orbit_ang = _ORBIT_ANGULAR_BASE
            return _ORBIT_LINEAR, orbit_ang

        # SERVO
        det = self.last_detection
        if det is None:
            return 0.0, 0.3

        if (time.time() - self.last_det_time) > self.LOST_TIMEOUT:
            self.last_detection = None
            return 0.0, 0.3

        cx, bw, bh = det["cx"], det["bw"], det["bh"]
        frame_w   = 640.0
        cx_norm   = cx / frame_w
        err_ang   = cx_norm - 0.5
        area_frac = (bw * bh) / (frame_w * frame_w * 0.5625)

        angular_z = float(
            max(-self.MAX_ANGULAR, min(self.MAX_ANGULAR, -self.KP_ANGULAR * err_ang))
        )

        if self.nav_maneuver == "behind" and area_frac >= _BEHIND_TRIGGER_AREA:
            self._phase       = _PHASE_ARC
            self._phase_start = time.time()
            return _ARC_LINEAR, _ARC_ANGULAR

        if self.nav_maneuver == "circle" and area_frac >= _ORBIT_TRIGGER_AREA:
            self._phase       = _PHASE_ORBIT
            self._phase_start = time.time()
            return _ORBIT_LINEAR, _ORBIT_ANGULAR_BASE

        if self.nav_maneuver == "approach" and area_frac >= self.STOP_AREA:
            self.nav_target = None
            return 0.0, 0.0

        align_factor = max(0.0, 1.0 - abs(err_ang) * 3.0)
        linear_x = float(max(0.0, min(self.MAX_LINEAR, self.KP_LINEAR * align_factor)))
        return linear_x, angular_z

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        self._recv_udp()

        if self.nav_target is not None:
            gz_model = _GZ_MODEL.get(self.nav_target)
            if gz_model is not None and self._own_pose is not None:
                lin, ang = self._navigate_pose(gz_model)
            else:
                lin, ang = self._navigate_camera()
        else:
            lin = self.manual_lin
            ang = self.manual_ang
            if (time.time() - self.last_udp_time) > self.timeout:
                lin = ang = 0.0
                self.manual_lin = self.manual_ang = 0.0

        msg = Twist()
        msg.linear.x  = float(lin)
        msg.angular.z = float(ang)
        self.pub.publish(msg)

        if lin != 0.0 or ang != 0.0:
            mode = f"NAV→{self.nav_target}" if self.nav_target else "MANUAL"
            self.get_logger().debug(f"[{mode}] lin={lin:.2f} ang={ang:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Хаагдаж байна...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
