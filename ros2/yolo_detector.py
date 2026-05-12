"""
yolo_detector.py — YOLOv8 object detection ROS2 Jazzy node.

Gazebo camera-с зураг авч YOLO-р объект илрүүлж /yolo/detections topic-т нийтэлнэ.

Суулгалт (WSL):
    pip install ultralytics
    sudo apt install ros-jazzy-cv-bridge ros-jazzy-vision-msgs

Ажиллуулах:
    ros2 run hri_multimodal yolo_detector

Camera topic тохируулах:
    ros2 run hri_multimodal yolo_detector --ros-args -p camera_topic:=/camera/image_raw
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header

import cv2
import numpy as np

# Classes whose detections get a color suffix appended to their label
_COLORIZE_CLASSES = {"car", "truck", "bus", "motorcycle"}


def _classify_color(bgr: np.ndarray, x1: float, y1: float,
                    x2: float, y2: float) -> tuple[str, float, float, float]:
    """Sample center 40% of bbox, return (color_name, h, s, v) for debugging."""
    h_img, w_img = bgr.shape[:2]
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    bw = max(4, int((x2 - x1) * 0.40))
    bh = max(4, int((y2 - y1) * 0.40))
    crop = bgr[max(0, cy - bh // 2):min(h_img, cy + bh // 2),
               max(0, cx - bw // 2):min(w_img, cx + bw // 2)]
    if crop.size < 12:
        return "unknown", 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = float(np.mean(hsv[:, :, 0]))
    s = float(np.mean(hsv[:, :, 1]))
    v = float(np.mean(hsv[:, :, 2]))
    # OpenCV H is 0–179 (half of standard 0–360)
    # green (120°) → H≈60,  blue (240°) → H≈120,  cyan (180°) → H≈90
    if s < 40:
        color = "white" if v > 180 else ("black" if v < 50 else "gray")
    elif h < 10 or h > 170: color = "red"
    elif h < 30:             color = "orange"
    elif h < 38:             color = "yellow"
    elif h < 95:             color = "green"   # wider: includes cyan-green (H 40–95)
    elif h < 135:            color = "blue"
    else:                    color = "purple"
    return color, h, s, v

try:
    from cv_bridge import CvBridge
    _CV_BRIDGE = True
except ImportError:
    _CV_BRIDGE = False

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# COCO class нэр → Монгол нэр харьцуулалт
COCO_TO_MN = {
    "person":        "хүн",
    "chair":         "сандал",
    "dining table":  "ширээ",
    "bottle":        "лонх",
    "cup":           "аяга",
    "car":           "машин",
    "dog":           "нохой",
    "cat":           "муур",
    "book":          "ном",
    "tv":            "телевизор",
    "laptop":        "зөөврийн компьютер",
    "cell phone":    "утас",
    "backpack":      "уут",
}


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        # Параметрүүд
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("model_name",   "yolov8n.pt")
        self.declare_parameter("confidence",   0.45)
        self.declare_parameter("publish_image", True)

        camera_topic  = self.get_parameter("camera_topic").value
        model_name    = self.get_parameter("model_name").value
        self.conf_thr = self.get_parameter("confidence").value
        pub_img       = self.get_parameter("publish_image").value

        if not _CV_BRIDGE:
            self.get_logger().error("cv_bridge олдсонгүй: sudo apt install ros-jazzy-cv-bridge")
            raise RuntimeError("cv_bridge missing")

        if not _YOLO_AVAILABLE:
            self.get_logger().error("ultralytics олдсонгүй: pip install ultralytics")
            raise RuntimeError("ultralytics missing")

        self.bridge = CvBridge()
        self.get_logger().info(f"YOLO загвар ачааллаж байна: {model_name}")
        self.model = YOLO(model_name)
        self.get_logger().info("YOLO бэлэн.")

        # Subscriber
        self.sub_img = self.create_subscription(
            Image, camera_topic, self._image_cb, 5)

        # Publisher — detections
        self.pub_det = self.create_publisher(Detection2DArray, "/yolo/detections", 10)

        # Publisher — annotated image (хэрэв идэвхтэй бол)
        self.pub_img = None
        if pub_img:
            self.pub_img = self.create_publisher(Image, "/yolo/image_annotated", 5)

        self.get_logger().info(
            f"YoloDetectorNode эхэллээ  camera:{camera_topic}  conf:{self.conf_thr}")

    # HSV ranges for Gazebo-rendered vehicle colors (OpenCV H: 0–179)
    # Gazebo vehicle_green renders as pale/light green → low saturation possible
    _HSV_COLORS = {
        "car_green": (np.array([35,  30, 60]), np.array([90,  255, 255])),
        "car_blue":  (np.array([95,  30, 60]), np.array([135, 255, 255])),
        "car_red":   (np.array([0,   30, 60]), np.array([10,  255, 255])),
        "car_yellow":(np.array([20,  30, 60]), np.array([35,  255, 255])),
    }
    _HSV_MIN_AREA = 200   # minimum blob pixel area to publish

    def _hsv_detections(self, frame: np.ndarray, header) -> list:
        """HSV color segmentation — finds Gazebo-rendered colored vehicles reliably."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        detections = []

        for label, (lo, hi) in self._HSV_COLORS.items():
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < self._HSV_MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(largest)
            self.get_logger().info(
                f"[hsv] {label}  area={area:.0f}  bbox=({x},{y},{w},{h})")

            det = Detection2D()
            det.header = header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score    = 0.90
            det.results.append(hyp)
            det.bbox.center.position.x = float(x + w / 2)
            det.bbox.center.position.y = float(y + h / 2)
            det.bbox.size_x = float(w)
            det.bbox.size_y = float(h)
            detections.append(det)

        return detections

    def _image_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge алдаа: {e}")
            return

        results = self.model(frame, conf=self.conf_thr, verbose=False)[0]

        det_array = Detection2DArray()
        det_array.header = msg.header

        for box in results.boxes:
            cls_id  = int(box.cls[0])
            label   = self.model.names[cls_id]
            conf    = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = x2 - x1
            h  = y2 - y1

            if label in _COLORIZE_CLASSES:
                color, h_val, s_val, v_val = _classify_color(frame, x1, y1, x2, y2)
                self.get_logger().info(
                    f"[yolo-color] {self.model.names[cls_id]} → {color}  "
                    f"H={h_val:.0f} S={s_val:.0f} V={v_val:.0f}")
                if color not in ("unknown",):
                    label = f"{label}_{color}"

            det = Detection2D()
            det.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score    = conf
            det.results.append(hyp)
            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.size_x = w
            det.bbox.size_y = h
            det_array.detections.append(det)

        # HSV fallback — Gazebo colored vehicles that YOLO misses
        for hsv_det in self._hsv_detections(frame, msg.header):
            det_array.detections.append(hsv_det)

        self.pub_det.publish(det_array)

        if self.pub_img and self.pub_img.get_subscription_count() > 0:
            annotated = results.plot()
            ann_msg   = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            ann_msg.header = msg.header
            self.pub_img.publish(ann_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
