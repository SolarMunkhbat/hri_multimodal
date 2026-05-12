#!/bin/bash
# HRI Simulator — WSL-д нэг командаар бүгдийг эхлүүлнэ
# Ажиллуулах: bash start_sim.sh
# Зогсоох:    Ctrl+C

source /opt/ros/jazzy/setup.bash

ROS2_DIR="$(cd "$(dirname "$0")" && pwd)"
CAMERA_TOPIC="/world/diff_drive/model/vehicle_blue/link/chassis/sensor/camera/image"
GZ_WORLD="diff_drive.sdf"

# ── Бүх child process-г цэвэр зогсоох ──────────────────────────────
PIDS=()
cleanup() {
    echo -e "\n[SIM] Зогсоож байна..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    pkill -f "gz sim" 2>/dev/null
    echo "[SIM] Дууслаа."
    exit 0
}
trap cleanup INT TERM

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Gazebo ───────────────────────────────────────────────────────
log "Gazebo эхлүүлж байна ($GZ_WORLD)..."
gz sim -v4 -r "$GZ_WORLD" &
PIDS+=($!)
sleep 6   # Gazebo дуусч load хийх хүртэл хүлээ

# ── 2. ROS ↔ GZ bridge ─────────────────────────────────────────────
log "ROS↔GZ bridge эхлүүлж байна..."
ros2 run ros_gz_bridge parameter_bridge \
  "/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist" \
  "${CAMERA_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image" \
  "/model/vehicle_blue/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry" \
  "/model/vehicle_green/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry" \
  2>&1 | sed 's/^/[bridge] /' &
PIDS+=($!)
sleep 2

# ── 3. YOLO detector ────────────────────────────────────────────────
log "YOLO detector эхлүүлж байна..."
python3 "$ROS2_DIR/yolo_detector.py" \
  --ros-args -p camera_topic:="$CAMERA_TOPIC" \
  2>&1 | sed 's/^/[yolo]   /' &
PIDS+=($!)
sleep 3   # YOLOv8 загвар татаж дуусах хүртэл хүлээ

# ── 4. Object navigator (UDP + YOLO nav) ────────────────────────────
log "Object navigator эхлүүлж байна..."
python3 "$ROS2_DIR/object_navigator.py" \
  2>&1 | sed 's/^/[nav]    /' &
PIDS+=($!)

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   HRI бүх node эхэллээ ✓             ║"
echo "║   Windows-с: py -m runtime.multimodal_control"
echo "║   Зогсоох: Ctrl+C                    ║"
echo "╚══════════════════════════════════════╝"
echo ""

wait
