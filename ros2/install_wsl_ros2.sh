#!/bin/bash
# ============================================================
# WSL Ubuntu 24.04 - ROS2 Jazzy + YOLO setup script
# Run inside WSL: bash install_wsl_ros2.sh
# ============================================================

set -e  # error on fail

echo "============================================"
echo " ROS2 Jazzy + YOLO suulgalt"
echo "============================================"

# ---- 1. System update ----
echo "[1/6] System update..."
sudo apt update && sudo apt upgrade -y

# ---- 2. ROS2 Jazzy suulgalt ----
echo "[2/6] ROS2 Jazzy suulgaj baina..."

# Locale
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ROS2 apt repo
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-jazzy-desktop

# ---- 3. ROS2 tools ----
echo "[3/6] ROS2 tools suulgaj baina..."
sudo apt install -y \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-pip \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-sim

# rosdep init
sudo rosdep init 2>/dev/null || true
rosdep update

# ---- 4. Python packages ----
echo "[4/6] Python packages suulgaj baina..."
pip3 install ultralytics opencv-python-headless

# ---- 5. ~/.bashrc setup ----
echo "[5/6] bashrc tohiruulj baina..."
BASHRC="$HOME/.bashrc"

add_if_missing() {
    grep -qF "$1" "$BASHRC" || echo "$1" >> "$BASHRC"
}

add_if_missing "source /opt/ros/jazzy/setup.bash"
add_if_missing "export ROS_DOMAIN_ID=0"
# Windows-WSL2 IP togloh
add_if_missing 'export ROS_LOCALHOST_ONLY=0'

source "$BASHRC" 2>/dev/null || true

# ---- 6. Workspace setup ----
echo "[6/6] ROS2 workspace tohiruulj baina..."
WS="$HOME/hri_ws"
mkdir -p "$WS/src"

# Symlink Windows proyektoos
WIN_PATH="/mnt/c/Users/tsogo/OneDrive/Desktop/Diploma/hri_multimodal/ros2"
if [ -d "$WIN_PATH" ]; then
    echo "Windows ros2 folder oldlaa: $WIN_PATH"
    ln -sfn "$WIN_PATH" "$WS/src/hri_ros2" 2>/dev/null || true
fi

echo ""
echo "============================================"
echo " Suulgalt duurslaa!"
echo "============================================"
echo ""
echo "Dagaj hiih:"
echo "  source ~/.bashrc"
echo "  cd ~/hri_ws"
echo "  colcon build"
echo ""
echo "Terminald test hiih:"
echo "  ros2 topic list"
echo ""
