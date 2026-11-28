#!/usr/bin/env bash
set -e
ADA_CAMERA_CHECKOUT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-87}"
export PYTHONPATH="$ADA_CAMERA_CHECKOUT/src:${PYTHONPATH:-}"
exec ros2 launch "$ADA_CAMERA_CHECKOUT/script/calibrated_cameras.launch.py" "$@"
