#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
WS=${B2_NAV_WS:-"$HOME/b2_nav_ws"}
if [[ ! -f "$WS/install/setup.bash" ]]; then
  echo "Build the migrated workspace first: scripts/workspace.sh build" >&2
  exit 1
fi
source "$WS/install/setup.bash"
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
exec ros2 launch b2_nav_bringup navigation_only.launch.py "$@"
