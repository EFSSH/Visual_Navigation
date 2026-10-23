#!/usr/bin/env bash
# Run inside Ubuntu. The mounted Mac checkout is the source of truth.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WS=${B2_NAV_WS:-"$HOME/b2_nav_ws"}
MODE=${1:-build}
mkdir -p "$WS/src"
if [[ ! -f "$WS/.navigation-migration-workspace" ]]; then
  if [[ -n "$(ls -A "$WS/src")" ]]; then
    echo "Refusing to overwrite an unmanaged workspace: $WS" >&2
    exit 1
  fi
  printf '%s\n' "$SOURCE" > "$WS/.navigation-migration-workspace"
fi
rsync -rlt --delete --exclude='._*' --exclude='__pycache__/' --exclude='.pytest_cache/' "$SOURCE/src/" "$WS/src/"
cd "$WS"
case "$MODE" in
  sync) ;;
  deps) rosdep install --from-paths src --ignore-src --rosdistro jazzy -y ;;
  build) colcon build --symlink-install --event-handlers console_direct+ ;;
  test)
    source install/setup.bash
    python3 -m pytest -q src/b2_nav_twin/test/test_local_world_model.py src/b2_nav_safety/test/test_lidar_safety.py src/b2_nav_planning/test/test_planner_geometry.py
    ;;
  *) echo "Usage: $0 {sync|deps|build|test}" >&2; exit 2 ;;
esac
