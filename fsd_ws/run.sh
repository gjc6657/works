#!/bin/bash
set -e
while [ -n "$CONDA_DEFAULT_ENV" ]; do
  conda deactivate 2>/dev/null || break
done
unset PYTHONPATH
export PATH="/usr/bin:/bin:/opt/ros/humble/bin:${PATH}"
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset CMAKE_PREFIX_PATH
source /opt/ros/humble/setup.bash
cd "$(dirname "$0")"
[ ! -f right_angle_stack/package.xml ] && echo "missing right_angle_stack package" && exit 1
[ ! -f fsd_bringup/models/simple_car/model.sdf ] && echo "missing F1 model.sdf" && exit 1
if [ ! -f install/setup.bash ] || [ "${1:-}" = "--rebuild" ]; then
  colcon build --symlink-install
  [ "${1:-}" = "--rebuild" ] && shift
fi
source install/setup.bash
pkill -f "f1_right_angle.launch.py" 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
pkill -f "ruby.*gz.*sim" 2>/dev/null || true
sleep 2
WORLD="$(ros2 pkg prefix right_angle_track)/share/right_angle_track/worlds/right_angle_harmonic.sdf"
exec ros2 launch fsd_bringup f1_right_angle.launch.py \
  use_rviz:=true \
  use_builtin_perception:=true \
  use_sim_perception:=false \
  gz_args:="-r -v 4 ${WORLD}" \
  "$@"
