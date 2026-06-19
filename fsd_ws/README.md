# Gazebo 仿真任务：直角弯

本项目实现 Formula Student Driverless 直角弯锥桶赛道的完整 Gazebo Harmonic 仿真：F1 风格赛车、GPS/IMU/轮速融合定位、锥桶感知与建图、中心线规划、Pure Pursuit 路径跟踪。开发环境为 Ubuntu 双系统 + ROS 2 Humble，Gazebo 物理模型与 RViz 显示模型统一，最终在实体机联调通过。

## 开发中遇到的困难

### 环境与工程

工程最初放在中文路径下，CMake 和 Gazebo 资源加载时不时报错；改到 `~/fsd_ws` 后稳定很多。另一个坑是 conda 和 ROS 混用：`PYTHONPATH` 被污染后节点能编译但 launch 起不来，最后在 `run.sh` 里强制 `conda deactivate` 并 `unset PYTHONPATH`。

Gazebo 和 bridge 进程若没杀干净，下次 launch 会连到旧仿真，表现为「改了代码但现象不变」。现在启动前会 `pkill` 相关进程，必要时直接重启。

### 仿真与可视化

launch 若带 `-s` 参数，Gazebo 只跑 server、不弹 3D 窗口，RViz 里仍有数据，容易误以为仿真坏了。双系统上改成 `-r -v 4` 后 Gazebo GUI 和 RViz 一起开。

曾出现 **RViz 里车在规划线上、Gazebo 里车却跑偏** 的情况。原因是 RViz 跟的是 `/localization/odom`（融合后的估计位姿），Gazebo 才是 DiffDrive 的真实运动；当时 Gazebo 用一套 SDF、RViz 用另一套 URDF，轮速话题和 DiffDrive 符号也对不齐。后来统一为：

- Gazebo：`fsd_bringup/models/simple_car/model.sdf`
- RViz：`fsd_bringup/urdf/simple_car.urdf.xacro`
- 传感器与里程计：全部 `/sensors/`*

F1 模型四轮关节轴曾不一致（前轮 `0 0 1`、后轮 `0 1 0`），弯里前轮几乎不滚动，Gazebo 侧滑严重而里程计仍显示「在前进」。四轮统一为 `0 1 0` 并补上轮胎摩擦后，物理与 `/sensors/wheel_odom` 才基本一致。

`ros_gz_sim create` 生成车辆有时失败、Entity 树里看不到车；改为 launch 里 `-file` 指定 SDF 生成，或在 world 里 include，才稳定看到 `racecar`。

### 定位

`/localization/pose` 曾跳到很远（x、y 到几万），车控完全失效。排查后是 NavSat 输出的经纬度与 world 里 `spherical_coordinates` 不完全对齐，按固定原点直接换算会把位置拉飞。现在的做法：

- 第一帧 GPS 映射到初始位姿 `(0, -15)`
- 后续 GPS 慢融合（`gps_gain=0.12`）
- 单帧跳变超过 `gps_reject_distance` 则拒绝
- 磁力计未标定前 `mag_gain=0`

短时位姿主要靠轮速 + IMU 积分；定位一漂，建图和感知都会跟着错。

### 感知与建图

建图节点依赖 `/localization/pose` 把 `base_link` 下锥桶变到 `world`。定位飞走或感知范围不对时，`/estimation/slam/map` 会空，规划只能走白色 fallback 线，弯心不准。

内置 `track_perception` 用静态锥桶表模拟观测，联调建图、规划、控制足够；`sim_perception` 依赖加密运行时，Python 版本不对时会缺 `pyarmor_runtime.so`，因此默认用内置感知。

### 规划与控制

直角弯入口需要更早转向。Pure Pursuit 最初用固定 `target_speed` 算角速度，弯里却又按曲率降速，实际线速度低、角速度按高速算，**弯中欠转、出弯贴不回绿线**。

改动包括：

- 用**实际线速度**参与曲率角速度计算，并设 `steer_speed_ratio` 下限
- 入弯时用 `path_heading_gain` 按前方路径切线提前转向
- 前方路径弯曲时自动缩短 lookahead
- 去掉「最近点横向修正」（直道/弯道交界处会把方向拉反，导致又转不过去）
- 提高 `max_yaw_rate` 上限

当前控制参数（`right_angle_stack/config/right_angle_stack.yaml`）：


| 参数                     | 值    | 作用             |
| ---------------------- | ---- | -------------- |
| target_speed           | 3.0  | 直道目标速度         |
| min_speed              | 1.5  | 弯中最低速度         |
| lookahead_distance     | 2.6  | 预瞄距离           |
| min_lookahead_distance | 1.3  | 弯中缩短后的预瞄       |
| max_yaw_rate           | 2.7  | 最大角速度          |
| yaw_error_gain         | 0.58 | 预瞄点航向修正        |
| path_heading_gain      | 0.84 | 路径切线方向修正（入弯）   |
| steer_speed_ratio      | 0.88 | 转向速度下限（相对目标速度） |
| turn_slowdown_factor   | 0.25 | 弯中降速幅度         |


若仍欠转可略增 `max_yaw_rate` 或 `path_heading_gain`；若 overshoot 可略减 `lookahead_distance`。

## 坐标系约定

- 全局：`world`，ENU，x 向东，y 向北，z 向上
- 车体：`base_link`，FLU，x 向前，y 向左，z 向上
- 相机光学系：`camera_optical_frame`
- 雷达：`lidar_link`
- TF：`world` → `base_link`

车辆从 `(0, -15)` 出发朝北，初始 yaw 为 $\frac{\pi}{2}$。

## 工程结构

工作空间：`~/fsd_ws`

```
fsd_ws/
├── run.sh                      # 一键编译 + 启动
├── README.md
├── fsd_bringup/                # F1 模型 + 主 launch
│   ├── launch/f1_right_angle.launch.py
│   ├── models/simple_car/      # Gazebo 车辆 SDF
│   └── urdf/simple_car.urdf.xacro
├── fsd_common_msgs/            # 锥桶 ROS 消息
├── tracks/                     # ROS 包名 right_angle_track
│   ├── worlds/right_angle_harmonic.sdf
│   ├── models/shixi/           # 赛道锥桶布局
│   └── meshes/                 # 锥桶网格
├── right_angle_stack/          # 定位 / 感知 / 建图 / 规划 / 控制
│   ├── config/right_angle_stack.yaml
│   ├── rviz/right_angle.rviz
│   └── right_angle_stack/*.py
└── sim_perception/             # 可选加密感知包（默认不用）
```

编译产物 `build/`、`install/`、`log/` 由 colcon 自动生成，可忽略或删除后重建。

### 各包作用


| 包名                  | 目录                   | 必需  | 作用                                |
| ------------------- | -------------------- | --- | --------------------------------- |
| `fsd_bringup`       | `fsd_bringup/`       | 是   | F1 模型、Gazebo/RViz 启动              |
| `fsd_common_msgs`   | `fsd_common_msgs/`   | 是   | 锥桶消息定义                            |
| `right_angle_track` | `tracks/`            | 是   | 赛道 world、锥桶模型                     |
| `right_angle_stack` | `right_angle_stack/` | 是   | 算法栈（5 个节点）                        |
| `sim_perception`    | `sim_perception/`    | 否   | 加密感知，需 `use_sim_perception:=true` |


主 launch 只启动 `f1_right_angle.launch.py`；算法参数统一在 `right_angle_stack/config/right_angle_stack.yaml`。

## 主要话题


| 话题                                              | 说明            |
| ----------------------------------------------- | ------------- |
| /cmd_vel                                        | 控制输出 → Gazebo |
| /sensors/wheel_odom                             | 轮速里程计         |
| /sensors/gps/fix、/sensors/imu/data_raw          | 定位传感器         |
| /localization/pose、/localization/odom           | 融合定位          |
| /perception/cones                               | 局部锥桶          |
| /estimation/slam/map                            | 世界系锥桶地图       |
| /planning/centerline                            | 规划路径          |
| /visualization/cone_map、/visualization/planning | RViz 可视化      |


## 运行

环境：ROS 2 Humble、Gazebo Harmonic、`ros_gz_sim`、`ros_gz_bridge`。

```bash
conda deactivate
cd ~/fsd_ws
chmod +x run.sh
./run.sh
```

改代码后：`./run.sh --rebuild`

清理旧进程：

```bash
pkill -INT -f "gz sim|rviz2|ros_gz_bridge|f1_right_angle" || true
sleep 2
pkill -9 -f "gz sim|rviz2|ros_gz_bridge|f1_right_angle" || true
```

手动 launch：

```bash
source /opt/ros/humble/setup.bash
source ~/fsd_ws/install/setup.bash
WORLD="$(ros2 pkg prefix right_angle_track)/share/right_angle_track/worlds/right_angle_harmonic.sdf"
ros2 launch fsd_bringup f1_right_angle.launch.py \
  use_rviz:=true use_builtin_perception:=true use_sim_perception:=false \
  gz_args:="-r -v 4 ${WORLD}"
```

切换为 `sim_perception`：设 `use_builtin_perception:=false use_sim_perception:=true`。

## 模块说明

**定位** — `localization_fusion.py`：GPS + 轮速 + IMU 融合，发 pose/odom/TF。

**感知** — 默认 `track_perception.py`；可选 `sim_perception`。

**建图** — `cone_mapper.py`：局部锥桶 → world 地图。

**规划** — `right_angle_planner.py`：锥桶中线优先，解析圆弧 fallback。

**控制** — `pure_pursuit_controller.py`：Pure Pursuit + 弯前航向修正。

## 联调顺序

1. 传感器与 `/clock` 有数据
2. `/localization/pose` 在赛道附近
3. `/perception/cones`、`/estimation/slam/map` 有锥桶
4. `/planning/centerline` 完整
5. `/cmd_vel` 线速度、角速度正常
6. Gazebo 与 RViz 中 F1 完成直角弯

## 调试命令

```bash
ros2 node list
ros2 topic echo /localization/pose --once
ros2 topic echo /planning/centerline --once
ros2 topic echo /cmd_vel
ros2 topic echo /sensors/wheel_odom --once
```

- pose 很远 → 定位 / GPS  
- centerline 空 → 建图或 fallback  
- angular.z 为 0 → 规划 / lookahead  
- 有角速度但不转 → DiffDrive / bridge

