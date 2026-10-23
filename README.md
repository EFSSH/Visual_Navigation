# 第一版导航框架

已从远端工程对应的 Git 基线迁入：`EFSSH/Visual_Navigation`，提交 `9c2475e`。
本地开发分支为 `codex/thor-migration`，目标是 Ubuntu 24.04 ARM64 / ROS 2 Jazzy，后续部署到 AGX Thor。

## 现在包含什么

保留原有 9 个已入库 ROS 包、HZ26 固定地图、双 RGB-D 模型、几何地图及安全门。
新增独立的 `navigation_only.launch.py` 和 Jazzy 参数，Mac 导航端不启动 Gazebo 或设备驱动。
原有 Humble 启动文件与参数保留，供旧电脑基线查阅。

```text
远端仿真／未来 B2：点云、激光、odom、TF、仿真 clock
                         ↓
双 RGB-D 预处理 → 2.5D 局部几何地图 → Nav2 / DWB
                                           ↓ /cmd_vel_nav
                                      速度平滑
                                           ↓ /cmd_vel
                                      原激光安全门
                                           ↓ /cmd_vel_safe
                                    远端仿真／未来 B2
```

## 在这台 Mac 上使用

- `构建导航框架.command`：启动已有 Ubuntu 环境，安装依赖、同步源码、构建并运行现有功能测试。
- `启动导航端.command`：启动导航端。需要设备数据、TF 和相应时钟到位；当前尚未配置跨机 ROS 通信。
- 停止导航进程：在其终端按 Ctrl+C。
- 退出并拔硬盘前，运行旁边 `nav-environment/停止导航环境.command`。

Mac 文件夹是源码主副本；Ubuntu `/home/efssh/b2_nav_ws` 是自动同步的构建副本。**编辑 Mac 文件夹中的源码；不要直接编辑 Ubuntu 的 src，下一次同步会覆盖。** 构建产物留在 Linux 文件系统中，以避免 ExFAT 的权限和符号链接限制。

Ubuntu 内执行：

```bash
bash /mnt/mac/第一版导航框架/scripts/workspace.sh build
bash /mnt/mac/第一版导航框架/scripts/workspace.sh test
bash /mnt/mac/第一版导航框架/scripts/run-navigation.sh use_sim_time:=true
```

默认使用仿真时间。未来接实机时使用 `use_sim_time:=false`，并替换对应的设备配置、机器人尺寸、外参和限速；当前参数仍是 Waffle 仿真基线，不能直接当作 B2 实机参数。

## 移植到 Thor

在 Thor 安装匹配设备的 JetPack、ROS 2 Jazzy 及 Git/colcon/rosdep。
将本仓库作为 Linux 工作区使用，在其根目录运行：

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y
colcon build --symlink-install
source install/setup.bash
ros2 launch b2_nav_bringup navigation_only.launch.py use_sim_time:=false
```

这是构建与启动方式，真实运行前仍需接入 B2 驱动、外参和实机配置。CUDA/TensorRT 模型适配不属于此次迁移。

## 当前边界

- 已迁入的是与远端提交一致的**入库框架**。远端未入库的 HZ26 场景资源、旧假 B2 节点、数据、权重和构建缓存未复制。
- 新入口需要设备端提供 `odom → base_footprint → base_link` 和传感器 TF；AMCL 负责 `map → odom`。导航端不重复发布设备 TF。
- 不自动发布初始位姿，需通过 RViz 或 `/initialpose` 初始化定位。
- 旧 Windows/WSL 仿真仍是 Humble/Gazebo Classic。跨版本通信与 Gazebo Harmonic 迁移尚未验收，不能直接认为已经打通双机导航。
- 学习模型、全局数字孪生、任务管理和末端图像查找尚未新增。

详细变更与验证范围见 [迁移记录](docs/MIGRATION.md)。
