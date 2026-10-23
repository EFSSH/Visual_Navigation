# 远端框架迁移记录

日期：2026-10-19。

## 来源与范围

从 `https://github.com/EFSSH/Visual_Navigation.git` 的 `codex/local-world-model` 获取完整 Git 历史及源码，基线为 `9c2475eb203596f54c70f822059e9be33b072eb7`。
此前远端只读检查确认：远端 `codex/semantic-world-model` 指向同一提交，工作区干净。因此此次使用已发布且与检查结果一致的源码，不改变远端分支或文件。
本地创建 `codex/thor-migration` 分支，迁移改动尚未提交和推送。

已入库的 9 个包：

- b2_nav_bringup
- b2_nav_perception
- b2_nav_twin
- b2_nav_safety
- b2_nav_planning
- b2_nav_localization
- b2_nav_robot
- b2_nav_sensors
- b2_nav_vla

最后四个包仍是原工程骨架；不代表已实现定位驱动、传感器驱动、真实底盘或 VLA。
远端本地的 b2_nav_sim 和 sim_assets 被原仓库忽略，未包含在本次获取的框架中。原 Gazebo 模型描述文件和固定地图已随仓库迁入，大型场景资产继续留在仿真电脑。

## 已做的兼容调整

1. 新增 `navigation_only.launch.py`：只启动导航端的感知、几何地图、安全门和基线 Nav2 服务器，不启动 Gazebo、RViz、底盘驱动或 robot_state_publisher。
2. 新增 `nav2_jazzy_params.yaml`，原 Humble 配置保留。更新 progress checker 列表、插件类型命名、行为服务器的地图和足迹参数，并使用 Jazzy 自动加载内置 BT 插件的方式。
3. 显式保持 Twist 消息。控制器和恢复行为输出 `/cmd_vel_nav`，速度平滑输出 `/cmd_vel`，原安全门输出 `/cmd_vel_safe`。新入口不提供绕开安全门的开关。
4. 将 Gazebo Classic、旧 b2_nav_sim 和 TurtleBot 仿真依赖限定为 Humble；补齐导航入口和已有子启动文件使用的依赖声明。
5. 固定地图安装到 bringup 包中，不依赖 `/home/dministrator` 或 sim_assets 路径。运行用地图是 `src/b2_nav_bringup/maps/`；根目录 maps 保留原基线。
6. 增加 Ubuntu 构建同步脚本。Mac 的 ExFAT 源码目录保持为主副本，构建、安装、日志及符号链接在 Ubuntu 的 ext4 文件系统内生成。
7. 增加启动/构建快捷脚本和无机器人烟雾检查。

未修改现有感知、几何地图、DWB 代价权重和安全门算法。

## 验证结果

- Ubuntu 24.04 ARM64 / Jazzy 下 9 个包构建成功。
- `rosdep check --from-paths src --ignore-src --rosdistro jazzy` 通过。
- 原有功能测试 14 项通过：局部世界模型、激光距离处理和早期规划几何。
- `navigation_only.launch.py --show-args` 可正常解析。
- 9 个 Nav2 生命周期服务器均成功 configure，插件加载成功；仅激活地图服务器，确认运行地图可加载。未激活运动规划服务器，也未发送导航目标。
- 前后感知、局部几何地图和安全节点成功启动。
- 在独立 ROS_DOMAIN_ID=142、LOCALHOST 范围内注入合成双路 PointCloud2 和 odom/base_link TF，确认障碍输出、占据栅格及停止输入后的 TTL 过期。
- 无激光/指令输入时安全门输出零速度，Twist 话题类型正确。

日志保存在 `migration/build.log`、`dependencies.log`、`tests.log`、`smoke.log`、`nav2-startup.log` 和 `launch-arguments.log`。

重新执行集成检查（Ubuntu 中）：

```bash
source /opt/ros/jazzy/setup.bash
source ~/b2_nav_ws/install/setup.bash
python3 /mnt/mac/第一版导航框架/scripts/smoke_jazzy.py
```

烟雾检查不连接机器人，仅使用测试域和合成数据；它不能代替相机标定、真实点云回放、双机导航或实机验收。

## 后续需要验证

1. 旧 Windows/WSL 端与新 Jazzy 导航端的通信方案，包含时钟、TF、QoS、激光、里程计和双路点云。
2. 现有 Gazebo Classic 资源向 Jazzy 配套仿真的迁移，或显式受测的过渡接口。
3. 机器人定位初始化，以及所有 Nav2 服务器在真实设备输入下的激活与完整导航。
4. AGX Thor 的实际 JetPack/驱动版本、B2 设备接口、机器人尺寸、限速和外参。
5. 学习模型适配、轨迹接口、训练数据录制、全局数字孪生与末端图像查找。

参考：

- https://docs.nav2.org/jazzy/configuration_and_development/migration_guides/
- https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/core_servers/controller_server/
