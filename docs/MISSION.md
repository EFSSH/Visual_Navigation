# 任务层架构

对应第一版实验方案第 5、6、7、8、11 节。数字孪生拓扑路线和模型适配器不在本次范围内：路线暂由任务文件给出，学习模型通过 `/execution/trajectory` 接入。

## 模块

| 包 | 运行位置 | 职责 |
| --- | --- | --- |
| `b2_nav_robot` | 设备端（仿真 / B2） | `cmd_watchdog_node`：转发 `/cmd_vel_safe` 到 `/cmd_vel_device`；0.5 s 无指令或 `/robot/estop` 为 True 时发零速 |
| `b2_nav_execution` | 导航端 | `trajectory_executor_node`：检查轨迹（时效、点距、长度、转向、局部代价地图碰撞），不修改轨迹；合格后经 Nav2 `FollowPath` 交给 `TrackPath`（RPP）跟踪；轨迹过期或被拒时取消执行 |
| `b2_nav_search` | 导航端 | `object_search_node`：参考库匹配（ORB + RANSAC）、深度反投影到地图、多视角一致性和可选 ArUco 编号确认、生成可达观察位姿 |
| `b2_nav_task` | 导航端 | `task_manager_node`：任务检查 → 定位初始化 → 跟踪子目标 → 区域搜索（窗口间原地转向）→ 观察位姿 → 停车并报告 |
| `b2_nav_experiment` | 导航端 | `experiment` 命令：记录（MCAP + zstd）、回放、分析、对比、导出训练片段 |
| `learning/` | 任意 Python 环境 | 不依赖 ROS：片段读取、按场景/路线划分、历史/预测窗口、基线预测器、评价指标 |

## 速度链路

所有速度都经过同一条链路，学习轨迹不绕开安全门：

```text
Nav2 (DWB 基线 | TrackPath) → /cmd_vel_nav → 速度平滑 → /cmd_vel
  → 激光安全门 → /cmd_vel_safe ─局域网→ 看门狗 → /cmd_vel_device → 底盘
```

`TrackPath` 只跟踪给定路径，碰撞检测只减速或停车。DWB 仍服务 Nav2 基线（`bt_navigator` 默认选择 `FollowPath`）。同一时间只应有一个轨迹来源。

## 话题接口

结构化输出均为 `std_msgs/String` JSON，带 `schema` 字段便于版本演进和记录。

| 话题 | 类型 | 方向 | 内容 |
| --- | --- | --- | --- |
| `/execution/trajectory` | `nav_msgs/Path` | 模型 → 执行层 | `odom` 或 `map` 坐标；时间戳后 1.5 s 内须更新 |
| `/execution/feedback` | JSON `execution_feedback/v1` | 执行层 → | 状态、跟踪误差、剩余长度、Nav2 反馈、安全门与看门狗状态、计数 |
| `/execution/events` | JSON `execution_event/v1` | 执行层 → | accepted / rejected(reason) / expired / succeeded / aborted / safety_stop |
| `/search/request` | JSON | 任务 → 搜索 | `request_id`、`device_id`、`timeout_sec` |
| `/search/result` | JSON `search_result/v1`（latched） | 搜索 → 任务 | confirmed / not_found / not_confirmed / lookalike_rejected / observation_unreachable，含证据与观察位姿 |
| `/search/observation_goal` | `geometry_msgs/PoseStamped` | 搜索 → | 确认后的观察位姿 |
| `/task/start`、`/task/cancel` | `String` | → 任务 | 任务 YAML/JSON 文本或 `file:<路径>`；取消原因 |
| `/task/state`、`/task/report` | JSON（latched） | 任务 → | 当前状态；最终报告（结果、失败原因、时间线、末位姿、搜索证据） |
| `/task/events` | JSON `task_event/v1` | 任务 → | 状态变化、事件、导航目标、搜索请求 |
| `/robot/watchdog_state` | JSON `watchdog_state/v1` | 设备端 → | waiting / forwarding / timeout / estop |
| `/robot/estop` | `std_msgs/Bool`（latched） | → 设备端 | True 急停，False 解除 |

搜索期间才订阅前相机 RGB、深度和内参，结束后取消订阅，减少局域网带宽。

## 参考库

默认位于 Ubuntu `~/b2_nav_data/reference_library/<device_id>/`，每个设备一个 `device.yaml` 和参考图，格式见 `b2_nav_search/library.py` 文件头。图像相似只产生候选；确认需要 `confirm.min_views` 个不同时刻的帧在 `cluster_radius_m` 内给出一致位置，设置了 `marker_id` 时还须在匹配框内看到该编号。匹配框内出现其他设备的编号时判为相似设备并拒绝。

## 任务文件

示例：`b2_nav_task/config/tasks/hz26_example.yaml`（位姿为占位值）。失败原因和对应的归因层：

| 失败原因 | 归因层 |
| --- | --- |
| `task_invalid`、`preflight_failed` | 接口 |
| `localization_failed` | 定位 |
| `subgoal_unreachable`、`observation_unreachable` | 轨迹生成 |
| `subgoal_timeout` | 控制 |
| `object_not_found`、`object_not_confirmed`、`lookalike_rejected` | 对象确认 |

分析时看门狗超时会改判为接口问题，多次轨迹拒绝改判为轨迹生成，频繁安全门停车改判为感知；`summary.json` 的 `failure_evidence` 保留全部依据供人工复核。

## 使用

导航端（Ubuntu）：

```bash
bash /mnt/mac/第一版导航框架/scripts/run-navigation.sh use_sim_time:=true
ros2 launch b2_nav_bringup mission.launch.py source_label:=nav2_baseline
ros2 topic pub --once /task/start std_msgs/msg/String "{data: 'file:/home/efssh/b2_nav_data/tasks/xxx.yaml'}"
```

实验记录（配置示例 `b2_nav_experiment/config/experiments/nav2_baseline_hz26.yaml`）：

```bash
ros2 run b2_nav_experiment experiment run <config.yaml>      # 记录，任务报告后自动停止并分析
ros2 run b2_nav_experiment experiment analyze <run_dir>      # 重新生成 summary.json、report.md
ros2 run b2_nav_experiment experiment compare <run_dir>...   # 按轨迹来源汇总成功率等
ros2 run b2_nav_experiment experiment export <run_dir>       # 导出 episode.npz
ros2 run b2_nav_experiment experiment replay <run_dir>       # 回放，需先停仿真
```

离线评价（不需要 ROS）：

```bash
PYTHONPATH=learning python3 -m b2_nav_learning.evaluate --data ~/b2_nav_data/episodes --split-by route --model action_kinematic
```

基线 `constant_velocity` 和 `action_kinematic` 不需要训练数据，是学习模型必须超过的参照。学习模型以 `--model module:Class` 接入，提供 `predict(samples)`，可选 `fit(train_samples)`。

## 数据量

开启 `include_images` 时前后 RGB 与深度约 3 MB/s（压缩前），10 分钟约 2 GB；Ubuntu 系统盘剩余约 90 GB。长期数据应移出虚拟机映像并备份。

## 验证

- `workspace.sh test`：69 项单元测试（ROS 包 61 项，含原有 14 项；`learning/` 8 项）。
- `scripts/smoke_mission.py`：隔离域内启动导航端、任务层和看门狗，检查看门狗转发与超时停车、TrackPath 加载、执行层决策、搜索请求处理、任务预检失败报告、实验记录/分析/导出往返。
- 尚未验证：与 WSL 仿真的双机闭环、真实相机画面上的匹配效果、实机。
