# SCI-1 2D 仿真：盲角下的随机核验与覆盖校准

代码在 `learning/sci1/`，只依赖 NumPy（画图用 Matplotlib），不依赖 ROS。对应 Notion「SCI-1｜盲角导航：用有限预算的随机核验，在保持覆盖率的同时少等、少停」。

## 运行

```bash
python3 -m venv ../.venv-sci1 && ../.venv-sci1/bin/pip install numpy matplotlib pytest
PYTHONPATH=learning ../.venv-sci1/bin/python -m pytest learning/tests
PYTHONPATH=learning ../.venv-sci1/bin/python -m sci1.experiments --out ../sci1_runs/<日期>_<名称>
```

默认 20 次重复、每次 4000 个校准回合、4000 个测试回合，单核约 10 秒。常用参数：
- `--scenario`：`occlusion`（宽墙角，v1 默认）、`wall_hug`（对方贴墙走、主走廊窄，出拐角时只剩 1.3–2 m）、`meeting`（无遮挡对照）；
- `--perception`：`ideal`（视线内一定检测到，等同 v1）或 `near_range_miss`（近距离漏检）；`--det-p-near`、`--det-corr`、`--det-near-range`、`--det-min-range`、`--det-p-far` 可单独覆盖预设值；
- `--wall-setback`（墙角退让，越小越常被遮挡）、`--alphas`、`--budgets`、`--allocations`、`--methods`、`--epsilon`。

输出：`config.json`、`rows.jsonl`（每次重复一行）、`summary.csv`（均值与标准差）、`diagnostics.json`（见下）、`example_calibration_log.npz`、`fig1_coverage_by_type.png`、`fig2_pareto.png`、`fig3_budget.png`。

检测模型敏感性扫描（近距离检测概率 × 漏检持续性，两种场景）：

```bash
PYTHONPATH=learning ../.venv-sci1/bin/python -m sci1.sweep --out ../sci1_runs/<日期>_detection_sweep
```

输出 `sweep.json` 和 `fig4_detection_sweep.png`。

## 感知模型

视线（几何）与检测（机器人的检测器）分开：对方按视线对机器人做出反应；机器人记录的一切（决策时是否遮挡、观测轨迹、可见步）都用检测结果。

- 检测概率：距离 ≥ `det_near_range` 时为 `det_p_far`，向 `det_min_range` 线性降到 `det_p_near`。对应相机人体检测器在近距离只看到被截断的身体。
- 漏检持续性 `det_corr`：每帧以该概率沿用本回合的同一个随机数（同一个人检测器总是认不出），否则重新抽取。随机数来自回合种子，各分支相同，分支回放仍然精确。
- `near_range_miss` 预设（2.0 m 开始下降，0.6 m 处 0.2，持续性 0.8）是占位值。下结论前应当用实物检测日志标定，所以结果以扫描曲线报告。
- 默认 `ideal` 与 v1 逐位一致（有回归检查）。

`diagnostics.json` 把近距离问题拆成两个独立的问题，按动作和决策时分层统计测试集里的险情：
1. 最近点有没有被记录：`closest_undetected`（最近那一帧没检测到）、`recorded_as_safe`（机载记录算出的最小距离 ≥ d_safe）、`mean_overestimate_m`；
2. 发现时还剩多少反应时间：`first_detect_dist_*`、`lead_time_*`（首次检测到最近点的时间）、`lead_below_stop_time`（短于从当前速度刹停所需时间的占比）。

## 模块

| 文件 | 内容 |
| --- | --- |
| `sim2d.py` | 走廊十字路口 + 左上墙角遮挡；对方从侧走廊驶来，看见机器人接近时按个体反应系数减速；每回合一个关键决策（go / slow / yield，保持整个时域），由此定义势结果 Y(a)；每回合独立种子，可分支回放；视线与检测分开的感知模型 |
| `records.py` | 决策记录格式（仿真、Gazebo、实物共用）：候选动作、倾向概率 μ(a\|s)、核验概率 ρ(s) 与核验结果、分支编号、种子、可见性、观测到的与完整的对方轨迹；`oracle_*` 字段仅仿真有 |
| `policies.py` | 名义策略 + ε 随机化得到 μ；按预算分配 ρ（`uniform` / `hidden_focus`） |
| `predictors.py` | 可见时恒速、遮挡时在遮挡边界放幽灵主体；分数 S = 预测最小距离 − 真实最小距离 |
| `conformal.py` | 按（动作，决策时是否遮挡）分层的分割共形；方法 `naive_partial`、`complete_case`、`verified_ipw`（权重只依赖状态，有限样本保证严格成立）、`observed_ipw`（实验性）、`oracle_full` |
| `planners.py` | 选下界 ≥ d_safe 的最快动作，否则让行；B0 最坏情况可达集（未检测时，视线外与近距离不可靠区都算未知） |
| `diagnostics.py` | 最近点是否被记录、首次检测时的剩余反应时间 |
| `collect.py` | 按 μ、ρ 采集决策记录；`reverify` 用公共随机数比较不同核验分配 |
| `experiments.py`、`sweep.py`、`plots.py` | 实验、检测模型扫描与四张图 |

## 与 ROS 的衔接

ROS 端只需要：在关键决策点按 μ 选动作并把 μ、ρ、Q 写进记录；按 `records.py` 的字段导出。校准、规划与评价代码不变。

## 当前边界

- 检测模型只有近距离漏检；遮挡边缘部分可见、跟踪丢失后的外推、检测延迟尚未加入。远距离漏检（`det_p_far` < 1）时 B0 不再保证有效。
- 只有一个遮挡块、一个对方主体。`wall_hug` 场景决策时几乎全被遮挡，看得到的那一层没有样本。
- 每回合只有一个关键决策，发现对方后不会重新规划，所以"发现太晚来不及让行"目前只统计、不参与规划；多步闭环（轨迹级权重）尚未实现。
- B0 忽略墙体，偏保守。
