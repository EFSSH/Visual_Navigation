# SCI-1 2D 仿真：盲角下的随机核验与覆盖校准

代码在 `learning/sci1/`，只依赖 NumPy（画图用 Matplotlib），不依赖 ROS。对应 Notion「SCI-1｜盲角导航：用有限预算的随机核验，在保持覆盖率的同时少等、少停」。

## 运行

```bash
python3 -m venv ../.venv-sci1 && ../.venv-sci1/bin/pip install numpy matplotlib pytest
PYTHONPATH=learning ../.venv-sci1/bin/python -m pytest learning/tests
PYTHONPATH=learning ../.venv-sci1/bin/python -m sci1.experiments --out ../sci1_runs/<日期>_<名称>
```

默认 20 次重复、每次 4000 个校准回合、4000 个测试回合，单核约 10 秒。常用参数：`--wall-setback`（墙角退让，越小越常被遮挡）、`--alphas`、`--budgets`、`--allocations`、`--methods`、`--epsilon`、`--scenario meeting`（无遮挡对照）。

输出：`config.json`、`rows.jsonl`（每次重复一行）、`summary.csv`（均值与标准差）、`example_calibration_log.npz`、`fig1_coverage_by_type.png`、`fig2_pareto.png`、`fig3_budget.png`。

## 模块

| 文件 | 内容 |
| --- | --- |
| `sim2d.py` | 走廊十字路口 + 左上墙角遮挡；对方从侧走廊驶来，看见机器人接近时按个体反应系数减速；每回合一个关键决策（go / slow / yield，保持整个时域），由此定义势结果 Y(a)；每回合独立种子，可分支回放 |
| `records.py` | 决策记录格式（仿真、Gazebo、实物共用）：候选动作、倾向概率 μ(a\|s)、核验概率 ρ(s) 与核验结果、分支编号、种子、可见性、观测到的与完整的对方轨迹；`oracle_*` 字段仅仿真有 |
| `policies.py` | 名义策略 + ε 随机化得到 μ；按预算分配 ρ（`uniform` / `hidden_focus`） |
| `predictors.py` | 可见时恒速、遮挡时在遮挡边界放幽灵主体；分数 S = 预测最小距离 − 真实最小距离 |
| `conformal.py` | 按（动作，决策时是否遮挡）分层的分割共形；方法 `naive_partial`、`complete_case`、`verified_ipw`（权重只依赖状态，有限样本保证严格成立）、`observed_ipw`（实验性）、`oracle_full` |
| `planners.py` | 选下界 ≥ d_safe 的最快动作，否则让行；B0 最坏情况可达集 |
| `collect.py` | 按 μ、ρ 采集决策记录；`reverify` 用公共随机数比较不同核验分配 |
| `experiments.py`、`plots.py` | 参数扫描与三张图 |

## 与 ROS 的衔接

ROS 端只需要：在关键决策点按 μ 选动作并把 μ、ρ、Q 写进记录；按 `records.py` 的字段导出。校准、规划与评价代码不变。

## 当前边界

- 感知是理想视线模型（无漏检、无跟踪丢失），只有一个遮挡块、一个对方主体。
- 每回合只有一个关键决策；多步闭环（轨迹级权重）尚未实现。
- B0 忽略墙体，偏保守。
