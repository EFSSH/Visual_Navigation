"""
Figures for the SCI-1 v6/v7a simulation records (static PNG, Chinese labels).

Reads the JSON results under ``--runs`` and writes PNGs to ``--out``:
fig1_onboard_miss, fig2_violation, fig3_daily_time, fig4_sample_size.

Run: PYTHONPATH=learning python -m sci1.plots_v7 --runs /home/xlz/sci1_runs --out /home/xlz/sci1_runs/figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical palette, fixed order (validated: CVD and normal-vision pass;
# slots 3-5 are below 3:1 contrast, so every bar group also has a legend and labels).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'
CRITICAL = '#d03b3b'

CAM_LABEL = {'front_87_h0.5': '87° 前视', 'front_120_h0.5': '120°×90° 前视',
             'ring4_87_h0.5': '四向环视 87°', 'gazebo_60_h0.11': 'Gazebo 60°'}
VAR_LABEL = {'ideal+extrap': '理想感知', 'realistic+extrap': '现实感知',
             'ideal': '理想检测', 'realistic': '现实组合', 'harsh': '恶劣组合',
             'latency 0.4s': '延迟 0.4 s', 'confirm 3 frames': '3 帧确认',
             'partial 40-80%': '局部门槛 40–80%', 'turn blur 0.5': '转弯降 50%'}
METHOD_LABEL = {'ours': '本方法', 'oursB': '刻画 B 分解界', 'point': '点估计',
                'ppi': 'PPI', 'naive': '只用机载日志'}


def style():
    from matplotlib import font_manager
    for path in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'):
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        'font.family': ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP', 'sans-serif'],
        'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK,
        'axes.titlesize': 10, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'legend.frameon': False, 'axes.unicode_minus': False,
    })


def load(path):
    with open(path) as f:
        return json.load(f)


def grouped_bars(ax, groups, series, values, labels=None, width=0.8, fmt=None):
    """values[i][j]: group i, series j. 2px-ish gaps between adjacent bars."""
    n = len(series)
    w = width / n
    x = np.arange(len(groups))
    for j, name in enumerate(series):
        vals = [values[i][j] for i in range(len(groups))]
        bars = ax.bar(x - width / 2 + w * (j + 0.5), vals, w * 0.9, color=SERIES[j],
                      label=name, edgecolor=SURFACE, linewidth=0.8)
        if fmt:
            for b, v in zip(bars, vals):
                if v is not None and not np.isnan(v):
                    ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                                ha='center', va='bottom', fontsize=6.5, color=INK2,
                                xytext=(0, 1), textcoords='offset points')
    ax.set_xticks(x, groups)
    ax.grid(axis='x', visible=False)
    return x


def fig1(runs, out):
    chk = load(os.path.join(runs, 'corner_check', 'att0.5.json'))['rows']
    per = load(os.path.join(runs, 'corner_perception', 'att0.5.json'))['rows']
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    cams = ['gazebo_60_h0.11', 'front_87_h0.5', 'front_120_h0.5', 'ring4_87_h0.5']
    acts = ['go', 'slow']
    vals = []
    for c in cams:
        row = []
        for act in acts:
            r = next(x for x in chk if x['camera'] == c and x['action'] == act)
            row.append(100 * (1 - r['near_moving_log'] / r['near_moving']) if r['near_moving'] else np.nan)
        vals.append(row)
    grouped_bars(a, [CAM_LABEL[c] for c in cams], ['go 1.0 m/s', 'slow 0.5 m/s'], vals,
                 fmt=lambda v: f'{v:.0f}%')
    a.set_ylim(0, 110)
    a.set_ylabel('机载日志漏掉的比例（%）')
    a.set_title('(a) 行驶中间隙 ≤ 10 cm 的事件，有多少不在机载日志里')
    a.legend(loc='lower left', fontsize=8)

    variants = ['ideal', 'latency 0.4s', 'turn blur 0.5', 'confirm 3 frames', 'partial 40-80%',
                'realistic', 'harsh']
    cams_b = ['front_87_h0.5', 'front_120_h0.5', 'ring4_87_h0.5']
    vals = [[100 * next(x for x in per if x['variant'] == v and x['camera'] == c
                        and x['action'] == 'go')['never'] for c in cams_b] for v in variants]
    grouped_bars(b, [VAR_LABEL[v] for v in variants], [CAM_LABEL[c] for c in cams_b], vals)
    b.set_ylabel('从未确认到人的相遇比例（%）')
    b.set_title('(b) 感知缺陷让人从机载记录中“消失”（go）')
    b.tick_params(axis='x', labelrotation=30)
    b.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig1_onboard_miss.png'), dpi=200)
    plt.close(fig)


def cases(cmp):
    seen = []
    for r in cmp['results']:
        key = (r['camera'], r['variant'])
        if key not in seen:
            seen.append(key)
    return seen


def fig2(runs, out):
    cmp = load(os.path.join(runs, 'corner_compare', 'base.json'))
    methods = ['ours', 'oursB', 'point', 'ppi', 'naive']
    keys = cases(cmp)
    fig, axes = plt.subplots(1, len(keys), figsize=(12, 3.3), sharey=True)
    for ax, (cam, var) in zip(axes, keys):
        rows = sorted([r for r in cmp['results'] if (r['camera'], r['variant']) == (cam, var)],
                      key=lambda r: -r['rho'])
        vals = [[100 * r[m]['violation'] for m in methods] for r in rows]
        x = grouped_bars(ax, [f'核验 {int(100*r["rho"])}%' for r in rows],
                         [METHOD_LABEL[m] for m in methods], vals)
        w = 0.8 / len(methods)
        for i, row in enumerate(vals):   # near-zero bars of our two bounds get a value label
            for j in (0, 1):
                ax.annotate(f'{row[j]:.1f}', (x[i] - 0.4 + w * (j + 0.5), row[j]), ha='center',
                            va='bottom', fontsize=6, color=INK2, xytext=(0, 1),
                            textcoords='offset points')
        ax.axhline(5, color=CRITICAL, lw=1, ls='--')
        ax.set_title(f'{CAM_LABEL[cam]}，{VAR_LABEL[var]}')
    axes[0].set_ylabel('超标率（%）：认证通过但实际一年 > 10 次')
    axes[0].annotate('承诺上限 δ = 5%', (0.02, 7), xycoords=('axes fraction', 'data'),
                     color=CRITICAL, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(methods), fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out, 'fig2_violation.png'), dpi=200)
    plt.close(fig)


def fig3(runs, out):
    cmp = load(os.path.join(runs, 'corner_compare', 'base.json'))
    E = 284.0  # encounters/year in the platform used by both scripts (printed by the runs)
    keys = cases(cmp)
    names = ['本方法', 'oracle', 'B0（最坏情况）', '全部 slow']
    vals, risks = [], []
    for cam, var in keys:
        r = next(x for x in cmp['results'] if (x['camera'], x['variant']) == (cam, var)
                 and x['rho'] == 1.0)
        q = r['q_true']
        b0q = q['crawl']
        b0s = 160 * (4 / 0.3 + (2.0 if r['b0'] == 'stop' else 0.0))
        ours_risk = r['ours']['true_risk'] if r['ours']['feasible'] >= 0.5 else None
        vals.append([r['ours']['seconds'] / 60, r['oracle']['seconds'] / 60, b0s / 60,
                     160 * 8.0 / 60])
        risks.append([ours_risk, r['oracle']['true_risk'], E * b0q, E * q['slow']])
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    x = grouped_bars(ax, [f'{CAM_LABEL[c]}\n{VAR_LABEL[v]}' for c, v in keys], names, vals)
    w = 0.8 / len(names)
    for i in range(len(keys)):
        for j in range(len(names)):
            rk = risks[i][j]
            label = '不可认证' if rk is None else f'{rk:.0f} 次/年'
            color = CRITICAL if (rk is not None and rk > 10) else INK2
            ax.annotate(label, (x[i] - 0.4 + w * (j + 0.5), vals[i][j]), ha='center',
                        va='bottom', fontsize=6.5, color=color, xytext=(0, 1),
                        textcoords='offset points')
    ax.set_ylabel('每天过 160 个拐角的时间（分钟）')
    ax.set_title('效率与真实险情：柱上数字为真实年险情次数（红色 = 超过 10 次）；核验 100%，1200 次标定')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.set_ylim(0, max(max(v) for v in vals) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig3_daily_time.png'), dpi=200)
    plt.close(fig)


def fig4(runs, out):
    pat = load(os.path.join(runs, 'corner_patrol', 'base.json'))['results']
    keys = [('front_87_h0.5', 'ideal+extrap'), ('ring4_87_h0.5', 'ideal+extrap'),
            ('front_120_h0.5', 'ideal+extrap'), ('front_120_h0.5', 'realistic+extrap')]
    fig, axes = plt.subplots(1, len(keys), figsize=(12, 3.0), sharey=True)
    for ax, (cam, var) in zip(axes, keys):
        for j, rho in enumerate([1.0, 0.3]):
            rows = sorted([r for r in pat if r['camera'] == cam and r['variant'] == var
                           and r['rho'] == rho], key=lambda r: r['n_cal'])
            ax.plot([r['n_cal'] for r in rows], [100 * r['ours']['feasible'] for r in rows],
                    color=SERIES[j], lw=2, marker='o', ms=5, label=f'核验 {int(100*rho)}%')
        ax.set_title(f'{CAM_LABEL[cam]}，{VAR_LABEL[var]}')
        ax.set_xlabel('标定相遇次数')
        ax.set_xticks([400, 1200, 2400])
        ax.set_ylim(-5, 105)
    axes[0].set_ylabel('能认证“每年 < 10 次”的比例（%）')
    axes[0].legend(loc='center right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig4_sample_size.png'), dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='/home/xlz/sci1_runs')
    ap.add_argument('--out', default='/home/xlz/sci1_runs/figures')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    style()
    for f in (fig1, fig2, fig3, fig4):
        f(a.runs, a.out)
        print('wrote', f.__name__)


if __name__ == '__main__':
    main()
