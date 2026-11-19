"""
Static figures for the SCI-1 2D experiments (PNG, light surface).

Colors follow the reference categorical palette in a fixed method order, so
a method keeps its color across figures and filters.
"""

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .sim2d import TRAJECTORY_TYPES  # noqa: E402

SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
GRID = '#e4e3df'

# Fixed method -> slot order (reference palette, light mode).
METHOD_STYLE = {
    'worst_case': ('#52514e', 'B0 worst case'),
    'naive_partial': ('#eb6834', 'naive (visible part)'),
    'complete_case': ('#eda100', 'complete case'),
    'verified_ipw': ('#2a78d6', 'verified IPW (exact)'),
    'observed_ipw': ('#1baf7a', 'observed IPW (exp.)'),
    'oracle_full': ('#6250d6', 'oracle (all truth)'),
}
TYPE_LABEL = {
    'always_visible': 'always visible',
    'lost_from_view': 'lost from view',
    'emerging': 'emerging from corner',
    'never_seen': 'never seen',
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.yaxis.label.set_color(TEXT_2)
    ax.xaxis.label.set_color(TEXT_2)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width=7.5, height=4.2, cols=1):
    fig, axes = plt.subplots(1, cols, figsize=(width, height), facecolor=SURFACE)
    axes = [axes] if cols == 1 else list(axes)
    for ax in axes:
        _style(ax)
    return fig, axes


def _pick(summary, **match):
    return [e for e in summary if all(e.get(k) == v for k, v in match.items())]


def coverage_by_type(summary, alpha, budget, allocation, path):
    """Figure 1: coverage per trajectory type, every method, at one alpha and budget."""
    methods = [m for m in METHOD_STYLE if m != 'worst_case']
    types = [t for t in TRAJECTORY_TYPES
             if any(e.get(f'coverage_{t}') is not None for e in summary)]
    fig, (ax,) = _figure(8.0, 4.2)
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        entry = (_pick(summary, method=method, alpha=alpha, budget=budget, allocation=allocation)
                 or [None])[0]
        if entry is None:
            continue
        xs = [j + (i - (len(methods) - 1) / 2) * width for j in range(len(types))]
        ys = [entry.get(f'coverage_{t}', float('nan')) for t in types]
        errs = [entry.get(f'coverage_{t}_std', 0.0) for t in types]
        color, label = METHOD_STYLE[method]
        ax.bar(xs, ys, width * 0.88, color=color, label=label, yerr=errs,
               error_kw={'ecolor': TEXT_2, 'elinewidth': 0.8, 'capsize': 0})
    ax.axhline(1 - alpha, color=TEXT, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(len(types) - 0.55, 1 - alpha + 0.005, f'target {1 - alpha:.2f}',
            color=TEXT_2, fontsize=8, ha='right', va='bottom')
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([TYPE_LABEL[t] for t in types], color=TEXT_2)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel('coverage of clearance bound')
    ax.set_title(f'Coverage by trajectory type (alpha {alpha}, budget {budget}, {allocation})',
                 color=TEXT, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='lower left', labelcolor=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def pareto(summary, budget, allocation, path):
    """Figure 2: near-miss rate vs time lost, one curve per method over alpha."""
    fig, (ax,) = _figure(7.5, 4.6)
    for method, (color, label) in METHOD_STYLE.items():
        if method == 'worst_case':
            pts = _pick(summary, method=method)
        else:
            pts = sorted(_pick(summary, method=method, budget=budget, allocation=allocation),
                         key=lambda e: e['alpha'])
        if not pts:
            continue
        xs = [p['time_lost_s'] for p in pts]
        ys = [100 * p['near_miss_rate'] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=2, marker='o', markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        if method == 'worst_case':
            ax.annotate(label, (xs[0], ys[0]), xytext=(-6, 6), textcoords='offset points',
                        fontsize=8, color=TEXT_2, ha='right')
    alphas = sorted({e['alpha'] for e in summary if e.get('alpha') is not None}, reverse=True)
    ax.text(0.99, 0.70, 'points per curve, left to right:\nalpha = '
            + ', '.join(str(a) for a in alphas), transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color=TEXT_2)
    ax.set_xlabel('mean time lost vs always-go over the 3 s horizon (s)')
    ax.set_ylabel('near-miss rate of chosen action (%)')
    ax.set_title(f'Safety-efficiency trade-off (budget {budget}, {allocation}); '
                 'lower-left is better', color=TEXT, fontsize=11, loc='left')
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc='upper right', labelcolor=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def budget_curves(summary, alpha, path):
    """Figure 3: hidden-stratum coverage, hidden-stratum and overall time lost vs budget."""
    fig, axes = _figure(13.0, 4.0, cols=3)
    dashes = {'uniform': (0, (4, 3)), 'hidden_focus': 'solid'}
    for method in ('verified_ipw', 'observed_ipw', 'complete_case'):
        color, label = METHOD_STYLE[method]
        for allocation, dash in dashes.items():
            pts = sorted(_pick(summary, method=method, alpha=alpha, allocation=allocation),
                         key=lambda e: e['budget'])
            if not pts:
                continue
            xs = [p['budget'] for p in pts]
            name = f'{label}, {allocation}'
            axes[0].plot(xs, [p['coverage_hidden0'] for p in pts], color=color,
                         linestyle=dash, linewidth=2, marker='o', markersize=4, label=name)
            axes[1].plot(xs, [p['time_lost_s_hidden0'] for p in pts], color=color,
                         linestyle=dash, linewidth=2, marker='o', markersize=4, label=name)
            axes[2].plot(xs, [p['time_lost_s'] for p in pts], color=color,
                         linestyle=dash, linewidth=2, marker='o', markersize=4, label=name)
    axes[0].axhline(1 - alpha, color=TEXT, linewidth=1.0, linestyle=(0, (4, 3)))
    axes[0].set_ylabel('coverage, agent hidden at t0')
    axes[1].set_ylabel('time lost, agent hidden at t0 (s)')
    axes[2].set_ylabel('time lost, all decisions (s)')
    for ax in axes:
        ax.set_xlabel('verification budget per decision')
    axes[0].set_title(f'Verification budget (alpha {alpha})', color=TEXT, fontsize=11, loc='left')
    axes[2].legend(frameon=False, fontsize=7, loc='upper right', labelcolor=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def make_figures(summary, args):
    alpha = 0.1 if 0.1 in args.alphas else args.alphas[0]
    budget = args.budgets[len(args.budgets) // 2]
    allocation = 'hidden_focus' if 'hidden_focus' in args.allocations else args.allocations[0]
    coverage_by_type(summary, alpha, budget, allocation,
                     os.path.join(args.out, 'fig1_coverage_by_type.png'))
    pareto(summary, budget, allocation, os.path.join(args.out, 'fig2_pareto.png'))
    if len(args.budgets) > 1:
        budget_curves(summary, alpha, os.path.join(args.out, 'fig3_budget.png'))


def detection_sweep(rows, alpha, path):
    """Figure 4: near-miss rate and coverage vs close-range detection probability."""
    scenarios = sorted({r['scenario'] for r in rows}, key=['occlusion', 'wall_hug'].index)
    fig, axes = plt.subplots(2, len(scenarios), figsize=(5.2 * len(scenarios), 7.0),
                             facecolor=SURFACE, sharex=True, sharey='row', squeeze=False)
    dashes = {0.0: (0, (4, 3)), 0.8: 'solid'}
    titles = {'occlusion': 'Wide corner (v1 geometry)',
              'wall_hug': 'Wall-hugging agent, narrow corner'}
    for col, scenario in enumerate(scenarios):
        for row_i, (metric, scale, ylabel) in enumerate((
                ('near_miss_rate', 100, 'near-miss rate of chosen action (%)'),
                ('coverage', 1, 'coverage of clearance bound'))):
            ax = axes[row_i][col]
            _style(ax)
            for method in ('naive_partial', 'verified_ipw', 'oracle_full'):
                color, label = METHOD_STYLE[method]
                for corr, dash in dashes.items():
                    pts = sorted((r for r in rows if r['scenario'] == scenario
                                  and r['method'] == method and r['det_corr'] == corr),
                                 key=lambda r: -r['det_p_near'])
                    if not pts:
                        continue
                    ax.plot([p['det_p_near'] for p in pts], [scale * p[metric] for p in pts],
                            color=color, linestyle=dash, linewidth=2, marker='o', markersize=5,
                            markeredgecolor=SURFACE, markeredgewidth=1.5)
            if metric == 'coverage':
                ax.axhline(1 - alpha, color=TEXT, linewidth=1.0, linestyle=(0, (4, 3)))
                ax.set_xlabel('detection probability at close range (<= 0.6 m)')
            ax.set_ylabel(ylabel)
            ax.invert_xaxis() if not ax.xaxis_inverted() else None
            if row_i == 0:
                ax.set_title(titles.get(scenario, scenario), color=TEXT, fontsize=11, loc='left')
    handles = [Line2D([], [], color=METHOD_STYLE[m][0], linewidth=2, label=METHOD_STYLE[m][1])
               for m in ('naive_partial', 'verified_ipw', 'oracle_full')]
    handles += [Line2D([], [], color=TEXT_2, linewidth=2, linestyle=dashes[0.8],
                       label='misses persist across frames (0.8)'),
                Line2D([], [], color=TEXT_2, linewidth=2, linestyle=dashes[0.0],
                       label='independent misses per frame')]
    axes[0][0].legend(handles=handles, frameon=False, fontsize=8, loc='upper left',
                      labelcolor=TEXT_2)
    fig.text(0.01, 0.005, 'Left to right: the detector gets worse at close range; 1.0 is ideal '
             'detection. Verified IPW and oracle do not use the onboard record, so they stay flat.',
             color=TEXT_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def sensor_comparison(rows, labels, path):
    """Figure 5: per sensor config, near-miss rate by method and what the log misses."""
    scenarios = sorted({r['scenario'] for r in rows}, key=['occlusion', 'wall_hug'].index)
    configs = list(dict.fromkeys(r['config'] for r in rows))
    titles = {'occlusion': 'Wide corner', 'wall_hug': 'Wall-hugging agent, narrow corner'}
    fig, axes = plt.subplots(2, len(scenarios), figsize=(6.4 * len(scenarios), 8.6),
                             facecolor=SURFACE, sharey=True, squeeze=False)
    ys = list(range(len(configs)))[::-1]
    methods = ('naive_partial', 'verified_ipw', 'oracle_full')
    height = 0.8 / len(methods)
    top_max = 1.08 * max(100 * r[f'{m}_near_miss_rate'] for r in rows for m in methods)
    for col, scenario in enumerate(scenarios):
        by_config = {r['config']: r for r in rows if r['scenario'] == scenario}
        ax = axes[0][col]
        _style(ax)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color=GRID, linewidth=0.8)
        for i, method in enumerate(methods):
            color, label = METHOD_STYLE[method]
            offs = [y + (len(methods) - 1 - i - (len(methods) - 1) / 2) * height for y in ys]
            vals = [100 * by_config[c][f'{method}_near_miss_rate'] if c in by_config else 0
                    for c in configs]
            ax.barh(offs, vals, height * 0.85, color=color, label=label)
        ax.set_xlim(0, top_max)
        ax.set_xlabel('near-miss rate of chosen action (%)')
        ax.set_title(titles.get(scenario, scenario), color=TEXT, fontsize=11, loc='left')
        ax = axes[1][col]
        _style(ax)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color=GRID, linewidth=0.8)
        get = lambda key: [100 * by_config[c][key] if c in by_config else 0 for c in configs]
        upper = [y + 0.19 for y in ys]
        lower = [y - 0.19 for y in ys]
        ax.barh(upper, get('logged_as_safe'), 0.34, color=TEXT_2,
                label='near miss logged as safe (closest approach not recorded)')
        never, late = get('never_detected'), get('lead_below_stop_time')
        ax.barh(lower, never, 0.34, color='#8a8984', label='never detected within 3 s')
        ax.barh(lower, late, 0.34, left=[n + 0.4 if n else 0 for n in never], color='#cfcec9',
                label='first detected < stopping time (1 s) before closest')
        ax.set_xlim(0, 100)
        ax.set_xlabel('share of go-action near misses (%)')
    for row_axes in axes:
        row_axes[0].set_yticks(ys)
        row_axes[0].set_yticklabels([labels.get(c, c) for c in configs], color=TEXT_2, fontsize=9)
    axes[0][-1].legend(frameon=False, fontsize=8, loc='lower right', labelcolor=TEXT_2)
    handles, names = axes[1][-1].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=8, loc='lower center', ncol=3,
               labelcolor=TEXT_2)
    axes[1][0].set_title('What the onboard log misses', color=TEXT, fontsize=11, loc='left')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def closed_loop_comparison(summary, labels, path):
    """Figure 6: near-miss rate held vs replanning, and time lost when replanning."""
    rules = ('naive_partial', 'verified_ipw', 'oracle_full', 'worst_case')
    keys = list(dict.fromkeys((e['config'], e['noise']) for e in summary))
    noise_label = {'none': 'no errors', 'camera_typical': 'noise + false det. + latency'}
    names = [f'{labels.get(c, c)}\n{noise_label.get(n, n)}' for c, n in keys]
    panels = (('hold_near_miss_rate', 100, 'near-miss rate, t0 decision held (%)'),
              ('replan_near_miss_rate', 100, 'near-miss rate, replanning (%; note the finer scale)'),
              ('replan_time_lost_s', 1, 'time lost over 5 s, replanning (s)'))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 0.62 * len(keys) + 1.8), facecolor=SURFACE,
                             sharey=True)
    ys = list(range(len(keys)))[::-1]
    height = 0.8 / len(rules)
    by_key = {(e['config'], e['noise'], e['rule']): e for e in summary}
    for ax, (metric, scale, xlabel) in zip(axes, panels):
        _style(ax)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color=GRID, linewidth=0.8)
        for i, rule in enumerate(rules):
            color, label = METHOD_STYLE[rule]
            offs = [y + ((len(rules) - 1) / 2 - i) * height for y in ys]
            vals = [scale * by_key[(c, n, rule)][metric] if (c, n, rule) in by_key else 0
                    for c, n in keys]
            ax.barh(offs, vals, height * 0.85, color=color, label=label)
        ax.set_xlabel(xlabel)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(names, color=TEXT_2, fontsize=8)
    axes[0].set_title('Mixed scenario: one decision held vs replanning at every frame',
                      color=TEXT, fontsize=11, loc='left')
    handles, names_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, names_, frameon=False, fontsize=8, loc='lower center', ncol=4,
               labelcolor=TEXT_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
