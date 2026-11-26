"""
SCI-1 design rule: certification probability of a preselected route vs the number
of independent lab checks, and the expected daily saving pi_N(a) [T_fb - T(a)].

Each check draws (corner type, speed) with the route's exposure probabilities, so
the pooled close-call count is Binomial(N, R / W_total). The route passes if the
count is at most the largest c with CP_upper(c, N; delta) * W_total <= F.

Note the saw-tooth: the pass probability is not monotone in N (discrete test);
we report the first N reaching the target and the curve itself.
"""

import argparse
import json

import numpy as np

from .certify_route import binom_cdf


_THR = {}


def threshold(N, W_total, budget, delta):
    """Largest c with CP_upper(c, N) * W_total <= budget; non-decreasing in N, so cached."""
    from .release_control_prototype import clopper_pearson_upper
    cache = _THR.setdefault((W_total, budget, delta), {})
    if N in cache:
        return cache[N]
    prev = max((n for n in cache if n < N), default=None)
    c = cache[prev] if prev is not None else -1
    while clopper_pearson_upper(c + 1, N, delta) * W_total <= budget:
        c += 1
    cache[N] = c
    return c


def pass_prob(N, R, W_total, budget, delta):
    c = threshold(N, W_total, budget, delta)
    return binom_cdf(c, N, R / W_total) if c >= 0 else 0.0


def first_n(R, target, W_total, budget, delta, n_max=20000):
    N = 10
    while N <= n_max:
        if pass_prob(N, R, W_total, budget, delta) >= target:
            return N
        N += 1 if N < 1000 else 5
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--risks', type=float, nargs='*', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--W-total', type=float, default=284.0)
    ap.add_argument('--budget', type=float, default=10.0)
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--grid', type=int, nargs='*',
                    default=list(range(50, 1001, 10)) + list(range(1020, 5001, 20)))
    ap.add_argument('--out', default=None)
    ap.add_argument('--fig', default=None)
    a = ap.parse_args(argv)
    curves, table = {}, []
    for R in a.risks:
        curves[R] = [pass_prob(N, R, a.W_total, a.budget, a.delta) for N in a.grid]
        n80 = first_n(R, 0.8, a.W_total, a.budget, a.delta)
        n90 = first_n(R, 0.9, a.W_total, a.budget, a.delta)
        p360 = pass_prob(360, R, a.W_total, a.budget, a.delta)
        p1000 = pass_prob(1000, R, a.W_total, a.budget, a.delta)
        table.append({'risk': R, 'p360': p360, 'p1000': p1000, 'n80': n80, 'n90': n90})
        print(f'R={R:4.1f}/yr: pass 360 {100*p360:5.1f}%  1000 {100*p1000:5.1f}%  '
              f'first N for 80% {n80}  90% {n90}')
    if a.out:
        with open(a.out, 'w') as f:
            json.dump({'args': vars(a), 'table': table,
                       'curves': {str(k): v for k, v in curves.items()}}, f, indent=1)
    if a.fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from .plots_v7 import style
        style()
        colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
        show = [3, 5, 7, 8, 9]
        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        for i, R in enumerate([r for r in show if r in curves]):
            ax.plot(a.grid, np.array(curves[R]) * 100, lw=2, color=colors[i % len(colors)],
                    label=f'真实年风险 {R:g} 次')
        ax.axhline(80, color='#999999', lw=1, ls='--')
        ax.set_xlabel('实验室核验次数 N')
        ax.set_ylabel('预选路线通过认证的概率 (%)')
        ax.set_title('核验次数与认证通过概率（预算 10 次/年，δ = 0.05）')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        fig.savefig(a.fig, dpi=180)
        print('wrote', a.fig)


if __name__ == '__main__':
    main()
