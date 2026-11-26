"""
SCI-1 v7e: does the margin rule frozen on v7d-3 transfer to held-out conditions?

Rule (frozen 2026-11-25 from v7d-3, minimax and mean regret over 12 training
cells): choose the preselected route under the prior inflated by m = 2.

Held-out conditions (none used to pick m):
  H1  120 deg camera, A corners: distracted and late-reacting workers
  H2  87 deg camera, A corners: backlight raises the recognition thresholds
      (the nominal simulator uses the Gazebo-measured thresholds everywhere)
  H3  87 deg camera, late-reacting factor, 60% of corners are type A
For each: priors correct / biased (A treated as B) / mix, N = 1000 / 2000,
margins m = 1, 1.5, 2, 3; regret of the frozen rule against the best margin.
"""

import argparse
import json

import numpy as np

from .corner_pass import CAMERA_CONFIGS, CornerScenario, simulate
from .corner_patrol import MODES, SECONDS, VARIANTS
from .corner_route import B_TRUTH, SEC, STOP_SECONDS, exposure, plan_for
from .corner_route2 import run_config

FROZEN_M = 2.0
MARGINS = (1.0, 1.5, 2.0, 3.0)
LATE = {'attentive_share': 0.2, 'react_range_inattentive': (0.45, 0.9)}
BACKLIGHT = {'attentive_share': 0.5, 'cam_ramps': ((87.0, 0.25, 0.75), (120.0, 0.35, 0.85))}
HELD_OUT = {
    'H1_120deg_late': ('front_120_h0.5', LATE, 0.35),
    'H2_87deg_backlight': ('front_87_h0.5', BACKLIGHT, 0.35),
    'H3_87deg_late_shareA0.6': ('front_87_h0.5', LATE, 0.6),
}


def rates(camera, variant, behaviour, n, seed):
    kw = {**VARIANTS[variant], **behaviour}
    scn = CornerScenario(cameras=CAMERA_CONFIGS[camera], **kw)
    q = []
    for j, mode in enumerate(MODES):
        out = simulate(scn, mode, n, seed + 1000 * j)
        q.append(float(out['near_moving'][out['encounter']].mean()))
    return np.array(q)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', nargs='*', default=list(HELD_OUT))
    ap.add_argument('--variant', default='gz+extrap')
    ap.add_argument('--priors', nargs='*', default=['correct', 'biased', 'mix'])
    ap.add_argument('--N', type=int, nargs='*', default=[1000, 2000])
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--pool', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--budget', type=float, default=10.0)
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    results = []
    for cname in a.conditions:
        camera, beh_A, share_A = HELD_OUT[cname]
        E = exposure(160, 0.006, 0.8, a.seed, 6.0)
        ctype = (np.random.default_rng(a.seed + 1).random(160) >= share_A).astype(int)
        truth = np.concatenate([rates(camera, a.variant, beh_A, a.pool, a.seed),
                                rates(camera, a.variant, B_TRUTH, a.pool, a.seed)])
        ok = np.concatenate([rates(camera, a.variant, beh_A, a.pool, a.seed + 50000),
                             rates(camera, a.variant, B_TRUTH, a.pool, a.seed + 51000)])
        nomB = rates(camera, a.variant, B_TRUTH, a.pool, a.seed + 90000)
        preds = {'correct': ok, 'biased': np.concatenate([nomB, nomB])}
        preds['mix'] = (preds['correct'] + preds['biased']) / 2
        oracle = plan_for(truth, E, ctype, a.budget)
        print(f'\n== {cname}: true q A ' + ' '.join(f'{100*v:.2f}%' for v in truth[:4])
              + ' | B ' + ' '.join(f'{100*v:.2f}%' for v in truth[4:])
              + f'; oracle {SEC[oracle].sum():.0f}s', flush=True)
        for pname in a.priors:
            for N in a.N:
                row = {}
                for m in MARGINS:
                    row[m] = run_config(truth, preds[pname], E, ctype, N, a.budget, a.delta,
                                        a.reps, a.seed + N, m, 1.0, 0.0)
                best = min(row, key=lambda m: row[m]['seconds'])
                regret = row[FROZEN_M]['seconds'] - row[best]['seconds']
                print(f'   prior={pname:7s} N={N}: ' + '  '.join(
                    f'm{m:g} {100*row[m]["certified"]:3.0f}%/{row[m]["seconds"]:.0f}s'
                    + (f' v{100*row[m]["violation"]:.0f}' if row[m]['violation'] else '')
                    for m in MARGINS) + f'  | best m{best:g}, frozen m2 regret {regret:.0f}s', flush=True)
                results.append({'condition': cname, 'prior': pname, 'N': N,
                                'by_margin': {str(m): row[m] for m in MARGINS},
                                'best_margin': best, 'frozen_regret_s': regret})
    regrets = [r['frozen_regret_s'] for r in results]
    print(f'\nfrozen m=2 regret over held-out cells: mean {np.mean(regrets):.0f}s, max {np.max(regrets):.0f}s')
    if a.out:
        with open(a.out, 'w') as f:
            json.dump({'args': vars(a), 'frozen_margin': FROZEN_M, 'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
