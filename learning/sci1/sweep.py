"""
Sensitivity of the close-range bias to the detector: sweep the detection
probability at close range and the persistence of misses.

Usage (from the repository root)::

    PYTHONPATH=learning python3 -m sci1.sweep --out ../sci1_runs/<name>

The detector parameters are placeholders until calibrated from the robot's
own logs, so results are reported as curves over them, not at one setting.
"""

import argparse
import contextlib
import io
import json
import os

from .experiments import parse_args as experiment_args, run

METHODS = ('naive_partial', 'verified_ipw', 'oracle_full')


def sweep(out, scenarios, p_nears, corrs, reps, alpha, budget, seed):
    rows = []
    for scenario in scenarios:
        for corr in corrs:
            for p_near in p_nears:
                path = os.path.join(out, f'{scenario}_corr{corr}_pnear{p_near}')
                args = experiment_args([
                    '--scenario', scenario, '--perception', 'near_range_miss',
                    '--det-p-near', str(p_near), '--det-corr', str(corr), '--out', path,
                    '--reps', str(reps), '--no-plots', '--alphas', str(alpha),
                    '--budgets', str(budget), '--allocations', 'uniform',
                    '--methods', ','.join(METHODS), '--seed', str(seed)])
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = run(args)
                with open(os.path.join(path, 'diagnostics.json')) as f:
                    diag = json.load(f)['go']['all']
                for entry in summary:
                    if entry['method'] not in METHODS:
                        continue
                    rows.append(dict(scenario=scenario, det_corr=corr, det_p_near=p_near,
                                     method=entry['method'],
                                     coverage=entry['coverage_all'],
                                     coverage_std=entry['coverage_all_std'],
                                     near_miss_rate=entry['near_miss_rate'],
                                     near_miss_rate_std=entry['near_miss_rate_std'],
                                     time_lost_s=entry['time_lost_s'],
                                     go_logged_as_safe=diag['recorded_as_safe'],
                                     go_closest_undetected=diag['closest_undetected']))
                r = {e['method']: e for e in rows[-len(METHODS):]}
                print(f'{scenario:<10} corr {corr:.1f} p_near {p_near:.1f}  '
                      f'logged-safe {diag["recorded_as_safe"]:.2f}  near-miss % '
                      + ' / '.join(f'{100 * r[m]["near_miss_rate"]:.2f}' for m in METHODS),
                      flush=True)
    with open(os.path.join(out, 'sweep.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    return rows


def main(argv=None):
    def floats(text):
        return [float(x) for x in text.split(',')]

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--out', required=True)
    parser.add_argument('--scenarios', default='occlusion,wall_hug')
    parser.add_argument('--p-near', type=floats, default=[1.0, 0.8, 0.5, 0.2])
    parser.add_argument('--corr', type=floats, default=[0.0, 0.8])
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--budget', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rows = sweep(args.out, args.scenarios.split(','), args.p_near, args.corr, args.reps,
                 args.alpha, args.budget, args.seed)
    if not args.no_plots:
        from .plots import detection_sweep
        detection_sweep(rows, args.alpha, os.path.join(args.out, 'fig4_detection_sweep.png'))


if __name__ == '__main__':
    main()
