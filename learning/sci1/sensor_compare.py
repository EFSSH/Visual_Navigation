"""
Camera vs 2D lidar vs fusion: how the detecting sensor changes what the
robot's own log misses, and what that does to naive calibration.

Usage (from the repository root)::

    PYTHONPATH=learning python3 -m sci1.sensor_compare --out ../sci1_runs/<name>

Camera variants span the mounting choices still open for the real robot:
the Gazebo model's low narrow camera, a wider depth camera at the same
height, the wider camera raised to 0.5 m, and a front + rear pair covering
360 deg at 0.5 m.
"""

import argparse
import contextlib
import io
import json
import os
from multiprocessing import Pool

from .experiments import parse_args as experiment_args, run

METHODS = ('naive_partial', 'verified_ipw', 'oracle_full')

WIDE = ['cam_hfov_deg=87', 'cam_vfov_deg=58']
RAISED = ['cam_height=0.5', 'cam_rear_height=0.5']
CONFIGS = {
    'ideal': ('abstract', []),
    'cam_60_low': ('camera', []),
    'cam_87_low': ('camera', WIDE + ['cam_rear_height=0.11']),
    'cam_87_0.5m': ('camera', WIDE + RAISED),
    'cam_360_0.5m': ('camera', ['cam_hfov_deg=180', 'cam_vfov_deg=58'] + RAISED),
    'lidar': ('lidar', []),
    'fusion_60_low': ('fusion', []),
    'fusion_87_0.5m': ('fusion', WIDE + RAISED),
}
LABELS = {
    'ideal': 'ideal detection',
    'cam_60_low': 'camera 60°, 0.11 m (Gazebo)',
    'cam_87_low': 'camera 87°, 0.11 m',
    'cam_87_0.5m': 'camera 87°, 0.5 m',
    'cam_360_0.5m': 'camera front+rear 360°, 0.5 m',
    'lidar': '2D lidar (Gazebo)',
    'fusion_60_low': 'fusion: Gazebo camera + lidar',
    'fusion_87_0.5m': 'fusion: camera 87°, 0.5 m + lidar',
}


def _one(job):
    out, scenario, name, reps, alpha, budget, seed = job
    sensor, settings = CONFIGS[name]
    path = os.path.join(out, f'{scenario}_{name}')
    argv = ['--scenario', scenario, '--sensor', sensor, '--out', path, '--reps', str(reps),
            '--no-plots', '--alphas', str(alpha), '--budgets', str(budget),
            '--allocations', 'uniform', '--methods', ','.join(METHODS), '--seed', str(seed)]
    for item in settings:
        argv += ['--set', item]
    with contextlib.redirect_stdout(io.StringIO()):
        summary = run(experiment_args(argv))
    with open(os.path.join(path, 'diagnostics.json')) as f:
        diag = json.load(f)['go']['all']
    with open(os.path.join(path, 'diagnostics.json')) as f:
        strata = json.load(f)['go']
    row = {'scenario': scenario, 'config': name, 'sensor': sensor, 'settings': settings,
           'hidden_share': strata['hidden0']['episodes'] / strata['all']['episodes'],
           'go_near_misses': diag['near_misses'],
           'closest_undetected': diag['closest_undetected'],
           'logged_as_safe': diag['recorded_as_safe'],
           'never_detected': diag['never_detected'],
           'first_detect_dist_p50': diag['first_detect_dist_p50'],
           'lead_below_stop_time': diag['lead_below_stop_time']}
    for entry in summary:
        key = 'B0' if entry['method'] == 'worst_case' else entry['method']
        row[f'{key}_near_miss_rate'] = entry['near_miss_rate']
        row[f'{key}_time_lost_s'] = entry['time_lost_s']
        row[f'{key}_stop_rate'] = entry['stop_rate']
        if key != 'B0':
            row[f'{key}_coverage'] = entry['coverage_all']
    return row


def format_table(rows):
    head = (f'{"scenario":<10}{"sensor config":<36}{"hidden":>7}{"logged":>7}{"lead<":>7}'
            f'{"near miss % naive/verif/oracle/B0":>36}{"cov naive/verif":>17}{"lost s verif":>13}')
    lines = [head, f'{"":<46}{"at t0":>7}{"safe":>7}{"stop":>7}']
    for r in rows:
        lines.append(
            f'{r["scenario"]:<10}{LABELS[r["config"]]:<36}{r["hidden_share"]:>7.2f}'
            f'{r["logged_as_safe"]:>7.2f}'
            f'{r["lead_below_stop_time"]:>7.2f}'
            f'{100 * r["naive_partial_near_miss_rate"]:>12.1f} /{100 * r["verified_ipw_near_miss_rate"]:>5.1f} /'
            f'{100 * r["oracle_full_near_miss_rate"]:>5.1f} /{100 * r["B0_near_miss_rate"]:>5.1f}'
            f'{r["naive_partial_coverage"]:>10.3f} /{r["verified_ipw_coverage"]:>5.3f}'
            f'{r["verified_ipw_time_lost_s"]:>13.2f}')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--out', required=True)
    parser.add_argument('--scenarios', default='occlusion,wall_hug')
    parser.add_argument('--configs', default=','.join(CONFIGS))
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--budget', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    jobs = [(args.out, s, c, args.reps, args.alpha, args.budget, args.seed)
            for s in args.scenarios.split(',') for c in args.configs.split(',')]
    with Pool(args.workers) as pool:
        rows = pool.map(_one, jobs)
    with open(os.path.join(args.out, 'sensor_compare.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(format_table(rows))
    if not args.no_plots:
        from .plots import sensor_comparison
        sensor_comparison(rows, LABELS, os.path.join(args.out, 'fig5_sensor_comparison.png'))


if __name__ == '__main__':
    main()
