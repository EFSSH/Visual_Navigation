"""
One decision held vs replanning at every frame, per sensor configuration
and measurement-error level, on the mixed scenario.

Usage (from the repository root)::

    PYTHONPATH=learning python3 -m sci1.closed_loop_compare --out ../sci1_runs/<name>

For each configuration and repetition: collect a calibration log (verification
budget 0.2, uniform), then roll out the same test worlds closed loop with
each decision rule, holding the t0 action or replanning at every sensor
frame. Also records the one-decision coverage of each calibration method on
the same configuration, for reference.
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from .closed_loop import (always_go, closed_loop_metrics, conformal_decider, run_closed_loop,
                          worst_case_decider)
from .collect import collect, reverify
from .conformal import calibration_table, lower_bounds
from .experiments import build_test_set, evaluate, parse_settings
from .policies import VerificationPolicy
from .predictors import ConstantVelocityPhantom
from .sensor_compare import CONFIGS, LABELS
from .sim2d import episode_seeds, mixed_scenario, NOISE, sample_episodes

METHODS = ('naive_partial', 'verified_ipw', 'oracle_full')


def _scenario(config, noise, steps):
    sensor, settings = CONFIGS[config]
    overrides = dict(NOISE[noise], sensor=sensor, closed_loop_steps=steps)
    overrides.update(parse_settings(settings))
    return mixed_scenario(**overrides)


def _one(job):
    config, noise, args = job
    scn = _scenario(config, noise, args['steps'])
    predictor = ConstantVelocityPhantom(scn)
    test_world = sample_episodes(scn, episode_seeds(args['seed'], args['n_test']))
    one_decision = build_test_set(scn, predictor, args['n_test_single'], args['seed'] + 7,
                                  args['epsilon'])
    rows = []
    rules = {'always_go': lambda cal, pol: always_go,
             'worst_case': lambda cal, pol: worst_case_decider(scn, args['d_safe'])}
    for method in METHODS:
        rules[method] = (lambda m: lambda cal, pol: conformal_decider(
            scn, predictor, cal, m, args['alpha'], args['epsilon'], pol, args['d_safe']))(method)
    rng = np.random.default_rng(args['seed'] + 1)
    for rep in range(args['reps']):
        log = collect(scn, episode_seeds(args['seed'] + 1000 + rep, args['n_cal']), predictor,
                      VerificationPolicy(1.0, 1.0), args['epsilon'], rng)
        policy = VerificationPolicy.from_budget(args['budget'], float(log['hidden0'].mean()),
                                                'uniform')
        log = reverify(log, policy, rng.random(len(log)))
        cal = calibration_table(log, scn, predictor)
        test_in = {'d_hat': one_decision['d_hat'], 'stratum': one_decision['stratum'],
                   'mu': one_decision['mu'], 'rho': policy(one_decision['state']['hidden0'])}
        for name, make in rules.items():
            if rep > 0 and name in ('always_go', 'worst_case'):
                continue                      # do not depend on calibration
            decide = make(cal, policy)
            row = {'config': config, 'noise': noise, 'rule': name, 'rep': rep}
            if name in METHODS:
                bounds, _ = lower_bounds(name, cal, test_in, args['alpha'])
                single = evaluate(bounds, one_decision, args['d_safe'])
                row.update(single_coverage=single['coverage_all'],
                           single_near_miss_rate=single['near_miss_rate'],
                           single_time_lost_s=single['time_lost_s'])
            for replan in (False, True):
                out = run_closed_loop(scn, test_world, decide, replan)
                metrics = closed_loop_metrics(scn, out, args['d_safe'])
                mode = 'replan' if replan else 'hold'
                row.update({f'{mode}_{k}': v for k, v in metrics.items()})
            rows.append(row)
        print(f'{config} / {noise}: rep {rep + 1}/{args["reps"]} done', flush=True)
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['config'], row['noise'], row['rule']), []).append(row)
    out = []
    for (config, noise, rule), members in groups.items():
        entry = {'config': config, 'noise': noise, 'rule': rule, 'reps': len(members)}
        for key in members[0]:
            if isinstance(members[0][key], float):
                values = [m[key] for m in members]
                entry[key] = float(np.mean(values))
                entry[key + '_std'] = float(np.std(values))
        out.append(entry)
    return out


def format_table(summary):
    lines = [f'{"config":<34}{"noise":<16}{"rule":<15}{"1-dec cov":>10}'
             f'{"near miss % hold/replan":>25}{"lost s hold/replan":>20}{"stop replan":>12}']
    for e in summary:
        cov = e.get('single_coverage')
        lines.append(
            f'{LABELS[e["config"]]:<34}{e["noise"]:<16}{e["rule"]:<15}'
            f'{"" if cov is None else format(cov, ".3f"):>10}'
            f'{100 * e["hold_near_miss_rate"]:>15.1f} /{100 * e["replan_near_miss_rate"]:>5.1f}'
            f'{e["hold_time_lost_s"]:>13.2f} /{e["replan_time_lost_s"]:>5.2f}'
            f'{e["replan_stop_rate"]:>12.2f}')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--out', required=True)
    parser.add_argument('--configs', default='ideal,cam_60_low,cam_87_0.5m,cam_360_0.5m,'
                                             'fusion_87_0.5m')
    parser.add_argument('--noises', default='none,camera_typical')
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--n-cal', type=int, default=4000)
    parser.add_argument('--n-test', type=int, default=1500)
    parser.add_argument('--n-test-single', type=int, default=4000)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--budget', type=float, default=0.2)
    parser.add_argument('--epsilon', type=float, default=0.15)
    parser.add_argument('--d-safe', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    shared = {k: getattr(args, k) for k in ('reps', 'n_cal', 'n_test', 'n_test_single', 'steps',
                                            'alpha', 'budget', 'epsilon', 'd_safe', 'seed')}
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    jobs = [(c, n, shared) for c in args.configs.split(',') for n in args.noises.split(',')]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.map(_one, jobs) for r in part]
    with open(os.path.join(args.out, 'rows.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    summary = summarize(rows)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(format_table(summary))
    if not args.no_plots:
        from .plots import closed_loop_comparison
        closed_loop_comparison(summary, LABELS, os.path.join(args.out, 'fig6_closed_loop.png'))


if __name__ == '__main__':
    main()
