"""
SCI-1 2D experiments: coverage by trajectory type and the coverage-efficiency trade-off.

Usage (from the repository root)::

    PYTHONPATH=learning python3 -m sci1.experiments --out ../sci1_runs/first

Writes ``rows.jsonl`` (one line per repetition x setting), ``summary.csv``
(mean and std over repetitions), ``config.json``, one example calibration
log, and the figures unless ``--no-plots`` is given.
"""

import argparse
import csv
import dataclasses
import json
import os
import time

import numpy as np

from .collect import collect, reverify
from .conformal import calibration_table, lower_bounds, METHODS
from .diagnostics import detection_diagnostics, format_diagnostics
from .planners import select_action, worst_case_bounds
from .policies import (ALLOCATIONS, epsilon_propensities, nominal_action,
                       VerificationPolicy)
from .predictors import ConstantVelocityPhantom, predicted_clearance
from .sensors import SENSORS
from .sim2d import (ACTION_SPEED, ACTIONS, clearance, episode_seeds, GO, perceive,
                    PERCEPTION, sample_episodes, Scenario, SCENARIOS, simulate_branches,
                    trajectory_type, TRAJECTORY_TYPES)

DET_FLAGS = ('det_p_far', 'det_near_range', 'det_min_range', 'det_p_near', 'det_corr')


def parse_settings(items):
    """``key=value`` pairs for any Scenario field, converted to the field's type."""
    types = {f.name: f.type for f in dataclasses.fields(Scenario)}
    out = {}
    for item in items or []:
        key, _, value = item.partition('=')
        if key not in types:
            raise ValueError(f'unknown scenario field {key!r}')
        kind = types[key] if isinstance(types[key], type) else eval(types[key])
        out[key] = value if kind is str else kind(value)
    return out


def make_scenario(args):
    """Scenario from the CLI: geometry preset, perception preset, overrides."""
    overrides = dict(PERCEPTION[args.perception])
    overrides.update({k: getattr(args, k) for k in DET_FLAGS if getattr(args, k) is not None})
    overrides['sensor'] = args.sensor
    overrides.update(parse_settings(args.set))
    if args.wall_setback is not None:
        if args.scenario == 'meeting':
            raise ValueError('--wall-setback does not apply to the meeting scenario')
        overrides['wall_setback'] = args.wall_setback
    return SCENARIOS[args.scenario](**overrides)


def build_test_set(scn, predictor, n, seed, epsilon):
    """Held-out episodes with every branch simulated (ground truth for all actions)."""
    world = sample_episodes(scn, episode_seeds(seed, n))
    state = perceive(scn, world)
    robots, agents, dets = simulate_branches(scn, world)
    d_hat = predicted_clearance(scn, state, predictor)
    candidate = np.ones(d_hat.shape, dtype=bool)
    mu = epsilon_propensities(nominal_action(d_hat, state['hidden0']), candidate, epsilon)
    return {
        'state': state,
        'stratum': state['hidden0'].astype(int),
        'd_hat': d_hat,
        'd_true': clearance(robots, agents),
        'mu': mu,
        'types': trajectory_type(dets),
        'progress': robots[:, :, -1, 0] - robots[:, :, 0, 0],
        'worst_case': worst_case_bounds(scn, state),
        'robots': robots,
        'agents': agents,
        'dets': dets,
    }


def evaluate(bounds, test, d_safe):
    """Safety, coverage and efficiency of the planner driven by ``bounds``."""
    n = len(test['stratum'])
    rows = np.arange(n)
    choice = select_action(bounds, d_safe)
    covered = test['d_true'] >= bounds
    chosen_d = test['d_true'][rows, choice]
    certified = bounds[rows, choice] >= d_safe
    out = {
        'coverage_all': float(covered.mean()),
        'coverage_selected': float(covered[rows, choice].mean()),
        'near_miss_rate': float((chosen_d < d_safe).mean()),
        # Predicted safe, actually a near miss: the optimistic-bias rate (H-A).
        'optimistic_rate': float((certified & (chosen_d < d_safe)).mean()),
        'stop_rate': float((choice == ACTIONS.index('yield')).mean()),
        'time_lost_s': float((test['progress'][:, GO] - test['progress'][rows, choice]).mean()
                             / ACTION_SPEED[GO]),
    }
    for stratum, label in ((0, 'visible0'), (1, 'hidden0')):
        mask = test['stratum'] == stratum
        out[f'coverage_{label}'] = float(covered[mask].mean()) if mask.any() else None
        out[f'near_miss_rate_{label}'] = float((chosen_d[mask] < d_safe).mean()) if mask.any() else None
        out[f'optimistic_rate_{label}'] = float(
            (certified[mask] & (chosen_d[mask] < d_safe)).mean()) if mask.any() else None
        out[f'time_lost_s_{label}'] = float(
            (test['progress'][mask, GO] - test['progress'][rows[mask], choice[mask]]).mean()
            / ACTION_SPEED[GO]) if mask.any() else None
    for k, label in enumerate(TRAJECTORY_TYPES):
        mask = test['types'] == k
        out[f'coverage_{label}'] = float(covered[mask].mean()) if mask.any() else None
    for a, name in enumerate(ACTIONS):
        out[f'share_{name}'] = float((choice == a).mean())
    return out


def run(args):
    scn = make_scenario(args)
    predictor = ConstantVelocityPhantom(scn)
    os.makedirs(args.out, exist_ok=True)
    config = dict(vars(args), scenario_params=scn.to_dict(), predictor=predictor.name)
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    started = time.time()
    test = build_test_set(scn, predictor, args.n_test, args.seed, args.epsilon)
    rows = []
    base = {'method': 'worst_case', 'alpha': None, 'budget': None,
            'allocation': None, 'rep': None}
    rows.append(dict(base, **evaluate(test['worst_case'], test, args.d_safe)))
    print(f'test set: {args.n_test} episodes, hidden share '
          f'{test["stratum"].mean():.2f}, built in {time.time() - started:.1f} s')
    diag = detection_diagnostics(scn, test['robots'], test['agents'], test['dets'],
                                 test['state']['hidden0'], args.d_safe)
    with open(os.path.join(args.out, 'diagnostics.json'), 'w') as f:
        json.dump(diag, f, indent=2)
    print(format_diagnostics(diag))

    rng = np.random.default_rng(args.seed + 1)
    for rep in range(args.reps):
        log = collect(scn, episode_seeds(args.seed + 1000 + rep, args.n_cal), predictor,
                      VerificationPolicy(1.0, 1.0), args.epsilon, rng)
        uniforms = rng.random(len(log))
        hidden_share = float(log['hidden0'].mean())
        for budget in args.budgets:
            for allocation in args.allocations:
                policy = VerificationPolicy.from_budget(budget, hidden_share, allocation)
                verified_log = reverify(log, policy, uniforms)
                if rep == 0 and budget == args.budgets[0] and allocation == args.allocations[0]:
                    verified_log.save(os.path.join(args.out, 'example_calibration_log.npz'))
                cal = calibration_table(verified_log, scn, predictor)
                test_in = {'d_hat': test['d_hat'], 'stratum': test['stratum'],
                           'mu': test['mu'], 'rho': policy(test['state']['hidden0'])}
                for method in args.methods:
                    for alpha in args.alphas:
                        bounds, n_eff = lower_bounds(method, cal, test_in, alpha)
                        row = {'method': method, 'alpha': alpha, 'budget': budget,
                               'allocation': allocation, 'rep': rep,
                               'rho_visible': policy.rho_visible,
                               'rho_hidden': policy.rho_hidden,
                               'verified_share': float(verified_log['verified'].mean()),
                               'n_eff_min': min(n_eff.values()) if n_eff else 0.0}
                        row.update(evaluate(bounds, test, args.d_safe))
                        rows.append(row)
        print(f'rep {rep + 1}/{args.reps} done, {time.time() - started:.1f} s')

    with open(os.path.join(args.out, 'rows.jsonl'), 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    summary = summarize(rows)
    write_csv(os.path.join(args.out, 'summary.csv'), summary)
    print_headline(summary, args)
    if not args.no_plots:
        from .plots import make_figures
        make_figures(summary, args)
    return summary


KEYS = ('method', 'alpha', 'budget', 'allocation')


def summarize(rows):
    """Mean and std over repetitions for every numeric metric."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in KEYS), []).append(row)
    out = []
    for key, members in groups.items():
        entry = dict(zip(KEYS, key), reps=len(members))
        for metric in members[0]:
            if metric in KEYS or metric == 'rep':
                continue
            values = [m[metric] for m in members if m[metric] is not None]
            if values and isinstance(values[0], (int, float)):
                entry[metric] = float(np.mean(values))
                entry[metric + '_std'] = float(np.std(values))
        out.append(entry)
    return out


def write_csv(path, summary):
    fields = []
    for entry in summary:
        fields += [k for k in entry if k not in fields]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def print_headline(summary, args):
    alpha = 0.1 if 0.1 in args.alphas else args.alphas[0]
    budget = args.budgets[len(args.budgets) // 2]
    print(f'\nalpha = {alpha}, budget = {budget} (target coverage {1 - alpha:.2f}); '
          f'B0 = worst case')
    header = (f'{"method":<15}{"alloc":<14}{"cov hidden0":>12}{"cov emerging":>13}'
              f'{"cov lost":>9}{"near miss":>10}{"optim.":>8}{"stop":>7}{"time lost":>10}')
    print(header)
    for entry in summary:
        if entry['method'] != 'worst_case' and (entry['alpha'] != alpha or entry['budget'] != budget):
            continue
        print(f'{entry["method"]:<15}{str(entry["allocation"]):<14}'
              f'{entry.get("coverage_hidden0", float("nan")):>12.3f}'
              f'{entry.get("coverage_emerging", float("nan")):>13.3f}'
              f'{entry.get("coverage_lost_from_view", float("nan")):>9.3f}'
              f'{entry["near_miss_rate"]:>10.3f}{entry["optimistic_rate"]:>8.3f}'
              f'{entry["stop_rate"]:>7.2f}'
              f'{entry["time_lost_s"]:>9.2f}s')


def parse_args(argv=None):
    def floats(text):
        return [float(x) for x in text.split(',')]

    def words(text):
        return [x for x in text.split(',') if x]

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--scenario', default='occlusion', choices=sorted(SCENARIOS))
    parser.add_argument('--wall-setback', type=float, default=None,
                        help='corner wall setback (m); smaller = more often hidden '
                             '(default: the scenario\'s own, 1.75 for occlusion)')
    parser.add_argument('--perception', default='ideal', choices=sorted(PERCEPTION),
                        help='detection preset; individual --det-* flags override it')
    for flag in DET_FLAGS:
        parser.add_argument('--' + flag.replace('_', '-'), type=float, default=None)
    parser.add_argument('--sensor', default='abstract', choices=SENSORS,
                        help='abstract: the --perception model; camera / lidar / fusion: '
                             'sensor-specific detection (sensors.py)')
    parser.add_argument('--set', action='append', metavar='FIELD=VALUE',
                        help='override any Scenario field, e.g. --set cam_hfov_deg=87')
    parser.add_argument('--n-cal', type=int, default=4000)
    parser.add_argument('--n-test', type=int, default=4000)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--alphas', type=floats, default=[0.3, 0.2, 0.1, 0.05])
    parser.add_argument('--budgets', type=floats, default=[0.05, 0.1, 0.2, 0.4])
    parser.add_argument('--allocations', type=words, default=list(ALLOCATIONS))
    parser.add_argument('--methods', type=words, default=list(METHODS))
    parser.add_argument('--epsilon', type=float, default=0.15)
    parser.add_argument('--d-safe', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--no-plots', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
