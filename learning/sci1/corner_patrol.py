"""
SCI-1 v7 whole-patrol experiment: certify per-mode close-call rates from a
calibration set, allocate pass modes over a platform's corners so that the
annual close-call bound stays below F, and check the plan against the true
rates.

Platform (placeholders until CCTV counts exist): K corners passed once a day;
per-corner probability that a worker is coming, lognormal with a given mean;
E_k = 365 * p_k yearly encounters.

Calibration (lab style): n_cal staged encounters; pass mode drawn uniformly
(known mu); each verified with probability rho before the outcome. Per-mode
bounds use delta / (number of modes).

Methods: ours (verified labels), naive (onboard log), oracle (true rates).
Baselines: every corner at one fixed mode.

Run: PYTHONPATH=learning python -m sci1.corner_patrol
"""

import argparse
import json

import numpy as np

from .certify import allocate, naive_upper, rate_upper
from .corner_pass import CAMERA_CONFIGS, CornerScenario, simulate

MODES = ('go', 'medium', 'slow', 'crawl')
APPROACH_M = 4.0                                     # decision point to corner exit
SECONDS = {'go': APPROACH_M / 1.0, 'medium': APPROACH_M / 0.7,
           'slow': APPROACH_M / 0.5, 'crawl': APPROACH_M / 0.3}
GZ_RAMPS = ((87.0, 0.06, 0.49), (120.0, 0.19, 0.63))
VARIANTS = {
    'ideal+extrap': {'extrapolate': True},
    # Recognition ramps measured with YOLOv8s on Gazebo renders (go + slow replay):
    'gz+extrap': {'cam_ramps': GZ_RAMPS, 'extrapolate': True},
    'gz+confirm3+extrap': {'cam_ramps': GZ_RAMPS, 'confirm_frames': 3, 'extrapolate': True},
    'realistic+extrap': {'recog_latency': 0.3, 'confirm_frames': 3, 'cam_area_lo': 0.4,
                         'cam_area_hi': 0.8, 'turn_blur': 0.5, 'extrapolate': True},
}
CAMERAS = ('front_87_h0.5', 'front_120_h0.5', 'ring4_87_h0.5')


def outcome_pools(camera, variant, n, seed):
    """Per mode: (true close call, onboard close call) for encounters only."""
    scn = CornerScenario(cameras=CAMERA_CONFIGS[camera], **VARIANTS[variant])
    pools = {}
    for j, mode in enumerate(MODES):
        out = simulate(scn, mode, n, seed + 1000 * j)
        enc = out['encounter']
        pools[mode] = (out['near_moving'][enc], out['near_moving_logged'][enc])
    return pools


def platform(corners, mean_p, sigma, seed):
    rng = np.random.default_rng(seed)
    p = rng.lognormal(np.log(mean_p) - sigma ** 2 / 2, sigma, corners)
    return 365.0 * np.clip(p, 0.0, 0.05)


def evaluate(plan, E, q_true):
    if plan is None:
        return None, None
    true_risk = float(sum(e * q_true[m] for e, m in zip(E, plan)))
    seconds = float(sum(SECONDS[m] for m in plan))
    return true_risk, seconds


def run_case(pools, E, n_cal, rho, budget, delta, reps, seed, mix=None):
    rng = np.random.default_rng(seed)
    q_true = {m: float(pools[m][0].mean()) for m in MODES}
    d = delta / len(MODES)
    stats = {k: [] for k in ('ours', 'naive')}
    q_hist = {k: {m: [] for m in MODES} for k in stats}
    for _ in range(reps):
        probs = np.full(len(MODES), 1.0 / len(MODES)) if mix is None else np.asarray(mix, float)
        action = rng.choice(len(MODES), n_cal, p=probs)
        mu = probs[action]
        rho_v = np.full(n_cal, rho)
        verified = rng.random(n_cal) < rho
        loss = np.zeros(n_cal, bool)
        onboard = np.zeros(n_cal, bool)
        for a, m in enumerate(MODES):
            idx = np.flatnonzero(action == a)
            draw = rng.integers(0, len(pools[m][0]), len(idx))
            loss[idx], onboard[idx] = pools[m][0][draw], pools[m][1][draw]
        q_ours = {m: rate_upper(action, mu, rho_v, verified, loss, a, d, rng).upper
                  for a, m in enumerate(MODES)}
        q_naive = {m: naive_upper(action, onboard, a, d).upper for a, m in enumerate(MODES)}
        for name, q in (('ours', q_ours), ('naive', q_naive)):
            plan, claimed = allocate(E, q, SECONDS, budget)
            true_risk, seconds = evaluate(plan, E, q_true)
            stats[name].append((plan is not None, claimed, true_risk, seconds))
            if plan is None:   # fall back to the slowest mode everywhere (uncertified)
                stats[name][-1] = (False, claimed, float(E.sum() * q_true[MODES[-1]]),
                                   float(len(E) * SECONDS[MODES[-1]]))
            for m in MODES:
                q_hist[name][m].append(q[m])
    oracle_plan, _ = allocate(E, q_true, SECONDS, budget)
    o_risk, o_sec = evaluate(oracle_plan, E, q_true)
    fixed = {m: (float(E.sum() * q_true[m]), float(len(E) * SECONDS[m])) for m in MODES}
    summary = {'q_true': q_true, 'oracle': {'feasible': oracle_plan is not None,
                                            'true_risk': o_risk, 'seconds': o_sec},
               'fixed': fixed}
    for name, rows in stats.items():
        feas = [r for r in rows if r[0]]
        summary[name] = {
            'feasible': len(feas) / len(rows),
            # Guarantee: P(certified plan exceeds the budget) <= delta, over all reps.
            'violation': sum(r[0] and r[2] > budget for r in rows) / len(rows),
            'violation_given_feasible': (sum(r[2] > budget for r in feas) / len(feas)) if feas else None,
            'true_risk': float(np.mean([r[2] for r in feas])) if feas else None,
            'seconds': float(np.mean([r[3] for r in feas])) if feas else None,
            'q_upper_median': {m: float(np.median(q_hist[name][m])) for m in MODES},
        }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=20000)
    ap.add_argument('--corners', type=int, default=160)
    ap.add_argument('--mean-p', type=float, default=0.006)
    ap.add_argument('--sigma', type=float, default=0.8)
    ap.add_argument('--budget', type=float, default=10.0)
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--n-cal', type=int, nargs='*', default=[400, 1200, 2400])
    ap.add_argument('--rho', type=float, nargs='*', default=[1.0, 0.3])
    ap.add_argument('--mix', type=float, nargs=4, default=None,
                    help='calibration probabilities for go medium slow crawl (default uniform)')
    ap.add_argument('--cameras', nargs='*', default=list(CAMERAS))
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    E = platform(a.corners, a.mean_p, a.sigma, a.seed)
    print(f'{a.corners} corners, {E.sum():.0f} encounters/year (max {E.max():.1f} per corner); '
          f'budget {a.budget}/year, delta {a.delta}, {a.reps} reps')
    print('time = corner passing seconds per day; risk = true close calls per year')
    results = []
    for cam in a.cameras:
        for var in a.variants:
            pools = outcome_pools(cam, var, a.pool, a.seed)
            q = {m: pools[m][0].mean() for m in MODES}
            print(f'\n== {cam} / {var}: true q ' + ', '.join(f'{m} {100*q[m]:.1f}%' for m in MODES))
            first = True
            for n_cal in a.n_cal:
                for rho in a.rho:
                    s = run_case(pools, E, n_cal, rho, a.budget, a.delta, a.reps,
                                 a.seed + n_cal + int(100 * rho), a.mix)
                    if first:
                        fx = s['fixed']
                        print('   fixed: ' + ', '.join(f'all-{m} risk {fx[m][0]:.1f} time {fx[m][1]:.0f}s'
                                                     for m in MODES))
                        o = s['oracle']
                        print(f'   oracle: ' + (f'risk {o["true_risk"]:.1f} time {o["seconds"]:.0f}s'
                                                if o['feasible'] else 'infeasible'))
                        first = False
                    for name in ('ours', 'naive'):
                        r = s[name]
                        body = (f'feasible {100*r["feasible"]:3.0f}%  violate {100*r["violation"]:4.1f}%  '
                                f'risk {r["true_risk"]:5.1f}  time {r["seconds"]:4.0f}s'
                                if r['feasible'] else 'infeasible in all reps')
                        qs = ' '.join(f'{100*r["q_upper_median"][m]:.1f}' for m in MODES)
                        print(f'   n={n_cal:5d} rho={rho:.1f} {name:>5}: {body}  | q_up% {qs}')
                    results.append({'camera': cam, 'variant': var, 'n_cal': n_cal, 'rho': rho, **s})
    if a.out:
        with open(a.out, 'w') as f:
            json.dump({'args': vars(a), 'encounters_per_year': E.tolist(), 'results': results},
                      f, indent=1)


if __name__ == '__main__':
    main()
