"""
SCI-1 v7 comparison of certification methods on the whole-patrol task.

Per pass mode, from the same calibration data:
- ours:   Clopper-Pearson on verified encounters (rejection-sampling reduction);
- oursB:  characterisation B decomposition: onboard rate (all logged
          encounters, including routine onboard-only logs) + missed-event rate
          (verified encounters);
- point:  plug-in rate from verified encounters, no margin;
- ppi:    prediction-powered CLT bound, onboard label as predictor;
- ppi++:  power-tuned PPI (CLT);
- ppiBet: finite-sample PPI, betting bound on the IPW pseudo-outcome;
- active: active statistical inference (risk-flag verification probabilities,
          same expected budget), CLT bound;
- actBet: the same active verification with the finite-sample betting bound
          (candidate for our method, not a published baseline);
- rg:     Rogan-Gladen correction with exact bounds (P(log) / sensitivity);
- naive:  Clopper-Pearson on the onboard log alone.
Plus oracle (true rates), fixed modes, and B0: the fastest mode with no close
call in a worst-case encounter (inattentive worker at 1.8 m/s on the
centreline, ideal detection), else stop-and-confirm (crawl + 2 s stop).

Each method's per-mode bounds feed the same allocation (annual budget F);
the plan is scored with the true rates. Violation = certified plan whose true
annual count exceeds F, over all repetitions (target <= delta).

Run: PYTHONPATH=learning python -m sci1.corner_compare
"""

import argparse
import json

import numpy as np

from .certify import (active_clt_upper, active_probs, allocate, betting_upper,
                      decomposed_upper, ipw_pseudo, naive_upper, point_estimate,
                      ppi_betting_upper, ppi_pp_upper, ppi_upper, rate_upper,
                      rogan_gladen_upper)
from .corner_pass import CAMERA_CONFIGS, CornerScenario, simulate
from .corner_patrol import MODES, SECONDS, VARIANTS, evaluate, outcome_pools, platform

METHODS = ('ours', 'oursB', 'point', 'ppi', 'ppi++', 'ppiBet', 'active', 'actBet', 'rg', 'naive')
STOP_SECONDS = SECONDS['crawl'] + 2.0
WORST_CASE = dict(attentive_share=0.0, person_speed_median=1.8, person_speed_sigma=0.0,
                  react_range_inattentive=(0.6, 0.6), person_lateral=0.0)


def b0_mode(camera, n=500, seed=11):
    """Fastest mode with zero moving close calls in the worst-case encounter, else 'stop'."""
    scn = CornerScenario(cameras=CAMERA_CONFIGS[camera], extrapolate=True, **WORST_CASE)
    for j, mode in enumerate(MODES):
        out = simulate(scn, mode, n, seed + 1000 * j)
        if not out['near_moving'][out['encounter']].any():
            return mode
    return 'stop'


def calibration(pools, n_cal, rho, mix, onboard_factor, rng):
    probs = np.asarray(mix, float)
    action = rng.choice(len(MODES), n_cal, p=probs)
    verified = rng.random(n_cal) < rho
    loss = np.zeros(n_cal, bool)
    onboard = np.zeros(n_cal, bool)
    flag = np.zeros(n_cal, bool)
    extra = {}
    for a, m in enumerate(MODES):
        idx = np.flatnonzero(action == a)
        draw = rng.integers(0, len(pools[m][0]), len(idx))
        loss[idx], onboard[idx], flag[idx] = pools[m][0][draw], pools[m][1][draw], pools[m][2][draw]
        # Routine onboard-only logs of the same mode (no verification).
        extra[m] = pools[m][1][rng.integers(0, len(pools[m][1]), onboard_factor * len(idx))]
    # Active verification: drawn from the robot's own log before the label is seen.
    pi = active_probs(onboard, flag, rho)
    verified_act = rng.random(n_cal) < pi
    return action, probs[action], verified, loss, onboard, extra, pi, verified_act


def bounds(action, mu, rho, verified, loss, onboard, extra, pi, verified_act, d, rng):
    out = {k: {} for k in METHODS}
    rho_v = np.full(len(action), rho)
    for a, m in enumerate(MODES):
        sel = action == a
        v = sel & verified
        on_all = np.concatenate([onboard[sel], extra[m]])
        out['ours'][m] = rate_upper(action, mu, rho_v, verified, loss, a, d, rng).upper
        out['oursB'][m] = decomposed_upper(on_all, loss[v], onboard[v], d)
        out['point'][m] = point_estimate(loss[v])
        out['ppi'][m] = ppi_upper(on_all, loss[v], onboard[v], d)
        out['ppi++'][m] = ppi_pp_upper(on_all, loss[v], onboard[v], d)
        out['ppiBet'][m] = ppi_betting_upper(onboard[sel], verified[sel], loss[sel], rho, d, rng)
        z = ipw_pseudo(onboard[sel], verified_act[sel], loss[sel], np.maximum(pi[sel], 1e-12))
        out['active'][m] = active_clt_upper(z, d)
        pmin = pi[sel][~onboard[sel]].min() if (~onboard[sel]).any() else 1.0
        out['actBet'][m] = betting_upper(z, 1.0 / max(pmin, 1e-12), d, rng)
        out['rg'][m] = rogan_gladen_upper(on_all, loss[v], onboard[v], d)
        out['naive'][m] = naive_upper(action, onboard, a, d).upper
    return out


def run_case(pools, E, n_cal, rho, mix, onboard_factor, budget, delta, reps, seed):
    rng = np.random.default_rng(seed)
    q_true = {m: float(pools[m][0].mean()) for m in MODES}
    d = delta / len(MODES)
    rows = {k: [] for k in METHODS}
    qs = {k: {m: [] for m in MODES} for k in METHODS}
    for _ in range(reps):
        action, mu, verified, loss, onboard, extra, pi, verified_act = calibration(
            pools, n_cal, rho, mix, onboard_factor, rng)
        per_mode = bounds(action, mu, rho, verified, loss, onboard, extra, pi, verified_act,
                          d, rng)
        for name, q in per_mode.items():
            plan, _ = allocate(E, q, SECONDS, budget)
            if plan is None:
                rows[name].append((False, float(E.sum() * q_true['crawl']),
                                   float(len(E) * STOP_SECONDS)))
            else:
                risk, sec = evaluate(plan, E, q_true)
                rows[name].append((True, risk, sec))
            for m in MODES:
                qs[name][m].append(q[m])
    summary = {'q_true': q_true}
    for name in METHODS:
        r = rows[name]
        feas = [x for x in r if x[0]]
        summary[name] = {
            'feasible': len(feas) / len(r),
            'violation': sum(x[0] and x[1] > budget for x in r) / len(r),
            'true_risk': float(np.mean([x[1] for x in feas])) if feas else None,
            'seconds': float(np.mean([x[2] for x in r])),   # infeasible -> stop everywhere
            'q_median': {m: float(np.median(qs[name][m])) for m in MODES},
        }
    oracle_plan, _ = allocate(E, q_true, SECONDS, budget)
    o_risk, o_sec = evaluate(oracle_plan, E, q_true)
    summary['oracle'] = {'feasible': oracle_plan is not None, 'true_risk': o_risk,
                         'seconds': o_sec if oracle_plan else float(len(E) * STOP_SECONDS)}
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
    ap.add_argument('--n-cal', type=int, nargs='*', default=[1200])
    ap.add_argument('--rho', type=float, nargs='*', default=[1.0, 0.3, 0.1])
    ap.add_argument('--mix', type=float, nargs=4, default=[0.1, 0.2, 0.35, 0.35])
    ap.add_argument('--onboard-factor', type=int, default=10,
                    help='routine onboard-only encounters per calibration encounter')
    ap.add_argument('--cameras', nargs='*', default=['front_87_h0.5', 'front_120_h0.5'])
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    E = platform(a.corners, a.mean_p, a.sigma, a.seed)
    print(f'{a.corners} corners, {E.sum():.0f} encounters/year; budget {a.budget}/year, '
          f'delta {a.delta}, {a.reps} reps, mix {a.mix}, onboard x{a.onboard_factor}')
    print('violate = P(certified plan truly exceeds budget); time = seconds/day '
          '(infeasible -> stop everywhere, %.0f s)' % (len(E) * STOP_SECONDS))
    results = []
    for cam in a.cameras:
        b0 = b0_mode(cam)
        for var in a.variants:
            pools = outcome_pools(cam, var, a.pool, a.seed)
            q = {m: pools[m][0].mean() for m in MODES}
            b0_q = q['crawl'] if b0 == 'stop' else q[b0]
            b0_sec = STOP_SECONDS if b0 == 'stop' else SECONDS[b0]
            print(f'\n== {cam} / {var}: true q ' + ', '.join(f'{m} {100*q[m]:.1f}%' for m in MODES))
            print(f'   B0 ({b0} everywhere): risk <= {E.sum()*b0_q:.1f}  time {len(E)*b0_sec:.0f}s;  '
                  + ', '.join(f'all-{m} risk {E.sum()*q[m]:.1f}' for m in MODES))
            for n_cal in a.n_cal:
                for rho in a.rho:
                    s = run_case(pools, E, n_cal, rho, a.mix, a.onboard_factor, a.budget,
                                 a.delta, a.reps, a.seed + n_cal + int(100 * rho))
                    o = s['oracle']
                    print(f'   n={n_cal} rho={rho:.1f}  oracle: risk {o["true_risk"] or 0:.1f} '
                          f'time {o["seconds"]:.0f}s')
                    for name in METHODS:
                        r = s[name]
                        risk = f'{r["true_risk"]:5.1f}' if r['true_risk'] is not None else '   - '
                        qm = ' '.join(f'{100*r["q_median"][m]:5.1f}' for m in MODES)
                        print(f'      {name:>6}: feasible {100*r["feasible"]:3.0f}%  '
                              f'violate {100*r["violation"]:5.1f}%  risk {risk}  '
                              f'time {r["seconds"]:5.0f}s  | q% {qm}')
                    results.append({'camera': cam, 'variant': var, 'n_cal': n_cal, 'rho': rho,
                                    'b0': b0, **s})
    if a.out:
        with open(a.out, 'w') as f:
            json.dump({'args': vars(a), 'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
