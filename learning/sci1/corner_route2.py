"""
SCI-1 v7d-3: two-stage route certification.

Same world as v7d-2 (sci1.corner_route, A corners with distracted / late-reacting
workers that the nominal simulator may miss).

Two-stage method, total error budget delta = delta1 + delta2:
  sampling  each lab trial independently: with prob (1 - r) draw a group with
            probability W_g(a0) / sum W(a0) (the preselected route's exposure),
            with prob r draw uniformly among the groups any candidate route uses.
            The component of every trial is recorded.
  stage 1   exact binomial test of the preselected route a0 on the route-component
            trials only (each is a Bernoulli with mean R(a0) / sum W(a0)) at delta1.
  stage 2   if stage 1 fails and delta2 > 0: per-group Clopper-Pearson with
            Bonferroni over groups at delta2 on all trials (counts per group are
            independent of outcomes), then route search; output if any route passes.
Validity: union bound over the two stages (the stage-2 bound is simultaneous over
routes, so choosing after looking at the data is covered).

The preselected route is chosen by expected time including fallback under the
prior inflated by a margin factor m (m > 1 picks safer routes).

Baselines: single-route test (delta1 = delta, r = 0, m = 1); per-group
simultaneous bound alone (v7c) with samples by the route's exposure.
"""

import argparse
import json

import numpy as np

from .certify_route import (allocate_general, cp_bounds, route_weights, single_route_threshold)
from .corner_patrol import SECONDS
from .corner_route import (A_FACTORS, B_TRUTH, SEC, STOP_SECONDS, TYPES, candidates,
                           choose_single, exposure, group_rates, plan_for)

CONFIGS = [  # (name, margin, delta1 share, coverage r)
    ('single', 1.0, 1.0, 0.0),
    ('single m1.5', 1.5, 1.0, 0.0),
    ('single m2', 2.0, 1.0, 0.0),
    ('single m3', 3.0, 1.0, 0.0),
    ('2stage .04/.01', 1.0, 0.8, 0.0),
    ('2stage .03/.02', 1.0, 0.6, 0.0),
    ('2stage .04/.01 r.15', 1.0, 0.8, 0.15),
    ('2stage .03/.02 r.15', 1.0, 0.6, 0.15),
    ('2stage .04/.01 m1.5', 1.5, 0.8, 0.0),
    ('2stage .04/.01 m1.5 r.15', 1.5, 0.8, 0.15),
    ('2stage .04/.01 m2', 2.0, 0.8, 0.0),
    ('v7c only', 1.0, 0.0, 0.0),
]


def run_config(truth, q_prior, E, ctype, N, budget, delta, reps, seed, margin, share1, r):
    rng = np.random.default_rng(seed)
    K = len(ctype)
    fallback = K * STOP_SECONDS
    crawl_time = K * SECONDS['crawl']
    G = len(truth)
    cands = candidates(np.minimum(q_prior * margin, 1), E, ctype, budget)
    n_route_exp = int(round(N * (1 - r)))
    _, pre, _, W_pre = choose_single(cands, np.minimum(q_prior * margin, 1), E, ctype,
                                     max(n_route_exp, 1), budget, delta * (share1 if share1 > 0 else 1.0),
                                     fallback)
    used = np.zeros(G, bool)
    for p in cands:
        used |= route_weights(p, E, ctype, 2, 4) > 0
    p_route = W_pre / W_pre.sum()
    p_cov = used / used.sum()
    d1, d2 = delta * share1, delta * (1 - share1)
    true_risk = lambda p: float(route_weights(p, E, ctype, 2, 4) @ truth)
    rows, rescued, slowed = [], 0, []
    stage1_fail = 0
    for _ in range(reps):
        comp_route = rng.random(N) >= r
        n1 = int(comp_route.sum())
        g1 = rng.choice(G, n1, p=p_route)
        g2 = rng.choice(G, N - n1, p=p_cov)
        y1 = rng.random(n1) < truth[g1]
        y2 = rng.random(N - n1) < truth[g2]
        chosen = None
        if d1 > 0:
            c = single_route_threshold(n1, W_pre.sum(), budget, d1)
            if c >= 0 and int(y1.sum()) <= c:
                chosen = pre
            else:
                stage1_fail += 1
        if chosen is None and d2 > 0:
            n_g = np.bincount(np.concatenate([g1, g2]), minlength=G)
            x_g = np.bincount(np.concatenate([g1[y1], g2[y2]]), minlength=G)
            q_up = cp_bounds(x_g, n_g, d2 / G)
            risk = E * q_up.reshape(2, 4)[ctype]
            plan, _ = allocate_general(risk, SEC, budget)
            if plan is not None:
                chosen = plan
                if d1 > 0 and SEC[plan].sum() < crawl_time:
                    rescued += 1
                    a = ctype == 0
                    slowed.append(((plan[a] > pre[a]).mean(), (plan[~a] > pre[~a]).mean()))
        if chosen is None:
            rows.append((False, None, fallback))
        else:
            rows.append((True, true_risk(chosen), float(SEC[chosen].sum())))
    cert = [z for z in rows if z[0]]
    sl = np.array(slowed) if slowed else None
    return {
        'pre_true_risk': true_risk(pre), 'pre_seconds': float(SEC[pre].sum()),
        'certified': len(cert) / reps,
        'violation': sum(z[1] > budget for z in cert) / reps,
        'faster_than_crawl': sum(z[2] < crawl_time for z in cert) / reps,
        'seconds': float(np.mean([z[2] for z in rows])),
        'stage1_fail': stage1_fail, 'rescued': rescued,
        'slowed_A_B': sl.mean(0).tolist() if sl is not None else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', default='front_87_h0.5')
    ap.add_argument('--variants', nargs='*', default=['gz+extrap', 'gz+confirm3+extrap'])
    ap.add_argument('--priors', nargs='*', default=['correct', 'biased', 'mix'])
    ap.add_argument('--N', type=int, nargs='*', default=[1000, 2000])
    ap.add_argument('--a-factor', default='late', choices=sorted(A_FACTORS))
    ap.add_argument('--share-A', type=float, default=0.35)
    ap.add_argument('--window', type=float, default=6.0)
    ap.add_argument('--corners', type=int, default=160)
    ap.add_argument('--mean-p', type=float, default=0.006)
    ap.add_argument('--sigma', type=float, default=0.8)
    ap.add_argument('--budget', type=float, default=10.0)
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--pool', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    E = exposure(a.corners, a.mean_p, a.sigma, a.seed, a.window)
    ctype = (np.random.default_rng(a.seed + 1).random(a.corners) >= a.share_A).astype(int)
    print(f'v7d-3 two-stage; A factor {a.a_factor}; {a.corners} corners; budget {a.budget}; '
          f'delta {a.delta}; {a.reps} reps; all-crawl {a.corners * SECONDS["crawl"]:.0f}s, '
          f'fallback {a.corners * STOP_SECONDS:.0f}s', flush=True)
    results = []
    for var in a.variants:
        beh = {'A': A_FACTORS[a.a_factor], 'B': B_TRUTH}
        truth = np.concatenate([group_rates(a.camera, var, beh[t], a.pool, a.seed) for t in TYPES])
        sim_ok = np.concatenate([group_rates(a.camera, var, beh[t], a.pool, a.seed + 50000)
                                 for t in TYPES])
        nominal_B = group_rates(a.camera, var, B_TRUTH, a.pool, a.seed + 90000)
        preds = {'correct': sim_ok, 'biased': np.concatenate([nominal_B, nominal_B])}
        preds['mix'] = (preds['correct'] + preds['biased']) / 2
        oracle = plan_for(truth, E, ctype, a.budget)
        print(f'\n== {var}: oracle {SEC[oracle].sum():.0f}s', flush=True)
        for pname in a.priors:
            for N in a.N:
                print(f'   prior={pname} N={N}', flush=True)
                for name, m, s1, r in CONFIGS:
                    res = run_config(truth, preds[pname], E, ctype, N, a.budget, a.delta, a.reps,
                                     a.seed + N, m, s1, r)
                    sl = res['slowed_A_B']
                    print(f'      {name:>26}: pre {res["pre_true_risk"]:4.1f}/{res["pre_seconds"]:.0f}s  '
                          f'cert {100*res["certified"]:5.1f}%  violate {100*res["violation"]:4.1f}%  '
                          f'time {res["seconds"]:5.0f}s  stage1 fail {res["stage1_fail"]:3d} '
                          f'rescued {res["rescued"]:3d}'
                          + (f' (slowed A {100*sl[0]:.0f}% B {100*sl[1]:.0f}%)' if sl else ''),
                          flush=True)
                    results.append({'variant': var, 'prior': pname, 'N': N, 'config': name,
                                    'margin': m, 'share1': s1, 'coverage': r, **res})
    if a.out:
        with open(a.out, 'w') as f:
            json.dump({'args': vars(a), 'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
