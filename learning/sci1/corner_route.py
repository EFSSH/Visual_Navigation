"""
SCI-1 v7d: route-level certification (v8 plan, E1 + E2) in 2D.

Setting
- 160 corners of two types. Type B behaves as the simulator assumes; at type A
  corners workers are more often distracted (attentive share 0.2 instead of
  0.5), which the nominal simulator does not know. That is the "factor the
  simulator gets wrong and that affects corner types differently" (E2).
- Groups g = (type, mode): 2 x 4. True per-encounter close-call rates come from
  large 2D pools; the lab draws x_g ~ Binomial(n_g, q_g), fully verified.
- Exposure depends on speed: E_k(m) = E_k * (T0 + d_m) / (T0 + d_go), d_m the
  corner passing time, T0 the encounter window (slower = longer dwell).
- Reference annual risk R_ref(a) = sum_k E_k(a_k) q_{type(k), a_k}; target < F.

Methods (same total lab budget N)
- shared data (one allocation: reserve r*N spread over all groups, the rest by
  the prior-best route's exposure):
  v7c    per-group Clopper-Pearson, Bonferroni over groups; any route;
  joint  likelihood-ratio region with a per-group prior (atom w0 at 0 + Beta
         mixture centred on the simulator), delta 0.04, plus a per-group CP
         safety net at 0.01; any route;
  fixseqW the same candidates, each tested with the weighted Chernoff route
         bound on the shared data (full delta, valid for any allocation);
  fixseq candidate routes fixed from the prior, conservative -> aggressive,
         each tested on its own groups (CP, Bonferroni over used groups),
         stop at first failure;
  point  plug-in rates (no margin), for reference.
- single: one route chosen from the prior by expected time including fallback;
  N encounters drawn by that route's exposure; one exact binomial test.
Rescue (E2): among repetitions where the prior-preselected route is not
certified by a shared-data method's own bound, how often does that method still
certify a route faster than all-crawl.

Run: PYTHONPATH=learning python -m sci1.corner_route --out ...
"""

import argparse
import json

import numpy as np

from .certify_route import (allocate_general, beta_prior, binom_cdf, cp_bounds, joint_upper,
                            route_chernoff_upper, route_weights, search_joint,
                            single_route_threshold)
from .corner_pass import CAMERA_CONFIGS, CornerScenario, simulate
from .corner_patrol import MODES, SECONDS, VARIANTS, platform

SEC = np.array([SECONDS[m] for m in MODES])
STOP_SECONDS = SECONDS['crawl'] + 2.0
TYPES = ('A', 'B')
# Truth per corner type; the nominal simulator assumes type B behaviour everywhere.
A_FACTORS = {
    'mild': {'attentive_share': 0.2},                                      # more distracted
    'late': {'attentive_share': 0.2, 'react_range_inattentive': (0.45, 0.9)},  # and react later
}
B_TRUTH = {'attentive_share': 0.5}


def group_rates(camera, variant, behaviour, n, seed):
    scn = CornerScenario(cameras=CAMERA_CONFIGS[camera], **behaviour, **VARIANTS[variant])
    q = []
    for j, mode in enumerate(MODES):
        out = simulate(scn, mode, n, seed + 1000 * j)
        enc = out['encounter']
        q.append(float(out['near_moving'][enc].mean()))
    return np.array(q)


def exposure(corners, mean_p, sigma, seed, window):
    E0 = platform(corners, mean_p, sigma, seed)
    factor = (window + SEC) / (window + SEC[0])
    return E0[:, None] * factor[None, :]


def plan_for(q, E, ctype, budget):
    risk = E * q.reshape(len(TYPES), len(MODES))[ctype]
    plan, _ = allocate_general(risk, SEC, budget)
    return plan


def candidates(q_prior, E, ctype, budget, scales=(2.0, 1.6, 1.3, 1.1, 1.0, 0.9, 0.8)):
    out, seen = [], set()
    for s in scales:
        p = plan_for(np.minimum(q_prior * s, 1.0), E, ctype, budget)
        if p is None:
            continue
        key = p.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out          # conservative -> aggressive


def allocation(N, reserve, W_target):
    G = len(W_target)
    n = np.full(G, int(reserve * N) // G)
    rest = N - n.sum()
    share = W_target / W_target.sum()
    extra = np.floor(share * rest).astype(int)
    extra[np.argmax(share)] += rest - extra.sum()
    return n + extra


def choose_single(cands, q_prior, E, ctype, N, budget, delta, fallback):
    best = None
    for p in cands:
        W = route_weights(p, E, ctype, len(TYPES), len(MODES))
        c = single_route_threshold(N, W.sum(), budget, delta)
        pr = binom_cdf(c, N, float(W @ q_prior / W.sum())) if c >= 0 else 0.0
        t = float(SEC[p].sum())
        exp_t = pr * t + (1 - pr) * fallback
        if best is None or exp_t < best[0]:
            best = (exp_t, p, c, W)
    return best


def run_cell(q_true, q_prior, E, ctype, N, reserve, budget, delta, reps, seed, prior):
    rng = np.random.default_rng(seed)
    K = len(ctype)
    fallback = K * STOP_SECONDS
    crawl_time = K * SECONDS['crawl']
    true_risk = lambda p: float(route_weights(p, E, ctype, 2, 4) @ q_true)
    cands = candidates(q_prior, E, ctype, budget)
    single = choose_single(cands, q_prior, E, ctype, N, budget, delta, fallback)
    _, p_single, c_single, W_single = single
    # The preselected route is the single-route choice; shared-data methods target the
    # same route for allocation (plus a reserve over all groups) so budgets are comparable.
    pre, W_pre = p_single, W_single
    n_g = allocation(N, reserve, W_pre)
    G = len(q_true)
    d_safe, d_joint = 0.01, delta - 0.01
    rows = {k: [] for k in ('v7c', 'joint', 'fixseqW', 'fixseq', 'point', 'single')}
    rescue = {k: [0, 0, []] for k in ('v7c', 'joint', 'fixseqW', 'fixseq')}   # failed pre, rescued, saved s
    slowedA = {k: [] for k in rescue}
    for _ in range(reps):
        x = rng.binomial(n_g, q_true)
        cp_all = cp_bounds(x, n_g, delta / G)
        cp_safe = cp_bounds(x, n_g, d_safe / G)

        def up_v7c(W):
            return float(W @ cp_all), cp_all

        def up_joint(W):
            Uj, qj = joint_upper(W, x, n_g, prior, d_joint)
            Us = float(W @ cp_safe)
            return (Uj, qj) if Uj <= Us else (Us, cp_safe)

        res = {}
        mle = np.where(n_g > 0, x / np.maximum(n_g, 1), 1.0)
        for name, up in (('v7c', up_v7c), ('joint', up_joint)):
            best = None
            for q0 in (np.maximum(mle, q_prior), up(W_pre)[1]):
                b = search_joint(E, ctype, SEC, budget, up, q0, 2)
                if b is not None and (best is None or b[1] < best[1]):
                    best = b
            # the preselected route itself is always a candidate
            if up(W_pre)[0] <= budget and (best is None or SEC[pre].sum() < best[1]):
                best = (pre, float(SEC[pre].sum()), None)
            res[name] = (best[0] if best else None, up)
        # fixed sequence over prior candidates
        chosen = None
        for p in cands:
            W = route_weights(p, E, ctype, 2, 4)
            used = W > 0
            cpu = cp_bounds(x, n_g, delta / max(used.sum(), 1))
            if float(W @ cpu) <= budget:
                chosen = p
            else:
                break
        res['fixseq'] = (chosen, None)
        chosen = None
        for p in cands:
            if route_chernoff_upper(route_weights(p, E, ctype, 2, 4), x, n_g, delta) <= budget:
                chosen = p
            else:
                break
        res['fixseqW'] = (chosen, None)
        # point estimate
        res['point'] = (plan_for(mle, E, ctype, budget), None)
        # single preselected route (own sampling)
        xs = rng.binomial(N, float(W_single @ q_true / W_single.sum()))
        res['single'] = (p_single if (c_single >= 0 and xs <= c_single) else None, None)

        for name, (p, up) in res.items():
            if p is None:
                rows[name].append((False, None, fallback))
            else:
                rows[name].append((True, true_risk(p), float(SEC[p].sum())))
        # rescue statistics (shared-data methods judge the preselected route themselves)
        for name in rescue:
            if name == 'fixseqW':
                pre_ok = route_chernoff_upper(W_pre, x, n_g, delta) <= budget
            elif name == 'fixseq':
                used = W_pre > 0
                pre_ok = float(W_pre @ cp_bounds(x, n_g, delta / used.sum())) <= budget
            else:
                pre_ok = res[name][1](W_pre)[0] <= budget
            if not pre_ok:
                rescue[name][0] += 1
                p = res[name][0]
                if p is not None and SEC[p].sum() < crawl_time:
                    rescue[name][1] += 1
                    rescue[name][2].append(fallback - float(SEC[p].sum()))
                    a = ctype == 0
                    slowedA[name].append((float((p[a] > pre[a]).mean()), float((p[~a] > pre[~a]).mean())))
    summary = {'n_alloc': n_g.tolist(), 'pre_route_true_risk': true_risk(pre),
               'pre_route_seconds': float(SEC[pre].sum()),
               'single_route_true_risk': true_risk(p_single), 'single_threshold': int(c_single),
               'single_route_seconds': float(SEC[p_single].sum())}
    for name, r in rows.items():
        cert = [z for z in r if z[0]]
        summary[name] = {
            'certified': len(cert) / reps,
            'violation': sum(z[1] > budget for z in cert) / reps,
            'faster_than_crawl': sum(z[2] < crawl_time for z in cert) / reps,
            'seconds': float(np.mean([z[2] for z in r])),
            'true_risk': float(np.mean([z[1] for z in cert])) if cert else None,
        }
    for name, (f, s, saved) in rescue.items():
        sl = np.array(slowedA[name]) if slowedA[name] else np.zeros((0, 2))
        summary[name]['pre_failed'] = f
        summary[name]['rescued'] = s
        summary[name]['rescued_saving_vs_fallback'] = float(np.mean(saved)) if saved else None
        summary[name]['slowed_share_A_B'] = sl.mean(0).tolist() if len(sl) else None
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', default='front_87_h0.5')
    ap.add_argument('--variants', nargs='*', default=['gz+extrap', 'gz+confirm3+extrap'])
    ap.add_argument('--N', type=int, nargs='*', default=[360, 1000, 2000])
    ap.add_argument('--priors', nargs='*', default=['correct', 'biased', 'mix'])
    ap.add_argument('--reserve', type=float, nargs='*', default=[0.0, 0.3, 0.6])
    ap.add_argument('--kappas', type=float, nargs='*', default=[10.0, 100.0, 1000.0])
    ap.add_argument('--w0', type=float, default=0.2)
    ap.add_argument('--share-A', type=float, default=0.35)
    ap.add_argument('--a-factor', default='mild', choices=sorted(A_FACTORS))
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
    ctype = (np.random.default_rng(a.seed + 1).random(a.corners) >= a.share_A).astype(int)  # 0 = A
    K = a.corners
    print(f'A-corner factor {a.a_factor}; {K} corners ({(ctype == 0).sum()} type A), go-mode encounters/year {E[:, 0].sum():.0f}; '
          f'exposure factor by mode {np.round(E.sum(0) / E[:, 0].sum(), 2).tolist()}; '
          f'budget {a.budget}, delta {a.delta}, {a.reps} reps')
    print(f'all-crawl time {K * SECONDS["crawl"]:.0f}s, fallback (uncertified) {K * STOP_SECONDS:.0f}s')
    results = []
    for var in a.variants:
        beh = {'A': A_FACTORS[a.a_factor], 'B': B_TRUTH}
        truth = np.concatenate([group_rates(a.camera, var, beh[t], a.pool, a.seed) for t in TYPES])
        sim_ok = np.concatenate([group_rates(a.camera, var, beh[t], a.pool, a.seed + 50000)
                                 for t in TYPES])
        nominal_B = group_rates(a.camera, var, B_TRUTH, a.pool, a.seed + 90000)
        sim_bad = np.concatenate([nominal_B, nominal_B])          # simulator thinks A behaves like B
        preds = {'correct': sim_ok, 'biased': sim_bad, 'mix': np.vstack([sim_ok, sim_bad])}
        oracle = plan_for(truth, E, ctype, a.budget)
        all_risk = {m: float(sum(E[k, j] * truth[ctype[k] * 4 + j] for k in range(K)))
                    for j, m in enumerate(MODES)}
        print(f'\n== {a.camera} / {var}')
        print('   true q  A: ' + ' '.join(f'{100*v:.2f}%' for v in truth[:4])
              + '   B: ' + ' '.join(f'{100*v:.2f}%' for v in truth[4:]))
        print('   biased prior A: ' + ' '.join(f'{100*v:.2f}%' for v in sim_bad[:4]))
        print('   all-mode risk ' + ', '.join(f'{m} {v:.1f}' for m, v in all_risk.items())
              + f';  oracle time {SEC[oracle].sum():.0f}s risk '
              f'{route_weights(oracle, E, ctype, 2, 4) @ truth:.1f}')
        for pname in a.priors:
            q_prior = preds[pname] if preds[pname].ndim == 1 else preds[pname].mean(0)
            prior = beta_prior(preds[pname], a.kappas, a.w0)
            for N in a.N:
                for r in a.reserve:
                    s = run_cell(truth, q_prior, E, ctype, N, r, a.budget, a.delta, a.reps,
                                 a.seed + N + int(100 * r), prior)
                    print(f'   prior={pname:7s} N={N:4d} reserve={r:.1f}  n={s["n_alloc"]}  '
                          f'preselected route: true risk {s["single_route_true_risk"]:.1f}, '
                          f'{s["single_route_seconds"]:.0f}s, single test passes if <= {s["single_threshold"]}')
                    for name in ('joint', 'v7c', 'fixseqW', 'fixseq', 'single', 'point'):
                        m = s[name]
                        extra = ''
                        if 'rescued' in m:
                            sv = m['rescued_saving_vs_fallback']
                            sl = m['slowed_share_A_B']
                            extra = (f'  | pre failed {m["pre_failed"]:3d}, rescued {m["rescued"]:3d}'
                                     + (f', saves {sv:.0f}s' if sv else '')
                                     + (f', slowed A {100*sl[0]:.0f}% B {100*sl[1]:.0f}%' if sl else ''))
                        print(f'      {name:>6}: cert {100*m["certified"]:5.1f}%  violate {100*m["violation"]:4.1f}%  '
                              f'faster-than-crawl {100*m["faster_than_crawl"]:5.1f}%  time {m["seconds"]:5.0f}s'
                              + extra)
                    results.append({'variant': var, 'prior': pname, 'N': N, 'reserve': r,
                                    'truth': truth.tolist(), 'q_prior': q_prior.tolist(), **s})
    if a.out:
        with open(a.out, 'w') as f:
            json.dump({'args': vars(a), 'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
