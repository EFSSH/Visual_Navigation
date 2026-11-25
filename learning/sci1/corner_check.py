"""
SCI-1 check 1: is a per-encounter close-call target certifiable, and does it
separate pass modes and camera set-ups?

For each hazard definition, camera set and pass speed, the simulator gives
the true per-encounter rate q and the rate the onboard log shows. A
calibration set of ``n_verified`` staged encounters with full truth
(independent view) is certified at level alpha if the one-sided
Clopper-Pearson bound is <= alpha. Reported:

- power: probability of certifying, ours (verified labels);
- naive: probability of certifying from onboard labels alone (same n);
  certifying a row whose true q > alpha is a false certification;
- n80: verified encounters needed for 80% power (q < alpha only).

Run: PYTHONPATH=learning python -m sci1.corner_check --n 20000
"""

import argparse
import json
from functools import lru_cache
from math import exp, lgamma, log, log1p

from .corner_pass import CAMERA_CONFIGS, CornerScenario, simulate
from .release_control_prototype import clopper_pearson_upper

HAZARDS = ('contact_mv', 'near_moving', 'late')
CONFIGS = ('gazebo_60_h0.11', 'front_87_h0.5', 'front_120_h0.5', 'ring4_87_h0.5')
ACTIONS = ('go', 'medium', 'slow', 'crawl')
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.05


@lru_cache(maxsize=None)
def k_max(n, alpha, delta=DELTA):
    """Largest error count still certified at alpha with n samples (-1: none)."""
    k = -1
    while k + 1 <= n and clopper_pearson_upper(k + 1, n, delta) <= alpha:
        k += 1
    return k


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return float(k >= n)
    lf, ls, lq = lgamma(n + 1), log(p), log1p(-p)
    return min(1.0, sum(exp(lf - lgamma(i + 1) - lgamma(n - i + 1) + i * ls + (n - i) * lq)
                        for i in range(k + 1)))


def power(q, n, alpha):
    return binom_cdf(k_max(n, alpha), n, q)


def n80(q, alpha, cap=5000):
    if q >= alpha:
        return None
    n = 10
    while n <= cap:
        if power(q, n, alpha) >= 0.8:
            lo, hi = n // 2, n
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if power(q, mid, alpha) >= 0.8 else (mid, hi)
            return hi
        n *= 2
    return None


def rates(n, seed, attentive_share):
    rows = []
    for cam in CONFIGS:
        scn = CornerScenario(cameras=CAMERA_CONFIGS[cam], attentive_share=attentive_share)
        for j, act in enumerate(ACTIONS):
            out = simulate(scn, act, n, seed + 1000 * j)
            enc = out['encounter']
            row = {'camera': cam, 'action': act, 'encounters': int(enc.sum())}
            for h in HAZARDS:
                row[h] = float(out[h][enc].mean())
                row[h + '_log'] = float(out[h + '_logged'][enc].mean())
            rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--attentive-share', type=float, default=0.5)
    ap.add_argument('--n-verified', type=int, default=100)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    rows = rates(a.n, a.seed, a.attentive_share)
    nv = a.n_verified
    print(f'attentive share {a.attentive_share}; per-encounter rates; '
          f'certification with {nv} verified encounters, delta {DELTA}')
    for h in HAZARDS:
        print(f'\n== hazard: {h} ==')
        head = f'{"camera":>16} {"act":>6} {"q_true":>7} {"q_log":>7}'
        for al in ALPHAS:
            head += f' | a={al:.2f} pow  naive   n80'
        print(head)
        for r in rows:
            q, ql = r[h], r[h + '_log']
            line = f'{r["camera"]:>16} {r["action"]:>6} {100 * q:6.1f}% {100 * ql:6.1f}%'
            cert = {}
            for al in ALPHAS:
                pw, nav, need = power(q, nv, al), power(ql, nv, al), n80(q, al)
                flag = '!' if (q > al and nav > 0.5) else ' '
                line += f' | {100 * pw:9.0f}% {100 * nav:4.0f}%{flag} {need if need else "  -":>5}'
                cert[str(al)] = {'power': pw, 'naive': nav, 'n80': need}
            r.setdefault('cert', {})[h] = cert
            print(line)
    print('\n! = naive onboard log certifies (>50%) a row whose true rate exceeds alpha')
    if a.out:
        with open(a.out, 'w') as f:
            json.dump({'attentive_share': a.attentive_share, 'n_verified': nv, 'rows': rows},
                      f, indent=1)


if __name__ == '__main__':
    main()
