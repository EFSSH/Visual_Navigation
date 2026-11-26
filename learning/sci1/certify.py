"""
SCI-1 v7 certification: per-encounter close-call rates and the annual
close-call count they imply.

A calibration encounter records the pass mode A, its known selection
probability mu(A|S), whether it was verified (Q ~ Bernoulli(rho), drawn before
the outcome) and, when verified, the true close-call indicator L. The
onboard indicator is always available but can only miss events.

Rate bound (rejection-sampling reduction to the full-label binomial case):
keep a verified encounter of the target mode with probability
c / (mu * rho), c = min over the calibration set of mu * rho. The kept
encounters are an i.i.d. sample of the target mode's outcomes with full
labels, so a one-sided Clopper-Pearson bound on them is valid.

Annual bound: sum_k E_k * q_upper(mode_k) with the confidence budget split
over the modes that are bounded (Bonferroni), E_k taken as known upper
bounds on yearly encounters per corner.
"""

from dataclasses import dataclass

import numpy as np

from .release_control_prototype import clopper_pearson_upper


@dataclass
class RateBound:
    upper: float
    kept: int
    events: int


def rate_upper(action, mu, rho, verified, loss, target, delta, rng):
    """Upper confidence bound on P(L = 1 | mode = target) from verified encounters."""
    action, mu, rho = np.asarray(action), np.asarray(mu, float), np.asarray(rho, float)
    verified, loss = np.asarray(verified, bool), np.asarray(loss, bool)
    sel = (action == target) & verified
    if not sel.any():
        return RateBound(1.0, 0, 0)
    c = float(np.min((mu * rho)[action == target]))
    keep = sel & (rng.random(len(action)) < c / (mu * rho))
    n, k = int(keep.sum()), int((loss & keep).sum())
    return RateBound(clopper_pearson_upper(k, n, delta) if n else 1.0, n, k)


def naive_upper(action, loss_onboard, target, delta):
    """Same bound computed from the onboard log alone (no verification)."""
    sel = np.asarray(action) == target
    n, k = int(sel.sum()), int((np.asarray(loss_onboard, bool) & sel).sum())
    return RateBound(clopper_pearson_upper(k, n, delta) if n else 1.0, n, k)


def allocate(encounters, q_upper, seconds, budget):
    """Choose a pass mode per corner: minimise time subject to sum E_k q(mode) <= budget.

    encounters: (K,) yearly encounters per corner; q_upper, seconds: dicts mode -> value.
    Greedy on risk reduction per extra second, starting from the fastest mode.
    Returns (modes, annual_risk) or (None, risk) if even the safest plan exceeds the budget.
    """
    modes = sorted(seconds, key=seconds.get)                 # fastest first
    # Drop modes that are slower but not safer than a faster one.
    usable = []
    for m in modes:
        if not usable or q_upper[m] < q_upper[usable[-1]]:
            usable.append(m)
    E = np.asarray(encounters, float)
    level = np.zeros(len(E), int)
    q = np.array([q_upper[m] for m in usable])
    t = np.array([seconds[m] for m in usable])
    risk = float((E * q[level]).sum())
    while risk > budget:
        can = level < len(usable) - 1
        if not can.any():
            return None, risk
        gain = np.where(can, E * (q[level] - q[np.minimum(level + 1, len(q) - 1)]), -1.0)
        cost = np.where(can, t[np.minimum(level + 1, len(t) - 1)] - t[level], 1.0)
        j = int(np.argmax(gain / np.maximum(cost, 1e-9)))
        risk -= gain[j]
        level[j] += 1
    return [usable[i] for i in level], risk


def point_estimate(loss_verified):
    """Plug-in rate from verified labels, no confidence margin."""
    loss_verified = np.asarray(loss_verified, bool)
    return float(loss_verified.mean()) if len(loss_verified) else 1.0


def ppi_upper(onboard_all, loss_verified, onboard_verified, delta):
    """Prediction-powered (CLT) upper bound with the onboard label as the predictor.

    theta = mean(onboard over all encounters) + mean(L - onboard over verified ones).
    Asymptotic normal interval; not valid in finite samples, which is the point
    of comparing it with the exact bounds when events are rare.
    """
    from statistics import NormalDist
    f = np.asarray(onboard_all, float)
    rect = np.asarray(loss_verified, float) - np.asarray(onboard_verified, float)
    if len(f) == 0 or len(rect) == 0:
        return 1.0
    theta = f.mean() + rect.mean()
    var = f.var(ddof=1) / len(f) if len(f) > 1 else 0.0
    var += rect.var(ddof=1) / len(rect) if len(rect) > 1 else 0.0
    return float(np.clip(theta + NormalDist().inv_cdf(1 - delta) * np.sqrt(var), 0.0, 1.0))


def decomposed_upper(onboard_all, loss_verified, onboard_verified, delta):
    """Characterisation B bound: onboard rate + rate of events the onboard log missed.

    Valid because the onboard log can only miss events (onboard <= L), so
    q = P(onboard = 1) + P(L = 1, onboard = 0). The first term uses every
    encounter the robot logged; the second only verified ones. delta is split
    equally between the two Clopper-Pearson bounds.
    """
    onboard_all = np.asarray(onboard_all, bool)
    missed = np.asarray(loss_verified, bool) & ~np.asarray(onboard_verified, bool)
    if len(onboard_all) == 0 or len(missed) == 0:
        return 1.0
    up = (clopper_pearson_upper(int(onboard_all.sum()), len(onboard_all), delta / 2)
          + clopper_pearson_upper(int(missed.sum()), len(missed), delta / 2))
    return float(min(1.0, up))


# ---------------------------------------------------------------------------
# Further baselines (v7c). All bound q = P(L = 1) for one pass mode.
# Structure used throughout: the onboard log can only miss events
# (onboard <= L), so the rectifier L - onboard is itself a 0/1 "missed" flag.
# ---------------------------------------------------------------------------

def _normal_upper(theta, var, delta):
    from statistics import NormalDist
    return float(np.clip(theta + NormalDist().inv_cdf(1 - delta) * np.sqrt(max(var, 0.0)), 0.0, 1.0))


def ppi_pp_upper(onboard_all, loss_verified, onboard_verified, delta):
    """PPI++ (power-tuned PPI, CLT): theta = lam*mean(f_all) + mean(L - lam*f) on verified.

    lam = Cov(L, f) / ((1 + n/N) Var(f)) from the verified sample, clipped to
    [0, 1]; lam = 0 is the verified-only mean, lam = 1 is plain PPI.
    """
    f = np.asarray(onboard_all, float)
    L, fv = np.asarray(loss_verified, float), np.asarray(onboard_verified, float)
    N, n = len(f), len(L)
    if N < 2 or n < 2:
        return 1.0
    vf = fv.var(ddof=1)
    lam = 0.0 if vf == 0 else float(np.cov(L, fv)[0, 1] / ((1 + n / N) * vf))
    lam = float(np.clip(lam, 0.0, 1.0))
    rect = L - lam * fv
    theta = lam * f.mean() + rect.mean()
    var = lam ** 2 * f.var(ddof=1) / N + rect.var(ddof=1) / n
    return _normal_upper(theta, var, delta)


def betting_upper(x, bound, delta, rng, grid=4000):
    """One-sided finite-sample upper confidence bound on E[X], X in [0, bound].

    Hedged-capital betting test (Waudby-Smith & Ramdas, predictable plug-in
    bets) for H0: E[X] >= m, inverted over a grid of m; valid for any sample
    size, no normal approximation. Order is randomised once.
    """
    x = np.asarray(x, float) / bound
    n = len(x)
    if n == 0:
        return 1.0
    x = x[rng.permutation(n)]
    i = np.arange(1, n + 1)
    sq = np.concatenate([[0.0], np.cumsum((x - (0.5 + np.cumsum(x)) / (i + 1)) ** 2)[:-1]])
    var_hat = (0.25 + sq) / i
    lam = np.sqrt(2 * np.log(1 / delta) / (var_hat * i * np.log1p(i)))
    m = np.unique(np.concatenate([np.geomspace(1e-5, 1.0, grid, endpoint=False),
                                  np.linspace(0.0, 1.0, grid, endpoint=False)[1:]]))[:, None]
    bet = np.minimum(lam[None, :], 0.5 / (1 - m))                # keeps 1 + bet*(m - x) > 0
    logk = np.cumsum(np.log1p(bet * (m - x[None, :])), axis=1)
    kept = ~(logk >= np.log(1 / delta)).any(axis=1)              # m not rejected
    if kept[-1]:
        return 1.0
    up = float(m[np.flatnonzero(kept)[-1] + 1, 0]) if kept.any() else float(m[0, 0])
    return float(min(1.0, up * bound))   # sup of the confidence set (next grid point up)


def ipw_pseudo(onboard, verified, loss, pi):
    """Z = f + Q (L - f) / pi: unbiased for q when Q ~ Bernoulli(pi) is drawn before L."""
    f = np.asarray(onboard, float)
    q = np.asarray(verified, bool)
    pi = np.broadcast_to(np.asarray(pi, float), f.shape)
    corr = np.where(q, (np.asarray(loss, float) - f) / np.where(q, pi, 1.0), 0.0)
    return f + corr


def ppi_betting_upper(onboard, verified, loss, rho, delta, rng):
    """Finite-sample PPI: betting bound on the IPW/PPI pseudo-outcome, uniform verification.

    (Splitting delta between exact bounds on E[f] and on the rectifier is the
    decomposition bound, decomposed_upper, since the rectifier is Bernoulli here.)
    """
    z = ipw_pseudo(onboard, verified, loss, rho)
    return betting_upper(z, 1.0 / rho, delta, rng)


def active_probs(onboard, flag, rho, ratio=3.0):
    """Verification probabilities for active inference, matching budget rho in expectation.

    Logged close calls need no check (onboard <= L, so L = 1); flagged encounters
    (late or no recognition in the robot's own log) get `ratio` times the rate of
    the rest. Capped at 1.
    """
    f, s = np.asarray(onboard, bool), np.asarray(flag, bool)
    w = np.where(f, 0.0, np.where(s, ratio, 1.0))
    if w.sum() == 0:
        return np.full(len(f), rho)
    budget = rho * len(f)
    pi = np.zeros(len(f))
    for _ in range(20):                                          # water-filling with cap 1
        free = (pi < 1.0) & (w > 0)
        left = budget - (pi == 1.0).sum()
        if left <= 0 or not free.any():
            break
        pi[free] = np.minimum(1.0, w[free] * left / w[free].sum())
        if (pi[free] < 1.0).all():
            break
    return pi


def active_clt_upper(z, delta):
    """Active statistical inference (Zrnic & Candes): CLT bound on the IPW mean."""
    z = np.asarray(z, float)
    if len(z) < 2:
        return 1.0
    return _normal_upper(z.mean(), z.var(ddof=1) / len(z), delta)


def rogan_gladen_upper(onboard_all, loss_verified, onboard_verified, delta):
    """Rogan-Gladen prevalence correction with exact bounds.

    q = P(f = 1) / Se, Se = P(f = 1 | L = 1) from verified close calls;
    specificity is 1 by construction (the log has no false close calls).
    CP upper on P(f = 1) at delta/2 divided by CP lower on Se at delta/2.
    """
    f = np.asarray(onboard_all, bool)
    L, fv = np.asarray(loss_verified, bool), np.asarray(onboard_verified, bool)
    pos = int(L.sum())
    if len(f) == 0 or pos == 0:
        return 1.0
    p_up = clopper_pearson_upper(int(f.sum()), len(f), delta / 2)
    k = int((fv & L).sum())
    se_lo = 1.0 - clopper_pearson_upper(pos - k, pos, delta / 2)
    return 1.0 if se_lo <= 0 else float(min(1.0, p_up / se_lo))
