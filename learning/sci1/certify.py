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
