"""
Split conformal calibration of the clearance lower bound, with and without weights.

Calibration is Mondrian: one quantile per (action, decision-time stratum),
where the stratum is whether the agent was hidden at t0. The target is the
natural state distribution of that stratum, so for action a the
calibration sample (drawn with probability mu(a|s)) is reweighted by
1 / mu(a|s); verified samples are further reweighted by 1 / rho(s).

Methods:

- ``naive_partial``: every logged decision, score from the visible part of
  the outcome only, no weights. This is what a robot gets from its own log.
- ``complete_case``: decisions whose outcome is fully known (R = 1), no
  weights.
- ``verified_ipw``: verified decisions only (Q = 1), weight 1 / (mu rho).
  Weights depend on the state alone, so the finite-sample guarantee of
  weighted conformal under covariate shift (Tibshirani et al. 2019) applies
  exactly.
- ``observed_ipw``: all decisions with R = 1, weight 1 / (mu (V + (1-V) rho)).
  Uses the naturally visible outcomes too, but the weight depends on the
  outcome through V; the test weight is replaced by its upper bound
  1 / (mu rho). Finite-sample validity is an open question (SCI-1 T1/L1),
  so this method is reported as experimental.
- ``oracle_full``: every logged decision with simulator truth, weight 1 / mu.
  Unattainable reference.
"""

import numpy as np

from .collect import state_of
from .predictors import predicted_clearance
from .sim2d import clearance

METHODS = ('naive_partial', 'complete_case', 'verified_ipw', 'observed_ipw', 'oracle_full')
EXACT_METHODS = ('verified_ipw',)


def weighted_quantiles(scores, weights, test_weights, alpha):
    """
    Weighted split-conformal quantile for each test point.

    Smallest calibration score s with sum_{S_i <= s} p_i >= 1 - alpha, where
    p_i = w_i / (sum_j w_j + w_test) and the test point's mass sits at +inf.
    Returns +inf where the calibration mass cannot reach 1 - alpha.
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    test_weights = np.asarray(test_weights, dtype=float)
    if scores.size == 0:
        return np.full(test_weights.shape, np.inf)
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    cumulative = np.cumsum(weights[order])
    need = (1.0 - alpha) * (cumulative[-1] + test_weights) * (1.0 - 1e-12)
    index = np.searchsorted(cumulative, need, side='left')
    return np.where(index < scores.size,
                    sorted_scores[np.minimum(index, scores.size - 1)], np.inf)


def calibration_table(log, scn, predictor):
    """Per-decision quantities every method needs, computed from the log alone."""
    n = len(log)
    rows = np.arange(n)
    action = log['action']
    d_hat = predicted_clearance(scn, state_of(log), predictor)[rows, action]
    observed_agent = log['agent_traj_observed']
    table = {
        'action': action,
        'stratum': log['hidden0'].astype(int),
        'mu': log['propensity'][rows, action],
        'rho': log['verify_prob'],
        'verified': log['verified'],
        'visible_full': log.visible_full,
        'observed': log.observed,
        'd_hat': d_hat,
        'd_full': clearance(log['robot_traj'], log['agent_traj_full']),
        'd_partial': clearance(log['robot_traj'], np.nan_to_num(observed_agent, nan=1e9),
                               log['visible_steps']),
    }
    if log.has_oracle:
        table['d_oracle'] = clearance(log['robot_traj'], log['oracle_agent_traj'][rows, action])
    return table


def _method_inputs(method, cal, select):
    if method == 'naive_partial':
        mask = select
        scores = cal['d_hat'] - cal['d_partial']
        weights = np.ones(mask.sum())
    elif method == 'complete_case':
        mask = select & cal['observed']
        scores = cal['d_hat'] - cal['d_full']
        weights = np.ones(mask.sum())
    elif method == 'verified_ipw':
        mask = select & cal['verified']
        scores = cal['d_hat'] - cal['d_full']
        weights = 1.0 / (cal['mu'] * cal['rho'])[mask]
    elif method == 'observed_ipw':
        mask = select & cal['observed']
        scores = cal['d_hat'] - cal['d_full']
        v = cal['visible_full'][mask]
        weights = 1.0 / (cal['mu'][mask] * (v + (1 - v) * cal['rho'][mask]))
    elif method == 'oracle_full':
        mask = select
        scores = cal['d_hat'] - cal['d_oracle']
        weights = 1.0 / cal['mu'][mask]
    else:
        raise ValueError(f'unknown method {method!r}')
    return scores[mask], weights


def _test_weights(method, mu_test, rho_test):
    if method in ('naive_partial', 'complete_case'):
        return np.ones_like(mu_test)
    if method in ('verified_ipw', 'observed_ipw'):
        return 1.0 / (mu_test * rho_test)
    return 1.0 / mu_test


def lower_bounds(method, cal, test, alpha):
    """
    Calibrated clearance lower bound for every test decision and action: (t, A).

    ``test`` needs ``d_hat`` (t, A), ``stratum`` (t,), ``mu`` (t, A) - the
    logging propensities at the test state, which define the importance
    weight - and ``rho`` (t,).
    """
    d_hat = test['d_hat']
    bounds = np.full(d_hat.shape, -np.inf)
    n_eff = {}
    for a in range(d_hat.shape[1]):
        for stratum in (0, 1):
            select = (cal['action'] == a) & (cal['stratum'] == stratum)
            target = test['stratum'] == stratum
            if not target.any():
                continue
            scores, weights = _method_inputs(method, cal, select)
            tw = _test_weights(method, test['mu'][target, a], test['rho'][target])
            q = weighted_quantiles(scores, weights, tw, alpha)
            bounds[target, a] = d_hat[target, a] - q
            total = weights.sum()
            n_eff[(a, stratum)] = float(total ** 2 / (weights ** 2).sum()) if total > 0 else 0.0
    return bounds, n_eff
