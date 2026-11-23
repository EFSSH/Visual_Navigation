"""Reproduce the v5 single-action release-risk experiment."""

from math import exp, lgamma, log, log1p

import numpy as np

from .collect import collect, reverify
from .conformal import calibration_table
from .experiments import build_test_set
from .policies import VerificationPolicy
from .predictors import ConstantVelocityPhantom
from .sim2d import ACTION_SPEED, GO, SCENARIOS, YIELD, episode_seeds


EPSILON = 0.15
SAFETY_DISTANCE = 0.8
DELTA = 0.05
RISK_TARGETS = (0.1, 0.05, 0.01)
AUDIT_CONFIGURATIONS = ((16000, 0.2), (16000, 0.5), (64000, 0.2))


def binomial_cdf(errors, trials, probability):
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return float(errors == trials)
    log_factorial = lgamma(trials + 1)
    log_success = log(probability)
    log_failure = log1p(-probability)
    return sum(
        exp(log_factorial - lgamma(i + 1) - lgamma(trials - i + 1)
            + i * log_success + (trials - i) * log_failure)
        for i in range(errors + 1)
    )


def clopper_pearson_upper(errors, trials, delta):
    """Invert the lower binomial tail for a one-sided confidence limit."""
    if trials == 0 or errors == trials:
        return 1.0
    lower, upper = errors / trials, 1.0
    for _ in range(60):
        midpoint = (lower + upper) / 2
        if binomial_cdf(errors, trials, midpoint) > delta:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def choose_threshold(scores, near_miss, grid, alpha, delta):
    """Choose the lowest candidate threshold that passes the risk test."""
    threshold = np.inf
    for candidate in grid:
        released = scores >= candidate
        errors = np.count_nonzero(near_miss & released)
        trials = np.count_nonzero(released)
        if clopper_pearson_upper(errors, trials, delta / len(grid)) <= alpha:
            threshold = min(threshold, candidate)
    return threshold


def run_scenario(node):
    scenario = SCENARIOS[node](sensor='camera', cam_hfov_deg=87, cam_vfov_deg=58,
                               cam_height=0.5, cam_rear_height=0.5)
    predictor = ConstantVelocityPhantom(scenario)
    test = build_test_set(scenario, predictor, 8000, 7, EPSILON)
    hidden = test['state']['hidden0']
    test_scores = test['d_hat'][:, GO]
    test_near_miss = test['d_true'][:, GO] < SAFETY_DISTANCE
    go_progress = test['progress'][:, GO] - test['progress'][:, YIELD]
    lost_time = go_progress / ACTION_SPEED[GO]

    print(f'\n=== {node} (camera 87 deg, 0.5 m); hidden decisions {hidden.mean():.2f}; '
          f'if always entering when not detected: error per release '
          f'{100 * test_near_miss[hidden].mean():.1f}%')
    print(f'{"calibration source":<40}{"alpha":>6}{"lambda":>8}{"release":>9}'
          f'{"error/release":>15}{"<= alpha":>9}{"time lost":>10}{"samples":>9}')

    for n_cal, budget in AUDIT_CONFIGURATIONS:
        rng = np.random.default_rng(1)
        log_data = collect(scenario, episode_seeds(1000, n_cal), predictor,
                           VerificationPolicy(1, 1), EPSILON, rng)
        uniforms = rng.random(len(log_data))
        policy = VerificationPolicy.from_budget(budget, log_data['hidden0'].mean(), 'uniform')
        cal = calibration_table(reverify(log_data, policy, uniforms), scenario, predictor)
        hidden_go = (cal['action'] == GO) & (cal['stratum'] == 1)

        # GO and audit probabilities are constant here. The grid is formed
        # from predictions alone, before inspecting any near-miss labels.
        grid = np.quantile(cal['d_hat'][hidden_go], np.linspace(0.9, 0.0, 10))
        verified = (f'ours: verified, n={n_cal}, budget {budget}',
                    hidden_go & cal['verified'], cal['d_full'])
        sources = [verified]
        if (n_cal, budget) == AUDIT_CONFIGURATIONS[0]:
            sources = [
                ('naive: onboard log', hidden_go, cal['d_partial']),
                verified,
                ('oracle: all truth (unattainable)', hidden_go, cal['d_oracle']),
            ]

        for name, mask, distances in sources:
            scores = cal['d_hat'][mask]
            near_miss = distances[mask] < SAFETY_DISTANCE
            for alpha in RISK_TARGETS:
                threshold = choose_threshold(scores, near_miss, grid, alpha, DELTA)
                released = hidden & (test_scores >= threshold)
                error = test_near_miss[released].mean() if released.any() else float('nan')
                meets_target = '-' if not released.any() else ('yes' if error <= alpha else 'NO')
                print(f'{name:<40}{alpha:>6}{threshold:>8.2f}{released.mean():>9.3f}'
                      f'{100 * error:>14.2f}%{meets_target:>9}'
                      f'{(lost_time * ~released).mean():>9.2f}s{len(scores):>9}')


if __name__ == '__main__':
    for scenario_name in ('door', 'mixed'):
        run_scenario(scenario_name)
