import numpy as np
import pytest

from sci1.collect import collect, reverify
from sci1.conformal import calibration_table, lower_bounds, METHODS, weighted_quantiles
from sci1.geometry import line_of_sight, segment_hits_box
from sci1.planners import select_action, worst_case_bounds
from sci1.policies import epsilon_propensities, sample_actions, VerificationPolicy
from sci1.predictors import ConstantVelocityPhantom, predicted_clearance
from sci1.records import DecisionLog
from sci1.sim2d import (clearance, episode_seeds, GO, meeting_scenario, occlusion_scenario,
                        perceive, sample_episodes, simulate, simulate_branches,
                        trajectory_type, YIELD)

BOX = ((-30.0, -1.0), (1.0, 30.0))


def test_segment_box_intersection():
    p = np.array([[-5.0, 0.0], [-5.0, 0.0], [-0.5, 0.0]])
    q = np.array([[0.0, 5.0], [0.0, 0.5], [0.0, 5.0]])
    assert segment_hits_box(p, q, BOX).tolist() == [True, False, False]
    # A segment entirely inside, and an axis-parallel one outside.
    assert segment_hits_box([-5.0, 5.0], [-4.0, 6.0], BOX)
    assert not segment_hits_box([0.0, 0.0], [0.0, 9.0], BOX)
    assert not line_of_sight([0.0, 0.0], [0.0, 20.0], [], 12.0)


def test_weighted_quantile_reduces_to_split_conformal():
    scores = np.arange(1.0, 20.0)  # n = 19
    q = weighted_quantiles(scores, np.ones(19), np.ones(3), alpha=0.1)
    # ceil((n + 1)(1 - alpha)) = 18th smallest.
    assert np.all(q == 18.0)
    assert np.isinf(weighted_quantiles(scores[:5], np.ones(5), [1.0], 0.1)[0])
    assert np.isinf(weighted_quantiles([], [], [1.0], 0.1)[0])


def test_weighted_conformal_covers_under_known_covariate_shift():
    # Calibration x ~ U(0, 1); test x ~ Beta(3, 1) (density 3x^2); y | x ~ N(0, (0.2 + x)^2).
    rng = np.random.default_rng(0)
    alpha, trials, n = 0.1, 400, 200
    hits_w, hits_u = [], []
    for _ in range(trials):
        x = rng.random(n)
        s = np.abs(rng.normal(0, 0.2 + x))
        xt = rng.beta(3, 1)
        st = abs(rng.normal(0, 0.2 + xt))
        q_w = weighted_quantiles(s, 3 * x ** 2, [3 * xt ** 2], alpha)[0]
        q_u = weighted_quantiles(s, np.ones(n), [1.0], alpha)[0]
        hits_w.append(st <= q_w)
        hits_u.append(st <= q_u)
    assert np.mean(hits_w) >= 1 - alpha - 0.03
    assert np.mean(hits_u) < np.mean(hits_w)


def test_propensities_and_sampling():
    mask = np.ones((4, 3), dtype=bool)
    mask[3, 1] = False
    mu = epsilon_propensities(np.array([0, 1, 2, 0]), mask, 0.15)
    assert np.allclose(mu.sum(-1), 1.0)
    assert mu[0, 0] == pytest.approx(0.7) and mu[3, 1] == 0.0 and mu[3, 0] == pytest.approx(0.85)
    rng = np.random.default_rng(1)
    draws = sample_actions(np.tile([0.2, 0.3, 0.5], (20000, 1)), rng)
    assert np.allclose(np.bincount(draws, minlength=3) / 20000, [0.2, 0.3, 0.5], atol=0.015)


def test_verification_budget():
    policy = VerificationPolicy.from_budget(0.2, hidden_share=0.6, allocation='hidden_focus')
    assert policy.expected_rate(0.6) == pytest.approx(0.2)
    assert policy.rho_hidden > policy.rho_visible
    assert VerificationPolicy.from_budget(0.2, 0.6).rho_hidden == 0.2


def test_branch_replay_is_deterministic():
    scn = occlusion_scenario()
    seeds = episode_seeds(7, 50)
    a = simulate(scn, sample_episodes(scn, seeds), GO)
    b = simulate(scn, sample_episodes(scn, seeds[::-1]), GO)
    for x, y in zip(a, b):
        assert np.array_equal(x, y[::-1])


def test_agent_reacts_only_to_the_robot():
    scn = occlusion_scenario()
    world = sample_episodes(scn, episode_seeds(3, 300))
    _, agents, _ = simulate_branches(scn, world)
    assert not np.allclose(agents[:, 0], agents[:, 2])
    # Without reaction the agent does not depend on the robot's action.
    _, agents, _ = simulate_branches(scn, dict(world, beta=np.zeros_like(world['beta'])))
    assert np.allclose(agents[:, 0], agents[:, 2])


def test_meeting_scenario_has_no_occlusion():
    scn = meeting_scenario(sensor_range=100.0)
    world = sample_episodes(scn, episode_seeds(4, 100))
    _, _, vis = simulate_branches(scn, world)
    assert vis.all()
    assert (trajectory_type(vis) == 0).all()


def test_worst_case_bound_is_valid():
    scn = occlusion_scenario()
    world = sample_episodes(scn, episode_seeds(5, 300))
    state = perceive(scn, world)
    robots, agents, _ = simulate_branches(scn, world)
    bounds = worst_case_bounds(scn, state)
    assert (clearance(robots, agents) >= bounds).all()


def test_selection_prefers_fastest_certified():
    bounds = np.array([[2.0, 2.0, 2.0], [0.1, 2.0, 2.0], [0.1, 0.1, -1.0]])
    assert select_action(bounds, 0.8).tolist() == [GO, 1, YIELD]


@pytest.fixture(scope='module')
def logged():
    scn = occlusion_scenario()
    predictor = ConstantVelocityPhantom(scn)
    rng = np.random.default_rng(0)
    log = collect(scn, episode_seeds(11, 600), predictor, VerificationPolicy(0.3, 0.3), 0.15, rng)
    return scn, predictor, log


def test_decision_log_roundtrip_and_validation(logged, tmp_path):
    _, _, log = logged
    path = tmp_path / 'log.npz'
    log.save(path)
    loaded = DecisionLog.load(path)
    assert len(loaded) == len(log) and loaded.has_oracle
    assert np.array_equal(loaded['seed'], log['seed'])
    real = log.without_oracle()
    assert not real.has_oracle
    broken = dict(real.columns, propensity=real['propensity'] * 2)
    with pytest.raises(ValueError):
        DecisionLog(real.meta, broken)
    # Unobserved decisions carry no full trajectory.
    assert np.isnan(log['agent_traj_full'][~log.observed]).all()


def test_reverify_keeps_world_and_changes_only_verification(logged):
    _, _, log = logged
    u = np.random.default_rng(2).random(len(log))
    low = reverify(log, VerificationPolicy(0.05, 0.05), u)
    high = reverify(log, VerificationPolicy(0.5, 0.5), u)
    assert np.array_equal(low['action'], high['action'])
    assert (low['verified'] <= high['verified']).all()
    assert low['verified'].mean() < high['verified'].mean()


def test_all_methods_produce_bounds(logged):
    scn, predictor, log = logged
    cal = calibration_table(log, scn, predictor)
    assert np.allclose(cal['d_full'][cal['observed']], cal['d_oracle'][cal['observed']])
    assert (cal['d_partial'] >= cal['d_oracle'] - 1e-9).all()
    test = {'d_hat': predicted_clearance(scn, perceive(scn, sample_episodes(
        scn, episode_seeds(12, 50))), predictor)}
    test.update(stratum=np.zeros(50, int), mu=np.full((50, 3), 1 / 3), rho=np.full(50, 0.3))
    for method in METHODS:
        bounds, _ = lower_bounds(method, cal, test, 0.2)
        assert bounds.shape == (50, 3)
        assert np.all(bounds <= test['d_hat'] + 1e2)
