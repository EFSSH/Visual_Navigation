import numpy as np
import pytest

from sci1.collect import collect, reverify
from sci1.conformal import calibration_table, lower_bounds, METHODS, weighted_quantiles
from sci1.geometry import line_of_sight, segment_hits_box
from sci1.planners import select_action, worst_case_bounds
from sci1.policies import epsilon_propensities, sample_actions, VerificationPolicy
from sci1.predictors import ConstantVelocityPhantom, predicted_clearance
from sci1.records import DecisionLog
from sci1.diagnostics import detection_diagnostics
from sci1.sim2d import (clearance, detection_prob, episode_seeds, GO, meeting_scenario,
                        occlusion_scenario, PERCEPTION, perceive, sample_episodes, simulate,
                        simulate_branches, trajectory_type, wall_hug_scenario, YIELD)

NEAR_MISS = PERCEPTION['near_range_miss']

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


@pytest.mark.parametrize('perception', ['ideal', 'near_range_miss'])
def test_branch_replay_is_deterministic(perception):
    scn = occlusion_scenario(**PERCEPTION[perception])
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


def test_ideal_detection_is_line_of_sight():
    scn = occlusion_scenario()
    world = sample_episodes(scn, episode_seeds(6, 300))
    _, _, det, los = simulate_branches(scn, world, with_los=True)
    assert np.array_equal(det, los)
    assert np.array_equal(perceive(scn, world)['hidden0'], ~los[:, GO, 0])


def test_detection_prob_degrades_only_at_close_range():
    scn = occlusion_scenario(**NEAR_MISS)
    p = detection_prob(scn, np.array([0.0, 0.6, 1.3, 2.0, 5.0]))
    assert p[0] == p[1] == pytest.approx(0.2) and p[3] == p[4] == 1.0
    assert 0.2 < p[2] < 1.0
    assert np.all(detection_prob(occlusion_scenario(), np.linspace(0, 10, 11)) == 1.0)


def test_misses_happen_only_in_line_of_sight_at_close_range():
    scn = occlusion_scenario(**NEAR_MISS)
    world = sample_episodes(scn, episode_seeds(8, 500))
    robots, agents, det, los = simulate_branches(scn, world, with_los=True)
    assert not (det & ~los).any()
    missed = los & ~det
    assert missed.any()
    dist = np.linalg.norm(robots - agents, axis=-1)
    assert (dist[missed] < scn.det_near_range).all()
    # Same agent motion as ideal detection: the agent reacts to line of sight.
    _, agents_ideal, _ = simulate_branches(occlusion_scenario(), world)
    assert np.array_equal(agents, agents_ideal)


def test_wall_hug_agents_emerge_close():
    scn = wall_hug_scenario()
    world = sample_episodes(scn, episode_seeds(9, 500))
    robots, agents, det = simulate_branches(scn, world)
    hidden0 = perceive(scn, world)['hidden0']
    diag = detection_diagnostics(scn, robots, agents, det, hidden0, 0.8)
    assert diag['go']['hidden0']['near_misses'] > 0
    assert diag['go']['hidden0']['first_detect_dist_p50'] < 2.0
    assert diag['go']['all']['closest_undetected'] == 0.0


def test_close_range_misses_hide_the_closest_approach():
    scn = wall_hug_scenario(**NEAR_MISS)
    world = sample_episodes(scn, episode_seeds(10, 500))
    robots, agents, det = simulate_branches(scn, world)
    diag = detection_diagnostics(scn, robots, agents, det, perceive(scn, world)['hidden0'], 0.8)
    assert diag['go']['all']['closest_undetected'] > 0.3
    assert diag['go']['all']['recorded_as_safe'] > 0.1


@pytest.mark.parametrize('scn', [wall_hug_scenario(), wall_hug_scenario(**NEAR_MISS),
                                 occlusion_scenario(**NEAR_MISS)])
def test_worst_case_bound_is_valid_with_close_range_misses(scn):
    world = sample_episodes(scn, episode_seeds(13, 300))
    state = perceive(scn, world)
    robots, agents, _ = simulate_branches(scn, world)
    assert (clearance(robots, agents) >= worst_case_bounds(scn, state)).all()


from sci1.sensors import camera_prob, lidar_prob, track_frames  # noqa: E402


def test_camera_field_of_view_and_truncation():
    scn = meeting_scenario(sensor='camera')
    robot = np.zeros(2)
    p = camera_prob(scn, robot, np.array([[3.0, 0.0], [3.0, 3.0], [0.6, 0.0], [-3.0, 0.0],
                                          [8.0, 0.0]]))
    assert p[0] == 1.0          # ahead, whole body in view
    assert p[1] == 0.0          # 45 deg to the side: outside both cameras
    assert p[2] < 0.2           # close: the low camera sees only the legs
    assert p[3] == 1.0          # behind: rear camera
    assert p[4] == 0.0          # beyond the depth limit
    raised = meeting_scenario(sensor='camera', cam_height=1.0)
    assert camera_prob(raised, robot, np.array([0.6, 0.0])) > p[2]


def test_lidar_range_density_and_wall_merge():
    scn = meeting_scenario(sensor='lidar')
    robot = np.zeros(2)
    p = lidar_prob(scn, robot, np.array([[3.0, 0.0], [0.0, 3.0], [15.0, 0.0]]))
    assert p[0] == 1.0 and p[1] == 1.0 and p[2] == 0.0   # 360 deg; sparse far away
    walled = wall_hug_scenario(sensor='lidar')
    hugging = lidar_prob(walled, np.array([-0.7, -3.0]), np.array([-0.75, 2.0]))
    free = lidar_prob(walled, np.array([-0.7, -3.0]), np.array([-0.3, 2.0]))
    assert hugging < free


def test_track_frames_fills_only_between_detected_frames():
    det = np.array([[1, 1, 1, 0, 0, 1, 1]], dtype=bool)
    assert track_frames(det, 2).tolist() == [[True, True, True, False, False, False, True]]
    det = np.array([[1, 0, 1, 1, 1]], dtype=bool)
    assert track_frames(det, 2).tolist() == [[True, True, True, True, True]]
    assert track_frames(det, 1) is det


def test_fusion_detects_at_least_what_the_camera_does():
    seeds = episode_seeds(14, 300)
    cam, fus = wall_hug_scenario(sensor='camera'), wall_hug_scenario(sensor='fusion')
    _, _, det_cam = simulate_branches(cam, sample_episodes(cam, seeds))
    _, _, det_fus = simulate_branches(fus, sample_episodes(fus, seeds))
    assert not (det_cam & ~det_fus).any() and (det_fus & ~det_cam).any()


@pytest.mark.parametrize('make', [occlusion_scenario, wall_hug_scenario])
@pytest.mark.parametrize('sensor', ['camera', 'lidar', 'fusion'])
def test_worst_case_bound_is_valid_for_sensor_models(make, sensor):
    scn = make(sensor=sensor)
    world = sample_episodes(scn, episode_seeds(15, 300))
    state = perceive(scn, world)
    robots, agents, _ = simulate_branches(scn, world)
    assert (clearance(robots, agents) >= worst_case_bounds(scn, state)).all()


from sci1.closed_loop import always_go, closed_loop_metrics, run_closed_loop  # noqa: E402
from sci1.collect import full_trajectory  # noqa: E402
from sci1.sim2d import measurement_noise, mixed_scenario, NOISE, observe  # noqa: E402


def test_observe_without_errors_is_the_detected_truth():
    scn = occlusion_scenario(sensor='camera')
    world = sample_episodes(scn, episode_seeds(16, 200))
    robots, agents, det = simulate_branches(scn, world)
    obs, mask = observe(scn, world, robots, agents, det)
    assert np.array_equal(mask, det)
    assert np.array_equal(obs[det], agents[det]) and np.isnan(obs[~det]).all()


def test_measurement_errors_and_false_detections():
    scn = mixed_scenario(sensor='camera', **NOISE['camera_typical'])
    world = sample_episodes(scn, episode_seeds(17, 400))
    robots, agents, det = simulate_branches(scn, world)
    obs, mask = observe(scn, world, robots, agents, det)
    false = mask & ~det
    assert false.any() and (np.arange(det.shape[-1]) % 2 == 0)[None, None].repeat(
        det.shape[0], 0).repeat(det.shape[1], 1)[false].all()   # only at sensor frames
    err = np.linalg.norm(obs[det] - agents[det], axis=-1)
    dist = np.linalg.norm(robots[det] - agents[det], axis=-1)
    far, near = err[dist > 4].mean(), err[dist < 2].mean()
    assert far > near > 0                                       # grows with range
    z = np.zeros((1, 2)); z[0, 0] = 1.0
    assert np.allclose(np.linalg.norm(measurement_noise(scn, [0, 0], [[3.0, 0.0]], z)),
                       scn.meas_noise_c0 + scn.meas_noise_c2 * 9)


def test_latency_uses_an_older_view():
    fresh = occlusion_scenario()
    late = occlusion_scenario(det_latency=0.5)
    world = sample_episodes(fresh, episode_seeds(18, 2000))
    assert perceive(late, world)['hidden0'].sum() > perceive(fresh, world)['hidden0'].sum()


def test_mixed_scenario_has_both_lanes_and_strata():
    scn = mixed_scenario()
    world = sample_episodes(scn, episode_seeds(19, 1000))
    hug = world['agent_p0'][:, 0] < -0.4
    assert 0.4 < hug.mean() < 0.6
    assert set(np.unique(world['robot_v0'])) == {0.0, 0.5, 1.0}
    hidden = perceive(scn, world)['hidden0']
    assert 0.1 < (~hidden).mean() < 0.4
    robots, agents, _ = simulate_branches(scn, world)
    assert (clearance(robots, agents) >= worst_case_bounds(scn, perceive(scn, world))).all()


def test_full_trajectory_prefers_verified_truth():
    truth = np.ones((3, 4, 2))
    onboard = np.full((3, 4, 2), 2.0)
    full = full_trajectory(np.array([True, False, False]), np.array([True, True, False]),
                           truth, onboard)
    assert (full[0] == 1).all() and (full[1] == 2).all() and np.isnan(full[2]).all()


def test_closed_loop_replay_and_metrics():
    scn = mixed_scenario(closed_loop_steps=30)
    world = sample_episodes(scn, episode_seeds(20, 200))
    a = run_closed_loop(scn, world, always_go)
    b = run_closed_loop(scn, world, always_go)
    assert np.array_equal(a['agent'], b['agent']) and np.array_equal(a['det'], b['det'])
    stop = lambda state: np.full(len(state['hidden0']), YIELD)  # noqa: E731
    m = closed_loop_metrics(scn, run_closed_loop(scn, world, stop), 0.8)
    assert m['yield_share'] == 1.0 and m['time_lost_s'] > 0
    assert closed_loop_metrics(scn, a, 0.8)['time_lost_s'] == pytest.approx(0.0, abs=1e-9)


from sci1.paths import path_length, path_point, path_project, path_tangent  # noqa: E402
from sci1.sim2d import door_scenario, walkway_scenario  # noqa: E402


def test_polyline_path_geometry():
    scn = walkway_scenario(length=8.0)
    assert path_length(scn) == pytest.approx(16.0)
    pts = path_point(scn, np.array([0.0, 8.0, 12.0, 16.0, 18.0]))
    assert np.allclose(pts, [[8, 8], [8, 0], [4, 0], [0, 0], [0, -2]])
    assert np.allclose(path_tangent(scn, np.array([4.0, 12.0, 18.0])), [[0, -1], [-1, 0], [0, -1]])
    assert np.allclose(path_project(scn, pts[:4]), [0, 8, 12, 16])
    # Offset is along the local normal: walking -x, the normal points -y.
    assert np.allclose(path_point(scn, 12.0, 0.1), [4.0, -0.1])


@pytest.mark.parametrize('make', [walkway_scenario, door_scenario])
def test_head_on_nodes(make):
    scn = make()
    world = sample_episodes(scn, episode_seeds(21, 600))
    present = world['agent_s0'] > -100
    assert 0.25 < present.mean() < 0.55
    robots, agents, det = simulate_branches(scn, world)
    d = clearance(robots, agents)
    assert (d[~present] > 5).all()                     # nobody coming: no conflict
    assert (d[:, YIELD] >= 0.8).mean() > 0.95          # waiting at the node is safe
    assert (d[present, GO] < 0.8).mean() > 0.4         # entering meets the oncoming worker
    state = perceive(scn, world)
    assert (d >= worst_case_bounds(scn, state)).all()
    # The agent walks toward the robot until the entrance, then steps aside.
    moved = agents[present, GO, -1] - agents[present, GO, 0]
    assert (moved[:, 0] <= 2 * scn.agent_lateral + 1e-9).all()   # corner offset jump only


def test_head_on_phantom_moves_toward_the_robot():
    scn = walkway_scenario(sensor='camera', cam_hfov_deg=87, cam_vfov_deg=58, cam_height=0.5)
    world = sample_episodes(scn, episode_seeds(22, 50))
    state = perceive(scn, world)
    hidden = state['hidden0']
    assert hidden.any()
    ph = state['phantom_p0'][hidden]
    on_walkway = np.abs(ph[:, 1]) < 1e-9
    assert on_walkway.any()
    # Camera depth limit: the phantom sits about 5 m ahead in the walkway.
    gap = ph[on_walkway, 0] - state['robot_state0'][hidden][on_walkway, 0]
    assert np.all((gap > 4.5) & (gap < 5.5))
    pred = ConstantVelocityPhantom(scn).predict(state)
    assert (pred[hidden][on_walkway, -1, 0] < ph[on_walkway, 0]).all()
