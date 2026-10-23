import json

from b2_nav_learning.baselines import ActionKinematic, ConstantVelocity, integrate_unicycle
from b2_nav_learning.episodes import ego_occupancy, held_command, load_episode
from b2_nav_learning.evaluate import evaluate
from b2_nav_learning.metrics import action_coverage, occupancy_metrics, pose_metrics
from b2_nav_learning.splits import assert_no_leakage, split_episodes
from b2_nav_learning.windows import build_samples, WindowSpec
import numpy as np
import pytest


def write_episode(path, route_id, v=0.2, w=0.0, duration=20.0, run_id=None):
    """
    Write a synthetic run in the ``experiment export`` format.

    An ideal unicycle drives past a static wall at y = 1 m.
    """
    dt = 0.02
    t = np.arange(0.0, duration, dt)
    pose = np.zeros((t.size, 3))
    for i in range(1, t.size):
        x, y, yaw = pose[i - 1]
        pose[i] = (x + v * np.cos(yaw) * dt, y + v * np.sin(yaw) * dt, yaw + w * dt)
    cmd_t = np.arange(0.0, duration, 0.05)
    grid_t = np.arange(0.0, duration, 0.5)
    res, size = 0.05, 200  # 10 m costmap centred near the start
    grid = np.zeros((size, size), np.int8)
    wall_row = int(round((1.0 + 5.0) / res))
    grid[wall_row:wall_row + 2, :] = 100  # y in [1.0, 1.1)
    np.savez_compressed(
        path,
        pose_t=t, pose=pose, twist=np.tile([v, w], (t.size, 1)),
        cmd_safe_t=cmd_t, cmd_safe=np.tile([v, w], (cmd_t.size, 1)),
        cmd_nav_t=cmd_t, cmd_nav=np.tile([v, w], (cmd_t.size, 1)),
        grid_t=grid_t, grid=np.repeat(grid[None], grid_t.size, axis=0),
        grid_origin=np.tile([-5.0, -5.0], (grid_t.size, 1)),
        grid_resolution=np.float64(res),
        scan_t=np.zeros(0), scan=np.zeros((0, 0), np.float32),
        safety_stop_t=np.zeros(0), safety_stop=np.zeros(0, bool),
        meta=np.asarray(json.dumps({
            'run_id': run_id or route_id, 'scenario': 'synthetic', 'route_id': route_id,
        })),
    )
    return load_episode(str(path))


def test_held_command_drops_stale_commands(tmp_path):
    episode = write_episode(tmp_path / 'a.npz', 'r1')
    assert held_command(episode, [1.0])[0] == pytest.approx([0.2, 0.0])
    assert held_command(episode, [25.0])[0] == pytest.approx([0.0, 0.0])


def test_ego_occupancy_sees_wall_to_the_left(tmp_path):
    episode = write_episode(tmp_path / 'a.npz', 'r1')
    occ, known = ego_occupancy(episode, 0, np.array([0.0, 0.0, 0.0]), cells=40, cell_m=0.1)
    assert known.all()
    # Columns run along +y (left); the wall at y = 1 m is 10 cells left of centre.
    wall_columns = np.where(occ.any(axis=0))[0]
    assert wall_columns.size == 1 and wall_columns[0] == 30


def test_windows_targets_match_ideal_motion(tmp_path):
    episode = write_episode(tmp_path / 'a.npz', 'r1')
    spec = WindowSpec()
    samples = build_samples(episode, spec)
    assert samples['target_pose'].shape[1:] == (spec.horizon, 3)
    # Straight driving: final target 10 steps * 0.2 s * 0.2 m/s ahead.
    assert samples['target_pose'][0, -1, 0] == pytest.approx(0.4, abs=1e-3)
    assert samples['target_occupancy'].shape[1:] == (spec.horizon, spec.cells, spec.cells)


def test_action_kinematic_beats_constant_velocity_on_turning_change(tmp_path):
    samples = build_samples(write_episode(tmp_path / 'a.npz', 'r1', w=0.3), WindowSpec())
    # Change the future actions: constant velocity cannot know about it.
    samples['future_action'][:, :, 1] = 0.0
    samples['target_pose'] = integrate_unicycle(samples['future_action'], 0.2)
    kinematic = pose_metrics(ActionKinematic(0.2).predict(samples)['pose'], samples['target_pose'])
    constant = pose_metrics(ConstantVelocity(0.2).predict(samples)['pose'], samples['target_pose'])
    assert kinematic['fde_m'] < 1e-9 < constant['fde_m']


def test_static_world_occupancy_is_scored_on_known_cells(tmp_path):
    samples = build_samples(write_episode(tmp_path / 'a.npz', 'r1'), WindowSpec())
    prediction = ActionKinematic(0.2).predict(samples)['occupancy']
    result = occupancy_metrics(prediction, samples['target_occupancy'], samples['target_known'])
    assert result['cells'] > 0 and result['iou'] is not None


def test_splits_keep_routes_together(tmp_path):
    episodes = [
        write_episode(tmp_path / f'{route}_{i}.npz', route, run_id=f'{route}-{i}', duration=6.0)
        for route in ('a', 'b', 'c', 'd', 'e', 'f') for i in range(2)
    ]
    splits = split_episodes(episodes, by='route', seed=1)
    assert_no_leakage(splits, by='route')
    assert sum(len(v) for v in splits.values()) == len(episodes)


def test_action_coverage_flags_unseen_actions():
    train = np.tile([0.2, 0.0], (100, 1))
    test = np.array([[0.2, 0.0], [-0.2, 0.9]])
    assert action_coverage(train, test)['test_outside_share'] == pytest.approx(0.5)


def test_evaluate_end_to_end(tmp_path):
    episodes = [
        write_episode(tmp_path / f'{route}.npz', route)
        for route in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
    ]
    result = evaluate(episodes, 'action_kinematic', WindowSpec(), split_by='route', seed=0)
    assert 'error' in result or result['pose']['fde_m'] < 1e-6
