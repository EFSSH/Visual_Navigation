"""
Closed-loop rollouts: the robot replans at every sensor frame.

The calibrated bound is still the one-decision quantity (clearance over the
next H steps if action a is held), but the robot re-decides from the
current perception at every frame, as a deployed planner would. This tests
two things the one-decision setting cannot: whether replanning rescues
agents that are detected late, and whether calibration quality still
matters once the robot can react.

Perception in the loop: detections at sensor frames (``cl_*`` draws),
measurement noise, false detections and latency as in ``observe`` and
``perceive``. A track keeps the last measurement and coasts it at constant
velocity for ``track_memory`` seconds; without an active track the agent is
treated as hidden and the phantom sits at the boundary of the region the
sensors cover for sure. The decision at t0 uses ``perceive``, so it matches
the one-decision setting.

There is no finite-sample guarantee here: the deployed decision states
differ from the logged ones (T3/T4 in the plan). Results are empirical.
"""

import numpy as np

from .planners import select_action, worst_case_bounds
from .policies import epsilon_propensities, nominal_action
from .predictors import predicted_clearance
from .conformal import lower_bounds
from .sensors import frame_period
from .sim2d import (ACTION_SPEED, agent_step, agent_velocity, detected,
                    false_detection_position, GO,
                    measurement_noise, perceive, phantom_points, visible)


def free_run_progress(scn, v0, steps):
    """Distance covered in ``steps`` holding GO from speed ``v0``."""
    v = np.asarray(v0, float).copy()
    x = np.zeros_like(v)
    dv = scn.robot_accel * scn.dt
    for _ in range(steps):
        v = v + np.clip(ACTION_SPEED[GO] - v, -dv, dv)
        x = x + v * scn.dt
    return x


def run_closed_loop(scn, world, decide, replan=True, phantom_step=0.1):
    """
    Roll out ``scn.closed_loop_steps`` steps; ``decide(state) -> actions (n,)``.

    With ``replan=False`` the t0 decision is held for the whole episode
    (the one-decision setting, over the longer window).
    """
    steps = scn.closed_loop_steps
    if steps <= 0:
        raise ValueError('set closed_loop_steps > 0')
    n = len(world['robot_x0'])
    dt, period = scn.dt, frame_period(scn)
    lag = int(round(scn.det_latency / dt))

    robot = np.zeros((n, steps + 1, 2))
    robot[:, 0, 0] = world['robot_x0']
    agent = np.zeros((n, steps + 1, 2))
    agent[:, 0] = world['agent_p0']
    speeds = np.zeros((n, steps + 1))
    speeds[:, 0] = world['robot_v0']
    los = np.zeros((n, steps + 1), dtype=bool)
    det = np.zeros((n, steps + 1), dtype=bool)
    meas = np.full((n, steps + 1, 4), np.nan)
    actions = np.zeros((n, steps), dtype=int)
    hidden_log = []

    state0 = perceive(scn, world)
    track = state0['agent_obs0'].copy()          # (x, y, vx, vy) at track_time
    track_time = np.where(state0['hidden0'], -np.inf, 0.0)
    agent_speed = world['agent_speed'].copy()
    s = world['agent_s0'].copy() if scn.agent_path else None
    action = None

    for k in range(steps + 1):
        los[:, k] = visible(scn, robot[:, k], agent[:, k])
        if k % period == 0:
            det[:, k] = detected(scn, world, k, robot[:, k], agent[:, k], los[:, k], pre='cl_')
            z = world['cl_meas_noise'][:, k]
            pos = agent[:, k] + scn.obs_noise * z + measurement_noise(scn, robot[:, k], agent[:, k], z)
            vel = (agent_velocity(scn, world, agent_speed, s)
                   + scn.obs_noise * world['cl_vel_noise'][:, k])
            frame = np.concatenate([pos, vel], -1)
            fp = ~det[:, k] & (world['cl_fp_u'][:, k] < scn.fp_rate)
            if fp.any():
                fp_pos = false_detection_position(scn, robot[:, k], world['cl_fp_pos_u'][:, k])
                frame[fp] = np.concatenate([fp_pos[fp], np.zeros((fp.sum(), 2))], -1)
            meas[:, k] = np.where((det[:, k] | fp)[:, None], frame, np.nan)
        if k == steps:
            break

        if k == 0 or (replan and k % period == 0):
            if k > 0:
                j = k - lag                              # newest frame available now
                if j >= 0 and j % period == 0:
                    fresh = ~np.isnan(meas[:, j, 0])
                    track[fresh] = meas[fresh, j]
                    track_time[fresh] = j * dt
                elif j >= 0:
                    j0 = j - j % period
                    fresh = ~np.isnan(meas[:, j0, 0]) & (track_time < j0 * dt)
                    track[fresh] = meas[fresh, j0]
                    track_time[fresh] = j0 * dt
                age = k * dt - track_time
                active = age <= scn.track_memory
                estimate = np.full((n, 4), np.nan)
                estimate[active, :2] = track[active, :2] + track[active, 2:] * age[active, None]
                estimate[active, 2:] = track[active, 2:]
                state = {'robot_state0': np.stack([robot[:, k, 0], speeds[:, k]], -1),
                         'hidden0': ~active, 'agent_obs0': estimate,
                         'phantom_p0': phantom_points(scn, robot[:, k], phantom_step)}
            else:
                state = state0
            action = np.asarray(decide(state))
            hidden_log.append(state['hidden0'])
        actions[:, k] = action

        dv = scn.robot_accel * dt
        speeds[:, k + 1] = speeds[:, k] + np.clip(ACTION_SPEED[action] - speeds[:, k], -dv, dv)
        robot[:, k + 1, 0] = robot[:, k, 0] + speeds[:, k + 1] * dt
        args = (scn, world, robot[:, k], robot[:, k + 1], agent[:, k], agent[:, k + 1],
                agent_speed, los[:, k], world['cl_accel_noise'][:, k])
        if s is None:
            agent_speed = agent_step(*args)
        else:
            agent_speed, s = agent_step(*args, s=s)

    return {'robot': robot, 'agent': agent, 'speed': speeds, 'det': det, 'los': los,
            'actions': actions, 'hidden_at_decisions': np.stack(hidden_log, 1)}


def closed_loop_metrics(scn, out, d_safe):
    """Safety and efficiency of one closed-loop run."""
    d = np.linalg.norm(out['robot'][:, 1:] - out['agent'][:, 1:], axis=-1)
    d_min = d.min(-1)
    k_min = d.argmin(-1)
    near = d_min < d_safe
    steps = scn.closed_loop_steps
    progress = out['robot'][:, -1, 0] - out['robot'][:, 0, 0]
    lost = (free_run_progress(scn, out['speed'][:, 0], steps) - progress) / ACTION_SPEED[GO]
    det = out['det'][:, 1:]
    ever = det.any(-1)
    first = det.argmax(-1)
    lead = (k_min - first) * scn.dt
    stop_time = scn.robot_speed0 / scn.robot_accel
    return {
        'near_miss_rate': float(near.mean()),
        'contact_rate': float((d_min < 0.4).mean()),
        'time_lost_s': float(lost.mean()),
        'stop_rate': float((out['speed'][:, 2:] < 0.05).any(-1).mean()),
        'yield_share': float((out['actions'] == 2).mean()),
        'hidden_decision_share': float(out['hidden_at_decisions'].mean()),
        'near_miss_late_or_never': float((near & (~ever | (lead < stop_time))).sum()
                                         / max(near.sum(), 1)),
    }


def conformal_decider(scn, predictor, cal, method, alpha, epsilon, verification, d_safe):
    def decide(state):
        d_hat = predicted_clearance(scn, state, predictor)
        mu = epsilon_propensities(nominal_action(d_hat, state['hidden0']),
                                  np.ones(d_hat.shape, dtype=bool), epsilon)
        test = {'d_hat': d_hat, 'stratum': state['hidden0'].astype(int), 'mu': mu,
                'rho': verification(state['hidden0'])}
        bounds, _ = lower_bounds(method, cal, test, alpha)
        return select_action(bounds, d_safe)
    return decide


def worst_case_decider(scn, d_safe, grid_step=0.5):
    def decide(state):
        return select_action(worst_case_bounds(scn, state, grid_step=grid_step), d_safe)
    return decide


def always_go(state):
    return np.full(len(state['hidden0']), GO)
