"""
Risk-constrained action selection and the worst-case baseline (B0).

Every method produces a clearance lower bound LB(a) per candidate action;
the planner takes the fastest action with LB(a) >= d_safe and otherwise
yields. The methods differ only in how LB is obtained, so efficiency
differences come from calibration alone.
"""

import numpy as np

from .predictors import candidate_robot_trajs
from .sim2d import hidden_lane_start, lanes, sensed_region, YIELD


def select_action(bounds, d_safe):
    """Index of the fastest certified action (actions ordered fast to slow)."""
    certified = bounds >= d_safe
    first = np.argmax(certified, axis=1)
    return np.where(certified.any(axis=1), first, YIELD)


def worst_case_bounds(scn, state, grid_step=0.25, chunk=500):
    """
    B0: reachable-set clearance bound, valid for any agent with speed <= v_max.

    Visible agent: everything within v_max * t of its estimated position.
    Undetected agent: everything within v_max * t of any point of the agent's
    lane that the robot cannot vouch for - out of line of sight, or, when the
    detector degrades at close range, closer than ``det_near_range``. The
    bound ignores walls, so it is conservative. It assumes detection beyond
    ``det_near_range`` is reliable (``det_p_far`` = 1). With a sensor model,
    the points not certainly covered by any sensor are unknown, dilated by
    one grid step along the lane.
    """
    robots = candidate_robot_trajs(scn, state)[:, :, 1:]          # (n, A, H, 2)
    reach = scn.agent_speed_max * scn.times[1:]                     # (H,)
    lateral = np.concatenate([c + np.array([-w, 0.0, w]) for c, w in lanes(scn)])
    spread = max(w for _, w in lanes(scn))
    ys = np.arange(hidden_lane_start(scn), scn.sensor_range + 3.0, grid_step)
    grid = np.stack(np.meshgrid(lateral, ys, indexing='ij'), -1).reshape(-1, 2)
    grid_margin = 0.5 * np.hypot(spread, grid_step)
    unreliable = scn.det_near_range if scn.det_near_range > scn.det_min_range else 0.0
    obs_margin = 4.0 * scn.obs_noise

    n = robots.shape[0]
    bounds = np.empty(robots.shape[:2])
    hidden = state['hidden0']
    robot0 = state['robot_state0'][:, :1] * np.array([[1.0, 0.0]])
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        r = robots[sl]
        # Visible agent.
        p = state['agent_obs0'][sl, None, None, :2]
        d_vis = np.linalg.norm(r - p, axis=-1) - reach - obs_margin
        lb_vis = np.nan_to_num(d_vis, nan=np.inf).min(-1)
        # Hidden agent: phantom anywhere in the hidden part of the corridor.
        unseen = ~sensed_region(scn, robot0[sl], grid)               # (m, P)
        if scn.sensor != 'abstract':
            columns = unseen.reshape(len(unseen), len(lateral), len(ys))
            grown = columns.copy()
            grown[..., 1:] |= columns[..., :-1]
            grown[..., :-1] |= columns[..., 1:]
            unseen = grown.reshape(unseen.shape)
        if unreliable > 0:
            near = np.linalg.norm(robot0[sl, None, :] - grid[None], axis=-1)
            unseen = unseen | (near < unreliable + grid_margin)
        d_hid = np.linalg.norm(r[:, :, :, None, :] - grid[None, None, None], axis=-1)
        d_hid = d_hid - reach[None, None, :, None] - grid_margin
        d_hid = np.where(unseen[:, None, None, :], d_hid, np.inf)
        lb_hid = d_hid.min(axis=(-1, -2))
        bounds[sl] = np.where(hidden[sl, None], lb_hid, lb_vis)
    return bounds
