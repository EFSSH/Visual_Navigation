"""
2D blind-corner interaction simulator.

Layout (metres): the robot drives along the main corridor (y = 0) towards a
crossing at the origin; the other agent walks down the side corridor
(x ~ 0) towards -y. A wall block in the upper-left quadrant hides the side
corridor from the robot until it gets close to the corner.

One key decision per episode: at t = 0 the robot picks an action from
``ACTIONS`` and holds its target speed for the whole horizon. This fixes the
continuation policy, so the potential outcome Y(a) - the agent trajectory
over the horizon had the robot executed ``a`` - is well defined.

The agent reacts to the robot: when it can see the robot approaching, it
slows down in proportion to an individual reaction gain. Outcomes therefore
depend on the robot's action, which is what makes the counterfactual and
closed-loop part of the problem real.

Perception: line of sight (geometry) and detection (the robot's detector)
are separate. The agent reacts to line of sight; everything the robot
records uses detection. Detection probability drops at close range, where a
camera-based person detector sees a truncated body, and misses are
correlated over time. The default is ideal detection (detected iff in line
of sight), which reproduces the first version exactly.

Branch replay: all randomness of an episode (agent parameters and
per-step acceleration noise) is drawn from that episode's own seed before
the rollout. Re-running any action with the same seed reproduces the same
world, so every candidate action can be rolled out as a separate branch to
obtain simulator ground truth for counterfactual actions.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

from .geometry import line_of_sight

ACTIONS = ('go', 'slow', 'yield')
GO, SLOW, YIELD = 0, 1, 2
ACTION_SPEED = np.array([1.0, 0.5, 0.0])

Box = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True)
class Scenario:
    """All parameters of one simulated scenario."""

    name: str = 'occlusion'
    dt: float = 0.1
    horizon: int = 30                 # steps; 3 s at dt = 0.1
    robot_speed0: float = 1.0
    robot_accel: float = 1.0
    robot_start: Tuple[float, float] = (2.0, 5.0)   # distance to crossing
    agent_start: Tuple[float, float] = (1.0, 9.0)   # initial y in side corridor
    agent_lateral: float = 0.3
    agent_speed_median: float = 1.0
    agent_speed_sigma: float = 0.35   # log-normal spread of preferred speed
    agent_speed_max: float = 2.2      # hard cap; also the worst-case bound
    agent_accel: float = 1.5
    agent_noise: float = 0.3          # std of acceleration noise (m/s^2)
    reaction: float = 1.0             # scales per-agent gain beta ~ U[0, 1]
    reaction_range: float = 4.0
    corridor_half_width: float = 1.0
    occluders: Tuple[Box, ...] = (((-30.0, -1.0), (1.0, 30.0)),)
    sensor_range: float = 12.0
    obs_noise: float = 0.05           # std of position / velocity estimates
    agent_x_center: float = 0.0       # lane of the agent in the side corridor
    # Detection model, see ``detection_prob``. Defaults: ideal detection.
    det_p_far: float = 1.0            # detection probability beyond det_near_range
    det_near_range: float = 0.0       # detection degrades below this distance (m)
    det_min_range: float = 0.5        # ... down to det_p_near at this distance
    det_p_near: float = 1.0
    det_corr: float = 0.0             # 0 independent misses per frame, 1 fully persistent

    @property
    def times(self):
        return np.arange(self.horizon + 1) * self.dt

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ('robot_start', 'agent_start'):
            data[key] = tuple(data[key])
        data['occluders'] = tuple(
            tuple(tuple(axis) for axis in box) for box in data['occluders'])
        return cls(**data)


def corner_block(wall_setback, setback_y=None):
    """Upper-left wall block whose corner sits at (-setback, +setback_y)."""
    setback_y = wall_setback if setback_y is None else setback_y
    return ((-30.0, -wall_setback), (setback_y, 30.0))


def occlusion_scenario(wall_setback=1.75, **overrides):
    """
    Blind-corner scenario.

    ``wall_setback`` is the knob for how often the agent is hidden at the
    decision time: about 0.86 / 0.74 / 0.59 / 0.42 of episodes for a setback
    of 1.25 / 1.5 / 1.75 / 2.0 m with the default start distributions.
    """
    overrides.setdefault('occluders', (corner_block(wall_setback),))
    return Scenario(**overrides)


def wall_hug_scenario(wall_setback=1.0, setback_y=0.7, **overrides):
    """
    Wall-hugging scenario: the agent walks close to the side-corridor wall
    (x = -setback) and the main corridor is narrow (wall 0.7 m from the
    robot's path), so the agent leaves the corner only 1.3 - 2 m from the
    robot.
    """
    overrides.setdefault('name', 'wall_hug')
    overrides.setdefault('occluders', (corner_block(wall_setback, setback_y),))
    overrides.setdefault('agent_x_center', -wall_setback + 0.3)
    overrides.setdefault('agent_lateral', 0.1)
    return Scenario(**overrides)


def meeting_scenario(**overrides):
    """Control scenario: same crossing, no occluder."""
    overrides.setdefault('name', 'meeting')
    overrides.setdefault('occluders', ())
    return Scenario(**overrides)


SCENARIOS = {'occlusion': occlusion_scenario, 'wall_hug': wall_hug_scenario,
             'meeting': meeting_scenario}

# Detection presets. ``near_range_miss`` is a placeholder for a camera-based
# person detector whose box is truncated at close range; calibrate it from
# the robot's own detection logs before drawing conclusions.
PERCEPTION = {
    'ideal': {},
    'near_range_miss': dict(det_near_range=2.0, det_min_range=0.6, det_p_near=0.2,
                            det_corr=0.8),
}


def episode_seeds(base_seed, n):
    """Independent per-episode seeds; recorded so any episode can be replayed."""
    return np.random.SeedSequence(base_seed).generate_state(n, dtype=np.uint64)


def sample_episodes(scn, seeds):
    """Draw each episode's world from its own seed (branch-replay safe)."""
    n = len(seeds)
    out = {
        'seed': np.asarray(seeds, dtype=np.uint64),
        'robot_x0': np.empty(n),
        'agent_p0': np.empty((n, 2)),
        'agent_speed': np.empty(n),
        'beta': np.empty(n),
        'accel_noise': np.empty((n, scn.horizon)),
        'obs_noise': np.empty((n, 4)),
        'det_u_episode': np.empty(n),
        'det_u': np.empty((n, scn.horizon + 1)),
        'det_share': np.empty((n, scn.horizon + 1)),
    }
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        out['robot_x0'][i] = -rng.uniform(*scn.robot_start)
        out['agent_p0'][i] = (scn.agent_x_center
                              + rng.uniform(-scn.agent_lateral, scn.agent_lateral),
                              rng.uniform(*scn.agent_start))
        speed = scn.agent_speed_median * np.exp(scn.agent_speed_sigma * rng.standard_normal())
        out['agent_speed'][i] = np.clip(speed, 0.2, scn.agent_speed_max)
        out['beta'][i] = scn.reaction * rng.uniform()
        out['accel_noise'][i] = scn.agent_noise * rng.standard_normal(scn.horizon)
        out['obs_noise'][i] = rng.standard_normal(4)
        # Detection draws come last so earlier draws match the first version.
        out['det_u_episode'][i] = rng.uniform()
        out['det_u'][i] = rng.uniform(size=scn.horizon + 1)
        out['det_share'][i] = rng.uniform(size=scn.horizon + 1)
    return out


def robot_rollout(scn, robot_x0, action):
    """Robot positions (n, H+1, 2) when holding ``action``; known to the robot."""
    robot_x0 = np.asarray(robot_x0, dtype=float)
    action = np.broadcast_to(np.asarray(action), robot_x0.shape)
    target = ACTION_SPEED[action]
    out = np.zeros(robot_x0.shape + (scn.horizon + 1, 2))
    x = robot_x0.copy()
    v = np.full(robot_x0.shape, scn.robot_speed0)
    out[..., 0, 0] = x
    step = scn.robot_accel * scn.dt
    for k in range(scn.horizon):
        v = v + np.clip(target - v, -step, step)
        x = x + v * scn.dt
        out[..., k + 1, 0] = x
    return out


def visible(scn, p, q):
    """Line of sight (geometry only)."""
    return line_of_sight(p, q, scn.occluders, scn.sensor_range)


def detection_prob(scn, dist):
    """
    Probability that the detector reports an agent in line of sight.

    ``det_p_far`` beyond ``det_near_range``, falling linearly to
    ``det_p_near`` at ``det_min_range`` and below.
    """
    dist = np.asarray(dist, dtype=float)
    if scn.det_near_range <= scn.det_min_range:
        return np.full(dist.shape, scn.det_p_far)
    frac = np.clip((dist - scn.det_min_range) / (scn.det_near_range - scn.det_min_range), 0, 1)
    return scn.det_p_near + (scn.det_p_far - scn.det_p_near) * frac


def detected(scn, world, step, robot_p, agent_p, los):
    """
    Detector output at ``step``, given line of sight ``los``.

    Each frame reuses the episode's uniform with probability ``det_corr``
    (persistent misses, e.g. a body the detector keeps failing on) and draws
    a fresh one otherwise. Draws are part of the episode seed, and the same
    across branches, so branch replay stays exact.
    """
    shared = world['det_share'][:, step] < scn.det_corr
    u = np.where(shared, world['det_u_episode'], world['det_u'][:, step])
    dist = np.linalg.norm(np.asarray(robot_p) - np.asarray(agent_p), axis=-1)
    return los & (u < detection_prob(scn, dist))


def simulate(scn, world, action, with_los=False):
    """
    Roll out one branch.

    Returns robot (n, H+1, 2), agent (n, H+1, 2) and ``det`` (n, H+1): whether
    the robot's detector reported the agent at each step. The agent reacts to
    line of sight (it sees the robot whether or not the robot detects it);
    with ``with_los`` the line-of-sight mask is returned as well.
    """
    robot = robot_rollout(scn, world['robot_x0'], action)
    n, steps = robot.shape[0], scn.horizon
    agent = np.zeros((n, steps + 1, 2))
    agent[:, 0] = world['agent_p0']
    los = np.zeros((n, steps + 1), dtype=bool)
    det = np.zeros((n, steps + 1), dtype=bool)
    los[:, 0] = visible(scn, robot[:, 0], agent[:, 0])
    det[:, 0] = detected(scn, world, 0, robot[:, 0], agent[:, 0], los[:, 0])
    speed = world['agent_speed'].copy()
    preferred = world['agent_speed']
    step = scn.agent_accel * scn.dt
    for k in range(steps):
        robot_v = (robot[:, k + 1, 0] - robot[:, k, 0]) / scn.dt
        dist = np.linalg.norm(robot[:, k] - agent[:, k], axis=-1)
        approaching = (robot_v > 0.2) & (robot[:, k, 0] < agent[:, k, 0] + scn.corridor_half_width)
        pressure = np.clip((scn.reaction_range - dist) / scn.reaction_range, 0.0, 1.0)
        pressure = pressure * los[:, k] * approaching
        desired = preferred * (1.0 - world['beta'] * pressure)
        speed = speed + np.clip(desired - speed, -step, step) + world['accel_noise'][:, k] * scn.dt
        speed = np.clip(speed, 0.0, scn.agent_speed_max)
        agent[:, k + 1, 0] = agent[:, k, 0]
        agent[:, k + 1, 1] = agent[:, k, 1] - speed * scn.dt
        los[:, k + 1] = visible(scn, robot[:, k + 1], agent[:, k + 1])
        det[:, k + 1] = detected(scn, world, k + 1, robot[:, k + 1], agent[:, k + 1], los[:, k + 1])
    return (robot, agent, det, los) if with_los else (robot, agent, det)


def simulate_branches(scn, world, with_los=False):
    """All candidate actions from the same world: (n, A, H+1, ...) arrays."""
    branches = zip(*(simulate(scn, world, a, with_los) for a in range(len(ACTIONS))))
    return tuple(np.stack(b, 1) for b in branches)


def hidden_lane_start(scn):
    """Lowest y at which a side-corridor point can be out of line of sight."""
    return min([scn.corridor_half_width] + [box[1][0] for box in scn.occluders])


def perceive(scn, world):
    """
    What the robot knows at the decision time.

    ``hidden0``: the detector did not report the agent (out of line of sight,
    or in line of sight but missed).
    ``agent_obs0``: noisy (x, y, vx, vy) of the agent if detected, else NaN.
    ``phantom_p0``: the nearest point of the agent's lane that is out of
    line of sight (occluded or out of range), i.e. the occlusion boundary.
    """
    n = len(world['robot_x0'])
    robot0 = np.stack([world['robot_x0'], np.zeros(n)], -1)
    los0 = visible(scn, robot0, world['agent_p0'])
    hidden0 = ~detected(scn, world, 0, robot0, world['agent_p0'], los0)
    noise = scn.obs_noise * world['obs_noise']
    obs = np.concatenate([
        world['agent_p0'] + noise[:, :2],
        np.stack([np.zeros(n), -world['agent_speed']], -1) + noise[:, 2:],
    ], -1)
    obs[hidden0] = np.nan
    grid_y = np.arange(hidden_lane_start(scn), scn.sensor_range + 10.0, 0.05)
    points = np.stack([np.full_like(grid_y, scn.agent_x_center), grid_y], -1)
    seen = visible(scn, robot0[:, None, :], points[None])
    first_hidden = np.argmax(~seen, axis=1)
    phantom = np.stack([np.full(n, scn.agent_x_center), grid_y[first_hidden]], -1)
    return {
        'robot_state0': np.stack([world['robot_x0'], np.full(n, scn.robot_speed0)], -1),
        'hidden0': hidden0,
        'agent_obs0': obs,
        'phantom_p0': phantom,
    }


def clearance(robot, agent, vis=None):
    """
    Minimum robot-agent distance over steps 1..H.

    With ``vis`` given, only those steps count (inf if none): the clearance
    a robot would compute from what its detector reported.
    """
    d = np.linalg.norm(robot[..., 1:, :] - agent[..., 1:, :], axis=-1)
    if vis is not None:
        d = np.where(vis[..., 1:], d, np.inf)
    return d.min(axis=-1)


TRAJECTORY_TYPES = ('always_visible', 'lost_from_view', 'emerging', 'never_seen')


def trajectory_type(vis):
    """
    Classify each outcome by what the robot detects of it.

    0 always visible; 1 visible at t0 then lost; 2 hidden at t0 and emerges
    within the horizon; 3 hidden for the whole horizon.
    """
    vis = np.asarray(vis)
    start, later = vis[..., 0], vis[..., 1:]
    full, ever = later.all(-1), later.any(-1)
    return np.select(
        [start & full, start & ~full, ~start & ever],
        [0, 1, 2], default=3)
