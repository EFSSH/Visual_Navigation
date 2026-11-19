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


def corner_block(wall_setback):
    """Upper-left wall block whose corner sits at (-setback, +setback)."""
    return ((-30.0, -wall_setback), (wall_setback, 30.0))


def occlusion_scenario(wall_setback=1.75, **overrides):
    """
    Blind-corner scenario.

    ``wall_setback`` is the knob for how often the agent is hidden at the
    decision time: about 0.86 / 0.74 / 0.59 / 0.42 of episodes for a setback
    of 1.25 / 1.5 / 1.75 / 2.0 m with the default start distributions.
    """
    overrides.setdefault('occluders', (corner_block(wall_setback),))
    return Scenario(**overrides)


def meeting_scenario(**overrides):
    """Control scenario: same crossing, no occluder."""
    overrides.setdefault('name', 'meeting')
    overrides.setdefault('occluders', ())
    return Scenario(**overrides)


SCENARIOS = {'occlusion': occlusion_scenario, 'meeting': meeting_scenario}


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
    }
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        out['robot_x0'][i] = -rng.uniform(*scn.robot_start)
        out['agent_p0'][i] = (rng.uniform(-scn.agent_lateral, scn.agent_lateral),
                              rng.uniform(*scn.agent_start))
        speed = scn.agent_speed_median * np.exp(scn.agent_speed_sigma * rng.standard_normal())
        out['agent_speed'][i] = np.clip(speed, 0.2, scn.agent_speed_max)
        out['beta'][i] = scn.reaction * rng.uniform()
        out['accel_noise'][i] = scn.agent_noise * rng.standard_normal(scn.horizon)
        out['obs_noise'][i] = rng.standard_normal(4)
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
    return line_of_sight(p, q, scn.occluders, scn.sensor_range)


def simulate(scn, world, action):
    """
    Roll out one branch.

    Returns robot (n, H+1, 2), agent (n, H+1, 2) and ``vis`` (n, H+1): whether
    the robot can see the agent at each step (visibility is symmetric, and the
    agent reacts only to a robot it can see).
    """
    robot = robot_rollout(scn, world['robot_x0'], action)
    n, steps = robot.shape[0], scn.horizon
    agent = np.zeros((n, steps + 1, 2))
    agent[:, 0] = world['agent_p0']
    vis = np.zeros((n, steps + 1), dtype=bool)
    vis[:, 0] = visible(scn, robot[:, 0], agent[:, 0])
    speed = world['agent_speed'].copy()
    preferred = world['agent_speed']
    step = scn.agent_accel * scn.dt
    for k in range(steps):
        robot_v = (robot[:, k + 1, 0] - robot[:, k, 0]) / scn.dt
        dist = np.linalg.norm(robot[:, k] - agent[:, k], axis=-1)
        approaching = (robot_v > 0.2) & (robot[:, k, 0] < agent[:, k, 0] + scn.corridor_half_width)
        pressure = np.clip((scn.reaction_range - dist) / scn.reaction_range, 0.0, 1.0)
        pressure = pressure * vis[:, k] * approaching
        desired = preferred * (1.0 - world['beta'] * pressure)
        speed = speed + np.clip(desired - speed, -step, step) + world['accel_noise'][:, k] * scn.dt
        speed = np.clip(speed, 0.0, scn.agent_speed_max)
        agent[:, k + 1, 0] = agent[:, k, 0]
        agent[:, k + 1, 1] = agent[:, k, 1] - speed * scn.dt
        vis[:, k + 1] = visible(scn, robot[:, k + 1], agent[:, k + 1])
    return robot, agent, vis


def simulate_branches(scn, world):
    """All candidate actions from the same world: (n, A, H+1, ...) arrays."""
    robots, agents, viss = zip(*(simulate(scn, world, a) for a in range(len(ACTIONS))))
    return np.stack(robots, 1), np.stack(agents, 1), np.stack(viss, 1)


def perceive(scn, world):
    """
    What the robot knows at the decision time.

    ``agent_obs0``: noisy (x, y, vx, vy) of the agent if visible, else NaN.
    ``phantom_p0``: the nearest point of the side corridor centreline that
    is hidden (occluded or out of range), i.e. the occlusion boundary.
    """
    n = len(world['robot_x0'])
    robot0 = np.stack([world['robot_x0'], np.zeros(n)], -1)
    hidden0 = ~visible(scn, robot0, world['agent_p0'])
    noise = scn.obs_noise * world['obs_noise']
    obs = np.concatenate([
        world['agent_p0'] + noise[:, :2],
        np.stack([np.zeros(n), -world['agent_speed']], -1) + noise[:, 2:],
    ], -1)
    obs[hidden0] = np.nan
    grid_y = np.arange(scn.corridor_half_width, scn.sensor_range + 10.0, 0.05)
    points = np.stack([np.zeros_like(grid_y), grid_y], -1)
    seen = visible(scn, robot0[:, None, :], points[None])
    first_hidden = np.argmax(~seen, axis=1)
    phantom = np.stack([np.zeros(n), grid_y[first_hidden]], -1)
    return {
        'robot_state0': np.stack([world['robot_x0'], np.full(n, scn.robot_speed0)], -1),
        'hidden0': hidden0,
        'agent_obs0': obs,
        'phantom_p0': phantom,
    }


def clearance(robot, agent, vis=None):
    """
    Minimum robot-agent distance over steps 1..H.

    With ``vis`` given, only visible steps count (inf if none): the clearance
    a robot would compute from what it actually saw.
    """
    d = np.linalg.norm(robot[..., 1:, :] - agent[..., 1:, :], axis=-1)
    if vis is not None:
        d = np.where(vis[..., 1:], d, np.inf)
    return d.min(axis=-1)


TRAJECTORY_TYPES = ('always_visible', 'lost_from_view', 'emerging', 'never_seen')


def trajectory_type(vis):
    """
    Classify each outcome by what the robot sees of it.

    0 always visible; 1 visible at t0 then lost; 2 hidden at t0 and emerges
    within the horizon; 3 hidden for the whole horizon.
    """
    vis = np.asarray(vis)
    start, later = vis[..., 0], vis[..., 1:]
    full, ever = later.all(-1), later.any(-1)
    return np.select(
        [start & full, start & ~full, ~start & ever],
        [0, 1, 2], default=3)
