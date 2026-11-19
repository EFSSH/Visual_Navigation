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
of sight), which reproduces the first version exactly. ``sensor`` selects
a sensor-specific model instead (camera, 2D lidar, or their fusion; see
``sensors.py``), with frames at ``sensor_rate_hz`` and tracker
interpolation between consecutive detections.

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
from .paths import path_length, path_point, path_tangent
from .sensors import channel_probs, frame_period, reliable, track_frames

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
    # Sensor-specific detection (sensors.py); 'abstract' uses det_* above.
    sensor: str = 'abstract'          # 'abstract', 'camera', 'lidar' or 'fusion'
    sensor_rate_hz: float = 5.0       # Gazebo: both depth cameras and the laser at 5 Hz
    person_height: float = 1.7
    body_half_width: float = 0.25
    cam_hfov_deg: float = 60.0        # Gazebo default optics of the RealSense model
    cam_vfov_deg: float = 46.8        # 320 x 240 at 60 deg horizontal
    cam_height: float = 0.11          # front camera above the floor
    cam_rear_height: float = 0.18     # rear camera, facing backwards; < 0 disables
    cam_max_range: float = 5.0        # depth limit of the camera plugin
    cam_area_lo: float = 0.25         # visible body fraction: never / always detected
    cam_area_hi: float = 0.55
    lidar_leg_half_span: float = 0.2  # half width of both legs at 0.15 m height
    lidar_res_deg: float = 1.0
    lidar_min_range: float = 0.12
    lidar_max_range: float = 20.0
    lidar_points_lo: float = 2.0      # points on the legs: never / always detected
    lidar_points_hi: float = 4.0
    lidar_gap_lo: float = 0.05        # leg-to-wall gap (m): merged / separate cluster
    lidar_gap_hi: float = 0.15
    # Second pedestrian lane: a share of agents hugs the side-corridor wall.
    agent_hug_share: float = 0.0
    agent_hug_x: float = -0.7
    agent_hug_lateral: float = 0.1
    # Per-episode initial robot speed drawn from these; empty: robot_speed0.
    robot_speed0_choices: Tuple[float, ...] = ()
    # Onboard measurement errors (all zero by default).
    meas_noise_c0: float = 0.0        # radial position std = c0 + c2 * d^2 (m)
    meas_noise_c2: float = 0.0
    meas_noise_tan: float = 0.0       # tangential position std per metre of range
    fp_rate: float = 0.0              # false detections per sensor frame
    fp_range: float = 5.0             # false detections appear within this range
    det_latency: float = 0.0          # detection-to-use delay (s)
    # Head-on nodes (paths.py): the agent follows this polyline to the node
    # entrance, then leaves along agent_exit_dir. agent_start is then an arc
    # length range and agent_lateral an offset along the normal. Empty: the
    # agent walks down a vertical lane at agent_x_center (crossing nodes).
    agent_path: Tuple[Tuple[float, float], ...] = ()
    agent_exit_dir: Tuple[float, float] = (0.0, -1.0)
    agent_presence: float = 1.0       # probability that anyone is coming at all
    # Closed loop (closed_loop.py): episode length and track coasting.
    closed_loop_steps: int = 0
    track_memory: float = 1.0         # keep a lost track for this long (s)

    @property
    def times(self):
        return np.arange(self.horizon + 1) * self.dt

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ('robot_start', 'agent_start', 'robot_speed0_choices', 'agent_exit_dir'):
            if key in data:
                data[key] = tuple(data[key])
        if 'agent_path' in data:
            data['agent_path'] = tuple(tuple(p) for p in data['agent_path'])
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


def mixed_scenario(wall_setback=1.0, setback_y=0.7, **overrides):
    """
    Narrow corner with two kinds of pedestrians: half walk mid-corridor, half
    hug the side-corridor wall. Some start near the crossing, so both
    decision-time strata (agent detected / not detected) have samples, and
    the robot starts at different speeds, as it would in closed loop.
    """
    overrides.setdefault('name', 'mixed')
    overrides.setdefault('occluders', (corner_block(wall_setback, setback_y),))
    overrides.setdefault('agent_hug_share', 0.5)
    overrides.setdefault('agent_hug_x', -wall_setback + 0.3)
    overrides.setdefault('agent_start', (0.3, 4.0))
    overrides.setdefault('robot_speed0_choices', (0.0, 0.5, 1.0))
    return Scenario(**overrides)


def walkway_scenario(length=8.0, half_width=0.5, **overrides):
    """
    Platform node N1: entrance of a single-lane walkway of ``length`` metres.

    The far end bends behind equipment, so an oncoming worker is hidden until
    entering the walkway, and the camera's depth limit hides the far part
    of the walkway too. A robot that enters cannot let anyone pass inside;
    the decision horizon covers the traversal (10 s). A worker leaving the
    walkway steps aside at the entrance.
    """
    overrides.setdefault('name', 'walkway')
    overrides.setdefault('agent_presence', 0.4)
    overrides.setdefault('occluders', (
        ((-30.0, length - 0.5), (half_width, 30.0)),            # equipment above the walkway
        ((0.6, length + 3.0), (-30.0, -half_width)),            # wall below the walkway
    ))
    overrides.setdefault('agent_path', ((length, 8.0), (length, 0.0), (0.0, 0.0)))
    overrides.setdefault('agent_start', (0.0, length + 6.0))
    overrides.setdefault('agent_lateral', 0.1)
    overrides.setdefault('corridor_half_width', half_width)
    overrides.setdefault('horizon', 100)
    overrides.setdefault('robot_start', (1.5, 4.0))
    overrides.setdefault('robot_speed0_choices', (0.0, 0.5, 1.0))
    return Scenario(**overrides)


def door_scenario(door_x=1.5, **overrides):
    """
    Platform node N3: a closed (blast) door on the robot's path at ``door_x``.

    A worker coming the other way is hidden until stepping through the
    door, close to the robot; the robot passing the door commits to the
    doorway. Horizon 5 s.
    """
    overrides.setdefault('name', 'door')
    overrides.setdefault('agent_presence', 0.4)
    overrides.setdefault('occluders', (((door_x, door_x + 0.1), (-30.0, 30.0)),))
    overrides.setdefault('agent_path', ((9.0, 0.0), (0.0, 0.0)))
    overrides.setdefault('agent_start', (0.0, 8.5))
    overrides.setdefault('agent_lateral', 0.1)
    overrides.setdefault('corridor_half_width', 0.5)
    overrides.setdefault('horizon', 50)
    overrides.setdefault('robot_start', (1.5, 4.0))
    overrides.setdefault('robot_speed0_choices', (0.0, 0.5, 1.0))
    return Scenario(**overrides)


def meeting_scenario(**overrides):
    """Control scenario: same crossing, no occluder."""
    overrides.setdefault('name', 'meeting')
    overrides.setdefault('occluders', ())
    return Scenario(**overrides)


SCENARIOS = {'occlusion': occlusion_scenario, 'wall_hug': wall_hug_scenario,
             'mixed': mixed_scenario, 'walkway': walkway_scenario, 'door': door_scenario,
             'meeting': meeting_scenario}

# Detection presets. ``near_range_miss`` is a placeholder for a camera-based
# person detector whose box is truncated at close range; calibrate it from
# the robot's own detection logs before drawing conclusions.
PERCEPTION = {
    'ideal': {},
    'near_range_miss': dict(det_near_range=2.0, det_min_range=0.6, det_p_near=0.2,
                            det_corr=0.8),
}

# Measurement-error presets. ``camera_typical``: RGB-D depth error growing
# with the square of range (about 4 cm at 2 m), bounding-box jitter of 1 % of
# range, one false person detection per 50 frames and 0.15 s from exposure
# to use. Placeholders until measured on the robot's own pipeline.
NOISE = {
    'none': {},
    'camera_typical': dict(meas_noise_c0=0.01, meas_noise_c2=0.01, meas_noise_tan=0.01,
                           fp_rate=0.02, det_latency=0.15),
}


def lanes(scn):
    """Pedestrian lanes as (centre x, half spread)."""
    out = [(scn.agent_x_center, scn.agent_lateral)]
    if scn.agent_hug_share > 0:
        out.append((scn.agent_hug_x, scn.agent_hug_lateral))
    return out


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
        'det2_u_episode': np.empty(n),
        'det2_u': np.empty((n, scn.horizon + 1)),
        'det2_share': np.empty((n, scn.horizon + 1)),
        'robot_v0': np.empty(n),
        'meas_noise': np.empty((n, scn.horizon + 1, 2)),
        'fp_u': np.empty((n, scn.horizon + 1)),
        'fp_pos_u': np.empty((n, scn.horizon + 1, 2)),
    }
    cl = scn.closed_loop_steps
    if cl:
        out.update({
            'cl_accel_noise': np.empty((n, cl)),
            'cl_det_u': np.empty((n, cl + 1)), 'cl_det_share': np.empty((n, cl + 1)),
            'cl_det2_u': np.empty((n, cl + 1)), 'cl_det2_share': np.empty((n, cl + 1)),
            'cl_meas_noise': np.empty((n, cl + 1, 2)), 'cl_vel_noise': np.empty((n, cl + 1, 2)),
            'cl_fp_u': np.empty((n, cl + 1)), 'cl_fp_pos_u': np.empty((n, cl + 1, 2)),
        })
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
        # Second detection channel (lidar in fusion); drawn last for the same reason.
        out['det2_u_episode'][i] = rng.uniform()
        out['det2_u'][i] = rng.uniform(size=scn.horizon + 1)
        out['det2_share'][i] = rng.uniform(size=scn.horizon + 1)
        # Later additions, again drawn last.
        lane_u, hug_offset, speed_u = rng.uniform(size=3)
        if lane_u < scn.agent_hug_share:
            out['agent_p0'][i, 0] = scn.agent_hug_x + (2 * hug_offset - 1) * scn.agent_hug_lateral
        choices = scn.robot_speed0_choices
        out['robot_v0'][i] = choices[int(speed_u * len(choices))] if choices else scn.robot_speed0
        out['meas_noise'][i] = rng.standard_normal((scn.horizon + 1, 2))
        out['fp_u'][i] = rng.uniform(size=scn.horizon + 1)
        out['fp_pos_u'][i] = rng.uniform(size=(scn.horizon + 1, 2))
        if cl:
            out['cl_accel_noise'][i] = scn.agent_noise * rng.standard_normal(cl)
            for key in ('cl_det_u', 'cl_det_share', 'cl_det2_u', 'cl_det2_share', 'cl_fp_u'):
                out[key][i] = rng.uniform(size=cl + 1)
            out['cl_meas_noise'][i] = rng.standard_normal((cl + 1, 2))
            out['cl_vel_noise'][i] = rng.standard_normal((cl + 1, 2))
            out['cl_fp_pos_u'][i] = rng.uniform(size=(cl + 1, 2))
        if scn.agent_presence < 1.0 and rng.uniform() >= scn.agent_presence:
            out['agent_p0'][i, 1] = -1000.0 if scn.agent_path else 1000.0   # nobody coming
    if scn.agent_path:
        # Reuse the lateral and start draws as normal offset and arc length.
        out['agent_offset'] = out['agent_p0'][:, 0] - scn.agent_x_center
        out['agent_s0'] = out['agent_p0'][:, 1].copy()
        out['agent_p0'] = path_point(scn, out['agent_s0'], out['agent_offset'])
    return out


def agent_velocity(scn, world, speed, s=None):
    """Agent velocity vectors (n, 2) for speeds (n,) at arc lengths s."""
    if scn.agent_path:
        return path_tangent(scn, world['agent_s0'] if s is None else s) * speed[:, None]
    return np.stack([np.zeros_like(speed), -speed], -1)


def robot_rollout(scn, robot_x0, action, v0=None):
    """Robot positions (n, H+1, 2) when holding ``action``; known to the robot."""
    robot_x0 = np.asarray(robot_x0, dtype=float)
    action = np.broadcast_to(np.asarray(action), robot_x0.shape)
    target = ACTION_SPEED[action]
    out = np.zeros(robot_x0.shape + (scn.horizon + 1, 2))
    x = robot_x0.copy()
    v = (np.full(robot_x0.shape, scn.robot_speed0) if v0 is None
         else np.broadcast_to(np.asarray(v0, dtype=float), robot_x0.shape).copy())
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


def detected(scn, world, step, robot_p, agent_p, los, pre=''):
    """
    Detector output at ``step``, given line of sight ``los``.

    Each frame reuses the episode's uniform with probability ``det_corr``
    (persistent misses, e.g. a body the detector keeps failing on) and draws
    a fresh one otherwise. Draws are part of the episode seed, and the same
    across branches, so branch replay stays exact.
    """
    def uniforms(prefix):
        shared = world[pre + prefix + '_share'][:, step] < scn.det_corr
        return np.where(shared, world[prefix + '_u_episode'], world[pre + prefix + '_u'][:, step])

    if scn.sensor == 'abstract':
        dist = np.linalg.norm(np.asarray(robot_p) - np.asarray(agent_p), axis=-1)
        return los & (uniforms('det') < detection_prob(scn, dist))
    probs = channel_probs(scn, robot_p, agent_p)
    return np.logical_or.reduce([uniforms(prefix) < p for prefix, p in zip(('det', 'det2'), probs)])


def agent_step(scn, world, robot_k, robot_next, agent_k, agent_next, speed, los_k, noise_k,
               s=None):
    """
    Advance the agent one step in place (``agent_next``); returns its new speed,
    and its new arc length on a path node (pass the current one as ``s``).

    The agent slows down, by its reaction gain, when it sees the robot
    approaching within ``reaction_range``.
    """
    robot_v = (robot_next[:, 0] - robot_k[:, 0]) / scn.dt
    dist = np.linalg.norm(robot_k - agent_k, axis=-1)
    approaching = (robot_v > 0.2) & (robot_k[:, 0] < agent_k[:, 0] + scn.corridor_half_width)
    pressure = np.clip((scn.reaction_range - dist) / scn.reaction_range, 0.0, 1.0)
    pressure = pressure * los_k * approaching
    desired = world['agent_speed'] * (1.0 - world['beta'] * pressure)
    step = scn.agent_accel * scn.dt
    speed = speed + np.clip(desired - speed, -step, step) + noise_k * scn.dt
    speed = np.clip(speed, 0.0, scn.agent_speed_max)
    if s is not None:
        s = s + speed * scn.dt
        agent_next[:] = path_point(scn, s, world['agent_offset'])
        return speed, s
    agent_next[:, 0] = agent_k[:, 0]
    agent_next[:, 1] = agent_k[:, 1] - speed * scn.dt
    return speed


def simulate(scn, world, action, with_los=False):
    """
    Roll out one branch.

    Returns robot (n, H+1, 2), agent (n, H+1, 2) and ``det`` (n, H+1): whether
    the robot's detector reported the agent at each step. The agent reacts to
    line of sight (it sees the robot whether or not the robot detects it);
    with ``with_los`` the line-of-sight mask is returned as well.
    """
    robot = robot_rollout(scn, world['robot_x0'], action, world.get('robot_v0'))
    n, steps = robot.shape[0], scn.horizon
    agent = np.zeros((n, steps + 1, 2))
    agent[:, 0] = world['agent_p0']
    los = np.zeros((n, steps + 1), dtype=bool)
    det = np.zeros((n, steps + 1), dtype=bool)
    los[:, 0] = visible(scn, robot[:, 0], agent[:, 0])
    det[:, 0] = detected(scn, world, 0, robot[:, 0], agent[:, 0], los[:, 0])
    speed = world['agent_speed'].copy()
    s = world['agent_s0'].copy() if scn.agent_path else None
    for k in range(steps):
        args = (scn, world, robot[:, k], robot[:, k + 1], agent[:, k], agent[:, k + 1],
                speed, los[:, k], world['accel_noise'][:, k])
        if s is None:
            speed = agent_step(*args)
        else:
            speed, s = agent_step(*args, s=s)
        los[:, k + 1] = visible(scn, robot[:, k + 1], agent[:, k + 1])
        det[:, k + 1] = detected(scn, world, k + 1, robot[:, k + 1], agent[:, k + 1], los[:, k + 1])
    det = track_frames(det, frame_period(scn))
    return (robot, agent, det, los) if with_los else (robot, agent, det)


def simulate_branches(scn, world, with_los=False):
    """All candidate actions from the same world: (n, A, H+1, ...) arrays."""
    branches = zip(*(simulate(scn, world, a, with_los) for a in range(len(ACTIONS))))
    return tuple(np.stack(b, 1) for b in branches)


def measurement_noise(scn, robot_p, agent_p, z):
    """Position error of one detection: radial c0 + c2 d^2, tangential tan * d."""
    rel = np.asarray(agent_p, float) - np.asarray(robot_p, float)
    dist = np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-9)
    radial = rel / dist
    tangent = np.stack([-radial[..., 1], radial[..., 0]], -1)
    sigma_r = scn.meas_noise_c0 + scn.meas_noise_c2 * dist ** 2
    sigma_t = scn.meas_noise_tan * dist
    return sigma_r * z[..., :1] * radial + sigma_t * z[..., 1:] * tangent


def false_detection_position(scn, robot_p, u):
    """Where a false person detection appears: inside the camera's view or anywhere nearby."""
    span = np.deg2rad(scn.cam_hfov_deg) if scn.sensor == 'camera' else 2 * np.pi
    angle = (u[..., 0] - 0.5) * span
    radius = 0.5 + u[..., 1] * (scn.fp_range - 0.5)
    return np.asarray(robot_p, float) + radius[..., None] * np.stack([np.cos(angle), np.sin(angle)], -1)


def observe(scn, world, robots, agents, dets, pre=''):
    """
    What the robot records: (positions, mask).

    Detected steps carry the agent position plus measurement noise. At sensor
    frames without a detection, a false detection appears with probability
    ``fp_rate`` and is recorded as the agent (the tracker cannot tell). With
    the default zero errors this is exactly the detected true positions.
    Accepts (n, S, 2) or (n, A, S, 2) trajectories.
    """
    steps = robots.shape[-2]

    def per_episode(key):
        arr = world[pre + key][:, :steps]
        return arr[:, None] if robots.ndim == 4 else arr

    pos = agents + measurement_noise(scn, robots, agents, per_episode('meas_noise'))
    frames = np.arange(steps) % frame_period(scn) == 0
    fp = (per_episode('fp_u') < scn.fp_rate) & frames & ~dets
    fp_pos = false_detection_position(scn, robots, per_episode('fp_pos_u'))
    obs = np.where(dets[..., None], pos, np.where(fp[..., None], fp_pos, np.nan))
    return obs, dets | fp


def hidden_lane_start(scn):
    """Lowest y at which a side-corridor point can be out of line of sight."""
    return min([scn.corridor_half_width] + [box[1][0] for box in scn.occluders])


def sensed_region(scn, robot_p, points, chunk=250):
    """
    (n, P): whether a static agent at each point would be detected for sure.

    Abstract model: line of sight (the first version's occlusion boundary).
    Sensor models: certain detection by at least one sensor, so the
    boundary also covers the field of view, depth limit and point density.
    """
    robot_p = np.asarray(robot_p, float)
    if scn.sensor == 'abstract':
        return visible(scn, robot_p[:, None, :], points[None])
    out = np.empty((len(robot_p), len(points)), dtype=bool)
    for start in range(0, len(robot_p), chunk):
        sl = slice(start, start + chunk)
        out[sl] = reliable(scn, robot_p[sl, None, :], points[None])
    return out


def perceive(scn, world):
    """
    What the robot knows at the decision time.

    ``hidden0``: the detector did not report the agent (out of line of sight,
    or in line of sight but missed).
    ``agent_obs0``: noisy (x, y, vx, vy) of the agent if detected, else NaN.
    ``phantom_p0``: the nearest point of the agent's lane that is out of
    line of sight (occluded or out of range), i.e. the occlusion boundary;
    with a sensor model, the nearest point the sensors do not cover for sure.
    """
    n = len(world['robot_x0'])
    v0 = world['robot_v0'] if 'robot_v0' in world else np.full(n, scn.robot_speed0)
    robot0 = np.stack([world['robot_x0'], np.zeros(n)], -1)
    # With latency, the frame in use was taken det_latency ago (the agent was
    # further up its corridor, the robot further back); the estimate is then
    # extrapolated to now with the estimated velocity.
    lag = scn.det_latency
    if scn.agent_path:
        agent_seen = path_point(scn, world['agent_s0'] - world['agent_speed'] * lag,
                                world['agent_offset'])
    else:
        agent_seen = world['agent_p0'] + np.stack([np.zeros(n), world['agent_speed'] * lag], -1)
    robot_seen = robot0 - np.stack([v0 * lag, np.zeros(n)], -1)
    los0 = visible(scn, robot_seen, agent_seen)
    detected0 = detected(scn, world, 0, robot_seen, agent_seen, los0)
    noise = scn.obs_noise * world['obs_noise']
    vel = agent_velocity(scn, world, world['agent_speed']) + noise[:, 2:]
    pos = agent_seen + noise[:, :2]
    if 'meas_noise' in world:
        pos = pos + measurement_noise(scn, robot_seen, agent_seen, world['meas_noise'][:, 0])
    obs = np.concatenate([pos + vel * lag, vel], -1)
    hidden0 = ~detected0
    if scn.fp_rate > 0:
        fp = hidden0 & (world['fp_u'][:, 0] < scn.fp_rate)
        fp_pos = false_detection_position(scn, robot_seen, world['fp_pos_u'][:, 0])
        obs[fp] = np.concatenate([fp_pos[fp], np.zeros((fp.sum(), 2))], -1)
        hidden0 = hidden0 & ~fp
    obs[hidden0] = np.nan
    return {
        'robot_state0': np.stack([world['robot_x0'], v0], -1),
        'hidden0': hidden0,
        'agent_obs0': obs,
        'phantom_p0': phantom_points(scn, robot0),
    }


def phantom_points(scn, robot_p, step=0.05):
    """
    Per robot position, the nearest lane point not covered for sure: (n, 2).

    For each pedestrian lane, the first point along it (from the crossing
    outwards) that is out of line of sight, or, with a sensor model, not
    certainly detected; across lanes, the one nearest the robot.
    """
    robot_p = np.asarray(robot_p, float)
    if scn.agent_path:
        # Along the path from the node entrance upstream.
        s = np.arange(path_length(scn), 0.0, -step)
        points = path_point(scn, s)
        unseen = ~sensed_region(scn, robot_p, points)
        first = np.where(unseen.any(1), np.argmax(unseen, axis=1), len(s) - 1)
        return points[first]
    grid_y = np.arange(hidden_lane_start(scn), scn.sensor_range + 10.0, step)
    best, best_dist = None, None
    for x_lane, _ in lanes(scn):
        points = np.stack([np.full_like(grid_y, x_lane), grid_y], -1)
        seen = sensed_region(scn, robot_p, points)
        phantom = points[np.argmax(~seen, axis=1)]
        dist = np.linalg.norm(phantom - robot_p, axis=-1)
        if best is None:
            best, best_dist = phantom, dist
        else:
            closer = dist < best_dist
            best = np.where(closer[:, None], phantom, best)
            best_dist = np.minimum(dist, best_dist)
    return best


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
