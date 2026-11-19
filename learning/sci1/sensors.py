"""
Sensor-specific person detection models for the 2D simulator.

Each model returns, per robot-agent pair, the probability that the sensor's
person detector reports the agent in one frame. The geometry comes from the
Gazebo model ``waffle_dual_rgbd`` used in this repository:

- ``camera``: front and rear RealSense depth cameras, Gazebo default optics
  (60 deg horizontal, 46.8 deg vertical field of view, 320 x 240), depth
  limit 5 m, mounted 0.11 m (front) and 0.18 m (rear) above the floor,
  5 Hz. A person is detectable when enough of the body is inside the image:
  the visible area fraction is the horizontal fraction (body points in line
  of sight and in the field of view) times the vertical fraction (body
  height inside the vertical field of view at that distance).
- ``lidar``: 2D laser at 0.15 m (ankle height), 360 deg, 1 deg resolution,
  5 Hz. A person is a leg cluster; it is detectable when it has enough
  points and is not merged with a wall by the clustering step.
- ``fusion``: late fusion, detected if either sensor detects.

Only physical and geometric failure modes are modelled. The ramps that turn
visible area, point count and wall gap into a probability (``*_lo`` /
``*_hi``) are placeholders for the detector actually deployed and must be
calibrated from its logs.
"""

import numpy as np

from .geometry import line_of_sight

SENSORS = ('abstract', 'camera', 'lidar', 'fusion')
BODY_SAMPLES = 7


def _ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def body_points(robot_p, agent_p, half_width, k=BODY_SAMPLES):
    """Points across the body, perpendicular to the sight line: (..., k, 2)."""
    robot_p, agent_p = np.broadcast_arrays(np.asarray(robot_p, float), np.asarray(agent_p, float))
    rel = agent_p - robot_p
    dist = np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-9)
    normal = np.stack([-rel[..., 1], rel[..., 0]], -1) / dist
    offsets = np.linspace(-half_width, half_width, k)
    return agent_p[..., None, :] + offsets[:, None] * normal[..., None, :]


def _visible_points(scn, robot_p, points):
    return line_of_sight(np.asarray(robot_p)[..., None, :], points, scn.occluders,
                         scn.sensor_range)


def camera_prob(scn, robot_p, agent_p):
    """Front + rear camera; the robot faces +x."""
    robot_p = np.asarray(robot_p, float)
    points = body_points(robot_p, agent_p, scn.body_half_width)
    seen = _visible_points(scn, robot_p, points)
    rel = points - robot_p[..., None, :]
    dist_pts = np.linalg.norm(rel, axis=-1)
    bearing = np.arctan2(rel[..., 1], rel[..., 0])
    dist = np.linalg.norm(np.asarray(agent_p, float) - robot_p, axis=-1)
    best = np.zeros(dist.shape)
    half_v = np.deg2rad(scn.cam_vfov_deg) / 2
    for heading, height in ((0.0, scn.cam_height), (np.pi, scn.cam_rear_height)):
        if height is None or height < 0:
            continue
        in_fov = np.abs(_wrap(bearing - heading)) <= np.deg2rad(scn.cam_hfov_deg) / 2
        in_range = dist_pts <= scn.cam_max_range
        f_h = (seen & in_fov & in_range).mean(-1)
        reach = dist * np.tan(half_v)
        top = np.minimum(scn.person_height, height + reach)
        bottom = np.maximum(0.0, height - reach)
        f_v = np.clip(top - bottom, 0.0, None) / scn.person_height
        best = np.maximum(best, f_h * f_v)
    return _ramp(best, scn.cam_area_lo, scn.cam_area_hi)


def _wall_gap(scn, agent_p):
    """Distance from the agent's centre to the nearest wall face, minus leg spread."""
    agent_p = np.asarray(agent_p, float)
    gap = np.full(agent_p.shape[:-1], np.inf)
    for (x0, x1), (y0, y1) in scn.occluders:
        dx = np.maximum.reduce([x0 - agent_p[..., 0], np.zeros(gap.shape), agent_p[..., 0] - x1])
        dy = np.maximum.reduce([y0 - agent_p[..., 1], np.zeros(gap.shape), agent_p[..., 1] - y1])
        gap = np.minimum(gap, np.hypot(dx, dy))
    return gap - scn.lidar_leg_half_span


def lidar_prob(scn, robot_p, agent_p):
    """2D laser leg detector."""
    robot_p = np.asarray(robot_p, float)
    points = body_points(robot_p, agent_p, scn.lidar_leg_half_span)
    seen = _visible_points(scn, robot_p, points).mean(-1)
    dist = np.maximum(np.linalg.norm(np.asarray(agent_p, float) - robot_p, axis=-1), 1e-6)
    in_range = (dist >= scn.lidar_min_range) & (dist <= scn.lidar_max_range)
    n_points = seen * 2 * scn.lidar_leg_half_span / (dist * np.deg2rad(scn.lidar_res_deg))
    p = _ramp(n_points, scn.lidar_points_lo, scn.lidar_points_hi)
    p = p * _ramp(_wall_gap(scn, agent_p), scn.lidar_gap_lo, scn.lidar_gap_hi)
    return p * in_range


def channel_probs(scn, robot_p, agent_p):
    """Per-sensor detection probabilities for the configured sensor set."""
    if scn.sensor == 'camera':
        return [camera_prob(scn, robot_p, agent_p)]
    if scn.sensor == 'lidar':
        return [lidar_prob(scn, robot_p, agent_p)]
    if scn.sensor == 'fusion':
        return [camera_prob(scn, robot_p, agent_p), lidar_prob(scn, robot_p, agent_p)]
    raise ValueError(f'unknown sensor {scn.sensor!r}')


def reliable(scn, robot_p, points):
    """Points where a static agent would be detected with certainty."""
    probs = channel_probs(scn, robot_p, points)
    return np.logical_or.reduce([p >= 1.0 - 1e-12 for p in probs])


def frame_period(scn):
    """Sensor frame period in simulation steps (1 for the abstract model)."""
    if scn.sensor == 'abstract':
        return 1
    return max(1, int(round(1.0 / (scn.sensor_rate_hz * scn.dt))))


def track_frames(det, period):
    """
    Detections at sensor frames (every ``period`` steps, from t0) with steps
    between two detected frames filled in, as a tracker interpolating
    between consecutive detections would report them.
    """
    if period == 1:
        return det
    steps = det.shape[-1]
    frames = np.arange(steps) % period == 0
    out = np.where(frames, det, False)
    idx = np.arange(steps)
    prev_f = (idx // period) * period
    next_f = np.minimum(prev_f + period, steps - 1)
    next_is_frame = (next_f % period == 0)
    fill = ~frames & next_is_frame
    out[..., fill] = det[..., prev_f[fill]] & det[..., next_f[fill]]
    return out
