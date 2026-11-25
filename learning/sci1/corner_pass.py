"""
SCI-1 v6 corner-pass study: how often does the onboard camera log miss a
contact (or a proxy event) when robot and worker meet at an L-shaped corner
of a narrow corridor?

Setting (all lengths in metres, placeholders until measured):

- L-shaped corridor of width ``width``. The robot drives along the
  horizontal leg (+x), turns left round the inner corner and continues up
  the vertical leg (+y), keeping to the inner side. A worker walks the other
  way: down the vertical leg, round the corner, along the horizontal leg.
- Sidestep: a worker who reacts steps to the outer wall; the gap between a
  sidestepped worker and the robot is then about 0.2 m (``robot_radius``,
  ``person_radius_side``, ``width``). Attentive workers react when they see
  the robot within ``react_range_attentive``; inattentive ones only very
  close (``react_range_inattentive``).
- Robot: holds the decided speed (go 1.0 / slow 0.5 m/s) under a fixed
  vision-only reactive safety layer: when a tracked person ahead is within
  the stopping distance plus a margin, brake to a stop, and resume after the
  track has been lost for ``resume_after`` seconds.
- Cameras: each is (heading relative to the robot, horizontal FOV, vertical
  FOV, mounting height). Detection follows ``sensors.camera_prob``: visible
  body fraction (line of sight x horizontal FOV) times vertical fraction,
  through the same ramp, at ``rate_hz`` with tracker interpolation.

Events per episode (ground truth vs onboard log):

- contact: outline distance <= 0 at some step;
- proxy: the person is in the robot's path ahead within the current
  stopping distance while the robot is moving and not braking;
- onboard versions: the same, evaluated only at steps the camera log
  covers (positions as detected). Missing a step can only remove events.

Run: PYTHONPATH=learning python -m sci1.corner_pass --n 20000
"""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from typing import Tuple

import numpy as np

from .geometry import line_of_sight
from .sensors import body_points, track_frames

Camera = Tuple[float, float, float, float]   # heading (deg), hfov, vfov, height

CAMERA_CONFIGS = {
    'gazebo_60_h0.11': ((0.0, 60.0, 46.8, 0.11), (180.0, 60.0, 46.8, 0.18)),
    'front_87_h0.5': ((0.0, 87.0, 58.0, 0.5),),
    'front_rear_87_h0.5': ((0.0, 87.0, 58.0, 0.5), (180.0, 87.0, 58.0, 0.5)),
    'front_120_h0.5': ((0.0, 120.0, 90.0, 0.5),),
    'ring4_87_h0.5': tuple((h, 87.0, 58.0, 0.5) for h in (0.0, 90.0, 180.0, 270.0)),
    'ring4_87_h1.0': tuple((h, 87.0, 58.0, 1.0) for h in (0.0, 90.0, 180.0, 270.0)),
}

ACTION_SPEEDS = {'go': 1.0, 'medium': 0.7, 'slow': 0.5, 'crawl': 0.3}


@dataclass(frozen=True)
class CornerScenario:
    dt: float = 0.05
    steps: int = 200                     # 10 s
    width: float = 1.0
    robot_radius: float = 0.22
    robot_inner_margin: float = 0.05     # robot edge to inner wall
    turn_radius: float = 0.25
    robot_start: Tuple[float, float] = (2.0, 4.0)    # distance before the turn
    person_start: Tuple[float, float] = (0.5, 5.0)   # distance up the side leg
    person_radius_walk: float = 0.25
    person_radius_side: float = 0.15
    person_height: float = 1.7
    person_speed_median: float = 1.0
    person_speed_sigma: float = 0.3
    person_speed_max: float = 1.8
    person_lateral: float = 0.1          # initial offset spread around the centreline
    attentive_share: float = 0.5
    react_range_attentive: float = 3.0
    react_range_inattentive: Tuple[float, float] = (0.6, 1.2)
    react_delay: float = 0.3
    sidestep_speed: float = 0.6
    react_speed_factor: float = 0.5
    robot_accel: float = 0.5
    brake_decel: float = 1.5
    stop_margin: float = 0.3
    closing_speed: float = 1.0           # assumed walking speed towards the robot
    path_margin: float = 0.1             # lateral margin of the swept path check
    near_gap: float = 0.10               # alternative proxy: gap below this
    track_memory: float = 0.5
    resume_after: float = 1.0
    rate_hz: float = 5.0
    cameras: Tuple[Camera, ...] = CAMERA_CONFIGS['front_87_h0.5']
    cam_max_range: float = 5.0
    cam_area_lo: float = 0.25
    cam_area_hi: float = 0.55
    # Optional per-camera ramps ((hfov, lo, hi), ...), e.g. measured in Gazebo;
    # cameras whose hfov is not listed use cam_area_lo / cam_area_hi.
    cam_ramps: Tuple[Tuple[float, float, float], ...] = ()
    sensor_range: float = 12.0
    # Perception imperfections (defaults reproduce the ideal detector):
    recog_latency: float = 0.0           # s from image to usable detection
    confirm_frames: int = 1              # consecutive hits before a track is confirmed
    turn_blur: float = 0.0               # detection prob x (1 - turn_blur * min(1, omega / 2))
    extrapolate: bool = False            # safety layer predicts the track forward at its velocity

    @property
    def half_width(self):
        return self.width / 2

    @property
    def occluders(self):
        hw = self.half_width
        return (((-30.0, -hw), (hw, 30.0)),      # inner corner block
                ((-30.0, 30.0), (-30.0, -hw)),   # outer wall of the horizontal leg
                ((hw, 30.0), (-hw, 30.0)))       # outer wall of the vertical leg

    @property
    def side_offset(self):
        """Lateral offset of a sidestepped worker (towards the outer wall)."""
        return self.half_width - self.person_radius_side

    def passing_gap(self):
        """Gap between the robot and a sidestepped worker on a straight leg."""
        robot_u = self.half_width - self.robot_radius - self.robot_inner_margin
        return self.side_offset - self.person_radius_side + robot_u - self.robot_radius


class Polyline:
    """Arc-length parametrised polyline with tangents and left normals."""

    def __init__(self, points):
        self.points = np.asarray(points, float)
        seg = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(seg, axis=1)
        self.dirs = seg / self.lengths[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.lengths)])

    def _idx(self, s):
        return np.clip(np.searchsorted(self.cum, s, side='right') - 1, 0, len(self.dirs) - 1)

    def point(self, s, offset=0.0):
        s = np.asarray(s, float)
        i = self._idx(s)
        tangent = self.dirs[i]
        normal = np.stack([-tangent[..., 1], tangent[..., 0]], -1)
        base = self.points[i] + tangent * (s - self.cum[i])[..., None]
        return base + np.asarray(offset, float)[..., None] * normal

    def tangent(self, s):
        return self.dirs[self._idx(np.asarray(s, float))]


def robot_path(scn, lead=20.0, arc_segments=12):
    """Inner-side path; arc length 0 is the start of the turn."""
    u = scn.half_width - scn.robot_radius - scn.robot_inner_margin
    r = scn.turn_radius
    centre = np.array([-u - r, u + r])
    angles = np.linspace(-np.pi / 2, 0.0, arc_segments + 1)
    arc = centre + r * np.stack([np.cos(angles), np.sin(angles)], -1)
    pts = np.vstack([[arc[0, 0] - lead, u], arc, [-u, arc[-1, 1] + lead]])
    return Polyline(pts), lead


def person_path(lead=20.0):
    """Centreline from far up the vertical leg to far down the horizontal leg.

    Positive offset along the left normal points to the outer walls on both legs.
    Arc length ``lead`` is the corner.
    """
    return Polyline([[0.0, lead], [0.0, 0.0], [-lead, 0.0]]), lead


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def camera_prob(scn, robot_p, heading, person_p, half_width, omega=None):
    """Per-frame detection probability for the configured camera set."""
    pts = body_points(robot_p, person_p, 1.0)
    # Scale the unit body span to each episode's current half width.
    centre = np.asarray(person_p, float)[:, None, :]
    pts = centre + (pts - centre) * half_width[:, None, None]
    seen = line_of_sight(robot_p[:, None, :], pts, scn.occluders, scn.sensor_range)
    rel = pts - robot_p[:, None, :]
    dist_pts = np.linalg.norm(rel, axis=-1)
    bearing = np.arctan2(rel[..., 1], rel[..., 0])
    dist = np.linalg.norm(person_p - robot_p, axis=-1)
    ramps = {float(h): (lo, hi) for h, lo, hi in scn.cam_ramps}
    prob = np.zeros(len(robot_p))
    for cam_heading, hfov, vfov, height in scn.cameras:
        in_fov = np.abs(_wrap(bearing - heading[:, None] - np.deg2rad(cam_heading))) \
            <= np.deg2rad(hfov) / 2
        f_h = (seen & in_fov & (dist_pts <= scn.cam_max_range)).mean(-1)
        reach = dist * np.tan(np.deg2rad(vfov) / 2)
        top = np.minimum(scn.person_height, height + reach)
        bottom = np.maximum(0.0, height - reach)
        f_v = np.clip(top - bottom, 0.0, None) / scn.person_height
        lo, hi = ramps.get(float(hfov), (scn.cam_area_lo, scn.cam_area_hi))
        prob = np.maximum(prob, np.clip((f_h * f_v - lo) / (hi - lo), 0.0, 1.0))
    if omega is not None and scn.turn_blur:
        prob = prob * (1.0 - scn.turn_blur * np.minimum(1.0, np.abs(omega) / 2.0))
    return prob


def in_path_ahead(robot_p, heading, person_p, reach, lateral):
    """Person centre inside the robot's straight swept path within ``reach``."""
    rel = person_p - robot_p
    fwd = np.stack([np.cos(heading), np.sin(heading)], -1)
    along = (rel * fwd).sum(-1)
    across = np.abs(rel[:, 0] * fwd[:, 1] - rel[:, 1] * fwd[:, 0])
    return (along > 0) & (along <= reach) & (across <= lateral)


def simulate(scn, action, n, seed, trajectories=False):
    """Roll out ``n`` corner encounters; returns per-episode arrays."""
    rng = np.random.default_rng(seed)
    rpath, r_lead = robot_path(scn)
    ppath, p_lead = person_path()
    dt, steps = scn.dt, scn.steps
    period = max(1, int(round(1.0 / (scn.rate_hz * dt))))
    target = ACTION_SPEEDS[action]

    r_s = r_lead - rng.uniform(*scn.robot_start, n)
    p_s = p_lead - rng.uniform(*scn.person_start, n)
    p_pref = np.clip(scn.person_speed_median * np.exp(scn.person_speed_sigma
                                                      * rng.standard_normal(n)),
                     0.3, scn.person_speed_max)
    attentive = rng.uniform(size=n) < scn.attentive_share
    react_range = np.where(attentive, scn.react_range_attentive,
                           rng.uniform(*scn.react_range_inattentive, n))
    offset = rng.uniform(-scn.person_lateral, scn.person_lateral, n)
    det_u = rng.uniform(size=(n, steps + 1))

    v = np.full(n, target)
    p_v = p_pref.copy()
    react_at = np.full(n, np.inf)
    braking = np.zeros(n, bool)
    last_det = np.full(n, -np.inf)
    track_pos = np.full((n, 2), np.nan)
    track_t = np.full(n, -np.inf)                  # image time of track_pos
    track_vel = np.zeros((n, 2))

    robot = np.zeros((n, steps + 1, 2))
    person = np.zeros((n, steps + 1, 2))
    heading = np.zeros((n, steps + 1))
    p_rad = np.zeros((n, steps + 1))
    speed = np.zeros((n, steps + 1))
    brake_log = np.zeros((n, steps + 1), bool)
    det_frames = np.zeros((n, steps + 1), bool)    # confirmed images, at image time
    eff_frames = np.zeros((n, steps + 1), bool)    # same, at the time they become usable
    lag = int(round(scn.recog_latency / dt))
    pending = {}                                   # usable step -> (detections, positions)
    streak = np.zeros(n, int)
    confirmed = np.zeros(n, bool)
    last_raw = np.full(n, -np.inf)
    n_arc = len(rpath.dirs) - 2

    for k in range(steps + 1):
        t = k * dt
        rp = rpath.point(r_s)
        tan = rpath.tangent(r_s)
        hdg = np.arctan2(tan[:, 1], tan[:, 0])
        pp = ppath.point(p_s, offset)
        prog = np.clip(offset / scn.side_offset, 0.0, 1.0)
        rad = scn.person_radius_walk + (scn.person_radius_side - scn.person_radius_walk) * prog
        robot[:, k], person[:, k], heading[:, k], p_rad[:, k], speed[:, k] = rp, pp, hdg, rad, v

        # Camera frame and track.
        if k % period == 0:
            seg = rpath._idx(r_s)
            omega = np.where((seg >= 1) & (seg <= n_arc), v / scn.turn_radius, 0.0)
            raw = det_u[:, k] < camera_prob(scn, rp, hdg, pp, rad, omega)
            confirmed &= (t - last_raw) <= scn.track_memory
            streak = np.where(raw, streak + 1, 0)
            last_raw = np.where(raw, t, last_raw)
            confirmed |= streak >= scn.confirm_frames
            ok = raw & confirmed
            det_frames[:, k] = ok
            pending[k + lag] = (ok, pp.copy(), t)
        if k in pending:
            ok, pos, img_t = pending.pop(k)
            # Finite-difference velocity from consecutive detections of a live track.
            live = ok & ((t - last_det) <= scn.track_memory) & np.isfinite(track_t)
            vel = (pos - np.nan_to_num(track_pos)) / np.maximum(img_t - track_t, dt)[:, None]
            spd = np.linalg.norm(vel, axis=-1, keepdims=True)
            vel = vel * np.minimum(1.0, 2.0 / np.maximum(spd, 1e-9))
            track_vel = np.where(live[:, None], vel, np.where(ok[:, None], 0.0, track_vel))
            track_t = np.where(ok, img_t, track_t)
            eff_frames[:, k] = ok
            last_det = np.where(ok, t, last_det)
            track_pos = np.where(ok[:, None], pos, track_pos)
        active = (t - last_det) <= scn.track_memory

        # Reactive safety layer (vision only): brake when the tracked person is
        # in the swept path within stopping distance + closing allowance + margin.
        t_stop = v / scn.brake_decel
        trig_reach = (v ** 2 / (2 * scn.brake_decel) + scn.closing_speed * t_stop
                      + scn.stop_margin + scn.robot_radius + scn.person_radius_walk)
        tp = track_pos
        if scn.extrapolate:
            tp = tp + track_vel * np.where(np.isfinite(track_t), np.clip(t - track_t, 0.0, None), 0.0)[:, None]
        tp = np.where(np.isnan(tp), 1e6, tp)
        trigger = active & in_path_ahead(rp, hdg, tp, trig_reach,
                                         scn.robot_radius + rad + scn.path_margin)
        braking = braking | trigger
        resume = braking & ~active & ((t - last_det) >= scn.resume_after)
        braking = braking & ~resume
        brake_log[:, k] = braking
        if k == steps:
            break
        dv_up = scn.robot_accel * dt
        v = np.where(braking, np.maximum(v - scn.brake_decel * dt, 0.0),
                     v + np.clip(target - v, -dv_up, dv_up))
        r_s = r_s + v * dt

        # Worker reaction: sees the robot (line of sight, centre to centre) within range.
        dist = np.linalg.norm(pp - rp, axis=-1)
        sees = line_of_sight(pp, rp, scn.occluders, scn.sensor_range) & (dist <= react_range)
        react_at = np.where(np.isinf(react_at) & sees, t + scn.react_delay, react_at)
        reacting = t >= react_at
        offset = np.where(reacting, np.minimum(offset + scn.sidestep_speed * dt, scn.side_offset),
                          offset)
        p_goal = np.where(reacting, scn.react_speed_factor * p_pref, p_pref)
        p_v = p_v + np.clip(p_goal - p_v, -1.5 * dt, 1.5 * dt)
        p_s = p_s + p_v * dt

    logged = track_frames(det_frames, period)
    gap = np.linalg.norm(robot - person, axis=-1) - scn.robot_radius - p_rad
    reach = speed ** 2 / (2 * scn.brake_decel) + scn.robot_radius
    lateral = scn.robot_radius + p_rad
    proxy_step = np.zeros_like(gap, bool)
    for k in range(steps + 1):
        proxy_step[:, k] = in_path_ahead(robot[:, k], heading[:, k], person[:, k],
                                         reach[:, k], lateral[:, k])
    proxy_step &= (speed > 0.1) & ~brake_log

    contact_step = gap <= 0.0
    near_step = gap <= scn.near_gap
    moving = speed > 0.05
    # Late detection: the first logged frame of the person comes after the
    # closest approach, or when the gap is already below what the robot needs
    # to stop at its current speed (braking + closing allowance + margin).
    # Never-logged encounters are late too, but invisible onboard.
    need = (speed ** 2 / (2 * scn.brake_decel) + scn.closing_speed * speed / scn.brake_decel
            + scn.stop_margin)
    first_contact = np.where(contact_step.any(1), contact_step.argmax(1), -1)
    ever_logged = logged.any(1)
    first_logged = np.where(ever_logged, logged.argmax(1), -1)
    closest = gap.argmin(1)
    idx = np.arange(n)
    encounter = gap.min(1) < 1.0
    ever_eff = eff_frames.any(1)
    first_eff = np.where(ever_eff, eff_frames.argmax(1), -1)
    fe = np.maximum(first_eff, 0)
    in_time = ever_eff & (first_eff <= closest) & (gap[idx, fe] > need[idx, fe])
    late = encounter & ~in_time
    moving_at_contact = np.where(first_contact >= 0, speed[idx, np.maximum(first_contact, 0)] > 0.05,
                                 False)
    return {
        'contact': contact_step.any(1),
        'contact_logged': (contact_step & logged).any(1),
        'contact_moving': moving_at_contact,
        'near': near_step.any(1),
        'near_logged': (near_step & logged).any(1),
        'contact_mv': (contact_step & moving).any(1),
        'contact_mv_logged': (contact_step & moving & logged).any(1),
        'near_moving': (near_step & moving).any(1),
        'near_moving_logged': (near_step & moving & logged).any(1),
        'late': late,
        'late_logged': late & ever_eff,
        'never_recognised': encounter & ~ever_eff,
        'proxy': proxy_step.any(1),
        'proxy_logged': (proxy_step & logged).any(1),
        'encounter': encounter,
        'ever_logged': ever_logged,
        'closest_logged': logged[idx, closest],
        'gap_true': gap.min(1),
        'gap_logged': np.where(logged, gap, np.inf).min(1),
        'warning_s': np.where((first_contact >= 0) & (first_logged >= 0)
                              & (first_logged <= first_contact),
                              (first_contact - first_logged) * dt, np.nan),
        'attentive': attentive,
        **({'robot': robot, 'person': person, 'heading': heading, 'radius': p_rad,
            'speed': speed, 'logged': logged, 'gap': gap, 'braking': brake_log}
           if trajectories else {}),
    }


def _rate(mask, given=None):
    given = np.ones_like(mask) if given is None else given
    k, m = int((mask & given).sum()), int(given.sum())
    return (k / m if m else float('nan')), m


def summarise(out):
    enc = out['encounter']
    c, p = out['contact'], out['proxy']
    contact_rate, _ = _rate(c)
    proxy_rate, _ = _rate(p)
    miss_c, n_c = _rate(~out['contact_logged'], c)
    miss_p, n_p = _rate(~out['proxy_logged'], p)
    proxy_cover, _ = _rate(p, c)
    near_rate, _ = _rate(out['near'])
    miss_near, _ = _rate(~out['near_logged'], out['near'])
    v1, n_enc = _rate(out['closest_logged'], enc)
    never, _ = _rate(~out['ever_logged'], enc)
    moving, _ = _rate(out['contact_moving'], c)
    late = np.nan_to_num(out['warning_s'], nan=0.0) < 1.0
    late_c, _ = _rate(late, c)
    both = enc & np.isfinite(out['gap_logged'])
    bias = float(np.median(out['gap_logged'][both] - out['gap_true'][both])) if both.any() else None
    return {
        'n': int(len(c)), 'encounters': n_enc,
        'contact_rate': contact_rate, 'contacts': n_c,
        'contact_while_moving': moving,
        'proxy_rate': proxy_rate, 'proxies': n_p,
        'proxy_covers_contact': proxy_cover,
        'miss_contact': miss_c, 'miss_proxy': miss_p,
        'near_rate': near_rate, 'miss_near': miss_near,
        'closest_point_logged': v1, 'person_never_logged': never,
        'contact_warning_lt_1s': late_c,
        'median_gap_bias_m': bias,
    }


def run(n, seed, configs=None, actions=('go', 'slow'), **overrides):
    rows = []
    for name in configs or CAMERA_CONFIGS:
        scn = CornerScenario(cameras=CAMERA_CONFIGS[name], **overrides)
        for j, action in enumerate(actions):
            out = simulate(scn, action, n, seed + 1000 * j)
            rows.append({'camera': name, 'action': action, **summarise(out)})
    return rows


def _fmt(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '   -  '
    return f'{100 * x:5.1f}%' if pct else f'{x:6.2f}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--attentive-share', type=float, default=None)
    parser.add_argument('--out', default=None, help='write rows as JSON here')
    args = parser.parse_args(argv)
    overrides = {}
    if args.attentive_share is not None:
        overrides['attentive_share'] = args.attentive_share
    rows = run(args.n, args.seed, **overrides)
    print(f"passing gap = {CornerScenario().passing_gap():.2f} m, n = {args.n} per row")
    head = ('camera', 'act', 'contact', 'moving', 'proxy', 'P(px|c)', 'miss_c', 'miss_px',
            'near10', 'miss_nr', 'V=1', 'never', 'warn<1s', 'bias_m')
    print(' '.join(f'{h:>8}' for h in head))
    for r in rows:
        cells = [r['camera'][:18].rjust(18), r['action'].rjust(4), _fmt(r['contact_rate']),
                 _fmt(r['contact_while_moving']), _fmt(r['proxy_rate']),
                 _fmt(r['proxy_covers_contact']), _fmt(r['miss_contact']),
                 _fmt(r['miss_proxy']), _fmt(r['near_rate']), _fmt(r['miss_near']),
                 _fmt(r['closest_point_logged']),
                 _fmt(r['person_never_logged']), _fmt(r['contact_warning_lt_1s']),
                 _fmt(r['median_gap_bias_m'], pct=False)]
        print(' '.join(f'{c:>8}' for c in cells))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'scenario': asdict(CornerScenario(**overrides)), 'rows': rows}, f, indent=1)


if __name__ == '__main__':
    main()
