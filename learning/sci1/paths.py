"""
Polyline agent paths for head-on nodes (single-lane walkway, door).

An agent moves along ``scn.agent_path`` (points from far upstream to the
node entrance) by arc length s, then continues along ``scn.agent_exit_dir``
(e.g. stepping aside out of the robot's way). A constant lateral offset is
applied along the local normal.
"""

import numpy as np


def _arrays(scn):
    pts = np.asarray(scn.agent_path, dtype=float)
    seg = np.diff(pts, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    return pts, cum, seg / lens[:, None]


def path_length(scn):
    return float(_arrays(scn)[1][-1])


def _locate(scn, s):
    pts, cum, dirs = _arrays(scn)
    s = np.asarray(s, dtype=float)
    idx = np.clip(np.searchsorted(cum, s, side='right') - 1, 0, len(dirs) - 1)
    return pts, cum, dirs, s, idx


def path_tangent(scn, s):
    pts, cum, dirs, s, idx = _locate(scn, s)
    exit_dir = np.asarray(scn.agent_exit_dir, dtype=float)
    return np.where((s > cum[-1])[..., None], exit_dir, dirs[idx])


def path_point(scn, s, offset=0.0):
    """Position at arc length s (beyond the end: along the exit direction)."""
    pts, cum, dirs, s, idx = _locate(scn, s)
    exit_dir = np.asarray(scn.agent_exit_dir, dtype=float)
    inside = pts[idx] + dirs[idx] * (s - cum[idx])[..., None]
    beyond = pts[-1] + exit_dir * (s - cum[-1])[..., None]
    point = np.where((s > cum[-1])[..., None], beyond, inside)
    tangent = path_tangent(scn, s)
    normal = np.stack([-tangent[..., 1], tangent[..., 0]], -1)
    return point + np.asarray(offset, dtype=float)[..., None] * normal


def path_project(scn, p):
    """Arc length of the path point nearest to each p (..., 2)."""
    pts, cum, dirs = _arrays(scn)
    p = np.asarray(p, dtype=float)
    best_s = np.zeros(p.shape[:-1])
    best_d = np.full(p.shape[:-1], np.inf)
    for i, d in enumerate(dirs):
        length = cum[i + 1] - cum[i]
        t = np.clip(((p - pts[i]) * d).sum(-1), 0.0, length)
        dist = np.linalg.norm(p - (pts[i] + t[..., None] * d), axis=-1)
        closer = dist < best_d
        best_d = np.where(closer, dist, best_d)
        best_s = np.where(closer, cum[i] + t, best_s)
    return best_s
