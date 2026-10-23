"""
Load exported episodes and resample them on a common clock.

Episodes are the ``.npz`` files written by ``experiment export``; this
module needs only NumPy, so training and evaluation run without ROS.
"""

from dataclasses import dataclass
import glob
import json
import os

import numpy as np

OCCUPIED_COST = 65  # OccupancyGrid value treated as occupied


@dataclass
class Episode:
    path: str
    meta: dict
    pose_t: np.ndarray
    pose: np.ndarray        # (N, 3) x, y, yaw in odom
    cmd_t: np.ndarray
    cmd: np.ndarray         # (M, 2) executed vx, wz after the safety gate
    grid_t: np.ndarray
    grid: np.ndarray        # (G, H, W) int8 local costmap
    grid_origin: np.ndarray  # (G, 2)
    grid_resolution: float

    @property
    def group(self):
        return self.meta.get('scenario', ''), self.meta.get('route_id', '')

    def time_range(self):
        return float(self.pose_t[0]), float(self.pose_t[-1])


def load_episode(path):
    with np.load(path, allow_pickle=False) as data:
        return Episode(
            path=path,
            meta=json.loads(str(data['meta'])),
            pose_t=data['pose_t'],
            pose=data['pose'],
            cmd_t=data['cmd_safe_t'],
            cmd=data['cmd_safe'],
            grid_t=data['grid_t'],
            grid=data['grid'],
            grid_origin=data['grid_origin'],
            grid_resolution=float(data['grid_resolution']),
        )


def load_episodes(directory):
    paths = sorted(glob.glob(os.path.join(directory, '**', '*.npz'), recursive=True))
    return [load_episode(path) for path in paths]


def interp_pose(episode, t):
    """Linearly interpolated (x, y, yaw) at times ``t`` (yaw unwrapped)."""
    t = np.atleast_1d(t)
    yaw = np.unwrap(episode.pose[:, 2])
    out = np.stack([
        np.interp(t, episode.pose_t, episode.pose[:, 0]),
        np.interp(t, episode.pose_t, episode.pose[:, 1]),
        np.interp(t, episode.pose_t, yaw),
    ], axis=-1)
    out[:, 2] = np.arctan2(np.sin(out[:, 2]), np.cos(out[:, 2]))
    return out


def held_command(episode, t, max_age=0.5):
    """
    Return the executed command in force at ``t``; zero when none is fresh.

    The device watchdog stops the base after 0.5 s without a command, so an
    older command is not what the robot was doing.
    """
    t = np.atleast_1d(t)
    if episode.cmd_t.size == 0:
        return np.zeros((t.size, 2))
    index = np.searchsorted(episode.cmd_t, t, side='right') - 1
    valid = (index >= 0)
    clipped = np.clip(index, 0, None)
    valid &= (t - episode.cmd_t[clipped]) <= max_age
    out = episode.cmd[clipped].copy()
    out[~valid] = 0.0
    return out


def latest_grid_index(episode, t):
    """Index of the newest costmap at or before ``t``, or -1."""
    return int(np.searchsorted(episode.grid_t, t, side='right') - 1)


def to_frame(points_xy, pose):
    """World (odom) points into the frame of ``pose`` = (x, y, yaw)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = np.asarray(points_xy) - pose[:2]
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def relative_pose(pose, anchor):
    xy = to_frame(pose[..., :2], anchor)
    yaw = np.arctan2(np.sin(pose[..., 2] - anchor[2]), np.cos(pose[..., 2] - anchor[2]))
    return np.concatenate([xy, yaw[..., None]], axis=-1)


def ego_occupancy(episode, grid_index, anchor, cells=40, cell_m=0.1):
    """
    Sample a costmap into a ``cells`` x ``cells`` grid around ``anchor``.

    Returns (occupied float array, known mask). Rows run along the anchor's
    x axis (forward), columns along y (left); the anchor is at the centre.
    """
    grid = episode.grid[grid_index]
    origin = episode.grid_origin[grid_index]
    res = episode.grid_resolution
    offsets = (np.arange(cells) - cells / 2.0 + 0.5) * cell_m
    ex, ey = np.meshgrid(offsets, offsets, indexing='ij')
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    wx = anchor[0] + c * ex - s * ey
    wy = anchor[1] + s * ex + c * ey
    col = np.floor((wx - origin[0]) / res).astype(int)
    row = np.floor((wy - origin[1]) / res).astype(int)
    inside = (col >= 0) & (row >= 0) & (col < grid.shape[1]) & (row < grid.shape[0])
    values = np.full(ex.shape, -1, dtype=np.int16)
    values[inside] = grid[row[inside], col[inside]]
    known = values >= 0
    return (values >= OCCUPIED_COST).astype(np.float32), known
