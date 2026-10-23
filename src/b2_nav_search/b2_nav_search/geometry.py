"""Search geometry: depth back-projection, multi-view agreement, stand-off pose."""

import math

import numpy as np


def depth_at(depth_image, u, v, window=5, min_m=0.05, max_m=10.0):
    """
    Median valid depth in metres around pixel (u, v), or None.

    Accepts 32FC1 metres or 16UC1 millimetres.
    """
    image = np.asarray(depth_image)
    scale = 0.001 if image.dtype == np.uint16 else 1.0
    height, width = image.shape[:2]
    col, row = int(round(u)), int(round(v))
    half = window // 2
    patch = image[max(0, row - half):min(height, row + half + 1),
                  max(0, col - half):min(width, col + half + 1)]
    values = patch.astype(np.float64).ravel() * scale
    values = values[np.isfinite(values) & (values >= min_m) & (values <= max_m)]
    if values.size == 0:
        return None
    return float(np.median(values))


def back_project(u, v, depth_m, fx, fy, cx, cy):
    """Pixel and depth to a point in the optical frame (z forward)."""
    return ((u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m)


def consensus(observations, min_views, radius_m):
    """
    Return the agreeing cluster centre (x, y) and its size, or (None, n).

    ``observations`` are (x, y) map positions from distinct frames. The
    cluster is anchored at the coordinate-wise median, which tolerates a
    minority of wrong matches.
    """
    if not observations:
        return None, 0
    points = np.asarray(observations, dtype=np.float64)
    centre = np.median(points, axis=0)
    members = points[np.hypot(*(points - centre).T) <= radius_m]
    if len(members) < min_views:
        return None, len(members)
    x, y = members.mean(axis=0)
    return (float(x), float(y)), len(members)


def observation_candidates(device_xy, robot_xy, distance_m, facing_yaw,
                           step_rad=math.radians(15.0), max_rad=math.radians(90.0)):
    """
    Yield (x, y, yaw) stand-off poses, preferred first.

    With ``facing_yaw`` the robot stands in front of the device along its
    normal; otherwise along the current line of sight. Alternatives rotate
    around the device in alternating directions up to ``max_rad``.
    """
    dx, dy = device_xy[0], device_xy[1]
    if facing_yaw is None:
        base = math.atan2(robot_xy[1] - dy, robot_xy[0] - dx)
    else:
        base = facing_yaw
    offsets = [0.0]
    k = 1
    while k * step_rad <= max_rad + 1e-9:
        offsets.extend([k * step_rad, -k * step_rad])
        k += 1
    for offset in offsets:
        angle = base + offset
        x = dx + distance_m * math.cos(angle)
        y = dy + distance_m * math.sin(angle)
        yaw = math.atan2(dy - y, dx - x)
        yield x, y, yaw


def choose_observation_pose(device_xy, robot_xy, distance_m, facing_yaw, is_free,
                            step_rad=math.radians(15.0), max_rad=math.radians(90.0)):
    """Return the first free stand-off pose (x, y, yaw), or None."""
    for x, y, yaw in observation_candidates(
        device_xy, robot_xy, distance_m, facing_yaw, step_rad, max_rad
    ):
        if is_free(x, y):
            return x, y, yaw
    return None
