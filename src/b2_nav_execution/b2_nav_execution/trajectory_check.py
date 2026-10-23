"""
Geometry checks for local trajectories, independent of ROS.

Points are (x, y) tuples in one frame. The checks only accept or reject a
trajectory; they never modify it, so a rejection is always attributable to
the trajectory source rather than to the executor.
"""

from dataclasses import dataclass
import math

REASON_OK = 'ok'
REASON_EMPTY = 'too_few_points'
REASON_NOT_FINITE = 'not_finite'
REASON_STALE = 'stale'
REASON_FUTURE = 'stamp_in_future'
REASON_FRAME = 'frame_not_allowed'
REASON_SPACING = 'spacing_too_large'
REASON_LENGTH = 'too_long'
REASON_TURN = 'turn_too_sharp'
REASON_COLLISION = 'collision'
REASON_NO_COSTMAP = 'costmap_unavailable'
REASON_OUTSIDE_COSTMAP = 'outside_costmap'
REASON_TRANSFORM = 'transform_failed'


@dataclass
class Limits:
    max_age_sec: float = 1.0
    max_future_sec: float = 0.2
    max_spacing_m: float = 0.5
    max_length_m: float = 6.0
    max_heading_change_rad_per_m: float = 3.0
    lethal_cost: int = 99
    # Points outside the local costmap cannot be checked; reject by default.
    reject_outside_costmap: bool = True


@dataclass
class Grid:
    """2D occupancy grid with Nav2 OccupancyGrid semantics (0..100, -1)."""

    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int
    data: list

    def cost_at(self, x, y):
        """Return the cell cost, or None outside the grid."""
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        if col < 0 or row < 0 or col >= self.width or row >= self.height:
            return None
        return self.data[row * self.width + col]


def path_length(points):
    return sum(
        math.hypot(x1 - x0, y1 - y0)
        for (x0, y0), (x1, y1) in zip(points, points[1:])
    )


def max_spacing(points):
    return max(
        (math.hypot(x1 - x0, y1 - y0)
         for (x0, y0), (x1, y1) in zip(points, points[1:])),
        default=0.0,
    )


def max_heading_change_per_m(points, min_segment_m=0.02):
    """
    Largest heading change between segments divided by their mean length.

    Segments shorter than ``min_segment_m`` are merged into the next one so
    that duplicated or nearly coincident points do not produce spikes.
    """
    segments = []
    start = points[0] if points else None
    for point in points[1:]:
        dx = point[0] - start[0]
        dy = point[1] - start[1]
        length = math.hypot(dx, dy)
        if length >= min_segment_m:
            segments.append((math.atan2(dy, dx), length))
            start = point
    worst = 0.0
    for (h0, l0), (h1, l1) in zip(segments, segments[1:]):
        change = abs(math.atan2(math.sin(h1 - h0), math.cos(h1 - h0)))
        worst = max(worst, change / (0.5 * (l0 + l1)))
    return worst


def densify(points, step):
    """Insert points so consecutive samples are at most ``step`` apart."""
    if not points:
        return []
    dense = [points[0]]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        count = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / step)))
        for i in range(1, count + 1):
            t = i / count
            dense.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return dense


def first_blocked_index(points, grid, lethal_cost, reject_outside):
    """
    Return (index, reason) of the first unsafe sample, or (None, None).

    The trajectory is resampled at the grid resolution so a segment cannot
    jump over a thin obstacle. Unknown cells (-1) are not treated as lethal:
    the local costmap marks unobserved space as unknown and the safety gate
    still stops on real obstacles.
    """
    for index, (x, y) in enumerate(densify(points, grid.resolution)):
        cost = grid.cost_at(x, y)
        if cost is None:
            if reject_outside:
                return index, REASON_OUTSIDE_COSTMAP
            continue
        if cost >= lethal_cost:
            return index, REASON_COLLISION
    return None, None


def check_trajectory(points, age_sec, limits, grid=None, require_grid=True):
    """Return (reason, details); reason is REASON_OK when executable."""
    if len(points) < 2:
        return REASON_EMPTY, {'points': len(points)}
    if not all(math.isfinite(v) for point in points for v in point):
        return REASON_NOT_FINITE, {}
    if age_sec < -limits.max_future_sec:
        return REASON_FUTURE, {'age_sec': age_sec}
    if age_sec > limits.max_age_sec:
        return REASON_STALE, {'age_sec': age_sec}
    spacing = max_spacing(points)
    if spacing > limits.max_spacing_m:
        return REASON_SPACING, {'max_spacing_m': spacing}
    length = path_length(points)
    if length > limits.max_length_m:
        return REASON_LENGTH, {'length_m': length}
    turn = max_heading_change_per_m(points)
    if turn > limits.max_heading_change_rad_per_m:
        return REASON_TURN, {'heading_change_rad_per_m': turn}
    if grid is None:
        if require_grid:
            return REASON_NO_COSTMAP, {}
        return REASON_OK, {'length_m': length}
    index, reason = first_blocked_index(
        points, grid, limits.lethal_cost, limits.reject_outside_costmap
    )
    if reason is not None:
        return reason, {'sample_index': index, 'length_m': length}
    return REASON_OK, {'length_m': length}


def cross_track(points, x, y):
    """Return (distance to polyline, remaining length from the nearest point)."""
    best = (math.inf, 0, 0.0)
    for index, ((x0, y0), (x1, y1)) in enumerate(zip(points, points[1:])):
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0.0 else max(
            0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / seg2)
        )
        px, py = x0 + t * dx, y0 + t * dy
        distance = math.hypot(x - px, y - py)
        if distance < best[0]:
            best = (distance, index, t)
    distance, index, t = best
    if not math.isfinite(distance):
        return None, None
    (x0, y0), (x1, y1) = points[index], points[index + 1]
    remaining = (1.0 - t) * math.hypot(x1 - x0, y1 - y0)
    remaining += path_length(points[index + 1:])
    return distance, remaining


def transform_points_2d(points, tx, ty, yaw):
    """Apply a planar rigid transform to (x, y) points."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [(tx + c * x - s * y, ty + s * x + c * y) for x, y in points]


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
