import math

from b2_nav_execution.trajectory_check import (
    check_trajectory,
    cross_track,
    Grid,
    Limits,
    max_heading_change_per_m,
    REASON_COLLISION,
    REASON_EMPTY,
    REASON_NO_COSTMAP,
    REASON_OK,
    REASON_OUTSIDE_COSTMAP,
    REASON_SPACING,
    REASON_STALE,
    REASON_TURN,
    transform_points_2d,
)


def _free_grid(size=40, resolution=0.1):
    # 4 m x 4 m centred on the origin.
    origin = -size * resolution / 2.0
    return Grid(resolution, origin, origin, size, size, [0] * (size * size))


def _line(length, step=0.1):
    count = int(round(length / step))
    return [(i * step, 0.0) for i in range(count + 1)]


def test_straight_path_in_free_space_is_accepted():
    reason, details = check_trajectory(_line(1.5), 0.1, Limits(), _free_grid())
    assert reason == REASON_OK
    assert math.isclose(details['length_m'], 1.5)


def test_single_point_is_rejected():
    assert check_trajectory([(0.0, 0.0)], 0.0, Limits(), _free_grid())[0] == REASON_EMPTY


def test_stale_path_is_rejected():
    assert check_trajectory(_line(1.0), 2.0, Limits(), _free_grid())[0] == REASON_STALE


def test_sparse_path_is_rejected():
    reason, _ = check_trajectory([(0, 0), (1.0, 0)], 0.0, Limits(), _free_grid())
    assert reason == REASON_SPACING


def test_obstacle_between_samples_is_detected():
    grid = _free_grid()
    # Thin wall one cell wide at x = 0.55 m, between 0.5 m-spaced samples.
    col = int((0.55 - grid.origin_x) / grid.resolution)
    for row in range(grid.height):
        grid.data[row * grid.width + col] = 100
    points = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]
    reason, details = check_trajectory(points, 0.0, Limits(), grid)
    assert reason == REASON_COLLISION
    assert details['sample_index'] > 0


def test_unknown_cells_are_not_lethal():
    grid = _free_grid()
    grid.data = [-1] * len(grid.data)
    assert check_trajectory(_line(1.0), 0.0, Limits(), grid)[0] == REASON_OK


def test_path_leaving_costmap_is_rejected():
    reason, _ = check_trajectory(_line(3.0), 0.0, Limits(), _free_grid())
    assert reason == REASON_OUTSIDE_COSTMAP


def test_missing_costmap_is_rejected_when_required():
    assert check_trajectory(_line(1.0), 0.0, Limits(), None)[0] == REASON_NO_COSTMAP
    reason, _ = check_trajectory(_line(1.0), 0.0, Limits(), None, require_grid=False)
    assert reason == REASON_OK


def test_hairpin_turn_is_rejected():
    points = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.2, 0.01), (0.0, 0.02)]
    assert max_heading_change_per_m(points) > Limits().max_heading_change_rad_per_m
    assert check_trajectory(points, 0.0, Limits(), _free_grid())[0] == REASON_TURN


def test_duplicate_points_do_not_create_turn_spikes():
    points = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.0), (0.2, 0.0)]
    assert max_heading_change_per_m(points) == 0.0


def test_cross_track_and_remaining_length():
    distance, remaining = cross_track(_line(2.0), 0.5, 0.3)
    assert math.isclose(distance, 0.3, abs_tol=1e-9)
    assert math.isclose(remaining, 1.5, abs_tol=1e-9)


def test_planar_transform():
    (x, y), = transform_points_2d([(1.0, 0.0)], 2.0, 1.0, math.pi / 2.0)
    assert math.isclose(x, 2.0, abs_tol=1e-9)
    assert math.isclose(y, 2.0, abs_tol=1e-9)
