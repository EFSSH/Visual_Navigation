import numpy as np
from geometry_msgs.msg import TransformStamped

from b2_nav_twin.local_world_model_node import (
    RollingGridModel,
    bresenham_cells,
    merge_fresh_clouds,
    transform_xyz,
)


def test_transform_xyz_applies_translation():
    transform = TransformStamped()
    transform.transform.rotation.w = 1.0
    transform.transform.translation.x = 2.0
    transform.transform.translation.y = -1.0
    points = np.array([[1.0, 2.0, 3.0]])

    result = transform_xyz(points, transform)

    np.testing.assert_allclose(result, [[3.0, 1.0, 3.0]])


def test_merge_fresh_clouds_drops_stale_source():
    cache = {
        'front': (10.0, np.array([[1.0, 0.0, 0.5]])),
        'rear': (9.0, np.array([[-1.0, 0.0, 0.5]])),
    }

    result = merge_fresh_clouds(cache, 10.0, 0.5)

    np.testing.assert_allclose(result, [[1.0, 0.0, 0.5]])


def test_merge_fresh_clouds_combines_recent_sources():
    cache = {
        'front': (10.0, np.array([[1.0, 0.0, 0.5]])),
        'rear': (9.8, np.array([[-1.0, 0.0, 0.5]])),
    }

    result = merge_fresh_clouds(cache, 10.0, 0.5)

    np.testing.assert_allclose(
        result,
        [[1.0, 0.0, 0.5], [-1.0, 0.0, 0.5]],
    )


def test_bresenham_cells_contains_both_endpoints():
    result = bresenham_cells((0, 0), (3, 1))

    assert result[0] == (0, 0)
    assert result[-1] == (3, 1)


def test_grid_records_occupied_height_and_source():
    model = RollingGridModel(resolution=0.1)
    points = np.array(
        [
            [1.01, 0.01, 0.10],
            [1.04, 0.04, 0.80],
        ]
    )

    model.mark_points(points, stamp=10.0, source_mask=1)

    occupied = model.occupied_items()
    assert len(occupied) == 1
    _, state = occupied[0]
    assert state.min_z == 0.10
    assert state.max_z == 0.80
    assert state.source_mask == 1
    assert state.hit_count == 1


def test_clearing_ray_removes_occupied_cell():
    model = RollingGridModel(resolution=0.1)
    model.mark_points(
        np.array([[1.05, 0.05, 0.50]]),
        stamp=10.0,
        source_mask=1,
    )

    model.clear_rays(
        origin=np.array([0.0, 0.0, 0.3]),
        endpoints=np.array([[2.0, 0.0, 0.0]]),
        stamp=10.1,
        max_range=3.2,
    )

    assert model.occupied_items() == []


def test_obstacle_expires_without_new_sensor_messages():
    model = RollingGridModel(
        resolution=0.1,
        obstacle_ttl_sec=1.0,
    )
    model.mark_points(
        np.array([[1.05, 0.05, 0.50]]),
        stamp=10.0,
        source_mask=1,
    )

    expired = model.expire_and_prune(
        now=11.1,
        center_x=0.0,
        center_y=0.0,
    )

    assert expired == 1
    assert model.occupied_items() == []


def test_expire_stale_returns_expired_obstacle_for_nav2_clearing():
    model = RollingGridModel(
        resolution=0.1,
        obstacle_ttl_sec=1.0,
    )
    model.mark_points(
        np.array([[1.05, 0.05, 0.50]]),
        stamp=10.0,
        source_mask=2,
    )

    removed, expired = model.expire_stale(now=11.1)

    assert removed == 1
    assert len(expired) == 1
    assert expired[0][0] == (10, 0)
    assert expired[0][1].source_mask == 2
    assert model.cells == {}


def test_rolling_window_prunes_distant_cells():
    model = RollingGridModel(
        resolution=0.1,
        width=8.0,
        height=8.0,
    )
    model.mark_points(
        np.array([[5.0, 0.0, 0.50]]),
        stamp=10.0,
        source_mask=2,
    )

    model.expire_and_prune(
        now=10.1,
        center_x=0.0,
        center_y=0.0,
    )

    assert model.cells == {}
