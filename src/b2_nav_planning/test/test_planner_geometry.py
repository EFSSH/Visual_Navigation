import math

from b2_nav_planning.simple_planner_node import point_to_segment_distance


def test_point_on_segment_has_zero_distance():
    distance = point_to_segment_distance((2.5, 0.0), (0.0, 0.0), (5.0, 0.0))
    assert math.isclose(distance, 0.0)


def test_point_beyond_segment_uses_nearest_endpoint():
    distance = point_to_segment_distance((6.0, 0.0), (0.0, 0.0), (5.0, 0.0))
    assert math.isclose(distance, 1.0)


def test_lateral_distance_is_measured_from_segment():
    distance = point_to_segment_distance((2.5, 2.0), (0.0, 0.0), (5.0, 0.0))
    assert math.isclose(distance, 2.0)
