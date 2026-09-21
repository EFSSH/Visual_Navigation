import math

from b2_nav_safety.lidar_safety_node import minimum_valid_range


def test_minimum_valid_range_ignores_invalid_values():
    ranges = [math.inf, math.nan, 0.02, 1.2, 0.7, 8.0]
    assert minimum_valid_range(ranges, 0.1, 5.0) == 0.7


def test_minimum_valid_range_returns_none_without_valid_values():
    assert minimum_valid_range([math.inf, math.nan, 0.01], 0.1, 5.0) is None
