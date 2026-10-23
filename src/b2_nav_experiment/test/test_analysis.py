from b2_nav_experiment.analysis import (
    aggregate,
    attribute_failure,
    path_length,
    rising_edges,
    summarize,
)
from b2_nav_experiment.config import ConfigError, IMAGE_TOPICS, parse_config
import pytest


def test_path_length_ignores_jitter():
    xy = [(0.0, 0.0), (0.001, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert path_length(xy) == pytest.approx(2.0)


def test_rising_edges_counts_and_times_stops():
    series = [(0.0, False), (1.0, True), (2.0, True), (3.0, False), (4.0, True), (4.5, False)]
    count, active = rising_edges(series)
    assert count == 2
    assert active == pytest.approx(2.5)


def test_success_has_no_failure_layer():
    assert attribute_failure({'outcome': 'succeeded'}, {}) == (None, [])


def test_watchdog_timeouts_point_at_interface():
    layer, evidence = attribute_failure(
        {'outcome': 'failed', 'failure_reason': 'subgoal_timeout'},
        {'watchdog_timeouts': 2},
    )
    assert layer == 'interface'
    assert any('watchdog' in e for e in evidence)


def test_frequent_safety_stops_point_at_perception():
    layer, _ = attribute_failure(
        {'outcome': 'failed', 'failure_reason': 'subgoal_timeout'},
        {'safety_stop_events': 5},
    )
    assert layer == 'perception'


def test_summarize_counts_events_and_contact():
    events = [
        {'schema': 'b2_nav/execution_event/v1', 'event': 'accepted'},
        {'schema': 'b2_nav/execution_event/v1', 'event': 'rejected'},
        {'schema': 'b2_nav/watchdog_state/v1', 'state': 'timeout'},
        {'schema': 'b2_nav/task_event/v1', 'event': 'nav_goal'},
    ]
    summary = summarize(
        {'outcome': 'failed', 'failure_reason': 'object_not_found'},
        [(0, 0), (3, 4)], [(0.0, False)], [(0.0, 0.5), (1.0, 0.1)], events,
    )
    assert summary['path_length_m'] == pytest.approx(5.0)
    assert summary['contact'] is True
    assert summary['trajectory_rejections'] == 1
    assert summary['watchdog_timeouts'] == 1
    assert summary['failure_layer'] == 'interface'


def test_record_only_run_is_not_a_failure():
    summary = summarize(None, [], [], [], [], expect_report=False)
    assert summary['outcome'] == 'recorded'
    assert summary['failure_layer'] is None
    missing = summarize(None, [], [], [], [], expect_report=True)
    assert missing['failure_layer'] == 'interface'


def test_aggregate_success_rate():
    result = aggregate([
        {'outcome': 'succeeded', 'sim_duration_sec': 10.0},
        {'outcome': 'failed', 'failure_layer': 'control'},
    ])
    assert result['success_rate'] == 0.5
    assert result['failure_layers'] == {'control': 1}


def test_config_validation_and_topics():
    config = parse_config({'name': 'run-1', 'scenario': 'hz26', 'extra_topics': ['/x']})
    assert '/x' in config.topics()
    assert IMAGE_TOPICS[0] in config.topics()
    with pytest.raises(ConfigError):
        parse_config({'name': '../escape', 'scenario': 'hz26'})
    with pytest.raises(ConfigError):
        parse_config({'name': 'ok', 'scenario': ''})
    with pytest.raises(ConfigError):
        parse_config({'name': 'ok', 'scenario': 's', 'unknown': 1})
