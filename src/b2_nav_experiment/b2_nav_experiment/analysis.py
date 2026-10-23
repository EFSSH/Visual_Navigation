"""
Run metrics and failure attribution, independent of ROS.

Inputs are plain Python series extracted from a run's bag and events, so
the same functions serve live runs, replays and unit tests.
"""

import math

# Failure reasons from b2_nav_task mapped to the plan's attribution layers.
FAILURE_LAYER = {
    'task_invalid': 'interface',
    'preflight_failed': 'interface',
    'localization_failed': 'localization',
    'subgoal_unreachable': 'trajectory_generation',
    'subgoal_timeout': 'control',
    'object_not_found': 'object_confirmation',
    'object_not_confirmed': 'object_confirmation',
    'lookalike_rejected': 'object_confirmation',
    'observation_unreachable': 'trajectory_generation',
    'cancelled': 'operator',
    'internal_error': 'interface',
}


def path_length(xy, min_step_m=0.005):
    """Travelled distance, ignoring sub-``min_step_m`` odometry jitter."""
    total = 0.0
    last = None
    for point in xy:
        if last is None:
            last = point
            continue
        step = math.hypot(point[0] - last[0], point[1] - last[1])
        if step >= min_step_m:
            total += step
            last = point
    return total


def rising_edges(series):
    """Count False->True transitions in [(t, bool)] and total True time."""
    count = 0
    active_time = 0.0
    previous = None
    for t, value in series:
        if previous is not None and previous[1]:
            active_time += t - previous[0]
        if value and (previous is None or not previous[1]):
            count += 1
        previous = (t, value)
    return count, active_time


def minimum(series):
    values = [v for _, v in series if v is not None and math.isfinite(v)]
    return min(values) if values else None


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low, high = int(math.floor(position)), int(math.ceil(position))
    return values[low] + (values[high] - values[low]) * (position - low)


def attribute_failure(report, metrics):
    """
    Return (layer, evidence) for a failed run, or (None, []) on success.

    The task's own failure reason gives the first layer. Recorded signals
    can move it: a run dominated by safety stops or near contacts points at
    perception/obstacle handling, repeated trajectory rejections at the
    trajectory source, and watchdog timeouts at the interface. The evidence
    list keeps every signal so the attribution can be reviewed by hand.
    """
    if report and report.get('outcome') == 'succeeded':
        return None, []
    reason = (report or {}).get('failure_reason') or 'no_report'
    layer = FAILURE_LAYER.get(reason, 'interface')
    evidence = [f'task failure_reason={reason}']
    if metrics.get('watchdog_timeouts', 0) > 0:
        evidence.append(f'watchdog timeouts={metrics["watchdog_timeouts"]}')
        layer = 'interface'
    elif metrics.get('trajectory_rejections', 0) >= 3:
        evidence.append(f'trajectory rejections={metrics["trajectory_rejections"]}')
        layer = 'trajectory_generation'
    elif (metrics.get('safety_stop_events', 0) >= 3
          and layer in ('control', 'trajectory_generation')):
        evidence.append(f'safety stops={metrics["safety_stop_events"]}')
        layer = 'perception'
    if metrics.get('contact'):
        evidence.append(f'min obstacle distance={metrics.get("min_obstacle_distance_m")}')
    return layer, evidence


def summarize(report, odom_xy, safety_stop, obstacle_distance, events,
              contact_distance_m=0.15, expect_report=True):
    """
    Combine report, signals and events into the run summary dict.

    ``expect_report`` is False for record-only runs without a task; those
    are marked ``recorded`` instead of being attributed as failures.
    """
    stops, stop_time = rising_edges(safety_stop)
    min_distance = minimum(obstacle_distance)
    execution = [e for e in events if e.get('schema') == 'b2_nav/execution_event/v1']
    watchdog = [e for e in events if e.get('schema') == 'b2_nav/watchdog_state/v1']
    nav_goals = [
        e for e in events
        if e.get('schema') == 'b2_nav/task_event/v1' and e.get('event') == 'nav_goal'
    ]
    metrics = {
        'outcome': (report or {}).get('outcome', 'no_report'),
        'failure_reason': (report or {}).get('failure_reason'),
        'sim_duration_sec': (report or {}).get('sim_duration_sec'),
        'path_length_m': path_length(odom_xy),
        'safety_stop_events': stops,
        'safety_stop_time_sec': stop_time,
        'min_obstacle_distance_m': min_distance,
        'contact': min_distance is not None and min_distance < contact_distance_m,
        'watchdog_timeouts': sum(1 for e in watchdog if e.get('state') == 'timeout'),
        'trajectory_accepted': sum(1 for e in execution if e.get('event') == 'accepted'),
        'trajectory_rejections': sum(1 for e in execution if e.get('event') == 'rejected'),
        'trajectory_expired': sum(1 for e in execution if e.get('event') == 'expired'),
        'nav_goals_sent': len(nav_goals),
        'search_windows': (report or {}).get('search_windows'),
        'search_evidence': ((report or {}).get('last_search') or {}).get('evidence'),
    }
    if report is None and not expect_report:
        metrics['outcome'] = 'recorded'
        layer, evidence = None, []
    else:
        layer, evidence = attribute_failure(report, metrics)
    metrics['failure_layer'] = layer
    metrics['failure_evidence'] = evidence
    return metrics


def aggregate(summaries):
    """Success rate and distribution over several run summaries."""
    total = len(summaries)
    succeeded = [s for s in summaries if s.get('outcome') == 'succeeded']
    durations = [s['sim_duration_sec'] for s in succeeded if s.get('sim_duration_sec')]
    layers = {}
    for s in summaries:
        if s.get('failure_layer'):
            layers[s['failure_layer']] = layers.get(s['failure_layer'], 0) + 1
    return {
        'runs': total,
        'success_rate': len(succeeded) / total if total else None,
        'contact_runs': sum(1 for s in summaries if s.get('contact')),
        'median_success_duration_sec': percentile(durations, 0.5),
        'mean_safety_stops': (
            sum(s.get('safety_stop_events', 0) for s in summaries) / total
            if total else None
        ),
        'failure_layers': layers,
    }
