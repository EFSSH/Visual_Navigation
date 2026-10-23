"""
Inspection task description and validation, independent of ROS.

Example task file::

    task_id: hz26-pump-07
    device_id: PUMP-07
    region: hz26_pump_room
    # Manual subgoals in the map frame until the digital-twin route
    # generator exists; the last one is the search-area entry pose.
    route:
      - {x: -7.0, y: -20.0, yaw: 1.57}
      - {x: -4.0, y: -15.0, yaw: 0.0}
    search:
      max_attempts: 4        # search windows, separated by in-place turns
      attempt_timeout_sec: 15.0
      spin_step_rad: 1.5708
    retries:
      subgoal: 1             # re-send a failed subgoal this many times
    timeouts_sec:
      preflight: 30.0
      localization: 60.0
      subgoal: 300.0
      observation: 120.0
"""

from dataclasses import dataclass, field
import math

import yaml


class TaskError(ValueError):
    pass


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass
class SearchSpec:
    max_attempts: int = 4
    attempt_timeout_sec: float = 15.0
    spin_step_rad: float = math.pi / 2.0


@dataclass
class TaskSpec:
    task_id: str
    device_id: str
    region: str
    route: list
    search: SearchSpec = field(default_factory=SearchSpec)
    subgoal_retries: int = 1
    timeouts_sec: dict = field(default_factory=lambda: {
        'preflight': 30.0,
        'localization': 60.0,
        'subgoal': 300.0,
        'observation': 120.0,
    })


def _pose(entry, index):
    try:
        pose = Pose2D(float(entry['x']), float(entry['y']), float(entry.get('yaw', 0.0)))
    except (KeyError, TypeError, ValueError) as error:
        raise TaskError(f'route[{index}] needs numeric x, y[, yaw]: {error}') from error
    if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw)):
        raise TaskError(f'route[{index}] is not finite')
    return pose


def parse_task(data):
    """Validate a task dict and return a TaskSpec."""
    if not isinstance(data, dict):
        raise TaskError('task must be a mapping')
    for key in ('task_id', 'device_id'):
        if not str(data.get(key, '')).strip():
            raise TaskError(f'{key} is required')
    route = data.get('route')
    if not isinstance(route, list) or not route:
        raise TaskError('route must list at least one subgoal')
    try:
        search = SearchSpec(**(data.get('search') or {}))
    except TypeError as error:
        raise TaskError(f'search: {error}') from error
    if search.max_attempts < 1 or search.attempt_timeout_sec <= 0.0:
        raise TaskError('search.max_attempts >= 1 and attempt_timeout_sec > 0')
    spec = TaskSpec(
        task_id=str(data['task_id']),
        device_id=str(data['device_id']),
        region=str(data.get('region', '')),
        route=[_pose(entry, i) for i, entry in enumerate(route)],
        search=search,
        subgoal_retries=int((data.get('retries') or {}).get('subgoal', 1)),
    )
    for key, value in (data.get('timeouts_sec') or {}).items():
        if key not in spec.timeouts_sec:
            raise TaskError(f'unknown timeout {key}')
        if float(value) <= 0.0:
            raise TaskError(f'timeout {key} must be positive')
        spec.timeouts_sec[key] = float(value)
    return spec


def load_task(text):
    """Parse a YAML or JSON task document."""
    try:
        return parse_task(yaml.safe_load(text))
    except yaml.YAMLError as error:
        raise TaskError(f'cannot parse task: {error}') from error
