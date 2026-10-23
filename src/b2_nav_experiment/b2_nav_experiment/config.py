"""
Experiment configuration, independent of ROS.

Example::

    name: nav2-baseline-hz26
    scenario: hz26_visual_deck     # scene; used to split train/test data
    route_id: deck-loop-a          # route; finer split key
    source_label: nav2_baseline    # trajectory source under test
    task_file: ~/b2_nav_data/tasks/hz26_pump_07.yaml   # optional
    warmup_sec: 5.0                # recording starts before the task
    max_duration_sec: 900.0
    include_images: true           # front/rear RGB and depth streams
    extra_topics: []
    param_files:                   # hashed into the manifest
      - b2_nav_bringup/config/nav2_jazzy_params.yaml
    notes: first baseline run after LAN migration
"""

from dataclasses import dataclass, field
import os
import re

import yaml

BASE_TOPICS = [
    '/clock', '/tf', '/tf_static',
    '/odom', '/scan', '/imu', '/amcl_pose', '/initialpose',
    '/cmd_vel_nav', '/cmd_vel', '/cmd_vel_safe',
    '/safety/lidar_min_distance', '/safety/lidar_stop', '/robot/watchdog_state',
    '/local_costmap/costmap', '/plan', '/goal_pose',
    '/local_world/front/obstacle_cloud', '/local_world/rear/obstacle_cloud',
    '/execution/trajectory', '/execution/feedback', '/execution/events',
    '/task/state', '/task/events', '/task/report',
    '/search/request', '/search/result', '/search/observation_goal',
]

IMAGE_TOPICS = [
    '/intel_realsense_r200_depth/image_raw',
    '/intel_realsense_r200_depth/depth/image_raw',
    '/intel_realsense_r200_depth/camera_info',
    '/rear_realsense_depth/image_raw',
    '/rear_realsense_depth/depth/image_raw',
    '/rear_realsense_depth/camera_info',
]

# Topics whose JSON payloads are also written to events.jsonl.
EVENT_TOPICS = [
    '/task/state', '/task/events', '/task/report',
    '/execution/events', '/search/result', '/robot/watchdog_state',
]

_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')


class ConfigError(ValueError):
    pass


@dataclass
class ExperimentConfig:
    name: str
    scenario: str
    route_id: str = ''
    source_label: str = 'unlabelled'
    task_file: str = ''
    warmup_sec: float = 5.0
    max_duration_sec: float = 900.0
    include_images: bool = True
    extra_topics: list = field(default_factory=list)
    param_files: list = field(default_factory=list)
    notes: str = ''

    def topics(self):
        topics = list(BASE_TOPICS)
        if self.include_images:
            topics += IMAGE_TOPICS
        for topic in self.extra_topics:
            if topic not in topics:
                topics.append(topic)
        return topics


def parse_config(data):
    if not isinstance(data, dict):
        raise ConfigError('experiment config must be a mapping')
    try:
        config = ExperimentConfig(**data)
    except TypeError as error:
        raise ConfigError(str(error)) from error
    # The name becomes part of a directory name.
    if not _NAME.match(config.name):
        raise ConfigError('name: letters, digits, "_", "-", "." only (max 64)')
    if not config.scenario:
        raise ConfigError('scenario is required (used to split datasets)')
    if config.max_duration_sec <= 0.0 or config.warmup_sec < 0.0:
        raise ConfigError('max_duration_sec > 0 and warmup_sec >= 0')
    if not all(isinstance(t, str) and t.startswith('/') for t in config.extra_topics):
        raise ConfigError('extra_topics must be absolute topic names')
    config.task_file = os.path.expanduser(config.task_file) if config.task_file else ''
    return config


def load_config(path):
    with open(os.path.expanduser(path), 'r', encoding='utf-8') as handle:
        try:
            return parse_config(yaml.safe_load(handle))
        except yaml.YAMLError as error:
            raise ConfigError(f'cannot parse {path}: {error}') from error
