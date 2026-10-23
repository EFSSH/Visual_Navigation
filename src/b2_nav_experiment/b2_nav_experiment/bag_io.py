"""Read recorded runs with rosbag2_py and convert them to plain arrays."""

import json
import math

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from rosidl_runtime_py.utilities import get_message


def iter_messages(bag_dir, topics=None):
    """
    Yield (topic, message, stamp_sec) in recording order.

    Stamps are the bag receive times; runs are recorded with sim time so
    all topics, including header-less Twist, share the simulation clock.
    """
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_dir, storage_id=''), ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = [t for t in (topics or types) if t in types]
    if not wanted:
        return
    reader.set_filter(StorageFilter(topics=wanted))
    classes = {name: get_message(types[name]) for name in wanted}
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        yield topic, deserialize_message(data, classes[topic]), stamp_ns * 1e-9


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def extract_signals(bag_dir):
    """Series needed for the run summary."""
    odom_xy, safety_stop, obstacle_distance = [], [], []
    for topic, msg, t in iter_messages(
        bag_dir, ['/odom', '/safety/lidar_stop', '/safety/lidar_min_distance']
    ):
        if topic == '/odom':
            p = msg.pose.pose.position
            odom_xy.append((p.x, p.y))
        elif topic == '/safety/lidar_stop':
            safety_stop.append((t, bool(msg.data)))
        else:
            obstacle_distance.append((t, float(msg.data)))
    return odom_xy, safety_stop, obstacle_distance


def export_episode(bag_dir, output_path, meta):
    """
    Write one run as an ``.npz`` episode for ROS-independent training.

    Arrays (each with its own ``*_t`` time vector in sim seconds):
    pose (N,3 x,y,yaw in odom) and twist (N,2 vx,wz) from /odom;
    cmd_safe (M,2) executed after the safety gate; cmd_nav (M',2) proposed
    by the controller; scan (K,R); grid (G,H,W int8) with grid_origin
    (G,2) from the local costmap; safety_stop (S,).
    """
    series = {key: [] for key in (
        'pose', 'twist', 'cmd_safe', 'cmd_nav', 'scan', 'grid', 'grid_origin',
        'safety_stop',
    )}
    times = {key: [] for key in ('pose', 'cmd_safe', 'cmd_nav', 'scan', 'grid', 'safety_stop')}
    grid_shape = None
    resolution = None
    topics = ['/odom', '/cmd_vel_safe', '/cmd_vel_nav', '/scan',
              '/local_costmap/costmap', '/safety/lidar_stop']
    for topic, msg, t in iter_messages(bag_dir, topics):
        if topic == '/odom':
            pose = msg.pose.pose
            series['pose'].append((pose.position.x, pose.position.y, _yaw(pose.orientation)))
            series['twist'].append((msg.twist.twist.linear.x, msg.twist.twist.angular.z))
            times['pose'].append(t)
        elif topic in ('/cmd_vel_safe', '/cmd_vel_nav'):
            key = 'cmd_safe' if topic == '/cmd_vel_safe' else 'cmd_nav'
            series[key].append((msg.linear.x, msg.angular.z))
            times[key].append(t)
        elif topic == '/scan':
            series['scan'].append(np.asarray(msg.ranges, dtype=np.float32))
            times['scan'].append(t)
        elif topic == '/local_costmap/costmap':
            shape = (msg.info.height, msg.info.width)
            if grid_shape is None:
                grid_shape, resolution = shape, msg.info.resolution
            if shape != grid_shape:
                continue  # resized costmap; keep a single shape per episode
            series['grid'].append(np.asarray(msg.data, dtype=np.int8).reshape(shape))
            series['grid_origin'].append(
                (msg.info.origin.position.x, msg.info.origin.position.y)
            )
            times['grid'].append(t)
        else:
            series['safety_stop'].append(bool(msg.data))
            times['safety_stop'].append(t)

    arrays = {
        'pose': np.asarray(series['pose'], np.float64).reshape(-1, 3),
        'twist': np.asarray(series['twist'], np.float64).reshape(-1, 2),
        'cmd_safe': np.asarray(series['cmd_safe'], np.float64).reshape(-1, 2),
        'cmd_nav': np.asarray(series['cmd_nav'], np.float64).reshape(-1, 2),
        'safety_stop': np.asarray(series['safety_stop'], bool),
        'grid_origin': np.asarray(series['grid_origin'], np.float64).reshape(-1, 2),
        'grid_resolution': np.float64(resolution or 0.0),
    }
    arrays['scan'] = (
        np.stack(series['scan']) if series['scan']
        and len({s.shape for s in series['scan']}) == 1
        else np.zeros((0, 0), np.float32)
    )
    arrays['grid'] = (
        np.stack(series['grid']) if series['grid'] else np.zeros((0, 0, 0), np.int8)
    )
    for key, values in times.items():
        arrays[f'{key}_t'] = np.asarray(values, np.float64)
    arrays['meta'] = np.asarray(json.dumps(meta))
    np.savez_compressed(output_path, **arrays)
    return {key: int(value.shape[0]) for key, value in arrays.items()
            if hasattr(value, 'shape') and value.ndim > 0}
