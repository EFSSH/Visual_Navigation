"""
Reference library of inspectable devices, independent of ROS.

Layout (one directory per device)::

    <library_dir>/<device_id>/device.yaml
    <library_dir>/<device_id>/scene_*.png   # optional, observation area
    <library_dir>/<device_id>/object_*.png  # required, the device itself

``device.yaml`` keys::

    device_id: PUMP-07            # must match the directory name
    region: hz26_pump_room
    scene_images: [scene_1.png]   # paths relative to the device directory
    object_images: [object_1.png, object_2.png]
    # Optional fiducial standing in for the nameplate number; a different
    # known marker inside the matched box rejects a look-alike device.
    marker_id: 7
    observation:
      distance_m: 1.2             # stand-off from the device centre
      facing_yaw: null            # device front normal in map, rad; null =
                                  # approach along the current line of sight
      yaw_tolerance_rad: 0.35
    confirm:
      min_views: 3                # distinct frames agreeing on position
      cluster_radius_m: 0.4
      min_inliers: 15
"""

from dataclasses import dataclass, field
import math
import os

import yaml


class LibraryError(ValueError):
    pass


@dataclass
class ObservationSpec:
    distance_m: float = 1.2
    facing_yaw: float = None
    yaw_tolerance_rad: float = 0.35


@dataclass
class ConfirmSpec:
    min_views: int = 3
    cluster_radius_m: float = 0.4
    min_inliers: int = 15


@dataclass
class DeviceProfile:
    device_id: str
    region: str
    directory: str
    object_images: list
    scene_images: list = field(default_factory=list)
    marker_id: int = None
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    confirm: ConfirmSpec = field(default_factory=ConfirmSpec)


def _resolve_images(directory, names, key, required):
    if not isinstance(names, list):
        raise LibraryError(f'{key} must be a list')
    if required and not names:
        raise LibraryError(f'{key} must not be empty')
    paths = []
    for name in names:
        path = os.path.normpath(os.path.join(directory, name))
        if not path.startswith(os.path.normpath(directory) + os.sep):
            raise LibraryError(f'{key} entry escapes the device directory: {name}')
        if not os.path.isfile(path):
            raise LibraryError(f'missing image: {path}')
        paths.append(path)
    return paths


def load_device(directory):
    """Load and validate one device directory."""
    meta_path = os.path.join(directory, 'device.yaml')
    try:
        with open(meta_path, 'r', encoding='utf-8') as handle:
            meta = yaml.safe_load(handle) or {}
    except OSError as error:
        raise LibraryError(f'cannot read {meta_path}: {error}') from error

    device_id = str(meta.get('device_id', ''))
    if device_id != os.path.basename(os.path.normpath(directory)):
        raise LibraryError(
            f'device_id {device_id!r} does not match directory {directory}'
        )
    try:
        observation = ObservationSpec(**(meta.get('observation') or {}))
        confirm = ConfirmSpec(**(meta.get('confirm') or {}))
    except TypeError as error:
        raise LibraryError(f'{meta_path}: {error}') from error
    if observation.distance_m <= 0.0:
        raise LibraryError('observation.distance_m must be positive')
    if observation.facing_yaw is not None and not math.isfinite(observation.facing_yaw):
        raise LibraryError('observation.facing_yaw must be finite or null')
    if confirm.min_views < 1 or confirm.min_inliers < 4:
        raise LibraryError('confirm.min_views >= 1 and confirm.min_inliers >= 4')
    marker_id = meta.get('marker_id')
    return DeviceProfile(
        device_id=device_id,
        region=str(meta.get('region', '')),
        directory=directory,
        object_images=_resolve_images(
            directory, meta.get('object_images', []), 'object_images', True
        ),
        scene_images=_resolve_images(
            directory, meta.get('scene_images', []), 'scene_images', False
        ),
        marker_id=None if marker_id is None else int(marker_id),
        observation=observation,
        confirm=confirm,
    )


def load_library(library_dir):
    """Return {device_id: DeviceProfile}; invalid devices raise LibraryError."""
    if not os.path.isdir(library_dir):
        raise LibraryError(f'library directory not found: {library_dir}')
    devices = {}
    for name in sorted(os.listdir(library_dir)):
        directory = os.path.join(library_dir, name)
        if name.startswith('.') or not os.path.isdir(directory):
            continue
        profile = load_device(directory)
        devices[profile.device_id] = profile
    return devices


def known_marker_ids(devices):
    return {d.marker_id: d.device_id for d in devices.values() if d.marker_id is not None}
