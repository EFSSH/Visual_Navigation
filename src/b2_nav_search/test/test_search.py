import math
import os

from b2_nav_search.geometry import (
    back_project,
    choose_observation_pose,
    consensus,
    depth_at,
)
from b2_nav_search.library import LibraryError, load_library
from b2_nav_search.matching import ReferenceMatcher
from b2_nav_search.session import (
    SearchSession,
    STATUS_CONFIRMED,
    STATUS_LOOKALIKE,
    STATUS_NOT_CONFIRMED,
    STATUS_NOT_FOUND,
)
import cv2
import numpy as np
import pytest


def _texture(seed=0, size=(160, 200)):
    rng = np.random.default_rng(seed)
    image = np.zeros((size[0], size[1], 3), np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, size[1]), rng.integers(0, size[0])
        w, h = rng.integers(8, 40), rng.integers(8, 40)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, -1)
    return image


def _write_device(root, device_id, marker_id=None, min_views=2):
    directory = root / device_id
    directory.mkdir()
    cv2.imwrite(str(directory / 'object_1.png'), _texture())
    lines = [
        f'device_id: {device_id}',
        'region: test_room',
        'object_images: [object_1.png]',
        'confirm: {min_views: %d, cluster_radius_m: 0.4, min_inliers: 15}' % min_views,
    ]
    if marker_id is not None:
        lines.append(f'marker_id: {marker_id}')
    (directory / 'device.yaml').write_text('\n'.join(lines) + '\n')


class _Match:
    def __init__(self, inliers=40):
        self.inliers = inliers
        self.corners = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])


def test_library_loads_and_validates(tmp_path):
    _write_device(tmp_path, 'PUMP-07', marker_id=7)
    devices = load_library(str(tmp_path))
    assert devices['PUMP-07'].marker_id == 7
    assert os.path.isfile(devices['PUMP-07'].object_images[0])


def test_library_rejects_mismatched_id(tmp_path):
    _write_device(tmp_path, 'PUMP-07')
    (tmp_path / 'PUMP-07' / 'device.yaml').write_text(
        'device_id: OTHER\nobject_images: [object_1.png]\n'
    )
    with pytest.raises(LibraryError):
        load_library(str(tmp_path))


def test_library_rejects_path_escape(tmp_path):
    _write_device(tmp_path, 'PUMP-07')
    (tmp_path / 'PUMP-07' / 'device.yaml').write_text(
        'device_id: PUMP-07\nobject_images: [../../etc/passwd]\n'
    )
    with pytest.raises(LibraryError):
        load_library(str(tmp_path))


def test_matcher_finds_reference_inside_larger_frame():
    reference = _texture(seed=3)
    frame = np.full((240, 320, 3), 127, np.uint8)
    frame[40:200, 60:260] = reference
    matcher = ReferenceMatcher()
    matcher.add_reference('ref', reference)
    match = matcher.match('ref', matcher.features(frame))
    assert match is not None and match.inliers >= 15
    u, v = match.centroid
    assert abs(u - 160) < 10 and abs(v - 120) < 10


def test_matcher_rejects_unrelated_frame():
    matcher = ReferenceMatcher()
    matcher.add_reference('ref', _texture(seed=3))
    match = matcher.match('ref', matcher.features(_texture(seed=11)))
    assert match is None or match.inliers < 15


def test_depth_median_ignores_invalid_values():
    depth = np.full((10, 10), np.nan, np.float32)
    depth[4:7, 4:7] = 2.0
    depth[5, 5] = 0.0
    assert depth_at(depth, 5, 5) == pytest.approx(2.0)
    assert depth_at(np.full((10, 10), 1500, np.uint16), 5, 5) == pytest.approx(1.5)


def test_back_projection_principal_point():
    assert back_project(160, 120, 2.0, 200, 200, 160, 120) == (0.0, 0.0, 2.0)


def test_consensus_tolerates_outlier():
    centre, agreeing = consensus([(1.0, 1.0), (1.1, 0.9), (5.0, 5.0)], 2, 0.4)
    assert agreeing == 2
    assert centre == pytest.approx((1.05, 0.95))


def test_observation_pose_faces_device_and_avoids_blocked_cells():
    def free(x, y):
        return y >= -0.01  # everything below the device is blocked

    x, y, yaw = choose_observation_pose((0.0, 0.0), (0.0, -3.0), 1.0, None, free)
    assert math.hypot(x, y) == pytest.approx(1.0)
    assert y >= -0.01
    # Heading points from the stand-off pose to the device.
    heading = math.atan2(-y, -x)
    assert math.cos(heading - yaw) == pytest.approx(1.0)


def _session(tmp_path, marker_id=None, known=None, min_views=2):
    _write_device(tmp_path, 'PUMP-07', marker_id=marker_id, min_views=min_views)
    profile = load_library(str(tmp_path))['PUMP-07']
    return SearchSession(profile, known or {}, min_view_interval_sec=0.5)


def test_session_confirms_after_agreeing_views(tmp_path):
    session = _session(tmp_path)
    session.add_frame(0.0, _Match(), (2.0, 1.0), {})
    assert not session.confirmed()
    session.add_frame(0.2, _Match(), (2.0, 1.0), {})  # too soon: same view
    assert not session.confirmed()
    session.add_frame(1.0, _Match(), (2.1, 1.0), {})
    assert session.final_status() == STATUS_CONFIRMED


def test_session_requires_own_marker(tmp_path):
    session = _session(tmp_path, marker_id=7, known={7: 'PUMP-07'})
    session.add_frame(0.0, _Match(), (2.0, 1.0), {})
    session.add_frame(1.0, _Match(), (2.0, 1.0), {})
    assert session.final_status() == STATUS_NOT_CONFIRMED
    session.add_frame(2.0, _Match(), (2.0, 1.0), {7: (50.0, 50.0)})
    assert session.final_status() == STATUS_CONFIRMED


def test_session_rejects_lookalike_with_other_marker(tmp_path):
    session = _session(tmp_path, marker_id=7, known={7: 'PUMP-07', 8: 'PUMP-08'})
    for t in range(3):
        session.add_frame(float(t), _Match(), (2.0, 1.0), {8: (50.0, 50.0)})
    assert session.final_status() == STATUS_LOOKALIKE


def test_session_without_matches_is_not_found(tmp_path):
    session = _session(tmp_path)
    session.add_frame(0.0, None, None, {})
    session.add_frame(1.0, _Match(inliers=5), (2.0, 1.0), {})
    assert session.final_status() == STATUS_NOT_FOUND
