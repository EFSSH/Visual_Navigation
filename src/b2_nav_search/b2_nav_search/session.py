"""Evidence accumulation for one device search request, independent of ROS."""

from b2_nav_search.geometry import consensus
from b2_nav_search.matching import point_in_quad

STATUS_CONFIRMED = 'confirmed'
STATUS_NOT_FOUND = 'not_found'
STATUS_NOT_CONFIRMED = 'not_confirmed'
STATUS_LOOKALIKE = 'lookalike_rejected'


class SearchSession:

    def __init__(self, profile, known_markers, min_view_interval_sec=0.5,
                 marker_margin_px=20.0):
        self.profile = profile
        self.known_markers = known_markers
        self.min_view_interval_sec = min_view_interval_sec
        self.marker_margin_px = marker_margin_px
        self.views = []
        self.frames = 0
        self.candidate_frames = 0
        self.lookalike_frames = 0
        self.no_depth_frames = 0
        self.marker_confirmed = profile.marker_id is None
        self.best_inliers = 0
        self.best_scene_inliers = 0
        self.last_view_stamp = None

    def add_frame(self, stamp_sec, match, map_xy, markers, scene_inliers=0):
        """
        Record one processed frame.

        ``match`` is the best object Match (or None), ``map_xy`` the device
        position estimated from depth for that match (or None), ``markers``
        {id: (u, v)} detected in the frame.
        """
        self.frames += 1
        self.best_scene_inliers = max(self.best_scene_inliers, scene_inliers)
        if match is None or match.inliers < self.profile.confirm.min_inliers:
            return
        self.candidate_frames += 1
        self.best_inliers = max(self.best_inliers, match.inliers)

        inside = {
            marker_id for marker_id, uv in markers.items()
            if point_in_quad(uv, match.corners, self.marker_margin_px)
        }
        others = {
            marker_id for marker_id in inside
            if marker_id in self.known_markers
            and self.known_markers[marker_id] != self.profile.device_id
        }
        if others:
            # A look-alike carrying another device's number: never a view.
            self.lookalike_frames += 1
            return
        if self.profile.marker_id is not None and self.profile.marker_id in inside:
            self.marker_confirmed = True
        if map_xy is None:
            self.no_depth_frames += 1
            return
        if (self.last_view_stamp is not None
                and stamp_sec - self.last_view_stamp < self.min_view_interval_sec):
            return
        self.last_view_stamp = stamp_sec
        self.views.append((float(map_xy[0]), float(map_xy[1])))

    def position(self):
        confirm = self.profile.confirm
        return consensus(self.views, confirm.min_views, confirm.cluster_radius_m)

    def confirmed(self):
        centre, _ = self.position()
        return centre is not None and self.marker_confirmed

    def final_status(self):
        if self.confirmed():
            return STATUS_CONFIRMED
        if self.candidate_frames == 0:
            return STATUS_NOT_FOUND
        if self.lookalike_frames and not self.views:
            return STATUS_LOOKALIKE
        return STATUS_NOT_CONFIRMED

    def summary(self):
        centre, agreeing = self.position()
        return {
            'frames': self.frames,
            'candidate_frames': self.candidate_frames,
            'lookalike_frames': self.lookalike_frames,
            'no_depth_frames': self.no_depth_frames,
            'views': len(self.views),
            'agreeing_views': agreeing,
            'marker_required': self.profile.marker_id is not None,
            'marker_confirmed': self.marker_confirmed,
            'best_inliers': self.best_inliers,
            'best_scene_inliers': self.best_scene_inliers,
            'device_xy': centre,
        }
