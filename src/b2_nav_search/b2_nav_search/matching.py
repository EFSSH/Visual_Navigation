"""
Reference-image matching with ORB features and a RANSAC homography.

Similarity only nominates candidates. Identity is decided later from
multi-view agreement and, when configured, a fiducial marker ID.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Match:
    reference: str
    inliers: int
    good_matches: int
    corners: np.ndarray  # 4x2 projected reference corners in the frame
    centroid: tuple      # (u, v) pixel


def to_gray(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class ReferenceMatcher:

    def __init__(self, n_features=1000, ratio=0.75, ransac_px=5.0, min_good=10):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio = ratio
        self.ransac_px = ransac_px
        self.min_good = min_good
        self.references = {}

    def add_reference(self, name, image):
        gray = to_gray(image)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < self.min_good:
            raise ValueError(f'reference {name} has too few features')
        height, width = gray.shape[:2]
        self.references[name] = (keypoints, descriptors, (width, height))

    def features(self, image):
        return self.orb.detectAndCompute(to_gray(image), None)

    def match(self, name, frame_features):
        """Return a Match for one reference, or None."""
        keypoints, descriptors = frame_features
        if descriptors is None or len(keypoints) < self.min_good:
            return None
        ref_kp, ref_desc, (width, height) = self.references[name]
        pairs = self.matcher.knnMatch(ref_desc, descriptors, k=2)
        good = [
            pair[0] for pair in pairs
            if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance
        ]
        if len(good) < self.min_good:
            return None
        src = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
        if homography is None:
            return None
        inliers = int(mask.sum())
        box = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        corners = cv2.perspectiveTransform(box.reshape(-1, 1, 2), homography)
        corners = corners.reshape(4, 2)
        if not plausible_quad(corners):
            return None
        u, v = corners.mean(axis=0)
        return Match(name, inliers, len(good), corners, (float(u), float(v)))

    def best_match(self, names, frame_features):
        best = None
        for name in names:
            if name not in self.references:
                # Reference failed to load; already reported at startup.
                continue
            candidate = self.match(name, frame_features)
            if candidate and (best is None or candidate.inliers > best.inliers):
                best = candidate
        return best


def plausible_quad(corners, min_area_px=100.0):
    """Reject degenerate or self-intersecting projections."""
    if not np.all(np.isfinite(corners)):
        return False
    contour = corners.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return False
    return cv2.contourArea(contour) >= min_area_px


def detect_markers(image):
    """Return {marker_id: (u, v)} for ArUco 4x4_50 markers in the frame."""
    gray = to_gray(image)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters()
        )
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None:
        return {}
    return {
        int(marker_id): tuple(float(c) for c in quad.reshape(4, 2).mean(axis=0))
        for marker_id, quad in zip(ids.flatten(), corners)
    }


def point_in_quad(point, corners, margin_px=0.0):
    contour = corners.astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, point, True) >= -margin_px
