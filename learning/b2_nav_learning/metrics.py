"""Prediction metrics: pose drift, occupancy quality, calibration, coverage."""

import numpy as np


def pose_errors(predicted, target):
    """Per-step position error (m) and absolute yaw error (rad), shape (S, K)."""
    position = np.linalg.norm(predicted[..., :2] - target[..., :2], axis=-1)
    dyaw = predicted[..., 2] - target[..., 2]
    yaw = np.abs(np.arctan2(np.sin(dyaw), np.cos(dyaw)))
    return position, yaw


def pose_metrics(predicted, target):
    position, yaw = pose_errors(predicted, target)
    return {
        'ade_m': float(position.mean()),
        'fde_m': float(position[:, -1].mean()),
        'yaw_error_rad': float(yaw.mean()),
        # Multi-step drift: mean error at each horizon step.
        'drift_m_per_step': position.mean(axis=0).round(4).tolist(),
    }


def occupancy_metrics(probability, target, known, threshold=0.5, eps=1e-6):
    """IoU, BCE and calibration over observed cells only."""
    p = np.clip(probability[known], eps, 1.0 - eps)
    y = target[known]
    if y.size == 0:
        return {'cells': 0}
    predicted = p >= threshold
    occupied = y >= 0.5
    union = np.logical_or(predicted, occupied).sum()
    return {
        'cells': int(y.size),
        'iou': float(np.logical_and(predicted, occupied).sum() / union) if union else None,
        'bce': float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        'ece': expected_calibration_error(p, y),
    }


def expected_calibration_error(probability, target, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(probability[mask].mean() - target[mask].mean())
    return float(total)


def action_coverage(train_actions, test_actions, v_bins=8, w_bins=8,
                    v_range=(-0.3, 0.3), w_range=(-1.0, 1.0)):
    """
    Share of test (vx, wz) actions falling in bins never seen in training.

    A model cannot be trusted on actions outside the training coverage; the
    plan asks to report this instead of assuming it.
    """
    def histogram(actions):
        actions = np.asarray(actions).reshape(-1, 2)
        counts, _, _ = np.histogram2d(
            actions[:, 0], actions[:, 1], bins=(v_bins, w_bins), range=(v_range, w_range)
        )
        return counts
    seen = histogram(train_actions) > 0
    test = histogram(test_actions)
    total = test.sum()
    return {
        'train_bins_covered': int(seen.sum()),
        'test_outside_share': float(test[~seen].sum() / total) if total else None,
    }
