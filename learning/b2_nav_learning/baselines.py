"""
Reference predictors that any learned model should beat.

A predictor maps a sample batch to ``{'pose': (S, K, 3), 'occupancy':
(S, K, C, C) probabilities}``. These baselines use no training data, which
makes them a fixed yardstick across datasets.
"""

import numpy as np


def integrate_unicycle(actions, dt):
    """Roll out (vx, wz) actions from the origin; returns (S, K, 3)."""
    actions = np.asarray(actions)
    samples, steps = actions.shape[:2]
    pose = np.zeros((samples, 3))
    out = np.zeros((samples, steps, 3))
    for k in range(steps):
        v, w = actions[:, k, 0], actions[:, k, 1]
        # Midpoint heading integrates a constant-rate arc well at small dt.
        heading = pose[:, 2] + 0.5 * w * dt
        pose = pose + np.stack([v * np.cos(heading) * dt,
                                v * np.sin(heading) * dt,
                                w * dt], axis=-1)
        pose[:, 2] = np.arctan2(np.sin(pose[:, 2]), np.cos(pose[:, 2]))
        out[:, k] = pose
    return out


def persistent_occupancy(samples, horizon, occupied_p=0.9, free_p=0.1):
    """Static-world assumption: the anchor-frame map does not change."""
    current = np.where(samples['occupancy'] >= 0.5, occupied_p, free_p)
    return np.repeat(current[:, None], horizon, axis=1)


class ConstantVelocity:
    """Keep executing the last command; ignores the given future actions."""

    name = 'constant_velocity'

    def __init__(self, dt):
        self.dt = dt

    def predict(self, samples):
        horizon = samples['future_action'].shape[1]
        last = samples['past_action'][:, -1:, :]
        actions = np.repeat(last, horizon, axis=1)
        return {
            'pose': integrate_unicycle(actions, self.dt),
            'occupancy': persistent_occupancy(samples, horizon),
        }


class ActionKinematic:
    """
    Integrate the executed future actions with an ideal unicycle.

    This is the action-conditioned baseline: a learned dynamics model adds
    value only where the real base deviates from ideal kinematics (slip,
    acceleration limits, gait-induced motion).
    """

    name = 'action_kinematic'

    def __init__(self, dt):
        self.dt = dt

    def predict(self, samples):
        horizon = samples['future_action'].shape[1]
        return {
            'pose': integrate_unicycle(samples['future_action'], self.dt),
            'occupancy': persistent_occupancy(samples, horizon),
        }


BASELINES = {cls.name: cls for cls in (ConstantVelocity, ActionKinematic)}
