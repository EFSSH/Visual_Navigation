"""
Agent predictors and the predicted clearance used as the planning signal.

The nonconformity score is S = D_hat(a) - D(a): how much the predictor
overestimates the minimum robot-agent distance under action a. Positive
scores are optimistic (dangerous); a calibrated quantile q gives the lower
bound D(a) >= D_hat(a) - q with the target probability. This one-dimensional,
action-conditioned score is what the planner needs, and it makes "predicted
safe but actually a near miss" directly measurable.
"""

import numpy as np

from .sim2d import ACTIONS, clearance, robot_rollout


class ConstantVelocityPhantom:
    """
    Visible agent: constant velocity from the noisy estimate (no reaction model).

    Hidden agent: a phantom at the occlusion boundary moving towards the
    crossing at a nominal speed. Neither branch models the agent's reaction
    to the robot, so the predictor is wrong in an action-dependent way.
    """

    name = 'cv_phantom'

    def __init__(self, scn, phantom_speed=1.0):
        self.scn = scn
        self.phantom_speed = phantom_speed

    def predict(self, state):
        """Predicted agent positions (n, H+1, 2)."""
        hidden = state['hidden0']
        obs = state['agent_obs0']
        n = len(hidden)
        pos = np.where(hidden[:, None], state['phantom_p0'], obs[:, :2])
        phantom_v = np.tile([0.0, -self.phantom_speed], (n, 1))
        vel = np.where(hidden[:, None], phantom_v, obs[:, 2:])
        t = self.scn.times
        return pos[:, None, :] + vel[:, None, :] * t[None, :, None]


def candidate_robot_trajs(scn, state):
    """Robot rollouts for every candidate action: (n, A, H+1, 2)."""
    x0 = state['robot_state0'][:, 0]
    return np.stack([robot_rollout(scn, x0, a) for a in range(len(ACTIONS))], 1)


def predicted_clearance(scn, state, predictor):
    """D_hat for every candidate action: (n, A)."""
    agent_hat = predictor.predict(state)
    robots = candidate_robot_trajs(scn, state)
    return clearance(robots, agent_hat[:, None])
