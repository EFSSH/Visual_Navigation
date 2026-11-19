"""
Known-probability randomization: the logging policy mu and verification rho.

Both probabilities are set by us and recorded with every decision, which is
what makes the calibration weights exact rather than estimated.
"""

from dataclasses import dataclass

import numpy as np

from .sim2d import ACTIONS, GO, SLOW, YIELD


def nominal_action(d_hat, hidden0, clear=2.0):
    """
    Hand-written nominal policy the logging policy randomizes around.

    Hidden agent: slow down. Visible agent: the fastest action whose
    predicted clearance is at least ``clear`` metres, else yield.
    """
    return np.where(
        hidden0, SLOW,
        np.where(d_hat[:, GO] >= clear, GO,
                 np.where(d_hat[:, SLOW] >= clear, SLOW, YIELD)))


def epsilon_propensities(nominal, candidate_mask, epsilon):
    """
    mu(a|s): ``epsilon`` on every other allowed action, the rest on the nominal one.

    Positivity: every action in the candidate set has probability >= epsilon.
    """
    n, actions = candidate_mask.shape
    if epsilon * (actions - 1) >= 1.0:
        raise ValueError('epsilon too large for the number of actions')
    mu = np.where(candidate_mask, epsilon, 0.0)
    mu[np.arange(n), nominal] = 0.0
    mu[np.arange(n), nominal] = 1.0 - mu.sum(-1)
    return mu


def sample_actions(propensity, rng):
    u = rng.random(len(propensity))
    return (propensity.cumsum(-1) < u[:, None]).sum(-1).clip(max=propensity.shape[1] - 1)


@dataclass(frozen=True)
class VerificationPolicy:
    """
    rho(s) for the two decision-time strata (agent visible / hidden at t0).

    Verification is drawn before the outcome happens and depends on the
    state only, so given (s, a) it is independent of the outcome.
    """

    rho_visible: float
    rho_hidden: float
    allocation: str = 'custom'

    def __call__(self, hidden0):
        return np.where(hidden0, self.rho_hidden, self.rho_visible)

    def expected_rate(self, hidden_share):
        return hidden_share * self.rho_hidden + (1 - hidden_share) * self.rho_visible

    @classmethod
    def from_budget(cls, budget, hidden_share, allocation='uniform', rho_min=0.02):
        """
        Spend an expected ``budget`` verifications per decision.

        ``uniform``: the same rho everywhere. ``hidden_focus``: rho_min where
        the agent is visible at t0 (the onboard view usually suffices there)
        and the rest of the budget where it is hidden.
        """
        if not 0 < budget <= 1:
            raise ValueError('budget must be in (0, 1]')
        if allocation == 'uniform':
            return cls(budget, budget, allocation)
        if allocation == 'hidden_focus':
            rho_visible = min(rho_min, budget)
            rho_hidden = (budget - (1 - hidden_share) * rho_visible) / max(hidden_share, 1e-9)
            return cls(rho_visible, float(np.clip(rho_hidden, rho_visible, 1.0)), allocation)
        raise ValueError(f'unknown allocation {allocation!r}')


ALLOCATIONS = ('uniform', 'hidden_focus')
ACTION_NAMES = ACTIONS
