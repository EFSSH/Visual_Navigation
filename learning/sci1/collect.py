"""Run the logging policy in the simulator and write a decision log."""

import numpy as np

from .policies import epsilon_propensities, nominal_action, sample_actions
from .predictors import predicted_clearance
from .records import DecisionLog
from .sim2d import ACTIONS, perceive, sample_episodes, simulate_branches


def collect(scn, seeds, predictor, verification, epsilon, rng, keep_oracle=True):
    """
    One randomized decision per episode.

    The robot perceives, draws its action from mu(.|s), draws the
    verification Q ~ Bernoulli(rho(s)) and executes. All branches are
    simulated so the log can carry simulator ground truth (``oracle_*``);
    the executed branch alone provides what a real robot would record.
    """
    world = sample_episodes(scn, seeds)
    state = perceive(scn, world)
    n, actions = len(seeds), len(ACTIONS)
    candidate = np.ones((n, actions), dtype=bool)
    d_hat = predicted_clearance(scn, state, predictor)
    mu = epsilon_propensities(nominal_action(d_hat, state['hidden0']), candidate, epsilon)
    action = sample_actions(mu, rng)
    rho = verification(state['hidden0'])
    verified = rng.random(n) < rho

    robots, agents, viss = simulate_branches(scn, world)
    rows = np.arange(n)
    robot, agent, vis = robots[rows, action], agents[rows, action], viss[rows, action]
    observed = vis[:, 1:].all(-1) | verified
    columns = {
        'episode_id': rows.astype(np.int64),
        'decision_index': np.zeros(n, np.int64),
        'seed': np.asarray(seeds, dtype=np.uint64),
        'branch_id': np.zeros(n, np.int64),
        'robot_state0': state['robot_state0'],
        'hidden0': state['hidden0'],
        'agent_obs0': state['agent_obs0'],
        'phantom_p0': state['phantom_p0'],
        'candidate_mask': candidate,
        'propensity': mu,
        'action': action.astype(np.int64),
        'verify_prob': rho.astype(float),
        'verified': verified,
        'visible_steps': vis,
        'robot_traj': robot,
        'agent_traj_observed': np.where(vis[..., None], agent, np.nan),
        'agent_traj_full': np.where(observed[:, None, None], agent, np.nan),
    }
    if keep_oracle:
        columns.update(oracle_robot_traj=robots, oracle_agent_traj=agents,
                       oracle_visible_steps=viss)
    meta = {
        'scenario': scn.to_dict(),
        'action_names': list(ACTIONS),
        'horizon': scn.horizon,
        'dt': scn.dt,
        'predictor': predictor.name,
        'epsilon': epsilon,
        'verification': {'rho_visible': verification.rho_visible,
                         'rho_hidden': verification.rho_hidden,
                         'allocation': verification.allocation},
    }
    return DecisionLog(meta, columns)


def reverify(log, verification, uniforms):
    """
    Redraw the verification of a simulated log under another policy.

    Uses common random numbers (``uniforms``, one per decision) so that
    allocations compared on the same log differ only in rho. Needs the
    oracle fields, hence simulation only.
    """
    if not log.has_oracle:
        raise ValueError('reverify needs a simulated log with oracle fields')
    rho = verification(log['hidden0']).astype(float)
    verified = uniforms < rho
    rows = np.arange(len(log))
    agent = log['oracle_agent_traj'][rows, log['action']]
    observed = log.visible_full | verified
    columns = dict(log.columns,
                   verify_prob=rho, verified=verified,
                   agent_traj_full=np.where(observed[:, None, None], agent, np.nan))
    meta = dict(log.meta, verification={'rho_visible': verification.rho_visible,
                                        'rho_hidden': verification.rho_hidden,
                                        'allocation': verification.allocation})
    return DecisionLog(meta, columns)


def state_of(log):
    return {k: log[k] for k in ('robot_state0', 'hidden0', 'agent_obs0', 'phantom_p0')}
