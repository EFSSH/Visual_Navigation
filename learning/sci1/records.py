"""
Decision log: the record format shared by the 2D simulator, Gazebo and the robot.

One row per randomized decision. The fields follow the second-version plan:
candidate actions, propensities mu(a|s), branch id, random seed, the
verification draw, and the agent ground truth when it was observed. Fields
prefixed with ``oracle_`` exist only in simulation (all-branch ground truth)
and must be absent in real logs; calibration code never reads them.

Files are ``.npz`` with a JSON ``meta`` entry, matching the episode export
of ``b2_nav_experiment``. NaN marks unobserved positions.
"""

import json

import numpy as np

SCHEMA_VERSION = 1

# name: (dtype kind, trailing shape as a function of (A, H), description)
FIELDS = {
    'episode_id': ('i', lambda A, H: (), 'episode index within the log'),
    'decision_index': ('i', lambda A, H: (), 'decision index within the episode'),
    'seed': ('u', lambda A, H: (), 'per-episode seed; replays the world exactly'),
    'branch_id': ('i', lambda A, H: (), '0 = executed branch'),
    'robot_state0': ('f', lambda A, H: (2,), 'robot x, speed at decision time'),
    'hidden0': ('b', lambda A, H: (), 'agent not detected at decision time'),
    'agent_obs0': ('f', lambda A, H: (4,), 'detected agent x, y, vx, vy (NaN if not)'),
    'phantom_p0': ('f', lambda A, H: (2,), 'occlusion boundary point on the agent lane'),
    'candidate_mask': ('b', lambda A, H: (A,), 'actions allowed by the safety gate'),
    'propensity': ('f', lambda A, H: (A,), 'logging policy mu(a|s), known exactly'),
    'action': ('i', lambda A, H: (), 'executed action index'),
    'verify_prob': ('f', lambda A, H: (), 'verification probability rho(s), known exactly'),
    'verified': ('b', lambda A, H: (), 'independent verification drawn (Q)'),
    'visible_steps': ('b', lambda A, H: (H + 1,), 'detector reported the agent at each step'),
    'robot_traj': ('f', lambda A, H: (H + 1, 2), 'executed robot trajectory'),
    'agent_traj_observed': ('f', lambda A, H: (H + 1, 2), 'onboard view; NaN where not detected'),
    'agent_traj_full': ('f', lambda A, H: (H + 1, 2), 'outcome if observed (R = 1): truth if verified, else onboard record; NaN otherwise'),
}

ORACLE_FIELDS = {
    'oracle_robot_traj': ('f', lambda A, H: (A, H + 1, 2), 'sim only: every branch'),
    'oracle_agent_traj': ('f', lambda A, H: (A, H + 1, 2), 'sim only: every branch'),
    'oracle_visible_steps': ('b', lambda A, H: (A, H + 1), 'sim only: every branch'),
}


class DecisionLog:
    """Columnar log; ``meta`` holds scenario parameters and action names."""

    def __init__(self, meta, columns):
        self.meta = dict(meta)
        self.columns = dict(columns)
        self.validate()

    def __len__(self):
        return len(self.columns['action'])

    def __getitem__(self, name):
        return self.columns[name]

    def __contains__(self, name):
        return name in self.columns

    @property
    def has_oracle(self):
        return all(name in self.columns for name in ORACLE_FIELDS)

    def validate(self):
        actions = len(self.meta['action_names'])
        horizon = int(self.meta['horizon'])
        missing = [name for name in FIELDS if name not in self.columns]
        if missing:
            raise ValueError(f'decision log is missing fields: {missing}')
        unknown = [name for name in self.columns if name not in FIELDS and name not in ORACLE_FIELDS]
        if unknown:
            raise ValueError(f'decision log has unknown fields: {unknown}')
        n = len(self.columns['action'])
        for name, column in self.columns.items():
            kind, shape, _ = FIELDS.get(name) or ORACLE_FIELDS[name]
            expected = (n,) + shape(actions, horizon)
            if column.shape != expected:
                raise ValueError(f'{name}: shape {column.shape}, expected {expected}')
            if column.dtype.kind != kind:
                raise ValueError(f'{name}: dtype {column.dtype}, expected kind {kind!r}')
        propensity = self.columns['propensity']
        if not np.allclose(propensity.sum(-1), 1.0):
            raise ValueError('propensities must sum to one per decision')
        if (propensity[~self.columns['candidate_mask']] != 0).any():
            raise ValueError('propensity must be zero outside the candidate set')
        taken = propensity[np.arange(n), self.columns['action']]
        if (taken <= 0).any():
            raise ValueError('an executed action has zero propensity')
        observed = self.observed
        if np.isnan(self.columns['agent_traj_full'][observed][:, 1:]).any():
            raise ValueError('observed decisions need a full agent trajectory')
        if not np.isnan(self.columns['agent_traj_full'][~observed]).all():
            raise ValueError('unobserved decisions must not carry a full trajectory')

    @property
    def visible_full(self):
        """V: the outcome (steps 1..H) was fully seen onboard."""
        return self.columns['visible_steps'][:, 1:].all(-1)

    @property
    def observed(self):
        """R = max(V, Q)."""
        return self.visible_full | self.columns['verified']

    def subset(self, mask):
        return DecisionLog(self.meta, {k: v[mask] for k, v in self.columns.items()})

    def without_oracle(self):
        return DecisionLog(self.meta, {k: v for k, v in self.columns.items() if k in FIELDS})

    def save(self, path):
        meta = dict(self.meta, schema_version=SCHEMA_VERSION)
        np.savez_compressed(path, meta=json.dumps(meta), **self.columns)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['meta']))
            if meta.get('schema_version') != SCHEMA_VERSION:
                raise ValueError(f'unsupported schema version {meta.get("schema_version")}')
            columns = {k: data[k] for k in data.files if k != 'meta'}
        return cls(meta, columns)
