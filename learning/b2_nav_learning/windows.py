"""
Fixed-rate history/horizon samples for action-conditioned prediction.

Each sample is anchored at the robot pose at time t. Inputs use only data
up to t; ``future_actions`` are the commands actually executed after t, so
a model predicts the consequence of a given action sequence. Targets are
future poses and future local occupancy, both in the anchor frame.
"""

from dataclasses import dataclass

from b2_nav_learning.episodes import (
    ego_occupancy,
    held_command,
    interp_pose,
    latest_grid_index,
    relative_pose,
)
import numpy as np


@dataclass
class WindowSpec:
    dt: float = 0.2
    history: int = 5
    horizon: int = 10
    stride: float = 0.5        # seconds between anchors
    cells: int = 40
    cell_m: float = 0.1
    max_grid_age: float = 1.0  # newest costmap must be this fresh


def build_samples(episode, spec):
    """Return a dict of stacked arrays for one episode (may be empty)."""
    start, end = episode.time_range()
    first = start + spec.history * spec.dt
    last = end - spec.horizon * spec.dt
    if last <= first or episode.grid.shape[0] == 0:
        return None
    columns = {key: [] for key in (
        'anchor_t', 'past_pose', 'past_action', 'future_action', 'occupancy',
        'occupancy_known', 'target_pose', 'target_occupancy', 'target_known',
    )}
    for t in np.arange(first, last, spec.stride):
        index = latest_grid_index(episode, t)
        if index < 0 or t - episode.grid_t[index] > spec.max_grid_age:
            continue
        anchor = interp_pose(episode, t)[0]
        past_t = t - spec.dt * np.arange(spec.history, 0, -1)
        future_t = t + spec.dt * np.arange(1, spec.horizon + 1)

        future_occ, future_known = [], []
        for ft in future_t:
            f_index = latest_grid_index(episode, ft)
            if f_index < 0 or ft - episode.grid_t[f_index] > spec.max_grid_age:
                break
            occ, known = ego_occupancy(episode, f_index, anchor, spec.cells, spec.cell_m)
            future_occ.append(occ)
            future_known.append(known)
        if len(future_occ) != spec.horizon:
            continue

        occ, known = ego_occupancy(episode, index, anchor, spec.cells, spec.cell_m)
        columns['anchor_t'].append(t)
        columns['past_pose'].append(relative_pose(interp_pose(episode, past_t), anchor))
        columns['past_action'].append(held_command(episode, past_t))
        columns['future_action'].append(held_command(episode, future_t - spec.dt))
        columns['occupancy'].append(occ)
        columns['occupancy_known'].append(known)
        columns['target_pose'].append(relative_pose(interp_pose(episode, future_t), anchor))
        columns['target_occupancy'].append(np.stack(future_occ))
        columns['target_known'].append(np.stack(future_known))
    if not columns['anchor_t']:
        return None
    samples = {key: np.stack(values) for key, values in columns.items()}
    samples['episode'] = np.array([episode.meta.get('run_id', episode.path)] * len(
        columns['anchor_t']
    ))
    return samples


def concat(sample_sets):
    sets = [s for s in sample_sets if s is not None]
    if not sets:
        return None
    return {key: np.concatenate([s[key] for s in sets]) for key in sets[0]}
