"""
Train/validation/test splits by scene or route, never by frame.

Adjacent frames of one run are nearly identical; splitting them randomly
leaks the test set into training. Whole groups go to one split.
"""

import hashlib


def group_key(episode, by):
    scenario, route = episode.group
    if by == 'scenario':
        return scenario
    if by == 'route':
        return f'{scenario}/{route}'
    if by == 'episode':
        return episode.meta.get('run_id', episode.path)
    raise ValueError(f'unknown split key {by!r}')


def _bucket(key, seed):
    digest = hashlib.sha256(f'{seed}:{key}'.encode('utf-8')).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_episodes(episodes, by='route', fractions=(0.7, 0.15, 0.15), seed=0):
    """Deterministically assign whole groups to train/val/test."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError('fractions must sum to 1')
    splits = {'train': [], 'val': [], 'test': []}
    for episode in episodes:
        value = _bucket(group_key(episode, by), seed)
        if value < fractions[0]:
            splits['train'].append(episode)
        elif value < fractions[0] + fractions[1]:
            splits['val'].append(episode)
        else:
            splits['test'].append(episode)
    return splits


def assert_no_leakage(splits, by='route'):
    seen = {}
    for name, episodes in splits.items():
        for episode in episodes:
            key = group_key(episode, by)
            if seen.setdefault(key, name) != name:
                raise AssertionError(f'group {key} appears in {seen[key]} and {name}')
