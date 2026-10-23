"""
Evaluate predictors on exported episodes with a leakage-free split.

    python3 -m b2_nav_learning.evaluate --data ~/b2_nav_data/episodes \
        --split-by route --model action_kinematic --output metrics.json

A learned model plugs in through ``--model module:Class``; the class takes
``dt`` and provides ``fit(train_samples)`` (optional) and
``predict(samples)`` returning the same keys as the baselines.
"""

import argparse
import importlib
import json
import sys

from b2_nav_learning.baselines import BASELINES
from b2_nav_learning.episodes import load_episodes
from b2_nav_learning.metrics import action_coverage, occupancy_metrics, pose_metrics
from b2_nav_learning.splits import assert_no_leakage, split_episodes
from b2_nav_learning.windows import build_samples, concat, WindowSpec


def make_model(name, dt):
    if name in BASELINES:
        return BASELINES[name](dt)
    module_name, _, class_name = name.partition(':')
    if not class_name:
        raise SystemExit(f'unknown model {name!r}; baselines: {sorted(BASELINES)}')
    return getattr(importlib.import_module(module_name), class_name)(dt)


def evaluate(episodes, model_name, spec, split_by='route', seed=0):
    splits = split_episodes(episodes, by=split_by, seed=seed)
    assert_no_leakage(splits, by=split_by)
    data = {
        name: concat(build_samples(e, spec) for e in items)
        for name, items in splits.items()
    }
    result = {
        'model': model_name,
        'split_by': split_by,
        'episodes': {name: len(items) for name, items in splits.items()},
        'samples': {name: 0 if d is None else len(d['anchor_t']) for name, d in data.items()},
        'window': vars(spec),
    }
    if data['test'] is None:
        result['error'] = 'no test samples; record more routes or change --split-by'
        return result
    model = make_model(model_name, spec.dt)
    if hasattr(model, 'fit') and data['train'] is not None:
        model.fit(data['train'])
    prediction = model.predict(data['test'])
    result['pose'] = pose_metrics(prediction['pose'], data['test']['target_pose'])
    result['occupancy'] = occupancy_metrics(
        prediction['occupancy'], data['test']['target_occupancy'],
        data['test']['target_known'],
    )
    if data['train'] is not None:
        result['action_coverage'] = action_coverage(
            data['train']['future_action'], data['test']['future_action']
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', required=True, help='directory of episode .npz files')
    parser.add_argument('--model', default='action_kinematic')
    parser.add_argument('--split-by', default='route', choices=['scenario', 'route', 'episode'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dt', type=float, default=0.2)
    parser.add_argument('--history', type=int, default=5)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--output')
    args = parser.parse_args(argv)

    spec = WindowSpec(dt=args.dt, history=args.history, horizon=args.horizon)
    episodes = load_episodes(args.data)
    if not episodes:
        print(f'no episodes under {args.data}', file=sys.stderr)
        return 2
    result = evaluate(episodes, args.model, spec, args.split_by, args.seed)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
