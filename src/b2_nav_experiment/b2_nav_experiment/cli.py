"""
Experiment runs: record, replay, analyze, compare and export.

    ros2 run b2_nav_experiment experiment run CONFIG.yaml
    ros2 run b2_nav_experiment experiment replay RUN_DIR [--rate 1.0]
    ros2 run b2_nav_experiment experiment analyze RUN_DIR
    ros2 run b2_nav_experiment experiment compare RUN_DIR [RUN_DIR ...]
    ros2 run b2_nav_experiment experiment export RUN_DIR [--output FILE.npz]

Each run directory under ``~/b2_nav_data/runs`` holds ``manifest.json``
(config, code version, parameter hashes), ``bag/`` (MCAP, zstd),
``events.jsonl`` (task, execution, search and watchdog JSON events),
``summary.json`` and ``report.md``.
"""

import argparse
import datetime
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time

from b2_nav_experiment.analysis import aggregate, summarize
from b2_nav_experiment.config import ConfigError, EVENT_TOPICS, load_config

DATA_ROOT = os.path.join(os.path.expanduser('~'), 'b2_nav_data', 'runs')


# -- manifest -------------------------------------------------------------

def _git(source, *args):
    try:
        return subprocess.run(
            ['git', '--no-optional-locks', '-C', source, *args],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def code_version():
    """Commit and dirty state of the source checkout behind the workspace."""
    marker = os.path.join(
        os.environ.get('B2_NAV_WS', os.path.expanduser('~/b2_nav_ws')),
        '.navigation-migration-workspace',
    )
    try:
        with open(marker, 'r', encoding='utf-8') as handle:
            source = handle.read().strip()
    except OSError:
        return {'source': None}
    status = _git(source, 'status', '--porcelain', '--untracked-files=no')
    return {
        'source': source,
        'commit': _git(source, 'rev-parse', 'HEAD'),
        'branch': _git(source, 'rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': bool(status) if status is not None else None,
    }


def param_hashes(entries):
    from ament_index_python.packages import (
        get_package_share_directory, PackageNotFoundError,
    )
    hashes = {}
    for entry in entries:
        package, _, relative = entry.partition('/')
        try:
            path = os.path.join(get_package_share_directory(package), relative)
            with open(path, 'rb') as handle:
                hashes[entry] = hashlib.sha256(handle.read()).hexdigest()
        except (PackageNotFoundError, OSError) as error:
            hashes[entry] = f'unavailable: {error}'
    return hashes


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


# -- run ------------------------------------------------------------------

def run(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import String

    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as error:
        print(f'invalid experiment config: {error}', file=sys.stderr)
        return 2
    task_text = None
    if config.task_file:
        with open(config.task_file, 'r', encoding='utf-8') as handle:
            task_text = handle.read()

    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(args.data_root, f'{stamp}_{config.name}')
    os.makedirs(run_dir)
    manifest = {
        'schema': 'b2_nav/run_manifest/v1',
        'run_id': os.path.basename(run_dir),
        'config': vars(config),
        'task': task_text,
        'code': code_version(),
        'param_hashes': param_hashes(config.param_files),
        'ros_distro': os.environ.get('ROS_DISTRO'),
        'ros_domain_id': os.environ.get('ROS_DOMAIN_ID'),
        'rmw': os.environ.get('RMW_IMPLEMENTATION'),
        'host': socket.gethostname(),
        'started_wall': datetime.datetime.now().isoformat(),
    }
    write_json(os.path.join(run_dir, 'manifest.json'), manifest)

    bag = subprocess.Popen([
        'ros2', 'bag', 'record', '--use-sim-time',
        '-o', os.path.join(run_dir, 'bag'),
        '--storage', 'mcap', '--storage-preset-profile', 'zstd_fast',
        '--topics', *config.topics(),
    ])

    rclpy.init()
    node = Node('experiment_runner', parameter_overrides=[])
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    events_path = os.path.join(run_dir, 'events.jsonl')
    events = open(events_path, 'a', encoding='utf-8')
    state = {'report': None}

    def on_event(msg, topic):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            payload = {'unparsed': msg.data}
        record = {'topic': topic, 'wall': time.time(), 'payload': payload}
        events.write(json.dumps(record, ensure_ascii=False) + '\n')
        events.flush()
        if topic == '/task/report' and task_text is not None:
            state['report'] = payload

    for topic in EVENT_TOPICS:
        qos = latched if topic in ('/task/state', '/task/report') else 10
        node.create_subscription(String, topic, lambda m, t=topic: on_event(m, t), qos)
    start_pub = node.create_publisher(String, '/task/start', 10)

    interrupted = {'flag': False}
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__('flag', True))
    started = time.monotonic()
    task_sent = False
    print(f'recording {run_dir}')
    try:
        while not interrupted['flag']:
            rclpy.spin_once(node, timeout_sec=0.2)
            elapsed = time.monotonic() - started
            if task_text and not task_sent and elapsed >= config.warmup_sec:
                if start_pub.get_subscription_count() == 0:
                    if elapsed > config.warmup_sec + 30.0:
                        print('task manager not running; recording only', file=sys.stderr)
                        task_sent = True
                    continue
                start_pub.publish(String(data=task_text))
                task_sent = True
                print('task started')
            if state['report'] is not None:
                # Keep a few seconds after the report for the stop to settle.
                time.sleep(2.0)
                break
            if elapsed > config.max_duration_sec:
                print('max_duration_sec reached', file=sys.stderr)
                break
            if bag.poll() is not None:
                print('ros2 bag record exited early', file=sys.stderr)
                break
    finally:
        events.close()
        if bag.poll() is None:
            bag.send_signal(signal.SIGINT)
            try:
                bag.wait(timeout=30)
            except subprocess.TimeoutExpired:
                bag.kill()
        node.destroy_node()
        rclpy.shutdown()

    manifest['ended_wall'] = datetime.datetime.now().isoformat()
    manifest['interrupted'] = interrupted['flag']
    write_json(os.path.join(run_dir, 'manifest.json'), manifest)
    analyze_run(run_dir)
    print(f'run finished: {run_dir}')
    return 0


# -- analysis -------------------------------------------------------------

def load_events(run_dir):
    events, report = [], None
    path = os.path.join(run_dir, 'events.jsonl')
    if not os.path.exists(path):
        return events, report
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            payload = record.get('payload', {})
            events.append(payload)
            if record.get('topic') == '/task/report':
                report = payload
    return events, report


def analyze_run(run_dir):
    from b2_nav_experiment.bag_io import extract_signals

    events, report = load_events(run_dir)
    bag_dir = os.path.join(run_dir, 'bag')
    if os.path.isdir(bag_dir):
        odom_xy, safety_stop, obstacle = extract_signals(bag_dir)
    else:
        odom_xy, safety_stop, obstacle = [], [], []
    manifest_path = os.path.join(run_dir, 'manifest.json')
    manifest = read_json(manifest_path) if os.path.exists(manifest_path) else None
    summary = summarize(
        report, odom_xy, safety_stop, obstacle, events,
        expect_report=bool(manifest and manifest.get('task')),
    )
    if manifest is not None:
        summary['run_id'] = manifest.get('run_id')
        summary['scenario'] = manifest['config'].get('scenario')
        summary['route_id'] = manifest['config'].get('route_id')
        summary['source_label'] = manifest['config'].get('source_label')
        summary['commit'] = (manifest.get('code') or {}).get('commit')
    write_json(os.path.join(run_dir, 'summary.json'), summary)
    with open(os.path.join(run_dir, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write(render_report(summary))
    return summary


def render_report(summary):
    rows = [
        ('结果', summary.get('outcome')),
        ('失败原因', summary.get('failure_reason')),
        ('失败归因层', summary.get('failure_layer')),
        ('归因依据', '; '.join(summary.get('failure_evidence') or [])),
        ('仿真耗时 (s)', summary.get('sim_duration_sec')),
        ('路径长度 (m)', round(summary.get('path_length_m') or 0.0, 2)),
        ('安全门停车次数', summary.get('safety_stop_events')),
        ('最小障碍距离 (m)', summary.get('min_obstacle_distance_m')),
        ('疑似接触', summary.get('contact')),
        ('看门狗超时', summary.get('watchdog_timeouts')),
        ('轨迹接受 / 拒绝 / 过期', '{} / {} / {}'.format(
            summary.get('trajectory_accepted'), summary.get('trajectory_rejections'),
            summary.get('trajectory_expired'))),
        ('导航目标数', summary.get('nav_goals_sent')),
        ('搜索窗口数', summary.get('search_windows')),
    ]
    lines = [f'# 实验运行 {summary.get("run_id")}', '',
             f'场景 {summary.get("scenario")}，路线 {summary.get("route_id")}，'
             f'轨迹来源 {summary.get("source_label")}，代码 {summary.get("commit")}', '',
             '| 指标 | 值 |', '| --- | --- |']
    lines += [f'| {name} | {value} |' for name, value in rows]
    return '\n'.join(lines) + '\n'


def analyze(args):
    print(json.dumps(analyze_run(args.run_dir), indent=2, ensure_ascii=False, default=str))
    return 0


def compare(args):
    summaries = []
    for run_dir in args.run_dirs:
        path = os.path.join(run_dir, 'summary.json')
        summaries.append(read_json(path) if os.path.exists(path) else analyze_run(run_dir))
    groups = {}
    for summary in summaries:
        groups.setdefault(summary.get('source_label'), []).append(summary)
    result = {label: aggregate(items) for label, items in groups.items()}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


# -- replay and export ----------------------------------------------------

def replay(args):
    bag_dir = os.path.join(args.run_dir, 'bag')
    command = ['ros2', 'bag', 'play', bag_dir, '--clock', '--rate', str(args.rate)]
    if args.topics:
        command += ['--topics', *args.topics]
    print('Stop the simulation first: replay publishes /clock and /tf.')
    return subprocess.call(command)


def export(args):
    from b2_nav_experiment.bag_io import export_episode

    manifest = read_json(os.path.join(args.run_dir, 'manifest.json'))
    _, report = load_events(args.run_dir)
    meta = {
        'run_id': manifest.get('run_id'),
        'scenario': manifest['config'].get('scenario'),
        'route_id': manifest['config'].get('route_id'),
        'source_label': manifest['config'].get('source_label'),
        'commit': (manifest.get('code') or {}).get('commit'),
        'outcome': (report or {}).get('outcome'),
    }
    output = args.output or os.path.join(args.run_dir, 'episode.npz')
    counts = export_episode(os.path.join(args.run_dir, 'bag'), output, meta)
    print(json.dumps({'output': output, 'counts': counts}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='experiment', description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='record one experiment run')
    p.add_argument('config')
    p.add_argument('--data-root', default=DATA_ROOT)
    p.set_defaults(func=run)
    p = sub.add_parser('replay', help='replay a run with its clock')
    p.add_argument('run_dir')
    p.add_argument('--rate', type=float, default=1.0)
    p.add_argument('--topics', nargs='*')
    p.set_defaults(func=replay)
    p = sub.add_parser('analyze', help='recompute summary.json and report.md')
    p.add_argument('run_dir')
    p.set_defaults(func=analyze)
    p = sub.add_parser('compare', help='aggregate runs by trajectory source')
    p.add_argument('run_dirs', nargs='+')
    p.set_defaults(func=compare)
    p = sub.add_parser('export', help='export a run as a training episode')
    p.add_argument('run_dir')
    p.add_argument('--output')
    p.set_defaults(func=export)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
