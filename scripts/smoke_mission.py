"""Check the mission layer end to end without a robot or simulation.

Starts navigation_only, mission and the command watchdog in an isolated
domain with a static odom -> base_link transform, then checks: watchdog
forwarding and timeout, TrackPath loading, executor events, search request
handling, task preflight failure reporting, and an experiment
record/analyze/export round trip.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Path as PathMsg
import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster


def launch(args, log_dir, name):
    log = open(log_dir / f'{name}.log', 'w')
    return subprocess.Popen(
        ['ros2', 'launch', *args], stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop(process):
    # Signal ros2 launch only; it forwards SIGINT to its nodes once.
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


def make_library(root):
    import cv2
    device = root / 'SMOKE-01'
    device.mkdir(parents=True)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    cv2.imwrite(str(device / 'object_1.png'), cv2.GaussianBlur(image, (3, 3), 0))
    (device / 'device.yaml').write_text(
        'device_id: SMOKE-01\nregion: smoke\nobject_images: [object_1.png]\n'
    )


def main():
    os.environ['ROS_DOMAIN_ID'] = '143'
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    os.environ.pop('CYCLONEDDS_URI', None)
    work = Path(tempfile.mkdtemp(prefix='b2-mission-smoke-'))
    make_library(work / 'library')
    # The watchdog is checked before navigation starts: once the safety gate
    # runs it publishes zero velocity on /cmd_vel_safe at 20 Hz.
    processes = [
        launch(['b2_nav_robot', 'cmd_watchdog.launch.py', 'use_sim_time:=false'],
               work, 'watchdog'),
    ]
    rclpy.init()
    node = rclpy.create_node('mission_smoke_check')
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    received = {key: [] for key in ('device', 'watchdog', 'exec', 'search', 'report')}
    node.create_subscription(Twist, '/cmd_vel_device', received['device'].append, 10)
    for key, topic, qos in [
        ('watchdog', '/robot/watchdog_state', latched),
        ('exec', '/execution/events', 50),
        ('search', '/search/result', latched),
        ('report', '/task/report', latched),
    ]:
        node.create_subscription(
            String, topic, lambda m, k=key: received[k].append(json.loads(m.data)), qos
        )
    cmd_pub = node.create_publisher(Twist, '/cmd_vel_safe', 10)
    path_pub = node.create_publisher(PathMsg, '/execution/trajectory', 10)
    search_pub = node.create_publisher(String, '/search/request', 10)
    task_pub = node.create_publisher(String, '/task/start', 10)
    tf = StaticTransformBroadcaster(node)
    static = TransformStamped()
    static.header.frame_id, static.child_frame_id = 'odom', 'base_link'
    static.transform.rotation.w = 1.0
    tf.sendTransform(static)

    def spin(seconds, until=None):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if until and until():
                return True
        return bool(until and until())

    results = {}
    try:
        spin(4.0)

        # 1. Watchdog forwards fresh commands and stops on timeout.
        command = Twist()
        command.linear.x = 0.1
        for _ in range(10):
            cmd_pub.publish(command)
            spin(0.1)
        forwarded = any(m.linear.x > 0.05 for m in received['device'])
        spin(1.5)
        stopped = bool(received['device']) and received['device'][-1].linear.x == 0.0
        timed_out = any(s.get('state') == 'timeout' for s in received['watchdog'])
        results['watchdog'] = forwarded and stopped and timed_out

        processes += [
            launch(['b2_nav_bringup', 'navigation_only.launch.py', 'use_sim_time:=false',
                    'use_initial_pose:=false'], work, 'navigation'),
            launch(['b2_nav_bringup', 'mission.launch.py', 'use_sim_time:=false',
                    f'library_dir:={work / "library"}', 'source_label:=smoke'],
                   work, 'mission'),
        ]
        spin(20.0)  # Nav2 lifecycle bring-up

        # 2. TrackPath controller plugin loaded by controller_server.
        nav_log = (work / 'navigation.log').read_text()
        results['track_path_loaded'] = 'TrackPath' in nav_log and (
            'Created controller : TrackPath' in nav_log
            or 'RegulatedPurePursuitController' in nav_log
        )

        # 3. Executor decides on a trajectory (accept or a named rejection).
        path = PathMsg()
        path.header.frame_id = 'odom'
        path.header.stamp = node.get_clock().now().to_msg()
        for i in range(11):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = 0.1 * i
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        path_pub.publish(path)
        spin(4.0, lambda: any(
            e.get('event') in ('accepted', 'rejected') for e in received['exec']
        ))
        decisions = [e for e in received['exec'] if e.get('event') in ('accepted', 'rejected')]
        results['executor_decision'] = (
            decisions[0]['event'] + ':' + str(decisions[0]['details'].get('reason', ''))
            if decisions else False
        )

        # 4. Search: unknown device rejected, known device times out not_found.
        search_pub.publish(String(data=json.dumps({'request_id': 'u', 'device_id': 'NOPE'})))
        search_pub.publish(String(data=json.dumps(
            {'request_id': 'k', 'device_id': 'SMOKE-01', 'timeout_sec': 2.0}
        )))
        spin(6.0, lambda: any(r.get('request_id') == 'k' for r in received['search']))
        by_id = {r.get('request_id'): r.get('status') for r in received['search']}
        results['search'] = by_id.get('u') == 'invalid_request' and by_id.get('k') == 'not_found'

        # 5. Task without sensors fails preflight with a reason.
        task_pub.publish(String(data=(
            'task_id: smoke\ndevice_id: SMOKE-01\nroute: [{x: 1.0, y: 0.0}]\n'
            'timeouts_sec: {preflight: 3.0}\n'
        )))
        spin(8.0, lambda: bool(received['report']))
        report = received['report'][-1] if received['report'] else {}
        results['task_preflight'] = (
            report.get('failure_reason') == 'preflight_failed' and 'not received'
            in report.get('detail', '')
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()
        for process in processes:
            stop(process)

    # 6. Experiment record, analyze and export round trip.
    config = work / 'experiment.yaml'
    config.write_text(
        'name: smoke\nscenario: smoke\nmax_duration_sec: 4.0\nwarmup_sec: 0.0\n'
        'include_images: false\n'
    )
    runs = work / 'runs'
    subprocess.run(['ros2', 'run', 'b2_nav_experiment', 'experiment', 'run', str(config),
                    '--data-root', str(runs)], capture_output=True, timeout=90)
    run_dirs = sorted(runs.glob('*_smoke'))
    ok = bool(run_dirs) and (run_dirs[0] / 'summary.json').exists() and (
        run_dirs[0] / 'bag' / 'metadata.yaml').exists()
    if ok:
        export = subprocess.run(['ros2', 'run', 'b2_nav_experiment', 'experiment', 'export',
                                 str(run_dirs[0])], capture_output=True, timeout=60)
        ok = export.returncode == 0 and (run_dirs[0] / 'episode.npz').exists()
    results['experiment_roundtrip'] = ok

    print(json.dumps({'logs': str(work), 'results': results}, indent=2))
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
