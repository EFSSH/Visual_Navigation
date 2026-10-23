"""Check migrated node startup and Nav2 plugin configuration without a robot."""

import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster


def main():
    # Keep this check separate from device networks and running experiments.
    os.environ['ROS_DOMAIN_ID'] = '142'
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'
    logfile = Path(tempfile.mkdtemp(prefix='b2-migration-')) / 'launch.log'
    with logfile.open('w') as output:
        process = subprocess.Popen([
            'ros2', 'launch', 'b2_nav_bringup', 'navigation_only.launch.py',
            'use_sim_time:=false', 'autostart:=false',
        ], stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        rclpy.init()
        node = rclpy.create_node('migration_smoke_check')
        safe_commands = []
        node.create_subscription(Twist, '/cmd_vel_safe', safe_commands.append, 10)

        def call(service_type, service, request):
            client = node.create_client(service_type, service)
            try:
                if not client.wait_for_service(timeout_sec=30.0):
                    raise RuntimeError(f'Service unavailable: {service}')
                future = client.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
                if not future.done() or future.result() is None:
                    raise RuntimeError(f'Service timed out: {service}')
                return future.result()
            finally:
                node.destroy_client(client)

        try:
            servers = [
                'map_server', 'amcl', 'controller_server', 'smoother_server',
                'planner_server', 'behavior_server', 'bt_navigator',
                'waypoint_follower', 'velocity_smoother',
            ]
            # Configure, but do not activate motion-producing servers or send goals.
            for name in servers:
                request = ChangeState.Request()
                request.transition.id = Transition.TRANSITION_CONFIGURE
                response = call(ChangeState, f'/{name}/change_state', request)
                if not response.success:
                    raise RuntimeError(f'Configuration failed: {name}')
                state = call(GetState, f'/{name}/get_state', GetState.Request())
                if state.current_state.label != 'inactive':
                    raise RuntimeError(f'Unexpected state: {name}')
                print(f'{name}: configured (plugins loaded)', flush=True)

            request = ChangeState.Request()
            request.transition.id = Transition.TRANSITION_ACTIVATE
            assert call(ChangeState, '/map_server/change_state', request).success
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            names = set(node.get_node_names())
            required = {
                'rgbd_obstacle_front', 'rgbd_obstacle_rear',
                'local_world_model_node', 'lidar_safety_node', 'safety_gate_node',
            }
            assert required <= names, f'Missing nodes: {required - names}'
            assert safe_commands, 'Safety gate did not publish'
            assert all(
                all(value == 0.0 for value in (
                    msg.linear.x, msg.linear.y, msg.linear.z,
                    msg.angular.x, msg.angular.y, msg.angular.z,
                )) for msg in safe_commands
            ), 'Safety gate emitted nonzero velocity without sensor/command input'
            topics = dict(node.get_topic_names_and_types())
            assert topics['/cmd_vel'] == ['geometry_msgs/msg/Twist']
            assert topics['/cmd_vel_safe'] == ['geometry_msgs/msg/Twist']

            # Synthetic local inputs check Jazzy's PointCloud2/NumPy path and
            # the original front/rear geometry pipeline without any device.
            grids = []
            front_clouds = []
            rear_clouds = []
            node.create_subscription(
                OccupancyGrid, '/local_world/occupancy_grid', grids.append, 10
            )
            for topic, received in [
                ('/vision/front/obstacle_cloud', front_clouds),
                ('/vision/rear/obstacle_cloud', rear_clouds),
            ]:
                node.create_subscription(
                    PointCloud2, topic, received.append, qos_profile_sensor_data
                )
            publishers = [node.create_publisher(
                PointCloud2, topic, qos_profile_sensor_data
            ) for topic in (
                '/intel_realsense_r200_depth/points',
                '/rear_realsense_depth/points',
            )]
            tf = TransformStamped()
            tf.header.stamp = node.get_clock().now().to_msg()
            tf.header.frame_id = 'odom'
            tf.child_frame_id = 'base_link'
            tf.transform.translation.x = 0.25
            tf.transform.rotation.w = 1.0
            broadcaster = StaticTransformBroadcaster(node)
            broadcaster.sendTransform(tf)
            deadline = time.monotonic() + 4.0
            next_publish = time.monotonic()
            while time.monotonic() < deadline:
                if time.monotonic() >= next_publish:
                    next_publish = time.monotonic() + 0.1
                    for publisher, x in zip(publishers, (1.0, -1.0)):
                        header = Header(
                            stamp=node.get_clock().now().to_msg(), frame_id='base_link'
                        )
                        publisher.publish(point_cloud2.create_cloud_xyz32(
                            header, [[x, 0.0, 0.5]] * 12
                        ))
                rclpy.spin_once(node, timeout_sec=0.1)
            assert any(c.width > 0 for c in front_clouds), 'No front obstacle output'
            assert any(c.width > 0 for c in rear_clouds), 'No rear obstacle output'
            assert any(any(v > 50 for v in g.data) for g in grids), 'No occupied grid'
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            assert grids and all(v <= 50 for v in grids[-1].data), 'TTL did not expire'
            print('Perception, local grid and safety nodes: running')
            print('Missing-input safety gate: zero velocity confirmed')
            print('Front/rear PointCloud2 -> occupied grid -> TTL expiry: confirmed')
            print('MIGRATION_SMOKE_PASSED')
        finally:
            node.destroy_node()
            rclpy.shutdown()
            if process.poll() is None:
                # ros2 launch forwards this once to its children. Signalling
                # the entire group here would interrupt their cleanup twice.
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
            print(f'Launch log: {logfile}', flush=True)
            if process.returncode not in (0, -signal.SIGINT):
                print(logfile.read_text()[-12000:])


if __name__ == '__main__':
    main()
