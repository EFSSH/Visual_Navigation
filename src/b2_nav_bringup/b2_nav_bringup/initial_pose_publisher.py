"""Publish the known HZ26 spawn pose after AMCL becomes available."""

import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rclpy.node import Node


class InitialPosePublisher(Node):
    def __init__(self):
        super().__init__('hz26_initial_pose_publisher')
        self.declare_parameter('initial_x', -7.0)
        self.declare_parameter('initial_y', -25.0)
        self.declare_parameter('initial_yaw', 1.5708)
        self.declare_parameter('timeout_sec', 30.0)

        self.initial_x = float(self.get_parameter('initial_x').value)
        self.initial_y = float(self.get_parameter('initial_y').value)
        self.initial_yaw = float(self.get_parameter('initial_yaw').value)
        self.timeout_sec = float(self.get_parameter('timeout_sec').value)
        self.started = time.monotonic()
        self.publish_count = 0
        self.confirmed = False
        self.recovery_attempted = False

        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10
        )
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._on_amcl_pose, 10
        )
        self.timer = self.create_timer(0.5, self._publish_when_ready)
        self.lifecycle_clients = {
            'map_server': (
                self.create_client(GetState, '/map_server/get_state'),
                self.create_client(ChangeState, '/map_server/change_state'),
            ),
            'amcl': (
                self.create_client(GetState, '/amcl/get_state'),
                self.create_client(ChangeState, '/amcl/change_state'),
            ),
        }

    def _on_amcl_pose(self, _message):
        if self.publish_count and not self.confirmed:
            self.confirmed = True
            self.get_logger().info('AMCL confirmed the automatic initial pose.')

    def _publish_when_ready(self):
        if self.confirmed or time.monotonic() - self.started > self.timeout_sec:
            return
        if self.publisher.get_subscription_count() == 0:
            return

        message = PoseWithCovarianceStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'map'
        message.pose.pose.position.x = self.initial_x
        message.pose.pose.position.y = self.initial_y
        message.pose.pose.orientation.z = math.sin(self.initial_yaw / 2.0)
        message.pose.pose.orientation.w = math.cos(self.initial_yaw / 2.0)
        message.pose.covariance[0] = 0.25
        message.pose.covariance[7] = 0.25
        message.pose.covariance[35] = math.radians(15.0) ** 2
        self.publisher.publish(message)
        self.publish_count += 1
        if self.publish_count == 1:
            self.get_logger().info(
                'Published automatic initial pose: '
                f'x={self.initial_x:.3f}, y={self.initial_y:.3f}, '
                f'yaw={self.initial_yaw:.4f}'
            )

    def _wait_for_future(self, future, timeout_sec=3.0):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return future.result() if future.done() else None

    def _state(self, node_name):
        get_state, _ = self.lifecycle_clients[node_name]
        if not get_state.wait_for_service(timeout_sec=1.0):
            return None
        response = self._wait_for_future(get_state.call_async(GetState.Request()))
        return response.current_state.id if response else None

    def _transition(self, node_name, transition_id):
        _, change_state = self.lifecycle_clients[node_name]
        if not change_state.wait_for_service(timeout_sec=1.0):
            return False
        request = ChangeState.Request()
        request.transition.id = transition_id
        response = self._wait_for_future(change_state.call_async(request))
        return bool(response and response.success)

    def recover_localization_lifecycle(self):
        """Recover if Nav2's localization manager timed out loading the map."""
        self.recovery_attempted = True
        states = {name: self._state(name) for name in self.lifecycle_clients}
        if all(value == State.PRIMARY_STATE_ACTIVE for value in states.values()):
            return

        self.get_logger().warning(
            f'Localization lifecycle did not activate in time: {states}; recovering.'
        )
        for name in ('map_server', 'amcl'):
            state = self._state(name)
            if state == State.PRIMARY_STATE_UNCONFIGURED:
                if not self._transition(name, Transition.TRANSITION_CONFIGURE):
                    self.get_logger().error(f'Failed to configure {name}.')
                    continue
                state = self._state(name)
            if state == State.PRIMARY_STATE_INACTIVE:
                if not self._transition(name, Transition.TRANSITION_ACTIVATE):
                    self.get_logger().error(f'Failed to activate {name}.')


def main():
    rclpy.init()
    node = InitialPosePublisher()
    deadline = time.monotonic() + node.timeout_sec
    while rclpy.ok() and not node.confirmed and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if not node.recovery_attempted and time.monotonic() - node.started > 5.0:
            node.recover_localization_lifecycle()
    if not node.confirmed:
        node.get_logger().warning('AMCL did not confirm the initial pose before timeout.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
