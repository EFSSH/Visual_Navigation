"""
Device-side command watchdog: stop the base when commands stop arriving.

The navigation host's safety gate already publishes zero velocity when its
own input times out, but it cannot do so if the navigation host, its VM or
the network goes away. This node runs next to the base (simulation or B2),
forwards /cmd_vel_safe to the base command topic, and publishes zero
velocity once commands are older than ``cmd_timeout_sec`` or an emergency
stop is latched.
"""

import json

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String

STATE_FORWARDING = 'forwarding'
STATE_TIMEOUT = 'timeout'
STATE_ESTOP = 'estop'
STATE_WAITING = 'waiting'


def select_command(cmd, cmd_age_sec, cmd_timeout_sec, estop_active):
    """Return (state, twist) to publish for the current inputs."""
    if estop_active:
        return STATE_ESTOP, Twist()
    if cmd is None or cmd_age_sec is None:
        return STATE_WAITING, Twist()
    if cmd_age_sec > cmd_timeout_sec:
        return STATE_TIMEOUT, Twist()
    return STATE_FORWARDING, cmd


class CmdWatchdogNode(Node):

    def __init__(self):
        super().__init__('cmd_watchdog_node')

        self.declare_parameter('input_cmd_topic', '/cmd_vel_safe')
        self.declare_parameter('output_cmd_topic', '/cmd_vel_device')
        self.declare_parameter('estop_topic', '/robot/estop')
        self.declare_parameter('state_topic', '/robot/watchdog_state')
        self.declare_parameter('cmd_timeout_sec', 0.5)
        self.declare_parameter('publish_rate_hz', 30.0)

        input_cmd_topic = self.get_parameter('input_cmd_topic').value
        output_cmd_topic = self.get_parameter('output_cmd_topic').value
        estop_topic = self.get_parameter('estop_topic').value
        state_topic = self.get_parameter('state_topic').value
        self.cmd_timeout_sec = float(self.get_parameter('cmd_timeout_sec').value)
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        if self.cmd_timeout_sec <= 0.0 or publish_rate_hz <= 0.0:
            raise ValueError('cmd_timeout_sec and publish_rate_hz must be positive')

        self.last_cmd = None
        self.last_cmd_time = None
        self.estop_active = False
        self.state = None
        self.timeout_count = 0

        self.cmd_pub = self.create_publisher(Twist, output_cmd_topic, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, state_topic, latched)
        self.cmd_sub = self.create_subscription(
            Twist, input_cmd_topic, self.on_cmd, 10
        )
        self.estop_sub = self.create_subscription(
            Bool, estop_topic, self.on_estop, latched
        )
        self.timer = self.create_timer(1.0 / publish_rate_hz, self.on_timer)
        self.get_logger().info(
            f'command watchdog: {input_cmd_topic} -> {output_cmd_topic}, '
            f'timeout={self.cmd_timeout_sec:.2f}s, estop={estop_topic}'
        )

    def on_cmd(self, msg):
        self.last_cmd = msg
        self.last_cmd_time = self.get_clock().now()

    def on_estop(self, msg):
        # True engages the stop; only an explicit False releases it.
        self.estop_active = bool(msg.data)

    def on_timer(self):
        now = self.get_clock().now()
        cmd_age = None
        if self.last_cmd_time is not None:
            cmd_age = (now - self.last_cmd_time).nanoseconds * 1e-9
        state, cmd = select_command(
            self.last_cmd, cmd_age, self.cmd_timeout_sec, self.estop_active
        )
        self.cmd_pub.publish(cmd)
        if state != self.state:
            if state == STATE_TIMEOUT:
                self.timeout_count += 1
                self.get_logger().warning(
                    f'no command for {cmd_age:.2f}s; publishing zero velocity'
                )
            self.state = state
            self.state_pub.publish(String(data=json.dumps({
                'schema': 'b2_nav/watchdog_state/v1',
                'state': state,
                'cmd_age_sec': cmd_age,
                'timeout_count': self.timeout_count,
                'stamp_sec': now.nanoseconds * 1e-9,
            })))


def main(args=None):
    rclpy.init(args=args)
    node = CmdWatchdogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
