import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class SafetyGateNode(Node):
    def __init__(self):
        super().__init__('safety_gate_node')

        self.declare_parameter('max_linear_x', 0.25)
        self.declare_parameter('max_linear_y', 0.20)
        self.declare_parameter('max_angular_z', 0.60)
        self.declare_parameter('cmd_timeout_sec', 0.5)
        self.declare_parameter('obstacle_stop_distance', 0.5)
        self.declare_parameter('obstacle_slow_distance', 1.0)
        self.declare_parameter('obstacle_timeout_sec', 1.0)
        self.declare_parameter('publish_rate_hz', 20.0)

        self.max_linear_x = self.get_parameter('max_linear_x').value
        self.max_linear_y = self.get_parameter('max_linear_y').value
        self.max_angular_z = self.get_parameter('max_angular_z').value
        self.cmd_timeout_sec = self.get_parameter('cmd_timeout_sec').value
        self.obstacle_stop_distance = self.get_parameter('obstacle_stop_distance').value
        self.obstacle_slow_distance = self.get_parameter('obstacle_slow_distance').value
        self.obstacle_timeout_sec = self.get_parameter('obstacle_timeout_sec').value
        self.publish_rate_hz = self.get_parameter('publish_rate_hz').value

        self.last_raw_cmd = Twist()
        self.last_raw_cmd_time = self.get_clock().now()
        self.last_obstacle_distance = None
        self.last_obstacle_time = self.get_clock().now()

        self.safe_cmd_pub = self.create_publisher(Twist, '/b2/cmd_vel', 10)
        self.raw_cmd_sub = self.create_subscription(
            Twist,
            '/nav/cmd_vel_raw',
            self.on_raw_cmd,
            10,
        )
        self.obstacle_sub = self.create_subscription(
            Float32,
            '/perception/nearest_obstacle_distance',
            self.on_obstacle_distance,
            10,
        )
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.on_timer)
        self.get_logger().info(
            'safety_gate_node started: /nav/cmd_vel_raw + obstacle distance -> /b2/cmd_vel'
        )

    def on_raw_cmd(self, msg):
        self.last_raw_cmd = msg
        self.last_raw_cmd_time = self.get_clock().now()

    def on_obstacle_distance(self, msg):
        self.last_obstacle_distance = msg.data
        self.last_obstacle_time = self.get_clock().now()

    def get_obstacle_speed_scale(self, now):
        if self.last_obstacle_distance is None:
            return 1.0

        obstacle_age = (now - self.last_obstacle_time).nanoseconds * 1e-9
        if obstacle_age > self.obstacle_timeout_sec:
            return 1.0

        distance = self.last_obstacle_distance
        if distance <= self.obstacle_stop_distance:
            return 0.0
        if distance >= self.obstacle_slow_distance:
            return 1.0

        slow_range = self.obstacle_slow_distance - self.obstacle_stop_distance
        return (distance - self.obstacle_stop_distance) / slow_range

    def on_timer(self):
        now = self.get_clock().now()
        cmd_age = (now - self.last_raw_cmd_time).nanoseconds * 1e-9

        safe_cmd = Twist()
        if cmd_age <= self.cmd_timeout_sec:
            safe_cmd.linear.x = clamp(
                self.last_raw_cmd.linear.x,
                -self.max_linear_x,
                self.max_linear_x,
            )
            safe_cmd.linear.y = clamp(
                self.last_raw_cmd.linear.y,
                -self.max_linear_y,
                self.max_linear_y,
            )
            safe_cmd.angular.z = clamp(
                self.last_raw_cmd.angular.z,
                -self.max_angular_z,
                self.max_angular_z,
            )

            obstacle_speed_scale = self.get_obstacle_speed_scale(now)
            safe_cmd.linear.x *= obstacle_speed_scale
            safe_cmd.linear.y *= obstacle_speed_scale
            safe_cmd.angular.z *= obstacle_speed_scale

        self.safe_cmd_pub.publish(safe_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyGateNode()
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
