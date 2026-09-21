import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32


def minimum_valid_range(ranges, range_min, range_max):
    """Return the nearest finite in-range reading, or None."""
    valid = (
        value for value in ranges
        if math.isfinite(value) and range_min <= value <= range_max
    )
    return min(valid, default=None)


class LidarSafetyNode(Node):
    """Convert the lidar scan into a small, independent safety signal."""

    def __init__(self):
        super().__init__('lidar_safety_node')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('distance_topic', '/safety/lidar_min_distance')
        self.declare_parameter('stop_topic', '/safety/lidar_stop')
        self.declare_parameter('stop_distance', 0.45)

        scan_topic = self.get_parameter('scan_topic').value
        distance_topic = self.get_parameter('distance_topic').value
        stop_topic = self.get_parameter('stop_topic').value
        self.stop_distance = float(self.get_parameter('stop_distance').value)

        self.distance_pub = self.create_publisher(Float32, distance_topic, 10)
        self.stop_pub = self.create_publisher(Bool, stop_topic, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.on_scan, qos_profile_sensor_data
        )
        self.get_logger().info(
            f'Lidar safety monitor: {scan_topic} -> {distance_topic}'
        )

    def on_scan(self, msg):
        distance = minimum_valid_range(msg.ranges, msg.range_min, msg.range_max)
        if distance is None:
            self.get_logger().warning(
                'No valid lidar ranges; safety gate will stop on timeout.',
                throttle_duration_sec=2.0,
            )
            return

        distance_msg = Float32()
        distance_msg.data = float(distance)
        self.distance_pub.publish(distance_msg)

        stop_msg = Bool()
        stop_msg.data = distance <= self.stop_distance
        self.stop_pub.publish(stop_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarSafetyNode()
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
