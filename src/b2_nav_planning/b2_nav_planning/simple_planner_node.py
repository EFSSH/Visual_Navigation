import math

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def point_to_segment_distance(point, start, end):
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    segment_length_squared = segment_x * segment_x + segment_y * segment_y
    if segment_length_squared == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])

    projection = (
        (point[0] - start[0]) * segment_x
        + (point[1] - start[1]) * segment_y
    ) / segment_length_squared
    projection = clamp(projection, 0.0, 1.0)
    nearest_x = start[0] + projection * segment_x
    nearest_y = start[1] + projection * segment_y
    return math.hypot(point[0] - nearest_x, point[1] - nearest_y)


class SimplePlannerNode(Node):
    def __init__(self):
        super().__init__('simple_planner_node')

        self.declare_parameter('goal_x', 5.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_tolerance', 0.1)
        self.declare_parameter('waypoint_tolerance', 0.15)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('max_lateral_speed', 0.2)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('yaw_gain', 1.5)
        self.declare_parameter('obstacle_radius', 0.35)
        self.declare_parameter('avoidance_clearance', 0.8)
        self.declare_parameter('obstacle_timeout_sec', 1.0)
        self.declare_parameter('frame_id', 'odom')

        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_lateral_speed = self.get_parameter('max_lateral_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.yaw_gain = self.get_parameter('yaw_gain').value
        self.obstacle_radius = self.get_parameter('obstacle_radius').value
        self.avoidance_clearance = self.get_parameter('avoidance_clearance').value
        self.obstacle_timeout_sec = self.get_parameter('obstacle_timeout_sec').value
        self.frame_id = self.get_parameter('frame_id').value

        self.obstacle_position = None
        self.last_obstacle_time = None
        self.avoidance_state = 'direct'
        self.waypoint = None

        self.cmd_pub = self.create_publisher(Twist, '/nav/cmd_vel_raw', 10)
        self.path_pub = self.create_publisher(Path, '/planning/path', 10)
        self.markers_pub = self.create_publisher(
            MarkerArray,
            '/planning/markers',
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/robot/odom',
            self.on_odom,
            10,
        )
        self.obstacle_position_sub = self.create_subscription(
            PointStamped,
            '/perception/nearest_obstacle_position',
            self.on_obstacle_position,
            10,
        )
        self.obstacle_radius_sub = self.create_subscription(
            Float32,
            '/perception/nearest_obstacle_radius',
            self.on_obstacle_radius,
            10,
        )
        self.get_logger().info(
            'avoidance planner started: '
            f'goal=({self.goal_x:.2f}, {self.goal_y:.2f})'
        )

    def on_obstacle_position(self, msg):
        self.obstacle_position = (msg.point.x, msg.point.y)
        self.last_obstacle_time = self.get_clock().now()

    def on_obstacle_radius(self, msg):
        self.obstacle_radius = max(0.0, float(msg.data))

    def obstacle_is_current(self):
        if self.obstacle_position is None or self.last_obstacle_time is None:
            return False
        age = (self.get_clock().now() - self.last_obstacle_time).nanoseconds * 1e-9
        return age <= self.obstacle_timeout_sec

    def route_is_blocked(self, start, end):
        if not self.obstacle_is_current():
            return False
        required_clearance = self.obstacle_radius + self.avoidance_clearance
        return point_to_segment_distance(
            self.obstacle_position,
            start,
            end,
        ) < required_clearance

    def choose_waypoint(self, robot_position):
        goal = (self.goal_x, self.goal_y)
        route_x = goal[0] - robot_position[0]
        route_y = goal[1] - robot_position[1]
        route_length = math.hypot(route_x, route_y)
        if route_length == 0.0:
            return goal

        perpendicular_x = -route_y / route_length
        perpendicular_y = route_x / route_length
        required_clearance = self.obstacle_radius + self.avoidance_clearance
        candidates = []

        for side in (1.0, -1.0):
            for step in range(1, 21):
                offset = required_clearance * (1.0 + 0.1 * step)
                candidate = (
                    self.obstacle_position[0] + side * perpendicular_x * offset,
                    self.obstacle_position[1] + side * perpendicular_y * offset,
                )
                first_clearance = point_to_segment_distance(
                    self.obstacle_position,
                    robot_position,
                    candidate,
                )
                second_clearance = point_to_segment_distance(
                    self.obstacle_position,
                    candidate,
                    goal,
                )
                if min(first_clearance, second_clearance) >= required_clearance:
                    route_cost = (
                        math.dist(robot_position, candidate)
                        + math.dist(candidate, goal)
                    )
                    candidates.append((route_cost, -side, candidate))
                    break

        if not candidates:
            return (
                self.obstacle_position[0] + perpendicular_x * required_clearance * 3.0,
                self.obstacle_position[1] + perpendicular_y * required_clearance * 3.0,
            )

        return min(candidates)[2]

    def update_route(self, robot_position):
        goal = (self.goal_x, self.goal_y)

        if self.avoidance_state == 'direct' and self.route_is_blocked(
            robot_position,
            goal,
        ):
            self.waypoint = self.choose_waypoint(robot_position)
            self.avoidance_state = 'to_waypoint'
            self.get_logger().info(
                'obstacle blocks direct path; detouring via '
                f'({self.waypoint[0]:.2f}, {self.waypoint[1]:.2f})'
            )

        if self.avoidance_state == 'to_waypoint':
            distance_to_waypoint = math.dist(robot_position, self.waypoint)
            if distance_to_waypoint <= self.waypoint_tolerance:
                self.avoidance_state = 'to_goal'
                self.get_logger().info('avoidance waypoint reached; returning to goal')

        if self.avoidance_state == 'to_waypoint':
            return self.waypoint
        return goal

    def on_odom(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        robot_position = (x, y)
        target = self.update_route(robot_position)

        self.publish_path_and_markers(robot_position)

        dx = target[0] - x
        dy = target[1] - y
        distance = math.hypot(dx, dy)

        cmd = Twist()
        target_tolerance = (
            self.waypoint_tolerance
            if self.avoidance_state == 'to_waypoint'
            else self.goal_tolerance
        )
        if distance > target_tolerance:
            target_yaw = math.atan2(dy, dx)
            yaw_error = math.atan2(
                math.sin(target_yaw - yaw),
                math.cos(target_yaw - yaw),
            )

            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            body_x_error = dx * cos_yaw + dy * sin_yaw
            body_y_error = -dx * sin_yaw + dy * cos_yaw

            cmd.linear.x = clamp(
                body_x_error,
                -self.max_linear_speed,
                self.max_linear_speed,
            )
            cmd.linear.y = clamp(
                body_y_error,
                -self.max_lateral_speed,
                self.max_lateral_speed,
            )
            cmd.angular.z = clamp(
                self.yaw_gain * yaw_error,
                -self.max_angular_speed,
                self.max_angular_speed,
            )

        self.cmd_pub.publish(cmd)

    def publish_path_and_markers(self, robot_position):
        stamp = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.frame_id

        points = [robot_position]
        if self.avoidance_state == 'to_waypoint' and self.waypoint is not None:
            points.append(self.waypoint)
        points.append((self.goal_x, self.goal_y))

        for point_x, point_y in points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(point_x)
            pose.pose.position.y = float(point_y)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

        markers = MarkerArray()
        markers.markers.append(
            self.create_marker(stamp, 0, self.goal_x, self.goal_y, 0.1, 0.4, 1.0)
        )
        if self.avoidance_state == 'to_waypoint' and self.waypoint is not None:
            markers.markers.append(
                self.create_marker(
                    stamp,
                    1,
                    self.waypoint[0],
                    self.waypoint[1],
                    1.0,
                    0.75,
                    0.0,
                )
            )
        else:
            delete_marker = Marker()
            delete_marker.header.stamp = stamp
            delete_marker.header.frame_id = self.frame_id
            delete_marker.ns = 'planning_targets'
            delete_marker.id = 1
            delete_marker.action = Marker.DELETE
            markers.markers.append(delete_marker)
        self.markers_pub.publish(markers)

    def create_marker(self, stamp, marker_id, x, y, red, green, blue):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.ns = 'planning_targets'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 1.0
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = SimplePlannerNode()
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
