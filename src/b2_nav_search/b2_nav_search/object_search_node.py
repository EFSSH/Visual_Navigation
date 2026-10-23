"""
End-stage device search: scene retrieval, object matching, confirmation.

A JSON request on ``request_topic`` starts a search for one device from the
reference library. While active, the node subscribes to RGB, depth and
camera info (and drops them afterwards to save LAN bandwidth), matches the
device's reference images, back-projects matches with depth into the map,
and confirms identity only when several frames agree on the position and,
if configured, the device's marker is seen inside the match. It never moves
the robot; on confirmation it publishes a reachable stand-off observation
pose for the task manager.
"""

import json
import math

from b2_nav_search.geometry import back_project, choose_observation_pose, depth_at
from b2_nav_search.library import known_marker_ids, LibraryError, load_library
from b2_nav_search.matching import detect_markers, ReferenceMatcher
from b2_nav_search.session import SearchSession, STATUS_CONFIRMED
import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener


class ObjectSearchNode(Node):

    def __init__(self):
        super().__init__('object_search_node')

        self.declare_parameter('library_dir', '')
        self.declare_parameter('request_topic', '/search/request')
        self.declare_parameter('result_topic', '/search/result')
        self.declare_parameter('observation_goal_topic', '/search/observation_goal')
        self.declare_parameter('image_topic', '/intel_realsense_r200_depth/image_raw')
        self.declare_parameter(
            'depth_topic', '/intel_realsense_r200_depth/depth/image_raw'
        )
        self.declare_parameter(
            'camera_info_topic', '/intel_realsense_r200_depth/camera_info'
        )
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('default_timeout_sec', 20.0)
        self.declare_parameter('max_process_rate_hz', 2.0)
        self.declare_parameter('min_view_interval_sec', 0.5)
        self.declare_parameter('map_free_threshold', 50)
        self.declare_parameter('n_features', 1000)

        param = self.get_parameter
        self.map_frame = param('map_frame').value
        self.robot_frame = param('robot_frame').value
        self.default_timeout_sec = float(param('default_timeout_sec').value)
        self.min_process_period = 1.0 / float(param('max_process_rate_hz').value)
        self.min_view_interval_sec = float(param('min_view_interval_sec').value)
        self.map_free_threshold = int(param('map_free_threshold').value)
        self.image_topic = param('image_topic').value
        self.depth_topic = param('depth_topic').value
        self.camera_info_topic = param('camera_info_topic').value

        library_dir = param('library_dir').value
        try:
            self.devices = load_library(library_dir) if library_dir else {}
        except LibraryError as error:
            self.get_logger().error(f'reference library invalid: {error}')
            self.devices = {}
        self.known_markers = known_marker_ids(self.devices)
        self.matcher = ReferenceMatcher(n_features=int(param('n_features').value))
        for device in self.devices.values():
            for path in device.object_images + device.scene_images:
                image = cv2.imread(path)
                if image is None:
                    self.get_logger().error(f'cannot decode {path}')
                    continue
                try:
                    self.matcher.add_reference(path, image)
                except ValueError as error:
                    self.get_logger().error(str(error))

        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.result_pub = self.create_publisher(
            String, param('result_topic').value, latched
        )
        self.goal_pub = self.create_publisher(
            PoseStamped, param('observation_goal_topic').value, latched
        )
        self.create_subscription(
            String, param('request_topic').value, self.on_request, 10
        )
        self.create_subscription(
            OccupancyGrid, param('map_topic').value, self.on_map, latched
        )
        self.map = None
        self.camera_info = None
        self.active = None
        self.sensor_subs = []
        self.sync = None
        self.last_processed = None
        self.create_timer(0.5, self.on_timer)
        self.get_logger().info(
            f'object search ready: {len(self.devices)} devices from '
            f'{library_dir or "(no library)"}'
        )

    # -- requests ---------------------------------------------------------

    def on_request(self, msg):
        try:
            request = json.loads(msg.data)
            request_id = str(request['request_id'])
            device_id = str(request['device_id'])
            timeout = float(request.get('timeout_sec', self.default_timeout_sec))
        except (ValueError, KeyError, TypeError) as error:
            self.publish_result({'status': 'invalid_request', 'reason': str(error)})
            return
        if self.active is not None:
            self.finish('cancelled', 'superseded by a new request')
        profile = self.devices.get(device_id)
        if profile is None:
            self.publish_result({
                'request_id': request_id, 'device_id': device_id,
                'status': 'invalid_request', 'reason': 'unknown device_id',
            })
            return
        now = self.get_clock().now()
        self.active = {
            'request_id': request_id,
            'profile': profile,
            'session': SearchSession(
                profile, self.known_markers, self.min_view_interval_sec
            ),
            'deadline': now + Duration(seconds=timeout),
            'started': now,
        }
        self.start_sensors()
        self.get_logger().info(f'search {request_id}: {device_id}, {timeout:.0f}s')

    def on_timer(self):
        if self.active is None:
            return
        if self.get_clock().now() > self.active['deadline']:
            self.finish(self.active['session'].final_status(), 'timeout')

    # -- sensors ----------------------------------------------------------

    def start_sensors(self):
        if self.sensor_subs:
            return
        image_sub = Subscriber(
            self, Image, self.image_topic, qos_profile=qos_profile_sensor_data
        )
        depth_sub = Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = ApproximateTimeSynchronizer([image_sub, depth_sub], 5, 0.1)
        self.sync.registerCallback(self.on_frame)
        info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.on_camera_info,
            qos_profile_sensor_data,
        )
        self.sensor_subs = [image_sub, depth_sub, info_sub]

    def stop_sensors(self):
        for sub in self.sensor_subs:
            if isinstance(sub, Subscriber):
                self.destroy_subscription(sub.sub)
            else:
                self.destroy_subscription(sub)
        self.sensor_subs = []
        self.sync = None

    def on_camera_info(self, msg):
        self.camera_info = msg

    def on_map(self, msg):
        self.map = msg

    def on_frame(self, image_msg, depth_msg):
        if self.active is None or self.camera_info is None:
            return
        now = self.get_clock().now()
        if (self.last_processed is not None
                and (now - self.last_processed).nanoseconds * 1e-9
                < self.min_process_period):
            return
        self.last_processed = now
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except CvBridgeError as error:
            self.get_logger().warning(f'image conversion failed: {error}')
            return

        profile = self.active['profile']
        session = self.active['session']
        features = self.matcher.features(image)
        scene = self.matcher.best_match(profile.scene_images, features)
        match = self.matcher.best_match(profile.object_images, features)
        markers = detect_markers(image) if self.known_markers else {}
        map_xy = None
        if match is not None:
            map_xy = self.locate(match.centroid, depth, image_msg.header)
        stamp = Time.from_msg(image_msg.header.stamp).nanoseconds * 1e-9
        session.add_frame(
            stamp, match, map_xy, markers, scene.inliers if scene else 0
        )
        if session.confirmed():
            self.finish(STATUS_CONFIRMED, 'multi-view agreement')

    def locate(self, centroid, depth, header):
        u, v = centroid
        depth_m = depth_at(depth, u, v)
        if depth_m is None:
            return None
        k = self.camera_info.k
        x, y, z = back_project(u, v, depth_m, k[0], k[4], k[2], k[5])
        point = PointStamped()
        point.header = header
        point.point.x, point.point.y, point.point.z = x, y, z
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, header.frame_id, Time.from_msg(header.stamp),
                timeout=Duration(seconds=0.1),
            )
        except TransformException:
            return None
        mapped = do_transform_point(point, transform).point
        return mapped.x, mapped.y

    # -- results ----------------------------------------------------------

    def robot_xy(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame, Time(),
                timeout=Duration(seconds=0.1),
            ).transform.translation
        except TransformException:
            return None
        return tf.x, tf.y

    def map_is_free(self, x, y):
        grid = self.map
        if grid is None:
            return True
        info = grid.info
        col = int(math.floor((x - info.origin.position.x) / info.resolution))
        row = int(math.floor((y - info.origin.position.y) / info.resolution))
        if col < 0 or row < 0 or col >= info.width or row >= info.height:
            return False
        cost = grid.data[row * info.width + col]
        return 0 <= cost < self.map_free_threshold

    def finish(self, status, reason):
        active = self.active
        self.active = None
        self.stop_sensors()
        session = active['session']
        profile = active['profile']
        summary = session.summary()
        result = {
            'request_id': active['request_id'],
            'device_id': profile.device_id,
            'status': status,
            'reason': reason,
            'elapsed_sec': (
                self.get_clock().now() - active['started']
            ).nanoseconds * 1e-9,
            'evidence': summary,
        }
        if status == STATUS_CONFIRMED:
            robot = self.robot_xy()
            pose = None
            if robot is not None:
                pose = choose_observation_pose(
                    summary['device_xy'], robot, profile.observation.distance_m,
                    profile.observation.facing_yaw, self.map_is_free,
                )
            if pose is None:
                result['status'] = 'observation_unreachable'
            else:
                result['observation_goal'] = {
                    'x': pose[0], 'y': pose[1], 'yaw': pose[2],
                    'frame': self.map_frame,
                    'yaw_tolerance_rad': profile.observation.yaw_tolerance_rad,
                }
                self.goal_pub.publish(self.pose_msg(*pose))
        self.publish_result(result)
        self.get_logger().info(
            f'search {active["request_id"]}: {result["status"]} ({reason})'
        )

    def pose_msg(self, x, y, yaw):
        msg = PoseStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        return msg

    def publish_result(self, result):
        result = dict(result, schema='b2_nav/search_result/v1')
        self.result_pub.publish(String(data=json.dumps(result)))


def main(args=None):
    rclpy.init(args=args)
    node = ObjectSearchNode()
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
