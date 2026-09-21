from dataclasses import dataclass
import math

from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener


def stamp_to_seconds(stamp):
    """Convert a ROS time message to floating-point seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def transform_xyz(xyz, transform_stamped):
    """Apply a TransformStamped to an N-by-3 XYZ array."""
    transform = transform_stamped.transform
    rotation = transform.rotation
    quaternion = np.array(
        [rotation.x, rotation.y, rotation.z, rotation.w],
        dtype=np.float64,
    )
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        raise ValueError('TF rotation quaternion has zero length')

    x, y, z, w = quaternion / norm
    rotation_matrix = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    translation = np.array(
        [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ],
        dtype=np.float64,
    )
    return xyz @ rotation_matrix.T + translation


def merge_fresh_clouds(cloud_cache, reference_stamp, max_age_sec):
    """Retain the first-stage point-cloud fusion helper."""
    fresh = [
        xyz for stamp, xyz in cloud_cache.values()
        if 0.0 <= reference_stamp - stamp <= max_age_sec
    ]
    if not fresh:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(fresh).astype(np.float32, copy=False)


def bresenham_cells(start, end):
    """Return all integer cells on a 2D line."""
    x0, y0 = start
    x1, y1 = end
    delta_x = abs(x1 - x0)
    step_x = 1 if x0 < x1 else -1
    delta_y = -abs(y1 - y0)
    step_y = 1 if y0 < y1 else -1
    error = delta_x + delta_y
    cells = []

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return cells
        doubled_error = 2 * error
        if doubled_error >= delta_y:
            error += delta_y
            x0 += step_x
        if doubled_error <= delta_x:
            error += delta_x
            y0 += step_y


@dataclass
class CellState:
    """Occupancy, height, source, and temporal state for one grid cell."""

    log_odds: float = 0.0
    last_observed: float = 0.0
    last_hit: float = -math.inf
    min_z: float = math.inf
    max_z: float = -math.inf
    source_mask: int = 0
    hit_count: int = 0


class RollingGridModel:
    """Odom-aligned rolling 2.5D occupancy grid."""

    def __init__(
        self,
        resolution=0.10,
        width=8.0,
        height=8.0,
        hit_log_odds=0.85,
        miss_log_odds=0.45,
        occupied_threshold=0.50,
        obstacle_ttl_sec=1.0,
        free_space_ttl_sec=2.0,
    ):
        if resolution <= 0.0:
            raise ValueError('resolution must be positive')
        if width <= 0.0 or height <= 0.0:
            raise ValueError('grid dimensions must be positive')
        if obstacle_ttl_sec <= 0.0 or free_space_ttl_sec <= 0.0:
            raise ValueError('TTL values must be positive')
        self.resolution = float(resolution)
        self.width = float(width)
        self.height = float(height)
        self.hit_log_odds = float(hit_log_odds)
        self.miss_log_odds = float(miss_log_odds)
        self.occupied_threshold = float(occupied_threshold)
        self.obstacle_ttl_sec = float(obstacle_ttl_sec)
        self.free_space_ttl_sec = float(free_space_ttl_sec)
        self.min_log_odds = -2.0
        self.max_log_odds = 3.0
        self.cells = {}

    def expire_stale(self, now):
        """Remove stale evidence and return occupied cells that expired."""
        remove_keys = []
        expired_obstacles = []
        for key, state in self.cells.items():
            time_reset = now < state.last_observed
            occupied = state.log_odds >= self.occupied_threshold
            stale_obstacle = occupied and (
                now - state.last_hit > self.obstacle_ttl_sec
            )
            stale_free = not occupied and (
                now - state.last_observed > self.free_space_ttl_sec
            )
            if time_reset or stale_obstacle or stale_free:
                remove_keys.append(key)
                if stale_obstacle:
                    expired_obstacles.append((key, state))
        for key in remove_keys:
            del self.cells[key]
        return len(remove_keys), expired_obstacles

    def prune_outside(self, center_x, center_y):
        """Remove cells outside the rolling window."""
        half_width = self.width * 0.5
        half_height = self.height * 0.5
        remove_keys = []
        for key in self.cells:
            cell_x, cell_y = self.cell_to_world(key)
            if (
                abs(cell_x - center_x) > half_width
                or abs(cell_y - center_y) > half_height
            ):
                remove_keys.append(key)
        for key in remove_keys:
            del self.cells[key]
        return len(remove_keys)

    def world_to_cell(self, x, y):
        """Convert odom coordinates to a stable integer grid key."""
        return (
            math.floor(float(x) / self.resolution),
            math.floor(float(y) / self.resolution),
        )

    def cell_to_world(self, key):
        """Return the odom position at the center of a grid cell."""
        return (
            (key[0] + 0.5) * self.resolution,
            (key[1] + 0.5) * self.resolution,
        )

    def mark_points(self, xyz, stamp, source_mask):
        """Integrate one obstacle hit per occupied cell and input frame."""
        grouped = {}
        for point in xyz:
            key = self.world_to_cell(point[0], point[1])
            z_value = float(point[2])
            if key not in grouped:
                grouped[key] = [z_value, z_value]
            else:
                grouped[key][0] = min(grouped[key][0], z_value)
                grouped[key][1] = max(grouped[key][1], z_value)

        for key, (min_z, max_z) in grouped.items():
            state = self.cells.setdefault(key, CellState())
            state.log_odds = min(
                self.max_log_odds,
                state.log_odds + self.hit_log_odds,
            )
            state.last_observed = float(stamp)
            state.last_hit = float(stamp)
            state.min_z = min(state.min_z, min_z)
            state.max_z = max(state.max_z, max_z)
            state.source_mask |= int(source_mask)
            state.hit_count += 1

    def clear_rays(self, origin, endpoints, stamp, max_range):
        """Integrate free space along each ray, excluding its endpoint."""
        if endpoints.size == 0:
            return
        origin_xy = np.asarray(origin[:2], dtype=np.float64)
        endpoint_xy = np.asarray(endpoints[:, :2], dtype=np.float64)
        vectors = endpoint_xy - origin_xy
        distances = np.linalg.norm(vectors, axis=1)
        valid = distances > self.resolution
        vectors = vectors[valid]
        distances = distances[valid]
        if distances.size == 0:
            return

        scales = np.minimum(1.0, float(max_range) / distances)
        clipped = origin_xy + vectors * scales[:, None]
        endpoint_cells = {
            self.world_to_cell(point[0], point[1]) for point in clipped
        }
        start_cell = self.world_to_cell(origin_xy[0], origin_xy[1])

        for endpoint_cell in endpoint_cells:
            ray = bresenham_cells(start_cell, endpoint_cell)
            for key in ray[1:-1]:
                state = self.cells.setdefault(key, CellState())
                state.log_odds = max(
                    self.min_log_odds,
                    state.log_odds - self.miss_log_odds,
                )
                state.last_observed = float(stamp)
                if state.log_odds < self.occupied_threshold:
                    state.min_z = math.inf
                    state.max_z = -math.inf
                    state.source_mask = 0
                    state.hit_count = 0

    def expire_and_prune(self, now, center_x, center_y):
        """Remove stale cells and cells outside the rolling window."""
        expired_count, _ = self.expire_stale(now)
        return expired_count + self.prune_outside(center_x, center_y)

    def occupied_items(self):
        """Return all cells whose occupancy exceeds the threshold."""
        return [
            (key, state)
            for key, state in self.cells.items()
            if state.log_odds >= self.occupied_threshold
        ]


class LocalWorldModelNode(Node):
    """Fuse four RGB-D streams into a rolling 2.5D local world model."""

    SOURCE_MASKS = {'front': 1, 'rear': 2}

    def __init__(self):
        super().__init__('local_world_model_node')
        self._declare_parameters()
        self._load_parameters()
        self.model = RollingGridModel(
            resolution=self.resolution,
            width=self.width,
            height=self.height,
            hit_log_odds=self.hit_log_odds,
            miss_log_odds=self.miss_log_odds,
            occupied_threshold=self.occupied_threshold,
            obstacle_ttl_sec=self.obstacle_ttl_sec,
            free_space_ttl_sec=self.free_space_ttl_sec,
        )

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        # Ray integration is CPU-heavy. Keep TF on its own executor thread so
        # point-cloud callbacks cannot starve the odom transform buffer.
        self.tf_listener = TransformListener(
            self.tf_buffer,
            None,
            spin_thread=True,
        )
        self.received_count = 0
        self.failed_count = 0
        self.tf_fallback_count = 0
        self.published_count = 0
        self.last_source_stamp = {}
        self.pending_decay_clears = []

        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            qos_profile_sensor_data,
        )
        self.grid_pub = self.create_publisher(
            OccupancyGrid,
            self.grid_topic,
            10,
        )
        self.obstacle_pubs = {
            'front': self.create_publisher(
                PointCloud2,
                self.front_output_topic,
                qos_profile_sensor_data,
            ),
            'rear': self.create_publisher(
                PointCloud2,
                self.rear_output_topic,
                qos_profile_sensor_data,
            ),
        }
        self.clearing_pubs = {
            'front': self.create_publisher(
                PointCloud2,
                self.front_clearing_output_topic,
                qos_profile_sensor_data,
            ),
            'rear': self.create_publisher(
                PointCloud2,
                self.rear_clearing_output_topic,
                qos_profile_sensor_data,
            ),
        }
        self._input_subscriptions = []
        self._subscribe_inputs()
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self.publish_model,
        )
        # Decay must continue even when no sensor callback or map publication
        # occurs. This timer also emits clearing rays for Nav2.
        self.decay_timer = self.create_timer(
            1.0 / self.decay_rate_hz,
            self.expire_obstacles,
        )

        self.get_logger().info(
            'Rolling 2.5D local world model started: '
            f'frame={self.target_frame}, '
            f'size={self.width}x{self.height} m, '
            f'resolution={self.resolution} m, '
            f'obstacle_ttl={self.obstacle_ttl_sec} s'
        )

    def _declare_parameters(self):
        defaults = {
            'front_topic': '/vision/front/obstacle_cloud',
            'rear_topic': '/vision/rear/obstacle_cloud',
            'front_clearing_topic': '/vision/front/clearing_cloud',
            'rear_clearing_topic': '/vision/rear/clearing_cloud',
            'output_topic': '/local_world/occupied_cloud',
            'grid_topic': '/local_world/occupancy_grid',
            'front_output_topic': '/local_world/front/obstacle_cloud',
            'rear_output_topic': '/local_world/rear/obstacle_cloud',
            'front_clearing_output_topic': (
                '/local_world/front/clearing_cloud'
            ),
            'rear_clearing_output_topic': (
                '/local_world/rear/clearing_cloud'
            ),
            'target_frame': 'odom',
            'base_frame': 'base_link',
            'resolution': 0.10,
            'width': 8.0,
            'height': 8.0,
            'publish_rate_hz': 5.0,
            'decay_rate_hz': 10.0,
            'obstacle_ttl_sec': 1.0,
            'free_space_ttl_sec': 5.0,
            'decay_clear_hold_sec': 1.0,
            'decay_clear_extension': 0.15,
            'hit_log_odds': 0.85,
            'miss_log_odds': 0.45,
            'occupied_threshold': 0.50,
            'clearing_ray_step': 1,
            'raytrace_max_range': 3.2,
            'tf_fallback_max_age_sec': 0.5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self):
        string_names = (
            'front_topic',
            'rear_topic',
            'front_clearing_topic',
            'rear_clearing_topic',
            'output_topic',
            'grid_topic',
            'front_output_topic',
            'rear_output_topic',
            'front_clearing_output_topic',
            'rear_clearing_output_topic',
            'target_frame',
            'base_frame',
        )
        float_names = (
            'resolution',
            'width',
            'height',
            'publish_rate_hz',
            'decay_rate_hz',
            'obstacle_ttl_sec',
            'free_space_ttl_sec',
            'decay_clear_hold_sec',
            'decay_clear_extension',
            'hit_log_odds',
            'miss_log_odds',
            'occupied_threshold',
            'raytrace_max_range',
            'tf_fallback_max_age_sec',
        )
        for name in string_names:
            setattr(self, name, str(self.get_parameter(name).value))
        for name in float_names:
            setattr(self, name, float(self.get_parameter(name).value))
        self.clearing_ray_step = int(
            self.get_parameter('clearing_ray_step').value
        )
        if self.publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be positive')
        if self.decay_rate_hz <= 0.0:
            raise ValueError('decay_rate_hz must be positive')
        if self.decay_clear_hold_sec <= 0.0:
            raise ValueError('decay_clear_hold_sec must be positive')
        if self.decay_clear_extension < 0.0:
            raise ValueError('decay_clear_extension cannot be negative')
        if self.clearing_ray_step < 1:
            raise ValueError('clearing_ray_step must be at least 1')
        if self.tf_fallback_max_age_sec < 0.0:
            raise ValueError('tf_fallback_max_age_sec cannot be negative')

    def _subscribe_inputs(self):
        obstacle_topics = {
            'front': self.front_topic,
            'rear': self.rear_topic,
        }
        clearing_topics = {
            'front_clear': self.front_clearing_topic,
            'rear_clear': self.rear_clearing_topic,
        }
        for source, topic in obstacle_topics.items():
            subscription = self.create_subscription(
                PointCloud2,
                topic,
                lambda msg, name=source: self.on_obstacle_cloud(name, msg),
                qos_profile_sensor_data,
            )
            self._input_subscriptions.append(subscription)
        for source, topic in clearing_topics.items():
            subscription = self.create_subscription(
                PointCloud2,
                topic,
                lambda msg, name=source: self.on_clearing_cloud(name, msg),
                qos_profile_sensor_data,
            )
            self._input_subscriptions.append(subscription)

    @staticmethod
    def _read_xyz(msg, step=1):
        points = point_cloud2.read_points(
            msg,
            field_names=['x', 'y', 'z'],
            skip_nans=False,
        )[::step]
        xyz = np.column_stack(
            (points['x'], points['y'], points['z'])
        ).astype(np.float64, copy=False)
        return xyz[np.isfinite(xyz).all(axis=1)]

    def _lookup_transform(self, source_frame, stamp):
        if source_frame == self.target_frame:
            return None
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time.from_msg(stamp),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exact_error:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.1),
                )
            except TransformException:
                raise exact_error
            requested = stamp_to_seconds(stamp)
            available = stamp_to_seconds(transform.header.stamp)
            if abs(requested - available) > self.tf_fallback_max_age_sec:
                raise exact_error
            self.tf_fallback_count += 1
            if (
                self.tf_fallback_count == 1
                or self.tf_fallback_count % 50 == 0
            ):
                self.get_logger().warning(
                    'Using latest TF because the cloud timestamp is ahead '
                    f'of the TF buffer: count={self.tf_fallback_count}'
                )
            return transform

    def on_obstacle_cloud(self, source, msg):
        """Integrate obstacle hits from one camera."""
        self.received_count += 1
        if not msg.header.frame_id:
            self._log_failure(source, 'empty frame_id')
            return
        try:
            xyz = self._read_xyz(msg)
            transform = self._lookup_transform(
                msg.header.frame_id,
                msg.header.stamp,
            )
            if transform is not None:
                xyz = transform_xyz(xyz, transform)
        except (AssertionError, TransformException, ValueError) as error:
            self._log_failure(source, error)
            return
        stamp = stamp_to_seconds(msg.header.stamp)
        self.last_source_stamp[source] = stamp
        self.model.mark_points(xyz, stamp, self.SOURCE_MASKS[source])

    def on_clearing_cloud(self, source, msg):
        """Integrate free-space rays from one camera."""
        self.received_count += 1
        if not msg.header.frame_id:
            self._log_failure(source, 'empty frame_id')
            return
        try:
            xyz = self._read_xyz(msg, self.clearing_ray_step)
            transform = self._lookup_transform(
                msg.header.frame_id,
                msg.header.stamp,
            )
            if transform is None:
                origin = np.zeros(3, dtype=np.float64)
                endpoints = xyz
            else:
                translation = transform.transform.translation
                origin = np.array(
                    [translation.x, translation.y, translation.z],
                    dtype=np.float64,
                )
                endpoints = transform_xyz(xyz, transform)
        except (AssertionError, TransformException, ValueError) as error:
            self._log_failure(source, error)
            return
        stamp = stamp_to_seconds(msg.header.stamp)
        self.last_source_stamp[source] = stamp
        self.model.clear_rays(
            origin,
            endpoints,
            stamp,
            self.raytrace_max_range,
        )
        camera = 'front' if source.startswith('front') else 'rear'
        # Preserve the optical frame so Nav2 raytraces from the real camera.
        self.clearing_pubs[camera].publish(msg)

    def _robot_position(self):
        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            self.base_frame,
            Time(),
            timeout=Duration(seconds=0.1),
        )
        translation = transform.transform.translation
        return float(translation.x), float(translation.y)

    def _world_to_base_transform(self):
        return self.tf_buffer.lookup_transform(
            self.base_frame,
            self.target_frame,
            Time(),
            timeout=Duration(seconds=0.1),
        )

    def expire_obstacles(self):
        """Expire stale cells and repeatedly clear them from Nav2."""
        now_message = self.get_clock().now().to_msg()
        now = stamp_to_seconds(now_message)
        _, expired = self.model.expire_stale(now)
        for key, state in expired:
            self.pending_decay_clears.append(
                (now + self.decay_clear_hold_sec, key, state.source_mask)
            )

        self.pending_decay_clears = [
            item for item in self.pending_decay_clears if item[0] >= now
        ]
        if not self.pending_decay_clears:
            return

        try:
            transform = self._world_to_base_transform()
        except TransformException as error:
            self._log_failure('decay_clear', error)
            return

        source_points = {'front': [], 'rear': []}
        for _, key, source_mask in self.pending_decay_clears:
            cell_x, cell_y = self.model.cell_to_world(key)
            world_point = np.array([[cell_x, cell_y, 0.10]])
            base_point = transform_xyz(world_point, transform)[0]
            distance = math.hypot(base_point[0], base_point[1])
            if distance > 1e-6 and self.decay_clear_extension > 0.0:
                scale = (distance + self.decay_clear_extension) / distance
                base_point[0] *= scale
                base_point[1] *= scale
            if source_mask & self.SOURCE_MASKS['front']:
                source_points['front'].append(base_point)
            if source_mask & self.SOURCE_MASKS['rear']:
                source_points['rear'].append(base_point)

        header = Header(stamp=now_message, frame_id=self.base_frame)
        for source, points in source_points.items():
            if not points:
                continue
            message = point_cloud2.create_cloud_xyz32(header, points)
            self.clearing_pubs[source].publish(message)

    def publish_model(self):
        """Expire stale evidence and publish cloud and occupancy grid."""
        now_message = self.get_clock().now().to_msg()
        try:
            center_x, center_y = self._robot_position()
            world_to_base = self._world_to_base_transform()
        except TransformException as error:
            self._log_failure('robot_pose', error)
            return

        self.model.prune_outside(center_x, center_y)
        occupied = self.model.occupied_items()
        self._publish_occupied_cloud(now_message, occupied)
        self._publish_nav2_obstacle_clouds(
            now_message,
            occupied,
            world_to_base,
        )
        self._publish_occupancy_grid(
            now_message,
            center_x,
            center_y,
            occupied,
        )
        self.published_count += 1
        if self.published_count == 1 or self.published_count % 50 == 0:
            self.get_logger().info(
                'Published local grid: '
                f'cells={len(self.model.cells)}, '
                f'occupied={len(occupied)}, '
                f'received={self.received_count}, '
                f'failed={self.failed_count}'
            )

    def _publish_occupied_cloud(self, stamp, occupied):
        points = []
        for key, state in occupied:
            cell_x, cell_y = self.model.cell_to_world(key)
            point_z = state.max_z if math.isfinite(state.max_z) else 0.10
            points.append((cell_x, cell_y, point_z))
        header = Header(stamp=stamp, frame_id=self.target_frame)
        message = point_cloud2.create_cloud_xyz32(header, points)
        self.cloud_pub.publish(message)

    def _publish_nav2_obstacle_clouds(
        self,
        stamp,
        occupied,
        world_to_base,
    ):
        """Publish current, source-separated model obstacles for Nav2."""
        source_points = {'front': [], 'rear': []}
        for key, state in occupied:
            cell_x, cell_y = self.model.cell_to_world(key)
            point_z = state.max_z if math.isfinite(state.max_z) else 0.10
            point = np.array([[cell_x, cell_y, point_z]])
            base_point = transform_xyz(point, world_to_base)[0]
            for source, source_mask in self.SOURCE_MASKS.items():
                if state.source_mask & source_mask:
                    source_points[source].append(base_point)

        header = Header(stamp=stamp, frame_id=self.base_frame)
        for source, points in source_points.items():
            message = point_cloud2.create_cloud_xyz32(header, points)
            self.obstacle_pubs[source].publish(message)

    def _publish_occupancy_grid(
        self,
        stamp,
        center_x,
        center_y,
        occupied,
    ):
        width_cells = int(round(self.width / self.resolution))
        height_cells = int(round(self.height / self.resolution))
        center_key = self.model.world_to_cell(center_x, center_y)
        origin_key = (
            center_key[0] - width_cells // 2,
            center_key[1] - height_cells // 2,
        )
        data = np.full(width_cells * height_cells, -1, dtype=np.int8)
        occupied_keys = {key for key, _ in occupied}

        for key in self.model.cells:
            grid_x = key[0] - origin_key[0]
            grid_y = key[1] - origin_key[1]
            if 0 <= grid_x < width_cells and 0 <= grid_y < height_cells:
                index = grid_y * width_cells + grid_x
                data[index] = 100 if key in occupied_keys else 0

        message = OccupancyGrid()
        message.header = Header(stamp=stamp, frame_id=self.target_frame)
        message.info.resolution = self.resolution
        message.info.width = width_cells
        message.info.height = height_cells
        message.info.origin.position.x = origin_key[0] * self.resolution
        message.info.origin.position.y = origin_key[1] * self.resolution
        message.info.origin.orientation.w = 1.0
        message.data = data.tolist()
        self.grid_pub.publish(message)

    def _log_failure(self, source, error):
        self.failed_count += 1
        if self.failed_count == 1 or self.failed_count % 25 == 0:
            self.get_logger().warning(
                'Local world-model update failed: '
                f'source={source}, count={self.failed_count}, reason={error}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = LocalWorldModelNode()
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
