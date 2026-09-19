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


class RGBDObstacleNode(Node):

    def __init__(self):
        super().__init__('rgbd_obstacle_node')

        self.declare_parameter(
            'input_topic',
            '/intel_realsense_r200_depth/points',
        )
        self.declare_parameter(
            'output_topic',
            '/vision/obstacle_cloud',
        )
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('x_min', 0.20)
        self.declare_parameter('x_max', 3.00)
        self.declare_parameter('y_min', -1.50)
        self.declare_parameter('y_max', 1.50)
        self.declare_parameter('z_min', 0.05)
        self.declare_parameter('z_max', 1.50)
        self.declare_parameter('point_step', 4)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        self.x_min = self.get_parameter('x_min').value
        self.x_max = self.get_parameter('x_max').value
        self.y_min = self.get_parameter('y_min').value
        self.y_max = self.get_parameter('y_max').value
        self.z_min = self.get_parameter('z_min').value
        self.z_max = self.get_parameter('z_max').value
        self.point_step = self.get_parameter('point_step').value

        self._validate_filter_parameters()

        self.received_count = 0
        self.transformed_count = 0
        self.failed_count = 0

        # TF Buffer 保存各坐标系之间的关系；Listener 持续接收 /tf 和 /tf_static。
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.on_cloud,
            qos_profile_sensor_data,
        )

        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            'RGB-D obstacle node started: '
            f'input={self.input_topic}, '
            f'output={self.output_topic}, '
            f'target_frame={self.target_frame}'
        )
        self.get_logger().info(
            'Point cloud filter: '
            f'x=[{self.x_min}, {self.x_max}] m, '
            f'y=[{self.y_min}, {self.y_max}] m, '
            f'z=[{self.z_min}, {self.z_max}] m, '
            f'point_step={self.point_step}'
        )

    def on_cloud(self, msg):
        self.received_count += 1
        source_frame = msg.header.frame_id

        if not source_frame:
            self.failed_count += 1
            self._log_processing_failure(
                'input point cloud has an empty frame_id'
            )
            return

        try:
            if source_frame == self.target_frame:
                transform = None
            else:
                # 使用点云自己的时间戳，取得该时刻“相机坐标系 -> base_link”的变换。
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    source_frame,
                    Time.from_msg(msg.header.stamp),
                    timeout=Duration(seconds=0.2),
                )
            filtered_cloud = self.transform_and_filter_cloud(msg, transform)
        except (AssertionError, TransformException, ValueError) as error:
            self.failed_count += 1
            self._log_processing_failure(str(error))
            return

        self.cloud_pub.publish(filtered_cloud)
        self.transformed_count += 1

        # 相机约5 Hz，每25帧打印一次，避免刷屏
        if self.transformed_count == 1 or self.transformed_count % 25 == 0:
            self.get_logger().info(
                'Transformed and filtered point cloud: '
                f'received={self.received_count}, '
                f'published={self.transformed_count}, '
                f'{source_frame} -> {filtered_cloud.header.frame_id}, '
                f'input_points={msg.width * msg.height}, '
                f'output_points={filtered_cloud.width * filtered_cloud.height}'
            )

    def transform_and_filter_cloud(self, cloud, transform):
        """Transform XYZ points and keep those inside the configured box."""
        points = point_cloud2.read_points(
            cloud,
            field_names=['x', 'y', 'z'],
            skip_nans=False,
        )

        # 相机点云通常很密，先均匀抽样以降低后续计算和通信负载。
        points = points[::self.point_step]
        xyz = np.column_stack(
            (points['x'], points['y'], points['z'])
        ).astype(np.float64, copy=False)

        # Gazebo uses NaN/Inf for pixels without a valid depth return. Remove
        # them before matrix multiplication to avoid warnings and wasted work.
        xyz = xyz[np.isfinite(xyz).all(axis=1)]

        if transform is not None:
            xyz = self._apply_transform(xyz, transform)

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        keep = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(z)
            & (x >= self.x_min)
            & (x <= self.x_max)
            & (y >= self.y_min)
            & (y <= self.y_max)
            & (z >= self.z_min)
            & (z <= self.z_max)
        )

        filtered_points = xyz[keep].astype(np.float32, copy=False)
        output_header = Header(
            stamp=cloud.header.stamp,
            frame_id=self.target_frame,
        )
        return point_cloud2.create_cloud_xyz32(
            output_header,
            filtered_points,
        )

    @staticmethod
    def _apply_transform(xyz, transform_stamped):
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

    def _validate_filter_parameters(self):
        if self.x_min >= self.x_max:
            raise ValueError('x_min must be smaller than x_max')
        if self.y_min >= self.y_max:
            raise ValueError('y_min must be smaller than y_max')
        if self.z_min >= self.z_max:
            raise ValueError('z_min must be smaller than z_max')
        if self.point_step < 1:
            raise ValueError('point_step must be at least 1')

    def _log_processing_failure(self, reason):
        # TF 尚未建立时，前几帧失败是正常现象；限制日志频率避免刷屏。
        if self.failed_count == 1 or self.failed_count % 25 == 0:
            self.get_logger().warning(
                'Point cloud processing failed: '
                f'failed={self.failed_count}, '
                f'target={self.target_frame}, '
                f'reason={reason}'
            )


def main(args=None):
    rclpy.init(args=args)

    node = RGBDObstacleNode()

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
