"""
Check and execute local trajectories from any trajectory source.

Model adapters publish ``nav_msgs/Path`` on the trajectory topic. Each path is
checked (age, spacing, length, turning, local-costmap collision) without
modification and, if accepted, sent to Nav2's controller server through
``FollowPath`` with a pure-tracking controller. Paths expire after
``validity_sec``; an expired or rejected path cancels the active goal so
the robot never keeps executing an old output. Execution feedback and
discrete events are published as JSON for recording and evaluation.
"""

import json

from action_msgs.msg import GoalStatus
from b2_nav_execution.trajectory_check import (
    check_trajectory,
    cross_track,
    Grid,
    Limits,
    REASON_OK,
    REASON_TRANSFORM,
    transform_points_2d,
    yaw_from_quaternion,
)
from nav2_msgs.action import FollowPath
from nav_msgs.msg import OccupancyGrid, Path
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener

STATE_IDLE = 'idle'
STATE_EXECUTING = 'executing'
STATE_STOPPED = 'stopped'


class TrajectoryExecutorNode(Node):

    def __init__(self):
        super().__init__('trajectory_executor_node')

        self.declare_parameter('trajectory_topic', '/execution/trajectory')
        self.declare_parameter('costmap_topic', '/local_costmap/costmap')
        self.declare_parameter('feedback_topic', '/execution/feedback')
        self.declare_parameter('events_topic', '/execution/events')
        self.declare_parameter('safety_stop_topic', '/safety/lidar_stop')
        self.declare_parameter('watchdog_state_topic', '/robot/watchdog_state')
        self.declare_parameter('follow_path_action', '/follow_path')
        self.declare_parameter('controller_id', 'TrackPath')
        self.declare_parameter('goal_checker_id', 'general_goal_checker')
        self.declare_parameter('progress_checker_id', 'progress_checker')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('allowed_frames', ['odom', 'map'])
        self.declare_parameter('validity_sec', 1.5)
        self.declare_parameter('costmap_max_age_sec', 2.0)
        self.declare_parameter('require_costmap', True)
        self.declare_parameter('feedback_rate_hz', 5.0)
        self.declare_parameter('max_age_sec', 1.0)
        self.declare_parameter('max_future_sec', 0.2)
        self.declare_parameter('max_spacing_m', 0.5)
        self.declare_parameter('max_length_m', 6.0)
        self.declare_parameter('max_heading_change_rad_per_m', 3.0)
        self.declare_parameter('lethal_cost', 99)
        self.declare_parameter('reject_outside_costmap', True)
        # Label recorded with every event so evaluation can tell sources apart.
        self.declare_parameter('source_label', 'unlabelled')

        param = self.get_parameter
        self.limits = Limits(
            max_age_sec=float(param('max_age_sec').value),
            max_future_sec=float(param('max_future_sec').value),
            max_spacing_m=float(param('max_spacing_m').value),
            max_length_m=float(param('max_length_m').value),
            max_heading_change_rad_per_m=float(
                param('max_heading_change_rad_per_m').value
            ),
            lethal_cost=int(param('lethal_cost').value),
            reject_outside_costmap=bool(param('reject_outside_costmap').value),
        )
        self.controller_id = param('controller_id').value
        self.goal_checker_id = param('goal_checker_id').value
        self.progress_checker_id = param('progress_checker_id').value
        self.robot_frame = param('robot_frame').value
        self.allowed_frames = list(param('allowed_frames').value)
        self.validity_sec = float(param('validity_sec').value)
        self.costmap_max_age_sec = float(param('costmap_max_age_sec').value)
        self.require_costmap = bool(param('require_costmap').value)
        self.source_label = param('source_label').value

        self.grid = None
        self.grid_frame = None
        self.grid_time = None
        self.active_points = None
        self.active_frame = None
        self.active_expiry = None
        self.active_seq = 0
        self.goal_handle = None
        self.nav2_feedback = {}
        self.state = STATE_IDLE
        self.safety_stop = False
        self.watchdog_state = None
        self.counters = {
            'received': 0,
            'accepted': 0,
            'rejected': {},
            'expired': 0,
            'succeeded': 0,
            'aborted': 0,
            'safety_stop_events': 0,
        }

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.follow_client = ActionClient(
            self, FollowPath, param('follow_path_action').value
        )
        self.feedback_pub = self.create_publisher(
            String, param('feedback_topic').value, 10
        )
        self.events_pub = self.create_publisher(
            String, param('events_topic').value, 50
        )
        self.create_subscription(
            Path, param('trajectory_topic').value, self.on_trajectory, 10
        )
        self.create_subscription(
            OccupancyGrid, param('costmap_topic').value, self.on_costmap, 1
        )
        self.create_subscription(
            Bool, param('safety_stop_topic').value, self.on_safety_stop, 10
        )
        self.create_subscription(
            String, param('watchdog_state_topic').value, self.on_watchdog, 10
        )
        rate = float(param('feedback_rate_hz').value)
        self.create_timer(1.0 / rate, self.on_timer)
        self.get_logger().info(
            f'trajectory executor: {param("trajectory_topic").value} -> '
            f'FollowPath[{self.controller_id}], source={self.source_label}'
        )

    # -- inputs -----------------------------------------------------------

    def on_costmap(self, msg):
        info = msg.info
        self.grid = Grid(
            resolution=info.resolution,
            origin_x=info.origin.position.x,
            origin_y=info.origin.position.y,
            width=info.width,
            height=info.height,
            data=list(msg.data),
        )
        self.grid_frame = msg.header.frame_id
        self.grid_time = self.get_clock().now()

    def on_safety_stop(self, msg):
        if msg.data and not self.safety_stop:
            self.counters['safety_stop_events'] += 1
            self.emit('safety_stop', {})
        self.safety_stop = bool(msg.data)

    def on_watchdog(self, msg):
        try:
            self.watchdog_state = json.loads(msg.data).get('state')
        except (ValueError, AttributeError):
            self.watchdog_state = 'unparseable'

    def on_trajectory(self, msg):
        self.counters['received'] += 1
        now = self.get_clock().now()
        frame = msg.header.frame_id
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        age = (now - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9

        if frame not in self.allowed_frames:
            self.reject('frame_not_allowed', {'frame': frame})
            return
        grid = self.current_grid(now)
        checked = points
        if grid is not None and frame != self.grid_frame:
            checked = self.transform(points, self.grid_frame, frame)
            if checked is None:
                self.reject(REASON_TRANSFORM, {
                    'from': frame, 'to': self.grid_frame,
                })
                return
        reason, details = check_trajectory(
            checked, age, self.limits, grid, self.require_costmap
        )
        if reason != REASON_OK:
            self.reject(reason, details)
            return
        if not self.follow_client.server_is_ready():
            self.reject('controller_unavailable', {})
            return

        self.counters['accepted'] += 1
        self.active_seq += 1
        self.active_points = points
        self.active_frame = frame
        self.active_expiry = Time.from_msg(msg.header.stamp) + Duration(
            seconds=self.validity_sec
        )
        goal = FollowPath.Goal()
        goal.path = msg
        goal.controller_id = self.controller_id
        goal.goal_checker_id = self.goal_checker_id
        if hasattr(goal, 'progress_checker_id'):
            goal.progress_checker_id = self.progress_checker_id
        seq = self.active_seq
        future = self.follow_client.send_goal_async(
            goal, feedback_callback=self.on_nav2_feedback
        )
        future.add_done_callback(lambda f: self.on_goal_response(f, seq))
        self.set_state(STATE_EXECUTING)
        self.emit('accepted', dict(details, seq=seq, points=len(points)))

    # -- Nav2 action ------------------------------------------------------

    def on_goal_response(self, future, seq):
        handle = future.result()
        if seq != self.active_seq:
            # A newer path replaced this one; Nav2 preempts automatically.
            return
        if not handle.accepted:
            self.emit('controller_rejected', {'seq': seq})
            self.clear_active(STATE_STOPPED)
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda f: self.on_result(f, seq)
        )

    def on_nav2_feedback(self, msg):
        feedback = msg.feedback
        self.nav2_feedback = {
            'distance_to_goal': feedback.distance_to_goal,
            'speed': feedback.speed,
        }

    def on_result(self, future, seq):
        if seq != self.active_seq:
            return
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.counters['succeeded'] += 1
            self.emit('succeeded', {'seq': seq})
            self.clear_active(STATE_IDLE)
        elif status == GoalStatus.STATUS_ABORTED:
            self.counters['aborted'] += 1
            result = future.result().result
            self.emit('aborted', {
                'seq': seq,
                'error_code': getattr(result, 'error_code', None),
            })
            self.clear_active(STATE_STOPPED)

    # -- periodic ---------------------------------------------------------

    def on_timer(self):
        now = self.get_clock().now()
        if self.state == STATE_EXECUTING and now > self.active_expiry:
            self.counters['expired'] += 1
            self.emit('expired', {'seq': self.active_seq})
            self.cancel_active(STATE_STOPPED)

        feedback = {
            'schema': 'b2_nav/execution_feedback/v1',
            'source': self.source_label,
            'state': self.state,
            'seq': self.active_seq,
            'safety_stop': self.safety_stop,
            'watchdog_state': self.watchdog_state,
            'counters': self.counters,
            'nav2': self.nav2_feedback,
            'stamp_sec': now.nanoseconds * 1e-9,
        }
        if self.state == STATE_EXECUTING and self.active_points:
            pose = self.robot_xy(self.active_frame)
            if pose is not None:
                error, remaining = cross_track(self.active_points, *pose)
                feedback['tracking_error_m'] = error
                feedback['remaining_m'] = remaining
        self.feedback_pub.publish(String(data=json.dumps(feedback)))

    # -- helpers ----------------------------------------------------------

    def current_grid(self, now):
        if self.grid is None:
            return None
        if (now - self.grid_time).nanoseconds * 1e-9 > self.costmap_max_age_sec:
            return None
        return self.grid

    def lookup_2d(self, target, source):
        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, Time(), timeout=Duration(seconds=0.05)
            ).transform
        except TransformException:
            return None
        q = tf.rotation
        return tf.translation.x, tf.translation.y, yaw_from_quaternion(
            q.x, q.y, q.z, q.w
        )

    def transform(self, points, target, source):
        tf = self.lookup_2d(target, source)
        return None if tf is None else transform_points_2d(points, *tf)

    def robot_xy(self, frame):
        tf = self.lookup_2d(frame, self.robot_frame)
        return None if tf is None else (tf[0], tf[1])

    def reject(self, reason, details):
        rejected = self.counters['rejected']
        rejected[reason] = rejected.get(reason, 0) + 1
        self.emit('rejected', dict(details, reason=reason))
        if self.state == STATE_EXECUTING:
            self.cancel_active(STATE_STOPPED)

    def cancel_active(self, state):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.clear_active(state)

    def clear_active(self, state):
        self.goal_handle = None
        self.active_points = None
        self.active_expiry = None
        self.nav2_feedback = {}
        self.set_state(state)

    def set_state(self, state):
        if state != self.state:
            self.state = state
            self.emit('state', {'state': state})

    def emit(self, event, details):
        self.events_pub.publish(String(data=json.dumps({
            'schema': 'b2_nav/execution_event/v1',
            'source': self.source_label,
            'event': event,
            'details': details,
            'stamp_sec': self.get_clock().now().nanoseconds * 1e-9,
        })))


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryExecutorNode()
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
