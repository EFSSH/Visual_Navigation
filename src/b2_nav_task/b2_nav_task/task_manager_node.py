"""
Run one inspection task through the plan's state machine.

Start a task by publishing its YAML/JSON text (or ``file:<path>``) on
``start_topic``, or set ``task_file`` with ``autostart``. The node checks
sensors and Nav2, waits for a confident AMCL pose, drives the route
subgoals with NavigateToPose, runs search windows separated by in-place
turns, drives to the confirmed observation pose and publishes a report.
Every failure stops the robot and is reported with a reason; nothing is
counted as a success unless the observation pose was reached.

Route subgoals come from the task file for now; the digital-twin route
generator will replace that source without changing this node.
"""

import json
import math
import os
import time

from action_msgs.msg import GoalStatus
from b2_nav_task.state_machine import TaskMachine, TERMINAL
from b2_nav_task.task_spec import load_task, TaskError
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose, Spin
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


def yaw_to_quaternion(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class TaskManagerNode(Node):

    def __init__(self):
        super().__init__('task_manager_node')

        self.declare_parameter('task_file', '')
        self.declare_parameter('autostart', False)
        self.declare_parameter('start_topic', '/task/start')
        self.declare_parameter('cancel_topic', '/task/cancel')
        self.declare_parameter('state_topic', '/task/state')
        self.declare_parameter('events_topic', '/task/events')
        self.declare_parameter('report_topic', '/task/report')
        self.declare_parameter('search_request_topic', '/search/request')
        self.declare_parameter('search_result_topic', '/search/result')
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('navigate_action', '/navigate_to_pose')
        self.declare_parameter('spin_action', '/spin')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        # "topic|package/msg/Type|max_age_sec" entries checked before start.
        self.declare_parameter('sensor_checks', [
            '/scan|sensor_msgs/msg/LaserScan|1.0',
            '/odom|nav_msgs/msg/Odometry|1.0',
        ])
        self.declare_parameter('max_position_variance', 0.5)
        self.declare_parameter('max_yaw_variance', 0.3)
        self.declare_parameter('spin_time_allowance_sec', 20.0)

        param = self.get_parameter
        self.map_frame = param('map_frame').value
        self.robot_frame = param('robot_frame').value
        self.max_position_variance = float(param('max_position_variance').value)
        self.max_yaw_variance = float(param('max_yaw_variance').value)
        self.spin_time_allowance = float(param('spin_time_allowance_sec').value)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, param('state_topic').value, latched)
        self.report_pub = self.create_publisher(String, param('report_topic').value, latched)
        self.events_pub = self.create_publisher(String, param('events_topic').value, 50)
        self.search_pub = self.create_publisher(
            String, param('search_request_topic').value, 10
        )
        self.create_subscription(String, param('start_topic').value, self.on_start, 10)
        self.create_subscription(String, param('cancel_topic').value, self.on_cancel, 10)
        self.create_subscription(
            String, param('search_result_topic').value, self.on_search_result, latched
        )
        self.create_subscription(
            PoseWithCovarianceStamped, param('amcl_pose_topic').value,
            self.on_amcl_pose, latched,
        )
        self.navigate_client = ActionClient(
            self, NavigateToPose, param('navigate_action').value
        )
        self.spin_client = ActionClient(self, Spin, param('spin_action').value)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sensor_last = {}
        self.sensor_max_age = {}
        for entry in param('sensor_checks').value:
            self.add_sensor_check(entry)

        self.machine = None
        self.amcl_pose = None
        self.pending = None      # what the machine is waiting for
        self.goal_handle = None
        self.goal_seq = 0
        self.search_request_id = None
        self.entries_seen = -1
        self.state_since = None
        self.started_wall = None
        self.timeline = []
        self.create_timer(0.5, self.on_timer)

        task_file = param('task_file').value
        if task_file and param('autostart').value:
            self.start_task_text(f'file:{task_file}')
        self.get_logger().info('task manager ready')

    # -- inputs -----------------------------------------------------------

    def add_sensor_check(self, entry):
        try:
            topic, type_name, max_age = entry.split('|')
            msg_type = get_message(type_name)
        except (ValueError, AttributeError, ModuleNotFoundError) as error:
            self.get_logger().error(f'bad sensor check {entry!r}: {error}')
            return
        self.sensor_max_age[topic] = float(max_age)
        self.create_subscription(
            msg_type, topic,
            lambda _msg, t=topic: self.sensor_last.__setitem__(
                t, self.get_clock().now()
            ),
            qos_profile_sensor_data,
        )

    def on_amcl_pose(self, msg):
        self.amcl_pose = msg

    def on_start(self, msg):
        self.start_task_text(msg.data)

    def on_cancel(self, msg):
        if self.machine is not None:
            self.apply('cancel', detail=msg.data or 'cancel requested')

    def on_search_result(self, msg):
        try:
            result = json.loads(msg.data)
        except ValueError:
            return
        if self.pending != 'search' or result.get('request_id') != self.search_request_id:
            return
        self.pending = None
        self.apply('search_result', **result)

    # -- task lifecycle ---------------------------------------------------

    def start_task_text(self, text):
        if self.machine is not None and self.machine.state not in TERMINAL:
            self.get_logger().warning('a task is already running; cancel it first')
            return
        try:
            if text.startswith('file:'):
                path = os.path.expanduser(text[len('file:'):].strip())
                with open(path, 'r', encoding='utf-8') as handle:
                    text = handle.read()
            spec = load_task(text)
        except (OSError, TaskError) as error:
            self.publish_report({
                'outcome': 'failed', 'failure_reason': 'task_invalid',
                'detail': str(error),
            })
            self.get_logger().error(f'task rejected: {error}')
            return
        self.machine = TaskMachine(spec)
        self.timeline = []
        self.started_wall = time.time()
        self.get_logger().info(f'task {spec.task_id}: device {spec.device_id}')
        self.run(self.machine.start())

    def apply(self, event, **data):
        self.emit('event', {'event': event, 'data': data})
        self.run(self.machine.handle(event, **data))

    def run(self, actions):
        self.note_state()
        for action in actions:
            getattr(self, f'do_{action.kind}')(**action.args)

    def note_state(self):
        machine = self.machine
        if machine.entries == self.entries_seen:
            return
        self.entries_seen = machine.entries
        now = self.get_clock().now()
        self.state_since = now
        record = {
            'state': machine.state,
            'detail': machine.detail,
            'stamp_sec': now.nanoseconds * 1e-9,
        }
        self.timeline.append(record)
        self.state_pub.publish(String(data=json.dumps(dict(
            record, schema='b2_nav/task_state/v1',
            task_id=machine.spec.task_id,
        ))))
        self.emit('state', record)
        self.get_logger().info(f'state -> {machine.state} {machine.detail}')

    # -- actions ----------------------------------------------------------

    def do_preflight(self):
        self.pending = 'preflight'

    def do_wait_localization(self):
        self.pending = 'localization'

    def do_navigate(self, purpose, pose, index=None):
        if not self.navigate_client.server_is_ready():
            self.pending = None
            self.apply('nav_failed', detail='navigate_to_pose unavailable')
            return
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = pose.get('frame', self.map_frame)
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(pose['x'])
        goal.pose.pose.position.y = float(pose['y'])
        z, w = yaw_to_quaternion(float(pose['yaw']))
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = z, w
        self.send_goal(self.navigate_client, goal, 'nav', {
            'purpose': purpose, 'index': index, 'pose': pose,
        })

    def do_spin(self, angle):
        if not self.spin_client.server_is_ready():
            self.pending = None
            self.apply('spin_failed', detail='spin unavailable')
            return
        goal = Spin.Goal()
        goal.target_yaw = float(angle)
        goal.time_allowance = Duration(seconds=self.spin_time_allowance).to_msg()
        self.send_goal(self.spin_client, goal, 'spin', {'angle': angle})

    def do_search(self, attempt, device_id, timeout_sec):
        self.pending = 'search'
        self.search_request_id = f'{self.machine.spec.task_id}-{attempt}'
        self.search_pub.publish(String(data=json.dumps({
            'request_id': self.search_request_id,
            'device_id': device_id,
            'timeout_sec': timeout_sec,
        })))
        self.emit('search_request', {'request_id': self.search_request_id})

    def do_stop(self, outcome, reason):
        self.pending = None
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
        self.goal_seq += 1  # ignore results of anything still in flight
        machine = self.machine
        report = {
            'outcome': outcome,
            'failure_reason': reason,
            'detail': machine.detail,
            'device_id': machine.spec.device_id,
            'region': machine.spec.region,
            'subgoals_total': len(machine.spec.route),
            'subgoals_reached': machine.subgoals_reached,
            'search_windows': machine.search_attempts,
            'last_search': machine.last_search,
            'timeline': self.timeline,
            'final_pose': self.robot_pose(),
            'wall_duration_sec': time.time() - self.started_wall,
        }
        if self.timeline:
            report['sim_duration_sec'] = (
                self.timeline[-1]['stamp_sec'] - self.timeline[0]['stamp_sec']
            )
        self.publish_report(report)

    # -- Nav2 goals -------------------------------------------------------

    def send_goal(self, client, goal, kind, info):
        self.pending = kind
        self.goal_seq += 1
        seq = self.goal_seq
        self.emit(f'{kind}_goal', info)
        future = client.send_goal_async(goal)
        future.add_done_callback(lambda f: self.on_goal_response(f, seq, kind))

    def on_goal_response(self, future, seq, kind):
        if seq != self.goal_seq:
            return
        handle = future.result()
        if not handle.accepted:
            self.pending = None
            self.apply(f'{kind}_failed', detail='goal rejected')
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda f: self.on_goal_result(f, seq, kind)
        )

    def on_goal_result(self, future, seq, kind):
        if seq != self.goal_seq:
            return
        self.goal_handle = None
        self.pending = None
        wrapped = future.result()
        if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
            event = 'nav_succeeded' if kind == 'nav' else 'spin_done'
            self.apply(event)
            return
        code = getattr(wrapped.result, 'error_code', None)
        self.apply(
            'nav_failed' if kind == 'nav' else 'spin_failed',
            detail=f'status {wrapped.status}, error_code {code}',
        )

    # -- periodic checks --------------------------------------------------

    def on_timer(self):
        machine = self.machine
        if machine is None or machine.state in TERMINAL:
            return
        now = self.get_clock().now()
        limit = machine.timeout_sec()
        if limit is not None and (now - self.state_since).nanoseconds * 1e-9 > limit:
            self.apply('state_timeout')
            return
        if self.pending == 'preflight':
            problems = self.preflight_problems(now)
            if not problems:
                self.pending = None
                self.apply('preflight_ok')
            else:
                machine.detail = '; '.join(problems)
        elif self.pending == 'localization' and self.localized():
            self.pending = None
            self.apply('localized')

    def preflight_problems(self, now):
        problems = []
        for topic, max_age in self.sensor_max_age.items():
            last = self.sensor_last.get(topic)
            if last is None:
                problems.append(f'{topic} not received')
            elif (now - last).nanoseconds * 1e-9 > max_age:
                problems.append(f'{topic} stale')
        if not self.navigate_client.server_is_ready():
            problems.append('navigate_to_pose unavailable')
        if not self.spin_client.server_is_ready():
            problems.append('spin unavailable')
        if self.search_pub.get_subscription_count() == 0:
            problems.append('object search node not running')
        return problems

    def localized(self):
        if self.amcl_pose is None:
            return False
        cov = self.amcl_pose.pose.covariance
        return (cov[0] <= self.max_position_variance
                and cov[7] <= self.max_position_variance
                and cov[35] <= self.max_yaw_variance)

    # -- output -----------------------------------------------------------

    def robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame, Time(),
                timeout=Duration(seconds=0.1),
            ).transform
        except TransformException:
            return None
        q = tf.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
        return {'x': tf.translation.x, 'y': tf.translation.y, 'yaw': yaw}

    def publish_report(self, report):
        machine = self.machine
        report = dict(
            report,
            schema='b2_nav/task_report/v1',
            task_id=machine.spec.task_id if machine else None,
        )
        self.report_pub.publish(String(data=json.dumps(report)))
        self.emit('report', report)
        self.get_logger().info(
            f'task report: {report.get("outcome")} {report.get("failure_reason") or ""}'
        )

    def emit(self, event, details):
        self.events_pub.publish(String(data=json.dumps({
            'schema': 'b2_nav/task_event/v1',
            'task_id': self.machine.spec.task_id if self.machine else None,
            'event': event,
            'details': details,
            'stamp_sec': self.get_clock().now().nanoseconds * 1e-9,
        })))


def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()
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
