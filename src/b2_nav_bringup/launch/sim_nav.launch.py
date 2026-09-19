from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    goal_x = LaunchConfiguration('goal_x')
    goal_y = LaunchConfiguration('goal_y')
    obstacle_x = LaunchConfiguration('obstacle_x')
    obstacle_y = LaunchConfiguration('obstacle_y')
    obstacle_radius = LaunchConfiguration('obstacle_radius')

    fake_b2_node = Node(
        package='b2_nav_sim',
        executable='fake_b2_node',
        name='fake_b2_node',
        output='screen',
        parameters=[
            {
                'rate_hz': 20.0,
                'cmd_timeout_sec': 0.5,
                'frame_id': 'odom',
                'child_frame_id': 'base_link',
            }
        ],
    )

    safety_gate_node = Node(
        package='b2_nav_safety',
        executable='safety_gate_node',
        name='safety_gate_node',
        output='screen',
        parameters=[
            {
                'max_linear_x': 0.25,
                'max_linear_y': 0.20,
                'max_angular_z': 0.60,
                'cmd_timeout_sec': 0.5,
                'obstacle_stop_distance': 0.5,
                'obstacle_slow_distance': 1.0,
                'obstacle_timeout_sec': 1.0,
                'publish_rate_hz': 20.0,
            }
        ],
    )

    fake_obstacle_node = Node(
        package='b2_nav_perception',
        executable='fake_obstacle_node',
        name='fake_obstacle_node',
        output='screen',
        parameters=[
            {
                'obstacle_x': obstacle_x,
                'obstacle_y': obstacle_y,
                'obstacle_radius': obstacle_radius,
                'publish_rate_hz': 10.0,
                'frame_id': 'odom',
            }
        ],
    )

    simple_planner_node = Node(
        package='b2_nav_planning',
        executable='simple_planner_node',
        name='simple_planner_node',
        output='screen',
        parameters=[
            {
                'goal_x': goal_x,
                'goal_y': goal_y,
                'goal_tolerance': 0.1,
                'max_linear_speed': 0.3,
                'max_lateral_speed': 0.2,
                'max_angular_speed': 0.8,
                'yaw_gain': 1.5,
                'obstacle_radius': obstacle_radius,
                'avoidance_clearance': 0.8,
                'obstacle_timeout_sec': 1.0,
                'frame_id': 'odom',
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('goal_x', default_value='5.0'),
            DeclareLaunchArgument('goal_y', default_value='0.0'),
            DeclareLaunchArgument('obstacle_x', default_value='2.5'),
            DeclareLaunchArgument('obstacle_y', default_value='0.0'),
            DeclareLaunchArgument('obstacle_radius', default_value='0.35'),
            fake_b2_node,
            safety_gate_node,
            fake_obstacle_node,
            simple_planner_node,
        ]
    )
