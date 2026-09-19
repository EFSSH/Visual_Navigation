from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
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
                'publish_rate_hz': 20.0,
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
                'goal_x': 5.0,
                'goal_y': 0.0,
                'goal_tolerance': 0.1,
                'max_linear_speed': 0.3,
                'max_lateral_speed': 0.2,
                'max_angular_speed': 0.8,
                'yaw_gain': 1.5,
            }
        ],
    )

    return LaunchDescription(
        [
            fake_b2_node,
            safety_gate_node,
            simple_planner_node,
        ]
    )
