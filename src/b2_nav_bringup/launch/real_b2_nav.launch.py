from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    goal_x = LaunchConfiguration('goal_x')
    goal_y = LaunchConfiguration('goal_y')

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
                'goal_x': goal_x,
                'goal_y': goal_y,
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
            DeclareLaunchArgument('goal_x', default_value='5.0'),
            DeclareLaunchArgument('goal_y', default_value='0.0'),
            LogInfo(
                msg=(
                    'real_b2_node is not implemented yet. Start the B2 '
                    'driver separately or add it to b2_nav_robot later.'
                )
            ),
            LogInfo(
                msg=(
                    'This launch waits for a real /robot/odom source and '
                    'publishes safe commands on /b2/cmd_vel.'
                )
            ),
            safety_gate_node,
            simple_planner_node,
        ]
    )
