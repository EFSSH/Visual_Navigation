"""Launch lidar monitoring and the final velocity safety gate."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory('b2_nav_safety')
    params_file = LaunchConfiguration('safety_params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    common_parameters = [params_file, {'use_sim_time': use_sim_time}]

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'safety_params_file',
            default_value=os.path.join(
                package_dir, 'config', 'lidar_safety.yaml'
            ),
        ),
        Node(
            package='b2_nav_safety',
            executable='lidar_safety_node',
            name='lidar_safety_node',
            output='screen',
            parameters=common_parameters,
        ),
        Node(
            package='b2_nav_safety',
            executable='safety_gate_node',
            name='safety_gate_node',
            output='screen',
            parameters=common_parameters,
        ),
    ])
