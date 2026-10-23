"""Launch the unified trajectory checker and executor."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory('b2_nav_execution')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('source_label', default_value='unlabelled'),
        DeclareLaunchArgument(
            'executor_params_file',
            default_value=os.path.join(
                package_dir, 'config', 'trajectory_executor.yaml'
            ),
        ),
        Node(
            package='b2_nav_execution',
            executable='trajectory_executor_node',
            name='trajectory_executor_node',
            output='screen',
            parameters=[
                LaunchConfiguration('executor_params_file'),
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'source_label': LaunchConfiguration('source_label'),
                },
            ],
        ),
    ])
