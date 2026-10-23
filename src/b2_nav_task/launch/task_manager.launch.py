"""Launch the inspection task manager."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory('b2_nav_task')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('task_file', default_value=''),
        DeclareLaunchArgument('autostart', default_value='false'),
        DeclareLaunchArgument(
            'task_params_file',
            default_value=os.path.join(package_dir, 'config', 'task_manager.yaml'),
        ),
        Node(
            package='b2_nav_task',
            executable='task_manager_node',
            name='task_manager_node',
            output='screen',
            parameters=[
                LaunchConfiguration('task_params_file'),
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'task_file': LaunchConfiguration('task_file'),
                    'autostart': LaunchConfiguration('autostart'),
                },
            ],
        ),
    ])
