"""Launch the end-stage device search node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory('b2_nav_search')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'library_dir',
            default_value=os.path.join(
                os.path.expanduser('~'), 'b2_nav_data', 'reference_library'
            ),
        ),
        DeclareLaunchArgument(
            'search_params_file',
            default_value=os.path.join(package_dir, 'config', 'object_search.yaml'),
        ),
        Node(
            package='b2_nav_search',
            executable='object_search_node',
            name='object_search_node',
            output='screen',
            parameters=[
                LaunchConfiguration('search_params_file'),
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'library_dir': LaunchConfiguration('library_dir'),
                },
            ],
        ),
    ])
