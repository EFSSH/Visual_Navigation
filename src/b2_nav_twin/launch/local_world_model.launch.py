"""Launch the rolling 2.5D local world model."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create the local world-model launch description."""
    package_dir = get_package_share_directory('b2_nav_twin')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('local_world_params_file')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='true',
                description='Use the simulation clock when true.',
            ),
            DeclareLaunchArgument(
                'local_world_params_file',
                default_value=os.path.join(
                    package_dir,
                    'config',
                    'local_world_model.yaml',
                ),
                description='Local world-model parameter file.',
            ),
            Node(
                package='b2_nav_twin',
                executable='local_world_model_node',
                name='local_world_model_node',
                output='screen',
                parameters=[params_file, {'use_sim_time': use_sim_time}],
            ),
        ]
    )
