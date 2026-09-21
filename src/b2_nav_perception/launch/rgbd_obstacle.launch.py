"""Launch RGB-D point-cloud transformation and obstacle filtering."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create the RGB-D obstacle-processing launch description."""
    package_dir = get_package_share_directory('b2_nav_perception')
    use_sim_time = LaunchConfiguration('use_sim_time')
    rgbd_params_file = LaunchConfiguration('rgbd_params_file')

    front_obstacle_node = Node(
        package='b2_nav_perception',
        executable='rgbd_obstacle_node',
        name='rgbd_obstacle_front',
        output='screen',
        parameters=[
            rgbd_params_file,
            {'use_sim_time': use_sim_time},
        ],
    )
    rear_obstacle_node = Node(
        package='b2_nav_perception',
        executable='rgbd_obstacle_node',
        name='rgbd_obstacle_rear',
        output='screen',
        parameters=[
            rgbd_params_file,
            {'use_sim_time': use_sim_time},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='true',
                description='Use the simulation clock when true.',
            ),
            DeclareLaunchArgument(
                'rgbd_params_file',
                default_value=os.path.join(
                    package_dir,
                    'config',
                    'rgbd_obstacles_dual.yaml',
                ),
                description='RGB-D obstacle-node parameter file.',
            ),
            front_obstacle_node,
            rear_obstacle_node,
        ]
    )
