"""Start the mission layer on the navigation host, next to navigation_only.

Runs the trajectory executor, the end-stage device search and the task
manager. Launch navigation_only.launch.py first (Nav2, AMCL, perception,
safety gate); this file adds no velocity source of its own except through
Nav2 actions, so every command still passes the safety gate.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _include(package, filename, arguments, condition=None):
    kwargs = {'condition': condition} if condition is not None else {}
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory(package), 'launch', filename
        )),
        launch_arguments=arguments.items(),
        **kwargs,
    )


def generate_launch_description():
    """Compose execution, search and task management."""
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('use_executor', default_value='true'),
        DeclareLaunchArgument('source_label', default_value='unlabelled'),
        DeclareLaunchArgument(
            'library_dir',
            default_value=os.path.join(
                os.path.expanduser('~'), 'b2_nav_data', 'reference_library'
            ),
        ),
        DeclareLaunchArgument('task_file', default_value=''),
        DeclareLaunchArgument('autostart', default_value='false'),
        _include('b2_nav_execution', 'trajectory_executor.launch.py', {
            'use_sim_time': use_sim_time,
            'source_label': LaunchConfiguration('source_label'),
        }, IfCondition(LaunchConfiguration('use_executor'))),
        _include('b2_nav_search', 'object_search.launch.py', {
            'use_sim_time': use_sim_time,
            'library_dir': LaunchConfiguration('library_dir'),
        }),
        _include('b2_nav_task', 'task_manager.launch.py', {
            'use_sim_time': use_sim_time,
            'task_file': LaunchConfiguration('task_file'),
            'autostart': LaunchConfiguration('autostart'),
        }),
    ])
