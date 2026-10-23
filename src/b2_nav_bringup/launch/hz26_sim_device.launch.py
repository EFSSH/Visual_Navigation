"""Run HZ26 as the simulated device host for a separate navigation host.

Starts Gazebo, robot_state_publisher and the TurtleBot3 spawn only. AMCL,
Nav2, perception and the safety gate run on the navigation host through
navigation_only.launch.py, so none of them are started here.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node


def generate_launch_description():
    """Keep localization, planning and the safety gate off the device host."""
    if os.environ.get('ROS_DISTRO') != 'humble':
        raise RuntimeError('hz26_sim_device.launch.py requires ROS 2 Humble.')

    hz26_bringup_dir = get_package_share_directory('b2_nav_bringup')
    robot_urdf = os.path.join(
        hz26_bringup_dir, 'urdf', 'turtlebot3_waffle_dual_rgbd.urdf'
    )
    robot_sdf = os.path.join(
        hz26_bringup_dir, 'models', 'waffle_dual_rgbd.model'
    )

    with open(robot_urdf, 'r', encoding='utf-8') as urdf_file:
        robot_description = urdf_file.read()

    model_home = LaunchConfiguration('model_home')
    world = LaunchConfiguration('world')
    headless = LaunchConfiguration('headless')
    use_rviz = LaunchConfiguration('use_rviz')

    default_world = PathJoinSubstitution(
        [model_home, 'hz26_station', 'worlds', 'HZ26_station.world']
    )
    gazebo_model_path = [
        model_home,
        ':',
        '/usr/share/gazebo-11/models',
        ':',
        '/opt/ros/humble/share/turtlebot3_gazebo/models',
        ':',
        EnvironmentVariable('GAZEBO_MODEL_PATH', default_value=''),
    ]

    gzserver = ExecuteProcess(
        cmd=[
            'gzserver',
            '-s',
            'libgazebo_ros_init.so',
            '-s',
            'libgazebo_ros_factory.so',
            world,
        ],
        output='screen',
    )
    gzclient = ExecuteProcess(
        condition=UnlessCondition(headless),
        cmd=['gzclient'],
        output='screen',
    )
    # The navigation host does not publish robot TF; base_link to sensor
    # frames come from here and odom to base_footprint from the diff drive.
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'robot_description': robot_description,
            }
        ],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_hz26_turtlebot3',
        output='screen',
        arguments=[
            '-entity',
            'turtlebot3_waffle',
            '-file',
            robot_sdf,
            '-x',
            LaunchConfiguration('x_pose'),
            '-y',
            LaunchConfiguration('y_pose'),
            '-z',
            LaunchConfiguration('z_pose'),
            '-Y',
            LaunchConfiguration('yaw'),
            '-timeout',
            '180.0',
        ],
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        condition=IfCondition(use_rviz),
        arguments=['-d', LaunchConfiguration('rviz_config_file')],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )
    # The diff drive listens to /cmd_vel_device; the watchdog forwards the
    # remote /cmd_vel_safe and stops the base if the navigation host is lost.
    cmd_watchdog = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('b2_nav_robot'),
            'launch', 'cmd_watchdog.launch.py',
        )),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'model_home',
                default_value=EnvironmentVariable(
                    'HZ26_MODEL_HOME',
                    default_value=[EnvironmentVariable('HOME'), '/b2_nav_ws/sim_assets'],
                ),
                description='Directory containing the hz26_station Gazebo model.',
            ),
            DeclareLaunchArgument('world', default_value=default_world),
            DeclareLaunchArgument('headless', default_value='False'),
            DeclareLaunchArgument(
                'use_rviz',
                default_value='False',
                description='RViz here subscribes to remote costmaps and paths over the LAN.',
            ),
            DeclareLaunchArgument(
                'rviz_config_file',
                default_value=os.path.join(
                    hz26_bringup_dir, 'rviz', 'hz26_nav.rviz'
                ),
            ),
            # Must match the initial pose given to navigation_only.launch.py.
            DeclareLaunchArgument('x_pose', default_value='-7.0'),
            DeclareLaunchArgument('y_pose', default_value='-25.0'),
            DeclareLaunchArgument('z_pose', default_value='0.65'),
            DeclareLaunchArgument('yaw', default_value='1.5708'),
            SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle'),
            SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),
            SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
            LogInfo(
                msg='HZ26 device host: Nav2 and the safety gate run on the navigation host.'
            ),
            LogInfo(msg=['HZ26 model directory: ', model_home]),
            gzserver,
            gzclient,
            robot_state_publisher,
            cmd_watchdog,
            spawn_robot,
            rviz,
        ]
    )
