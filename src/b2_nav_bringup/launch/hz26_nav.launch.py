"""Start HZ26, then spawn TurtleBot3 before bringing up SLAM/Nav2 and RViz."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node


def generate_launch_description():
    bringup_dir = get_package_share_directory('nav2_bringup')
    hz26_bringup_dir = get_package_share_directory('b2_nav_bringup')
    perception_dir = get_package_share_directory('b2_nav_perception')
    launch_dir = os.path.join(bringup_dir, 'launch')
    robot_urdf = os.path.join(bringup_dir, 'urdf', 'turtlebot3_waffle.urdf')
    robot_sdf = os.path.join(bringup_dir, 'worlds', 'waffle.model')

    with open(robot_urdf, 'r', encoding='utf-8') as urdf_file:
        robot_description = urdf_file.read()

    model_home = LaunchConfiguration('model_home')
    world = LaunchConfiguration('world')
    slam = LaunchConfiguration('slam')
    headless = LaunchConfiguration('headless')
    use_rviz = LaunchConfiguration('use_rviz')
    use_rgbd_obstacles = LaunchConfiguration('use_rgbd_obstacles')
    params_file = LaunchConfiguration('params_file')
    rgbd_params_file = LaunchConfiguration('rgbd_params_file')
    map_file = LaunchConfiguration('map')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    z_pose = LaunchConfiguration('z_pose')
    yaw = LaunchConfiguration('yaw')

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
            x_pose,
            '-y',
            y_pose,
            '-z',
            z_pose,
            '-Y',
            yaw,
            '-timeout',
            '180.0',
        ],
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'bringup_launch.py')
        ),
        launch_arguments={
            'namespace': '',
            'use_namespace': 'False',
            'slam': slam,
            'map': map_file,
            'use_sim_time': 'True',
            'params_file': params_file,
            'autostart': 'True',
            'use_composition': 'False',
            'use_respawn': 'False',
        }.items(),
    )
    rgbd_perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                perception_dir,
                'launch',
                'rgbd_obstacle.launch.py',
            )
        ),
        condition=IfCondition(use_rgbd_obstacles),
        launch_arguments={
            'use_sim_time': 'true',
            'rgbd_params_file': rgbd_params_file,
        }.items(),
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        condition=IfCondition(use_rviz),
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )
    initial_pose_publisher = Node(
        package='b2_nav_bringup',
        executable='hz26_initial_pose',
        name='hz26_initial_pose_publisher',
        condition=UnlessCondition(slam),
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'initial_x': x_pose,
                'initial_y': y_pose,
                'initial_yaw': yaw,
                'timeout_sec': 30.0,
            }
        ],
    )

    start_navigation_after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[
                LogInfo(
                    msg=(
                        'Robot spawn completed; starting RGB-D perception, '
                        'SLAM/Nav2 and RViz.'
                    )
                ),
                rgbd_perception,
                nav2,
                rviz,
                initial_pose_publisher,
            ],
        )
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
            DeclareLaunchArgument(
                'slam',
                default_value='False',
                description='True for online SLAM; False for fixed-map AMCL localization.',
            ),
            DeclareLaunchArgument(
                'map',
                default_value=PathJoinSubstitution(
                    [
                        model_home,
                        'hz26_station',
                        'maps',
                        'hz26_final.yaml',
                    ]
                ),
                description='Fixed occupancy map used when slam is False.',
            ),
            DeclareLaunchArgument('headless', default_value='False'),
            DeclareLaunchArgument('use_rviz', default_value='True'),
            DeclareLaunchArgument(
                'use_rgbd_obstacles',
                default_value='True',
                description='Start RGB-D filtering and feed it to local costmap.',
            ),
            DeclareLaunchArgument(
                'params_file',
                default_value=os.path.join(
                    hz26_bringup_dir,
                    'config',
                    'nav2_rgbd_params.yaml',
                ),
            ),
            DeclareLaunchArgument(
                'rgbd_params_file',
                default_value=os.path.join(
                    perception_dir,
                    'config',
                    'rgbd_obstacle.yaml',
                ),
            ),
            DeclareLaunchArgument(
                'rviz_config_file',
                default_value=os.path.join(
                    hz26_bringup_dir, 'rviz', 'hz26_nav.rviz'
                ),
            ),
            # Staging point on the visual deck after the station's world offset.
            DeclareLaunchArgument('x_pose', default_value='-7.0'),
            DeclareLaunchArgument('y_pose', default_value='-25.0'),
            DeclareLaunchArgument('z_pose', default_value='0.65'),
            DeclareLaunchArgument('yaw', default_value='1.5708'),
            SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle'),
            SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),
            SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
            SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
            LogInfo(
                msg='Loading HZ26; the first GUI start can take one to three minutes.'
            ),
            LogInfo(msg=['HZ26 model directory: ', model_home]),
            gzserver,
            gzclient,
            robot_state_publisher,
            start_navigation_after_spawn,
            spawn_robot,
        ]
    )
