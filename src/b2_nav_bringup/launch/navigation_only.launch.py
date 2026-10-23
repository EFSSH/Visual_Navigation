"""Run the Jazzy navigation host, receiving sensors and TF from a device host."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    """Keep Gazebo, device drivers and robot TF publishing outside this host."""
    if os.environ.get('ROS_DISTRO') != 'jazzy':
        raise RuntimeError('navigation_only.launch.py requires ROS 2 Jazzy.')

    share = get_package_share_directory('b2_nav_bringup')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params = ParameterFile(
        RewrittenYaml(
            source_file=LaunchConfiguration('params_file'),
            root_key='',
            param_rewrites={
                'use_sim_time': use_sim_time,
                'yaml_filename': LaunchConfiguration('map'),
            },
            convert_types=True,
        ),
        allow_substs=True,
    )

    actions = [
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('use_nav2', default_value='true'),
        DeclareLaunchArgument('use_perception', default_value='true'),
        DeclareLaunchArgument('use_local_world_model', default_value='true'),
        DeclareLaunchArgument(
            'map', default_value=os.path.join(share, 'maps', 'hz26_final.yaml')
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(share, 'config', 'nav2_jazzy_params.yaml'),
        ),
        DeclareLaunchArgument('use_initial_pose', default_value='true'),
        # Must match the spawn pose given to hz26_sim_device.launch.py.
        DeclareLaunchArgument('initial_x', default_value='-7.0'),
        DeclareLaunchArgument('initial_y', default_value='-25.0'),
        DeclareLaunchArgument('initial_yaw', default_value='1.5708'),
    ]

    # Every generated velocity passes through the existing independent gate.
    # This entry point deliberately uses the baseline Nav2 servers rather than
    # enabling unrelated route/docking servers in newer nav2_bringup releases.
    servers = [
        ('nav2_map_server', 'map_server', 'map_server', []),
        ('nav2_amcl', 'amcl', 'amcl', []),
        ('nav2_controller', 'controller_server', 'controller_server',
         [('cmd_vel', 'cmd_vel_nav')]),
        ('nav2_smoother', 'smoother_server', 'smoother_server', []),
        ('nav2_planner', 'planner_server', 'planner_server', []),
        ('nav2_behaviors', 'behavior_server', 'behavior_server',
         [('cmd_vel', 'cmd_vel_nav')]),
        ('nav2_bt_navigator', 'bt_navigator', 'bt_navigator', []),
        ('nav2_waypoint_follower', 'waypoint_follower', 'waypoint_follower', []),
        ('nav2_velocity_smoother', 'velocity_smoother', 'velocity_smoother',
         [('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')]),
    ]
    for package, executable, name, remappings in servers:
        actions.append(Node(
            package=package, executable=executable, name=name,
            parameters=[params], output='screen',
            condition=IfCondition(LaunchConfiguration('use_nav2')),
            remappings=remappings,
        ))
    actions.append(Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen',
        condition=IfCondition(LaunchConfiguration('use_nav2')),
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'node_names': [entry[2] for entry in servers],
        }],
    ))
    # Runs here rather than on the device host so the AMCL and map_server
    # lifecycle services it may call stay within the same ROS distribution.
    # The timeout is long because the remote simulation may start later.
    actions.append(Node(
        package='b2_nav_bringup', executable='hz26_initial_pose',
        name='hz26_initial_pose_publisher', output='screen',
        condition=IfCondition(LaunchConfiguration('use_initial_pose')),
        parameters=[{
            'use_sim_time': use_sim_time,
            'initial_x': LaunchConfiguration('initial_x'),
            'initial_y': LaunchConfiguration('initial_y'),
            'initial_yaw': LaunchConfiguration('initial_yaw'),
            'timeout_sec': 180.0,
        }],
    ))

    for package, filename, enabled in [
        ('b2_nav_perception', 'rgbd_obstacle.launch.py', 'use_perception'),
        ('b2_nav_twin', 'local_world_model.launch.py', 'use_local_world_model'),
        ('b2_nav_safety', 'lidar_safety.launch.py', None),
    ]:
        include_kwargs = {}
        if enabled is not None:
            include_kwargs['condition'] = IfCondition(LaunchConfiguration(enabled))
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory(package), 'launch', filename
            )),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
            **include_kwargs,
        ))
    return LaunchDescription(actions)
