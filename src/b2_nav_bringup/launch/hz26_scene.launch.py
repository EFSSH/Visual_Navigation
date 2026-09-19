"""Start only the HZ26 Gazebo world for scene inspection."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.conditions import UnlessCondition
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)


def generate_launch_description():
    model_home = LaunchConfiguration('model_home')
    world = LaunchConfiguration('world')
    headless = LaunchConfiguration('headless')
    default_world = PathJoinSubstitution(
        [model_home, 'hz26_station', 'worlds', 'HZ26_station.world']
    )

    gazebo_model_path = [
        model_home,
        ':',
        '/usr/share/gazebo-11/models',
        ':',
        EnvironmentVariable('GAZEBO_MODEL_PATH', default_value=''),
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'model_home',
                default_value=EnvironmentVariable(
                    'HZ26_MODEL_HOME',
                    default_value=[EnvironmentVariable('HOME'), '/b2_nav_ws/sim_assets'],
                ),
            ),
            DeclareLaunchArgument('world', default_value=default_world),
            DeclareLaunchArgument('headless', default_value='False'),
            SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),
            SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
            LogInfo(msg='Starting the HZ26 Gazebo scene without a robot.'),
            ExecuteProcess(
                cmd=[
                    'gzserver',
                    '-s',
                    'libgazebo_ros_init.so',
                    '-s',
                    'libgazebo_ros_factory.so',
                    '--verbose',
                    world,
                ],
                output='screen',
            ),
            ExecuteProcess(
                condition=UnlessCondition(headless),
                cmd=['gzclient'],
                output='screen',
            ),
        ]
    )
