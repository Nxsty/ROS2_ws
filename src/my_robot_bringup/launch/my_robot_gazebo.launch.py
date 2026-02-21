
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Get the package share directories
    my_robot_description_pkg = FindPackageShare(package='my_robot_description').find('my_robot_description')
    my_robot_bringup_pkg = FindPackageShare(package='my_robot_bringup').find('my_robot_bringup')
    ros_gz_sim_pkg = FindPackageShare(package='ros_gz_sim').find('ros_gz_sim')

    # File paths
    urdf_path = os.path.join(my_robot_description_pkg, 'urdf/my_robot.urdf.xacro')
    gazebo_config_path = os.path.join(my_robot_bringup_pkg, 'config/gazebo_bridge.yaml')
    rviz_config_path = os.path.join(my_robot_description_pkg, 'rviz/urdf_config.rviz')
    world_path = os.path.join(my_robot_bringup_pkg, 'worlds/test_world.sdf')

    # Robot state publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': ParameterValue(Command(['xacro ', urdf_path]), value_type=str)}]
    )

    # Gazebo
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': world_path + ' -r'}.items()
    )

    # Spawn robot
    spawn_robot_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description'],
        output='screen'
    )

    # Gazebo Bridge
    gazebo_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': gazebo_config_path}],
        output='screen'
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    my_robot_custom_controller_pkg = FindPackageShare(package='my_robot_custom_controller').find('my_robot_custom_controller')

    # Custom controller
    custom_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(my_robot_custom_controller_pkg, 'launch', 'my_robot_custom_controller.launch.py'))
    )

    return LaunchDescription([
        robot_state_publisher_node,
        gazebo_sim,
        spawn_robot_node,
        gazebo_bridge_node,
        rviz_node,
        custom_controller_launch
    ])
