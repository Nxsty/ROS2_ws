import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get the package share directories
    my_robot_description_pkg = FindPackageShare(package='my_robot_normal_description').find('my_robot_normal_description')
    my_robot_bringup_pkg = FindPackageShare(package='my_robot_normal_bringup').find('my_robot_normal_bringup')
    ros_gz_sim_pkg = FindPackageShare(package='ros_gz_sim').find('ros_gz_sim')

    # File paths
    urdf_path = os.path.join(my_robot_description_pkg, 'urdf/my_robot.urdf.xacro')
    gazebo_config_path = os.path.join(my_robot_bringup_pkg, 'config/mecanum_gazebo_bridge.yaml')
    rviz_config_path = os.path.join(my_robot_description_pkg, 'rviz/urdf_config.rviz')
    world_path = os.path.join(my_robot_bringup_pkg, 'worlds/test_world.sdf')

    # Robot state publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', urdf_path])}]
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
        arguments=['-topic', 'robot_description', '-name', 'my_robot'],
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




    # Custom controller
    mecanum_controller_node = Node(
        package='my_robot_mecanum_controller',
        executable='mecanum_drive_controller',        output='screen',
        parameters=[{
            'wheel_radius': 0.1,
            'wheel_separation_x': 0.272,
            'wheel_separation_y': 0.215,
            'fl_wheel_topic': '/base_left_front_wheel_joint/cmd_vel',
            'fr_wheel_topic': '/base_right_front_wheel_joint/cmd_vel',
            'bl_wheel_topic': '/base_left_back_wheel_joint/cmd_vel',
            'br_wheel_topic': '/base_right_back_wheel_joint/cmd_vel'
        }]
    )

    return LaunchDescription([
        robot_state_publisher_node,
        gazebo_sim,
        spawn_robot_node,
        gazebo_bridge_node,
        rviz_node,
        mecanum_controller_node
    ])