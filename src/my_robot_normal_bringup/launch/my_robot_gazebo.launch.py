import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Get the package share directories
    my_robot_description_pkg = FindPackageShare(package='my_robot_normal_description').find('my_robot_normal_description')
    my_robot_bringup_pkg = FindPackageShare(package='my_robot_normal_bringup').find('my_robot_normal_bringup')
    ros_gz_sim_pkg = FindPackageShare(package='ros_gz_sim').find('ros_gz_sim')

    # Add the parent directory of the packages to GZ_SIM_RESOURCE_PATH
    # This allows Gazebo to find models using model://package_name/
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[os.path.dirname(my_robot_bringup_pkg)]
    )

    # File paths
    urdf_path = os.path.join(my_robot_description_pkg, 'urdf/my_robot.urdf.xacro')
    gazebo_config_path = os.path.join(my_robot_bringup_pkg, 'config/mecanum_gazebo_bridge.yaml')
    rviz_config_path = os.path.join(my_robot_description_pkg, 'rviz/urdf_config.rviz')
    world_path = os.path.join(my_robot_bringup_pkg, 'worlds/empty_world.sdf')

    # Robot state publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(Command(['xacro ', urdf_path]), value_type=str),
            'use_sim_time': True
        }]
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
        arguments=[
            '-topic', 'robot_description',
            '-name', 'my_robot',
            '-x', '-0.8116',
            '-y', '0.9690',
            '-z', '0.1',
            '-Y', '0.0'
        ],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Gazebo Bridge
    gazebo_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': gazebo_config_path,
            'use_sim_time': True
        }],
        output='screen'
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Diff Drive Controller
    diff_drive_controller_node = Node(
        package='my_robot_custom_controller',
        executable='diff_drive_controller',
        output='screen',
        parameters=[{
            'wheel_radius': 0.1,
            'wheel_separation': 0.47,
            'left_front_wheel_joint': 'base_left_front_wheel_joint',
            'left_back_wheel_joint': 'base_left_back_wheel_joint',
            'right_front_wheel_joint': 'base_right_front_wheel_joint',
            'right_back_wheel_joint': 'base_right_back_wheel_joint',
            'left_front_wheel_topic': '/base_left_front_wheel_joint/cmd_vel',
            'left_back_wheel_topic': '/base_left_back_wheel_joint/cmd_vel',
            'right_front_wheel_topic': '/base_right_front_wheel_joint/cmd_vel',
            'right_back_wheel_topic': '/base_right_back_wheel_joint/cmd_vel',
            'base_frame': 'base_footprint',
            'use_sim_time': True
        }]
    )

    return LaunchDescription([
        set_gz_resource_path,
        robot_state_publisher_node,
        gazebo_sim,
        spawn_robot_node,
        gazebo_bridge_node,
        rviz_node,
        diff_drive_controller_node
    ])