import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Launch configuration
    use_vio_arg = DeclareLaunchArgument(
        'use_vio',
        default_value='false',
        description='Whether to use Visual-Inertial Odometry instead of wheel encoders'
    )
    use_vio = LaunchConfiguration('use_vio')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='open_textured_world.sdf',
        description='World file name under worlds/'
    )
    world = LaunchConfiguration('world')

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
    world_path = PathJoinSubstitution([my_robot_bringup_pkg, 'worlds', world])

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
        launch_arguments={'gz_args': [world_path, ' -r']}.items()
    )

    # Spawn coordinates
    spawn_x_arg = DeclareLaunchArgument('spawn_x', default_value='0.0', description='Robot spawn X')
    spawn_y_arg = DeclareLaunchArgument('spawn_y', default_value='0.0', description='Robot spawn Y')
    spawn_z_arg = DeclareLaunchArgument('spawn_z', default_value='0.1', description='Robot spawn Z')
    spawn_yaw_arg = DeclareLaunchArgument('spawn_yaw', default_value='0.0', description='Robot spawn Yaw')

    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    # Spawn robot
    spawn_robot_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'my_robot',
            '-x', spawn_x,
            '-y', spawn_y,
            '-z', spawn_z,
            '-Y', spawn_yaw
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

    # Diff Drive Controller (Standard Encoder Mode)
    diff_drive_controller_node = Node(
        package='my_robot_custom_controller',
        executable='diff_drive_controller',
        output='screen',
        condition=UnlessCondition(use_vio),
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
            'publish_odom': True,
            'publish_tf': True,
            'use_sim_time': True
        }]
    )

    # Diff Drive Controller (VIO Actuator-only Mode)
    diff_drive_controller_vio_mode_node = Node(
        package='my_robot_custom_controller',
        executable='diff_drive_controller',
        output='screen',
        condition=IfCondition(use_vio),
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
            'publish_odom': False,
            'publish_tf': False,
            'use_sim_time': True
        }]
    )

    # VIO Odometry Node (Active when use_vio:=true)
    vio_odometry_node = Node(
        package='my_robot_custom_controller',
        executable='vio_odometry.py',
        output='screen',
        condition=IfCondition(use_vio),
        parameters=[{
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'publish_tf': True,
            'use_sim_time': True
        }]
    )

    return LaunchDescription([
        use_vio_arg,
        world_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_z_arg,
        spawn_yaw_arg,
        set_gz_resource_path,
        robot_state_publisher_node,
        gazebo_sim,
        spawn_robot_node,
        gazebo_bridge_node,
        rviz_node,
        diff_drive_controller_node,
        diff_drive_controller_vio_mode_node,
        vio_odometry_node
    ])