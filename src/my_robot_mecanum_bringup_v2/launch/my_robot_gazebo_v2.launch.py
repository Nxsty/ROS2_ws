import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions.set_environment_variable import SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch_ros.parameter_descriptions import ParameterValue

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression

def generate_launch_description():
    # Launch arguments
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='False',
        description='Run Gazebo in server-only (headless) mode'
    )
    headless = LaunchConfiguration('headless')

    # Get the package share directories
    my_robot_description_pkg = FindPackageShare(package='my_robot_mecanum_description_v2').find('my_robot_mecanum_description_v2')
    my_robot_bringup_pkg = FindPackageShare(package='my_robot_mecanum_bringup_v2').find('my_robot_mecanum_bringup_v2')
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
        parameters=[{'robot_description': ParameterValue(Command(['xacro ', urdf_path]), value_type=str)}]
    )

    # Gazebo - Pass '-s' if headless is true
    gz_args = PythonExpression([
        "'", world_path, " -r'", 
        " if '", headless, "' == 'False' else ",
        "'", world_path, " -r -s'"
    ])

    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args}.items()
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


    # Custom controller
    mecanum_controller_node = Node(
        package='my_robot_mecanum_controller',
        executable='mecanum_drive_controller',
        output='screen',
        parameters=[{
            'wheel_radius': 0.029,
            'wheel_separation_x': 0.224,
            'wheel_separation_y': 0.275,
            'fl_wheel_topic': '/rim_left_front_joint/cmd_vel',
            'fr_wheel_topic': '/rim_right_front_joint/cmd_vel',
            'bl_wheel_topic': '/rim_left_back_joint/cmd_vel',
            'br_wheel_topic': '/rim_right_back_joint/cmd_vel'
        }]
    )

    # Set GZ_SIM_RESOURCE_PATH for Gazebo to find meshes
    # We join paths with ':' as required by Linux environment variables
    description_pkg_share = os.path.dirname(my_robot_description_pkg)
    
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=description_pkg_share + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    )

    # Set GAZEBO_MODEL_PATH for Gazebo to find models (including meshes)
    set_gazebo_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=description_pkg_share + ':' + os.environ.get('GAZEBO_MODEL_PATH', '')
    )

    return LaunchDescription([
        headless_arg,
        set_gz_resource_path,
        set_gazebo_model_path,
        robot_state_publisher_node,
        gazebo_sim,
        spawn_robot_node,
        gazebo_bridge_node,
        rviz_node,
        mecanum_controller_node
    ])