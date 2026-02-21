import os
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Get the package share directories
    my_robot_description_pkg = FindPackageShare(package='my_robot_mecanum_description_v2').find('my_robot_mecanum_description_v2')
    
    # File paths
    urdf_path = os.path.join(my_robot_description_pkg, 'urdf/my_robot.urdf.xacro')
    rviz_config_path = os.path.join(my_robot_description_pkg, 'rviz/urdf_config.rviz')

    # Robot state publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(
                Command(['xacro ', urdf_path]), value_type=str
            )
        }]
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    # Joint state publisher GUI
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
    )


    return LaunchDescription([
        robot_state_publisher_node,
        rviz_node,
        joint_state_publisher_gui_node,
    ])