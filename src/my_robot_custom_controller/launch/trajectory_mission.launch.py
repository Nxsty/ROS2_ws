from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    # Launch Arguments
    controller_type_arg = DeclareLaunchArgument(
        'controller_type',
        default_value='mpc',
        description='Controller type to use: "mpc" (Model Predictive Control) or "pose" (Legacy PID)'
    )
    controller_type = LaunchConfiguration('controller_type')

    trajectory_mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='spline',
        description='Trajectory mission mode: "spline" (continuous C^2 loop with cornering speed limit), "spline_stations" (spline with shelf dwell pauses), "stations", or "continuous"'
    )
    trajectory_mode = LaunchConfiguration('mode')

    heading_mode_arg = DeclareLaunchArgument(
        'heading_mode',
        default_value='face_travel',
        description='Heading mode: "face_travel", "fixed" (0 yaw strafe), or "face_stations"'
    )
    heading_mode = LaunchConfiguration('heading_mode')

    max_vel_arg = DeclareLaunchArgument(
        'max_velocity',
        default_value='1.0',
        description='Maximum trajectory cruising velocity (m/s)'
    )
    max_vel = LaunchConfiguration('max_velocity')

    max_accel_arg = DeclareLaunchArgument(
        'max_acceleration',
        default_value='0.8',
        description='Maximum trajectory acceleration (m/s^2)'
    )
    max_accel = LaunchConfiguration('max_acceleration')

    max_lat_accel_arg = DeclareLaunchArgument(
        'max_lateral_accel',
        default_value='0.6',
        description='Maximum lateral centripetal acceleration on curves (m/s^2)'
    )
    max_lat_accel = LaunchConfiguration('max_lateral_accel')

    dwell_time_arg = DeclareLaunchArgument(
        'dwell_time',
        default_value='2.0',
        description='Dwell time at each warehouse station (seconds)'
    )
    dwell_time = LaunchConfiguration('dwell_time')

    # Condition checks
    use_mpc = PythonExpression(["'", controller_type, "' == 'mpc'"])
    use_pose = PythonExpression(["'", controller_type, "' == 'pose'"])

    # -------------------------------------------------------------
    # 1. Advanced Model Predictive Controller (MPC) Stack
    # -------------------------------------------------------------
    mpc_controller_node = Node(
        package='my_robot_custom_controller',
        executable='mpc_controller.py',
        name='mpc_controller',
        output='screen',
        condition=IfCondition(use_mpc),
        parameters=[{
            'horizon': 15,
            'dt': 0.05,
            'max_linear_vel': 1.20,
            'max_lateral_vel': 0.40,
            'max_angular_vel': 1.50,
            'max_linear_accel': 1.00,
            'max_angular_accel': 1.80,
            'max_wheel_speed': 20.0,
            'wheel_radius': 0.10,
            'wheel_separation_x': 0.272,
            'wheel_separation_y': 0.225,
            'q_x': 35.0,
            'q_y': 40.0,
            'q_yaw': 25.0,
            'r_vx': 0.3,
            'r_vy': 1.2,
            'r_wz': 0.2,
            's_vx': 2.5,
            's_vy': 3.0,
            's_wz': 1.2,
            'q_terminal_mult': 2.0,
            'use_sim_time': True
        }]
    )

    trajectory_generator_node = Node(
        package='my_robot_custom_controller',
        executable='trajectory_generator.py',
        name='trajectory_generator',
        output='screen',
        condition=IfCondition(use_mpc),
        parameters=[{
            'mode': trajectory_mode,
            'heading_mode': heading_mode,
            'max_velocity': max_vel,
            'max_acceleration': max_accel,
            'max_lateral_accel': max_lat_accel,
            'max_angular_vel': 1.2,
            'max_angular_accel': 1.5,
            'dwell_time': dwell_time,
            'fillet_radius': 2.2,
            'dt': 0.05,
            'auto_start': True,
            'use_sim_time': True
        }]
    )

    # -------------------------------------------------------------
    # 2. Legacy Pose Controller & Waypoint Manager Stack
    # -------------------------------------------------------------
    pose_controller_node = Node(
        package='my_robot_custom_controller',
        executable='pose_controller',
        name='pose_controller',
        output='screen',
        condition=IfCondition(use_pose),
        parameters=[{
            'kp_x': 1.5,
            'kp_y': 1.5,
            'kp_yaw': 2.0,
            'dist_tolerance': 0.35,
            'yaw_tolerance': 0.10,
            'max_linear_vel': 1.20,
            'max_angular_vel': 1.50,
            'max_linear_accel': 1.0,
            'max_angular_accel': 1.8,
            'max_wheel_speed': 20.0,
            'control_yaw': True,
            'face_direction_of_travel': True,
            'use_sim_time': True
        }]
    )

    trajectory_manager_node = Node(
        package='my_robot_custom_controller',
        executable='trajectory_manager.py',
        name='trajectory_manager',
        output='screen',
        condition=IfCondition(use_pose),
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        controller_type_arg,
        trajectory_mode_arg,
        heading_mode_arg,
        max_vel_arg,
        max_accel_arg,
        max_lat_accel_arg,
        dwell_time_arg,
        mpc_controller_node,
        trajectory_generator_node,
        pose_controller_node,
        trajectory_manager_node
    ])
