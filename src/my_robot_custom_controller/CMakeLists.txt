cmake_minimum_required(VERSION 3.8)
project(my_robot_custom_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(diff_drive_controller src/diff_drive_controller.cpp)
add_executable(mecanum_drive_controller src/mecanum_drive_controller.cpp)
add_executable(pose_controller src/pose_controller.cpp)

ament_target_dependencies(diff_drive_controller
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  std_msgs
  tf2_ros
)

ament_target_dependencies(mecanum_drive_controller
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  std_msgs
  tf2_ros
)

ament_target_dependencies(pose_controller
  rclcpp
  geometry_msgs
  nav_msgs
  tf2_ros
)

install(TARGETS
  diff_drive_controller
  mecanum_drive_controller
  pose_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  scripts
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/world_oriented_teleop.py
  scripts/maze_rviz_visualizer.py
  scripts/generate_h5_dataset.py
  scripts/vio_odometry.py
  scripts/trajectory_manager.py
  scripts/plot_trajectories.py
  scripts/trajectory_generator.py
  scripts/mpc_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
