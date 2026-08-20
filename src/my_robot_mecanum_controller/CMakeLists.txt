cmake_minimum_required(VERSION 3.8)
project(my_robot_mecanum_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(mecanum_drive_controller src/mecanum_drive_controller.cpp)

ament_target_dependencies(mecanum_drive_controller
  rclcpp
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
  tf2_ros
)

install(TARGETS
  mecanum_drive_controller
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
