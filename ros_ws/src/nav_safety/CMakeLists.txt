cmake_minimum_required(VERSION 3.22)
project(nav_safety)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
# Headers only, for spec_generated.hpp: the safety thresholds must come from the
# same generated contract the policy and the simulator use, so that intervention
# counts measured in sim mean the same thing as on hardware. No linking, and in
# particular no ExecuTorch dependency -- this node must be buildable and
# runnable even when the inference stack is broken.
find_package(nav_policy_node REQUIRED)

add_executable(nav_safety_node src/nav_safety_node.cpp)

ament_target_dependencies(nav_safety_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  nav_policy_node
)

install(TARGETS nav_safety_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
