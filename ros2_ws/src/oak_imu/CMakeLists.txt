cmake_minimum_required(VERSION 3.22)
project(oak_imu)

# C++17 is a depthai-core v3 requirement, not a preference.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# ── DepthAI v3 ────────────────────────────────────────────────────────────────
# Installed on the board via ros-jazzy-depthai-v3, whose cmake config lives
# under the arch-specific lib directory rather than anywhere find_package looks
# by default. The training host has no OAK and no depthai package, so this
# whole package is skipped there rather than failing the workspace build --
# see the COLCON_IGNORE note in the README.
find_package(depthai_v3 QUIET CONFIG
  PATHS
    /opt/ros/$ENV{ROS_DISTRO}/lib/aarch64-linux-gnu/cmake/depthai_v3
    /opt/ros/$ENV{ROS_DISTRO}/lib/x86_64-linux-gnu/cmake/depthai_v3
)

if(NOT depthai_v3_FOUND)
  message(WARNING
    "oak_imu: depthai_v3 not found, skipping. Install ros-$ENV{ROS_DISTRO}-depthai-v3 "
    "to build the OAK-D IMU driver. Nothing else in the workspace depends on it.")
  ament_package()
  return()
endif()

add_executable(oak_imu_node src/oak_imu_node.cpp)

target_include_directories(oak_imu_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(oak_imu_node depthai_v3::core)
ament_target_dependencies(oak_imu_node rclcpp sensor_msgs)

install(TARGETS oak_imu_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
# config only. This package ships no launch files -- sim2edge_bringup owns
# launching, and this node is only ever started as part of the robot. The
# directory was listed here for a `launch/` that existed but was empty, which
# git does not track, so the build worked on a copied tree and failed on the
# first clean clone.
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
