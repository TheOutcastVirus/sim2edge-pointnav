cmake_minimum_required(VERSION 3.22)
project(sim2edge_nodes)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -ffp-contract=off)

# Bake the resolved link directories into the installed binaries' RPATH.
#
# ExecuTorch's QNN delegate is a *shared* library (libqnn_executorch_backend.so)
# living outside any standard path, so without this every node dies at startup
# with "error while loading shared libraries" -- and it does so as exit code 127
# through ros2 launch, which reports it as "process has died" and nothing else.
# Requiring LD_LIBRARY_PATH to be right instead would make the stack depend on
# an environment variable that is easy to lose and silent when wrong.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sim2edge_core REQUIRED)
find_package(sim2edge_msgs REQUIRED)

set(DEPENDENCIES
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  std_srvs
  sim2edge_core
  sim2edge_msgs
)

# Three executables, not a component library.
#
# Composition would let these share a process and hand over messages without
# serializing them -- but nothing here wants that. The navigator already owns
# the whole tick internally, so no sweep ever crosses a node boundary, and the
# safety layer is deliberately a SEPARATE process precisely so that it survives
# the navigator dying. Building them as components would also pull ExecuTorch's
# static archives into a shared library, which requires every one of them to
# have been compiled -fPIC.
foreach(node navigator_node safety_node replay_node)
  add_executable(${node} src/${node}.cpp)
  target_include_directories(${node} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  ament_target_dependencies(${node} ${DEPENDENCIES})
endforeach()

install(TARGETS navigator_node safety_node replay_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_dependencies(${DEPENDENCIES})
ament_package()
