cmake_minimum_required(VERSION 3.22)
project(sim2edge_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
