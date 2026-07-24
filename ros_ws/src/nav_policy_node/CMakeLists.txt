cmake_minimum_required(VERSION 3.22)
project(nav_policy_node)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── ROS 2 ─────────────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# ── ExecuTorch ────────────────────────────────────────────────────────────────
# Point CMAKE_PREFIX_PATH at the ExecuTorch install, or set it explicitly:
#   colcon build --cmake-args -DEXECUTORCH_INSTALL_DIR=/opt/executorch
#
# Pin the ExecuTorch commit. The board bring-up patches /opt/executorch in two
# places (a fake QCS8300 chipset enum mapping to HTP V75, and enabling the QNN
# AOT pybind on aarch64) and both will break on an upgrade.
if(DEFINED EXECUTORCH_INSTALL_DIR)
  list(APPEND CMAKE_PREFIX_PATH ${EXECUTORCH_INSTALL_DIR})
endif()

# The native Qualcomm ExecuTorch build exports optional LLM/tokenizer targets
# even when a package only links the runtime module API. Predefining them lets
# ExecuTorch's package config load without dragging in the whole tokenizer
# dependency graph. Inherited verbatim from the reference project's
# collision_classifier/CMakeLists.txt:29-39.
foreach(_tokenizer_target
  tokenizers::regex_lookahead
  tokenizers::pcre2-8-static
  tokenizers::tokenizers
  tokenizers::re2
  tokenizers::sentencepiece-static
)
  if(NOT TARGET ${_tokenizer_target})
    add_library(${_tokenizer_target} INTERFACE IMPORTED)
  endif()
endforeach()

find_package(executorch REQUIRED)

# ── Backend selection ─────────────────────────────────────────────────────────
option(BUILD_QNN_BACKEND "Build the Qualcomm QNN/HTP NPU backend" ON)
# ON, unlike the reference project: the CPU path is this project's Gate-A
# fallback, so it has to actually work rather than being a dead parameter value.
option(BUILD_XNNPACK_BACKEND "Build the ExecuTorch XNNPACK CPU backend" ON)

if(BUILD_XNNPACK_BACKEND)
  foreach(_xnnpack_target xnnpack_backend XNNPACK xnnpack-microkernels-prod)
    if(NOT TARGET ${_xnnpack_target})
      message(FATAL_ERROR
        "BUILD_XNNPACK_BACKEND=ON requires an ExecuTorch build with "
        "EXECUTORCH_BUILD_XNNPACK=ON so backend:=cpu works as a live fallback. "
        "Missing target: ${_xnnpack_target}")
    endif()
  endforeach()
  if(COMMAND executorch_target_link_options_shared_lib)
    executorch_target_link_options_shared_lib(xnnpack_backend)
  endif()
endif()

# ── Preprocessing library ─────────────────────────────────────────────────────
# Separate target so the parity harness links exactly the code the node runs.
# If the parity tool had its own copy, the test would prove nothing.
add_library(nav_preprocess STATIC src/preprocess.cpp)
target_include_directories(nav_preprocess PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# ── Node ──────────────────────────────────────────────────────────────────────
set(NODE_SOURCES
  src/main.cpp
  src/nav_policy_node.cpp
  src/cpu_backend.cpp
)
if(BUILD_QNN_BACKEND)
  list(APPEND NODE_SOURCES src/qnn_backend.cpp)
endif()

add_executable(nav_policy_node ${NODE_SOURCES})

target_include_directories(nav_policy_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EXECUTORCH_INCLUDE_DIRS}
)

target_compile_definitions(nav_policy_node PRIVATE
  $<$<BOOL:${BUILD_QNN_BACKEND}>:HAVE_QNN_BACKEND>
)

target_link_libraries(nav_policy_node
  nav_preprocess
  executorch
  executorch_core
  extension_module_static
  extension_tensor
  portable_ops_lib
  $<$<BOOL:${BUILD_XNNPACK_BACKEND}>:xnnpack_backend>
  $<$<BOOL:${BUILD_XNNPACK_BACKEND}>:XNNPACK>
  $<$<BOOL:${BUILD_XNNPACK_BACKEND}>:xnnpack-microkernels-prod>
  $<$<BOOL:${BUILD_QNN_BACKEND}>:qnn_executorch_backend>
)

ament_target_dependencies(nav_policy_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
)

# ── Parity harness ────────────────────────────────────────────────────────────
# No ROS, no ExecuTorch: the parity gate can run on a laptop, not only on the
# board. See tools/parity_check.py.
add_executable(preprocess_parity src/preprocess_parity.cpp)
target_link_libraries(preprocess_parity nav_preprocess)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS nav_policy_node preprocess_parity
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_package()
