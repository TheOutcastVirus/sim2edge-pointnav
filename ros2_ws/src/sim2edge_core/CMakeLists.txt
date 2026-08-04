cmake_minimum_required(VERSION 3.22)
project(sim2edge_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  # This library is the control loop. A default -O0 build would report a
  # localizer latency several times the real one and quietly invalidate every
  # measurement taken with it, so the default is Release rather than empty.
  set(CMAKE_BUILD_TYPE Release)
endif()

# -O3 with the A78C's baseline features. Deliberately NOT -march=native: the
# workspace is built on the board today, but the same source is built on the
# training host for the parity harness, and a host-tuned binary that silently
# uses AVX would make "same code path" a lie.
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


# -ffp-contract=off matters more than it looks. With contraction on, GCC fuses
# a*b+c into fma at its own discretion, which changes results in the last bits
# and, because the two builds are different compilers on different ISAs, changes
# them *differently* on host and board. The parity harness compares against
# Python at a tolerance; unpredictable fusion is exactly the kind of drift that
# makes a tolerance meaningless.

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

# ── ExecuTorch (optional) ─────────────────────────────────────────────────────
# Present on the board at /opt/executorch, absent on the training host. The
# preprocessing, deskew and localizer paths -- which is everything the parity
# test compares -- build and run without it.
#
#   colcon build --cmake-args -DEXECUTORCH_INSTALL_DIR=/opt/executorch/build-x86
#
# (`build-x86` is a native aarch64 build; the directory name is misleading and
# is inherited from the board's bring-up.)
option(BUILD_QNN_BACKEND "Link the Qualcomm QNN/HTP delegate" ON)
option(BUILD_XNNPACK_BACKEND "Link the ExecuTorch XNNPACK CPU backend" ON)

set(SIM2EDGE_HAVE_EXECUTORCH FALSE)
if(NOT DEFINED EXECUTORCH_INSTALL_DIR)
  set(EXECUTORCH_INSTALL_DIR "")
endif()
if(NOT EXECUTORCH_INSTALL_DIR STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH ${EXECUTORCH_INSTALL_DIR})

  # The native Qualcomm ExecuTorch build exports optional LLM/tokenizer targets
  # even when a consumer only links the runtime module API. Predefining them
  # lets ExecuTorch's package config load without dragging in the whole
  # tokenizer dependency graph. Inherited from the reference project's
  # collision_classifier/CMakeLists.txt.
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
  set(SIM2EDGE_HAVE_EXECUTORCH TRUE)

  # XNNPACK is detected, not demanded.
  #
  # The Ventuno's ExecuTorch at /opt/executorch was built without it, and that
  # is not a blocker: `portable_ops_lib` below is a complete CPU implementation
  # and is what the 5.15 ms figure in docs/daily/2026_08_04_log.md was measured
  # on -- 10.3% of the 50 ms period, against 0.15-0.18 ms on the NPU. So
  # backend:=cpu remains a live fallback either way; XNNPACK would only make it
  # a faster one, and the actor was never the constraint.
  #
  # Requiring it would mean rebuilding ExecuTorch on the board, and that build
  # carries two hand-applied patches (the QCS8300 -> HtpArch::V75 enum and the
  # aarch64 pybind guard) which a rebuild would be an opportunity to lose.
  if(BUILD_XNNPACK_BACKEND)
    set(_have_xnnpack TRUE)
    foreach(_xnnpack_target xnnpack_backend XNNPACK xnnpack-microkernels-prod)
      if(NOT TARGET ${_xnnpack_target})
        set(_have_xnnpack FALSE)
      endif()
    endforeach()
    if(NOT _have_xnnpack)
      message(STATUS
        "sim2edge_core: this ExecuTorch build has no XNNPACK; backend:=cpu "
        "will use portable kernels (measured at 5.15 ms on this board, 10.3% "
        "of the control period).")
      set(BUILD_XNNPACK_BACKEND OFF)
    elseif(COMMAND executorch_target_link_options_shared_lib)
      executorch_target_link_options_shared_lib(xnnpack_backend)
    endif()
  endif()

  # The QNN delegate, likewise: absent means the NPU path cannot be built, which
  # is worth saying out loud rather than discovering as a link error.
  if(BUILD_QNN_BACKEND AND NOT TARGET qnn_executorch_backend)
    message(WARNING
      "sim2edge_core: this ExecuTorch build exports no qnn_executorch_backend "
      "target, so the NPU path is unavailable and backend:=qnn will fail to "
      "initialize its delegate at load. Building the CPU path only.")
    set(BUILD_QNN_BACKEND OFF)
  endif()
else()
  message(STATUS
    "sim2edge_core: no EXECUTORCH_INSTALL_DIR -- building without an "
    "inference runtime. Preprocessing, deskew and the localizer still build; "
    "loadActor() will report that it cannot execute a .pte.")
endif()

# ── Library ───────────────────────────────────────────────────────────────────
add_library(sim2edge_core
  src/preprocess.cpp
  src/deskew.cpp
  src/point_grid.cpp
  src/localizer.cpp
  src/navigator.cpp
  src/replay.cpp
  src/actor.cpp
)

target_include_directories(sim2edge_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(sim2edge_core PUBLIC Eigen3::Eigen)

if(SIM2EDGE_HAVE_EXECUTORCH)
  target_compile_definitions(sim2edge_core PUBLIC SIM2EDGE_HAVE_EXECUTORCH)
  target_include_directories(sim2edge_core PUBLIC ${EXECUTORCH_INCLUDE_DIRS})
  target_link_libraries(sim2edge_core PUBLIC
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
  target_link_libraries(sim2edge_core PUBLIC ${CMAKE_DL_LIBS})
endif()

# ── Standalone tools ──────────────────────────────────────────────────────────
# No ROS and (unless ExecuTorch is present) no accelerator: these run on a
# laptop. A parity gate that can only be executed on the robot is a parity gate
# nobody runs.
add_executable(sim2edge_parity tools/parity_main.cpp)
target_link_libraries(sim2edge_parity sim2edge_core)

add_executable(sim2edge_bench tools/bench_main.cpp)
target_link_libraries(sim2edge_bench sim2edge_core)

install(TARGETS sim2edge_core
  EXPORT export_sim2edge_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS sim2edge_parity sim2edge_bench
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_sim2edge_core HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)
ament_export_include_directories(include)

# sim2edge_core is a static archive, so a consumer that links it must also
# resolve ExecuTorch itself. The extras file does that on their behalf, with
# the install prefix baked in at build time.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sim2edge_core-extras.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/sim2edge_core-extras.cmake
  @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sim2edge_core-extras.cmake
  DESTINATION share/${PROJECT_NAME}/cmake)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest QUIET)
  if(ament_cmake_gtest_FOUND)
    ament_add_gtest(test_sim2edge_core
      test/test_preprocess.cpp
      test/test_point_grid.cpp
      test/test_deskew.cpp
    )
    if(TARGET test_sim2edge_core)
      target_link_libraries(test_sim2edge_core sim2edge_core)
    endif()
  endif()
endif()

ament_package(CONFIG_EXTRAS
  "${CMAKE_CURRENT_BINARY_DIR}/sim2edge_core-extras.cmake")
