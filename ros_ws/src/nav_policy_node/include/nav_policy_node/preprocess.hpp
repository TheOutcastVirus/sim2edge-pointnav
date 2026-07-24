// Observation preprocessing -- the C++ half of a two-implementation contract.
//
// The other half is sim2edge/preprocess.py. tools/parity_check.py runs both on
// identical raw sensor data and asserts they agree to 1e-5 before any robot
// trial. Every constant used here comes from spec_generated.hpp, which is
// generated from sim2edge/spec.py -- do not introduce a literal.
//
// If you change anything in this file, change sim2edge/preprocess.py to match
// and re-run the parity check. A divergence here does not crash and does not
// log; it silently feeds the network inputs that differ from the ones the
// quantizer was calibrated on, and the robot drives confidently into a wall.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "nav_policy_node/spec_generated.hpp"

namespace nav_policy_node
{

/// Preprocessed observation, laid out as the exported graph expects.
struct Observation
{
  std::vector<float> depth;        // 1 * kDepthHeight * kDepthWidth
  std::vector<float> lidar;        // kLidarBins
  std::vector<float> goal;         // 3
  std::vector<float> last_action;  // 2

  Observation()
  : depth(spec::kDepthNumel, spec::kDepthInvalidFill),
    lidar(spec::kLidarBins, spec::kLidarInvalidFill),
    goal(spec::kGoalDim, 0.0f),
    last_action(spec::kActionDim, 0.0f) {}
};

/// Index boundaries mapping `n_in` samples onto `n_out` bins.
///
/// Mirrors sim2edge.preprocess._block_bounds, including the integer arithmetic:
/// `edge(i) = (i * n_in) / n_out` with truncating division. `end` is forced to
/// at least `start + 1` so upsampling degenerates to nearest-neighbour instead
/// of producing empty bins.
void blockBounds(int n_in, int n_out, std::vector<int> & starts,
                 std::vector<int> & ends);

/// Normalize raw metric depth to [0, 1] and min-pool onto the policy grid.
///
/// Downsampling is min-pooling over integer index blocks, NOT cv::resize.
/// Interpolating across a depth discontinuity invents a surface that exists in
/// neither the foreground nor the background, and min-pooling is the only
/// reduction that cannot delete a thin obstacle. Invalid pixels (non-positive,
/// NaN, infinite) normalize to kDepthInvalidFill before pooling, so "invalid"
/// and "far" reduce identically and there is no separate mask to keep in sync.
void preprocessDepth(const float * depth_m, int height, int width,
                     std::vector<float> & out);

/// Same, from a raw 16UC1 millimetre buffer as the OAK-D publishes it.
void preprocessDepthMillimetres(const uint16_t * depth_mm, int height, int width,
                                std::vector<float> & out);

/// Bin a raw LaserScan into kLidarBins normalized bins by min-pooling.
///
/// Beams are binned by their actual angle, not by index, so the result does not
/// depend on scan direction, start angle, or beam count. The RPLIDAR's start
/// angle is not stable across driver versions, and binning by index would
/// silently rotate the policy's entire view of the world.
///
/// `range_min` / `range_max` are the message's own validity limits; returns
/// outside them are treated as invalid rather than clamped.
void preprocessLidar(const float * ranges, std::size_t count, float angle_min,
                     float angle_increment, float range_min, float range_max,
                     std::vector<float> & out);

/// Goal in the ROBOT frame -> (normalized distance, sin bearing, cos bearing).
///
/// sin/cos rather than a raw angle so the input stays continuous across the
/// +/-pi wrap, which is exactly where the robot needs to turn around and
/// exactly where an INT8-quantized discontinuity would hurt most.
void preprocessGoal(float goal_x, float goal_y, std::vector<float> & out);

/// Previous physical command -> normalized [-1, 1]^2.
void preprocessLastAction(float v, float omega, std::vector<float> & out);

/// Transform an odom-frame goal into the robot frame.
void worldGoalToRobotFrame(float goal_x, float goal_y, float robot_x,
                           float robot_y, float robot_yaw,
                           float & out_x, float & out_y);

/// Wrap an angle to [-pi, pi). Matches Python's modulo semantics, which differ
/// from fmod for negative operands.
float wrapAngle(float a);

}  // namespace nav_policy_node
