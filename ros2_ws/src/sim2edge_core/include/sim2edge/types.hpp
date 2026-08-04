// The data structures every stage of the on-board pipeline hands to the next.
//
// Everything is structure-of-arrays `std::vector<float>` rather than an array
// of point structs. Two reasons, both measured concerns rather than taste:
// the localizer's inner loop touches x and y of forty thousand submap points
// per ICP iteration and wants them contiguous, and every buffer here is reused
// across control ticks so that a 20 Hz loop performs no allocation in steady
// state.
//
// There is no torch, no ROS and no Eigen in this header. It is included by the
// parity harness, which must build and run on a laptop.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sim2edge/spec_generated.hpp"

namespace sim2edge
{

/// One lidar revolution, every return keeping its exact body-frame bearing.
///
/// The direct transcription of `sim2edge.sensors.PolarSweep`. Uniform angular
/// spacing is deliberately *not* assumed: the A1's beam count varies with spin
/// rate and its start angle is not stable across driver versions, so bearings
/// travel with the ranges rather than being reconstructed from an increment.
///
/// `ages` is seconds since each beam was acquired, zero at the beam the sweep
/// just crossed and rising to a full revolution at the beam ahead of it. That
/// gradient around the circle is the whole reason this type exists instead of a
/// bare range array.
struct PolarSweep
{
  std::vector<float> ranges;    ///< metres, in the *current* body frame
  std::vector<float> bearings;  ///< radians, base_link, wrapped to [-pi, pi)
  std::vector<float> ages;      ///< seconds since acquisition
  std::vector<uint8_t> valid;   ///< 1 where the return is usable

  std::size_t size() const { return ranges.size(); }

  void resize(std::size_t n)
  {
    ranges.resize(n);
    bearings.resize(n);
    ages.resize(n);
    valid.resize(n);
  }

  void clear()
  {
    ranges.clear();
    bearings.clear();
    ages.clear();
    valid.clear();
  }

  /// True when every field agrees on length. Cheap, and the one invariant that
  /// silently corrupts geometry if it is ever violated.
  bool consistent() const
  {
    return bearings.size() == ranges.size() && ages.size() == ranges.size() &&
           valid.size() == ranges.size();
  }
};

/// Planar pose: x, y in metres, yaw in radians.
struct Pose2
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

/// What one localizer update produced.
///
/// `observable` is the field that matters and it is not a confidence score: it
/// is a hard statement that the local geometry made all three planar degrees of
/// freedom identifiable this tick. A rejected scan still advances the prior --
/// the robot did move -- but its correction never masquerades as a lidar
/// measurement, and `covariance` keeps growing to say so.
struct LocalizerEstimate
{
  Pose2 pose;                  ///< relative to wherever the tracker was reset
  double covariance[9]{};      ///< row-major 3x3 over (x, y, yaw)
  double motion_scale{1.0};    ///< estimated drivetrain scale, slowly averaged
  bool observable{false};
  Pose2 correction;            ///< applied scan correction; zero when rejected
  int points_used{0};
  int iterations{0};
  std::string motion_prior_source{spec::kMotionPriorSource};
};

/// The flat 731-number vector the exported actor consumes.
///
/// Layout is fixed by spec.py and asserted there:
///     [ lidar(4, 180) | goal(3) | previous_action(4, 2) ]
/// oldest to newest within each history block. It is stored flat rather than as
/// three arrays because that is exactly the buffer the ExecuTorch tensor wraps,
/// so the inference call copies nothing.
struct Observation
{
  std::vector<float> data;

  Observation()
  : data(static_cast<std::size_t>(spec::kObsDim), 0.0f) {}

  float * lidar() { return data.data(); }
  float * goal() { return data.data() + spec::kGoalOffset; }
  float * action() { return data.data() + spec::kActionOffset; }

  const float * lidar() const { return data.data(); }
  const float * goal() const { return data.data() + spec::kGoalOffset; }
  const float * action() const { return data.data() + spec::kActionOffset; }
};

/// A physical command in the units the Twist carries: m/s and rad/s.
struct Command
{
  float v{0.0f};
  float omega{0.0f};
};

}  // namespace sim2edge
