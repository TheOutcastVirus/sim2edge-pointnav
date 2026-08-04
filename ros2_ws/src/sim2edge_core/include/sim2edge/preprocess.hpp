// Raw sensor readings -> the actor's input vector. The C++ half of a
// two-implementation contract whose other half is sim2edge/preprocess.py.
//
// tools/export_replay.py records what Python produced; sim2edge_parity replays
// it through this code and asserts agreement. A divergence here does not crash
// and does not log -- it feeds the network inputs that differ from the ones the
// quantizer was calibrated on, and the robot drives confidently into a wall.
//
// Two choices below look arbitrary and neither is:
//
// **Beams are binned by wrapped bearing, never by array index.** The RPLIDAR's
// start angle is not stable across driver versions and its beam count varies
// with spin rate. Index binning would silently rotate the policy's entire view
// of the world the day the driver was upgraded.
//
// **Ranges are normalized before pooling, not after.** "Invalid" and "far" both
// map to kInvalidFill, so pooling needs no companion mask -- there is no second
// data structure to keep in sync between the two implementations.

#pragma once

#include <cstddef>
#include <vector>

#include "sim2edge/spec_generated.hpp"
#include "sim2edge/types.hpp"

namespace sim2edge
{

/// Wrap to [-pi, pi).
///
/// Written with floor rather than fmod on purpose. C's `fmod` keeps the sign of
/// the dividend and Python's `%` keeps the sign of the divisor, and the bin
/// index depends on which. Getting this wrong is an off-by-one-hemisphere error
/// that presents as a badly trained policy.
float wrapAngle(float a);
double wrapAngle(double a);

/// Bin raw ranges + bearings into kLidarBins normalized values in [0, 1].
///
/// Reduction is **minimum**, the only choice that cannot delete a thin
/// obstacle: a table leg covering one beam in ten survives to the policy, where
/// averaging would blend it into the wall behind it.
///
/// Bins with no beam in them come out as kInvalidFill -- "nothing out to max
/// range" -- which is also what an out-of-range or non-finite return becomes.
/// Absent data and distant data are the same statement to the policy.
///
/// `out` is resized to kLidarBins. Passing a buffer that is already that size
/// costs no allocation, which is the steady-state path.
void binLidar(const float * ranges, const float * bearings, std::size_t count,
              std::vector<float> & out);

/// Same reduction over a PolarSweep, honouring its validity flags.
void binLidar(const PolarSweep & sweep, std::vector<float> & out);

/// Goal offset already in the robot frame -> (distance/kGoalMaxM, sin, cos).
///
/// sin/cos rather than a raw angle: an angle is discontinuous at +/-pi, exactly
/// where the robot needs to turn around, and quantizing a wrapping input to
/// INT8 fails silently.
///
/// Distance is clamped at 1.0 rather than left to grow. A goal beyond
/// kGoalMaxM is "far away in that direction"; the exact number carries nothing
/// the policy can act on, and an unbounded input quantizes badly.
///
/// The epsilon guards atan2(0, 0) at the instant the robot stands on the goal.
/// Direction is meaningless there and the distance feature is already zero, so
/// any finite value is as good as another. NaN is not.
void relativeGoalFeatures(float goal_x, float goal_y, float * out3);

/// World-frame goal and robot pose -> the same three features.
///
/// The rotation happens here rather than by subtracting yaw from a bearing
/// later, so that this entry point and the belief path share one encoder.
void goalFeatures(double goal_x, double goal_y, const Pose2 & robot,
                  float * out3);

/// Rotate a world-frame offset into the robot's body frame.
void toBodyFrame(double dx, double dy, double yaw, double & out_x,
                 double & out_y);

/// The actor's rolling input history: kHistoryFrames of lidar bins and actions.
///
/// Mirrors `sim2edge.preprocess.ObservationHistory`, including its reset rule:
/// the first frame after a reset is *repeated* across the whole history rather
/// than left zeroed. Zeros denormalize to an obstacle 15 cm dead ahead in every
/// bin, so a zero-filled history is not a neutral start -- it is a wall.
class ObservationHistory
{
public:
  ObservationHistory();

  /// Append one frame. `action` is the previous *command*, normalized to
  /// [-1, 1] by kVMax / kOmegaMax the way the trainer stored it.
  void push(const std::vector<float> & lidar_bins, float norm_v,
            float norm_omega);

  /// Repeat the next pushed frame across the whole history.
  void reset() { initialized_ = false; }

  bool initialized() const { return initialized_; }

  /// Write [lidar | goal | action] into `obs`. `goal3` is the output of
  /// relativeGoalFeatures.
  void fill(const float * goal3, Observation & obs) const;

private:
  // Ring buffer rather than a memmove of 720 floats per tick. `head_` indexes
  // the oldest frame, so frame k of the flat layout is (head_ + k) % frames.
  std::vector<float> lidar_;   // kHistoryFrames * kLidarBins
  std::vector<float> action_;  // kHistoryFrames * 2
  int head_{0};
  bool initialized_{false};
};

/// Unbounded head output -> bounded SI (v, omega).
///
/// Transcribes `nav_policy.physical_action`. Only needed when the exported
/// graph stops at the pre-squash logit; the production actor.pte already
/// applies this, so the policy node does not call it. It exists for the parity
/// harness and for a CPU reference path.
Command physicalAction(float raw_v, float raw_omega);

}  // namespace sim2edge
