// Motion deskew: undo the smear a spinning lidar bakes into every revolution.
//
// The measured A1 turns at 7.32 Hz, so one revolution takes 137 ms -- 2.7 control
// periods at 20 Hz. The beams in one "scan" were therefore acquired at three or
// four different robot poses, and presenting them all at their nominal body
// bearing is exactly the error the hardware makes. The smear is a function of
// the command just issued, so it is largest precisely when the robot turns hard
// to avoid something: a systematic, action-correlated sim-to-real gap rather
// than noise, and it cannot be averaged away.
//
// This is the C++ counterpart of `sim2edge.sensors.deskew_history`, and it is
// fixed, non-learned preprocessing in the same category as binLidar. It does
// not make the policy less end-to-end; it declines to spend network capacity
// relearning a rotation matrix that is exactly known.
//
// One asymmetry worth stating, because it is the difference between this file
// and its Python sibling. In simulation a sweep must first be *reconstructed*
// (`sensors.sweep_scan` stitches one revolution out of stored instantaneous
// scans, because Isaac produces a snapshot). On hardware there is nothing to
// reconstruct: the driver hands over a real revolution with real per-beam
// acquisition times. Only the correction half of the sim's pipeline has a job
// here, and it is this one.

#pragma once

#include <cstddef>
#include <vector>

#include "sim2edge/spec_generated.hpp"
#include "sim2edge/types.hpp"

namespace sim2edge
{

/// Rolling motion history plus the deskew transform built from it.
///
/// Deskew spans a full lidar revolution, which is longer than the actor's
/// four-frame feature history. The two contracts are kept separate here for the
/// same reason they are separate in the environment: changing the actor's
/// history length must not silently shorten deskew.
class SweepDeskewer
{
public:
  SweepDeskewer();

  /// Record the motion over the control tick that just ended.
  ///
  /// `gyro_increment_rad` is the measured yaw change; `forward_speed_mps` is
  /// the body-frame forward velocity. Translation is corrected from velocity
  /// and rotation from the gyro -- the two things the robot actually has. The
  /// residual left behind is precisely the wheel slip the localizer exists to
  /// estimate, not a deskew artifact.
  void pushMotion(double gyro_increment_rad, double forward_speed_mps);

  /// Forget the history. Anything that teleports the robot must call this, or
  /// the next revolution is corrected with motion that never happened.
  void reset();

  /// True once a full revolution of motion history has accumulated. Before
  /// that the transform is built from zeros, which under-corrects rather than
  /// mis-corrects -- the honest failure of the two.
  bool ready() const { return filled_ >= spec::kSpinTicks; }

  /// Deskew one revolution into the current body frame.
  ///
  /// `bearings` are nominal body-frame bearings with the lidar's quarter-turn
  /// mount rotation **already applied**; `ages` are seconds since acquisition,
  /// zero at the beam the sweep just crossed. `out` is resized to `count` and
  /// carries corrected ranges, corrected bearings and the validity mask.
  ///
  /// `subtick` interpolates the accumulated transform at each beam's true age
  /// rather than rounding to the nearest control tick. These quantities are
  /// integrals of velocity, so they are smooth in time and linear interpolation
  /// is exact to second order -- there is no depth-edge analogue to the hazard
  /// that makes range interpolation dangerous. It matches the simulator's
  /// `subtick=True` path, which is what the deployed checkpoint trained on.
  void deskew(const float * ranges, const float * bearings, const float * ages,
              std::size_t count, PolarSweep & out, bool subtick = true) const;

private:
  /// Accumulated acquisition->current transform for `back` control ticks of
  /// beam age, for back in [0, kSpinTicks).
  struct Transform
  {
    double dx[spec::kSpinTicks]{};
    double dy[spec::kSpinTicks]{};
    double yaw[spec::kSpinTicks]{};
  };

  Transform buildTransform() const;

  // Ring buffers, newest written at `head_`. kSpinTicks entries is one
  // revolution plus the partial tick at each end.
  double gyro_[spec::kSpinTicks]{};
  double speed_[spec::kSpinTicks]{};
  int head_{0};
  int filled_{0};
};

/// Convert a LaserScan-style beam layout into body-frame bearings and ages.
///
/// `angle_min` / `angle_increment` are the driver's own, in the **lidar** frame;
/// the quarter-turn mount rotation (spec::kLidarMountYaw) is applied here, once,
/// so that no downstream stage has to remember it.
///
/// Ages come from the beam index and `time_increment`: beam i was acquired
/// `(count - 1 - i) * time_increment + trailing_age_s` seconds ago, where
/// `trailing_age_s` is the delay between the last beam of the revolution and
/// the control tick being served. A driver that reports time_increment as zero
/// (some do) falls back to spreading kSpinPeriodS evenly across the beams,
/// which is the correct model for a constant-rate motor.
void beamGeometry(std::size_t count, float angle_min, float angle_increment,
                  float time_increment, double trailing_age_s,
                  std::vector<float> & bearings, std::vector<float> & ages);

}  // namespace sim2edge
