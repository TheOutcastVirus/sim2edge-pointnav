// CPU local pose tracking from exact-bearing lidar sweeps.
//
// A C++ port of `sim2edge.localization.LocalPoseTracker`, and the reason this
// workspace exists in C++ at all: on the board the Python tracker costs 21-27 ms
// of a 50 ms control period against the actor's 0.15-0.18 ms on the NPU. The
// localizer is roughly 150x the model. Every remaining latency question on this
// robot is a CPU scan-matching question.
//
// The algorithm is unchanged and deliberately so. Gyro and measured velocity
// form a prior; robust point-to-plane ICP against a rolling submap supplies a
// correction *only* when the local geometry makes all three planar degrees of
// freedom observable. A rejected scan still advances the prior -- the robot did
// move -- but never masquerades as a lidar estimate, and the covariance grows
// to say so.
//
// What changed is how the same arithmetic is executed:
//
//   * scipy's cKDTree, rebuilt over ~46,000 points every tick, becomes a
//     uniform bucket grid (see point_grid.hpp) whose build is O(N) and whose
//     radius-limited queries are exact for the rejection radius ICP already
//     enforces;
//   * surface normals are memoized per submap point for the duration of a tick
//     instead of recomputed from a fresh 7-NN query on every ICP iteration.
//     Successive iterations re-match largely the same submap points, so this is
//     pure memoization -- identical values, roughly a fifth of the queries;
//   * submap points live in one contiguous float array rebuilt from the scan
//     ring each tick, so the inner loop walks memory linearly;
//   * the 2x2 normal eigenproblem is solved in closed form rather than through
//     a general symmetric eigensolver.
//
// The pose state stays in double. It is an integrator over hundreds of ticks,
// and float there would accumulate rounding into exactly the drift term this
// tracker exists to measure. Point clouds are float: they are measurements with
// centimetre noise, and halving their memory traffic is most of the speedup.

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "sim2edge/point_grid.hpp"
#include "sim2edge/spec_generated.hpp"
#include "sim2edge/types.hpp"

namespace sim2edge
{

class LocalPoseTracker
{
public:
  /// Defaults are the production configuration measured in
  /// docs/daily/2026_08_04_log.md: 128 submap scans at point_stride 2.
  struct Config
  {
    int max_submap_scans = 128;
    float max_correspondence_m = 0.20f;
    int min_points = 60;
    double min_information = 25.0;
    double max_condition = 2.0e4;
    /// A full per-sweep correction turns independent centimetre-scale range
    /// noise into a pose random walk. The submap error actually being chased is
    /// slowly accumulated drivetrain scale, so the correction is applied as a
    /// conservative complementary filter; gyro and velocity remain the fast
    /// prior.
    double correction_gain = 0.10;
    double yaw_correction_gain = 0.05;
    double max_correction_m = 0.12;
    double max_correction_yaw_rad = 0.06981317007977318;  // 4 degrees
    double motion_scale_update_rate = 0.02;
    double max_yaw_departure_rad = 1.0e300;               // effectively off
    int point_stride = 2;
    int max_iterations = 6;
    /// Returns beyond this contribute little local constraint and make
    /// nearest-neighbour association needlessly ambiguous. Deliberately tighter
    /// than the policy's kLidarMaxM: the two consume the same scan for
    /// different purposes.
    float point_max_range_m = 8.0f;
    float point_min_range_m = 0.15f;
  };

  /// Where the time went, in microseconds. Populated on every update so the
  /// node can publish it without a second code path -- the cost of reading a
  /// steady clock four times is far below the noise floor of what it measures.
  struct Timings
  {
    double extract_us = 0.0;   ///< sweep -> strided body-frame points
    double submap_us = 0.0;    ///< flatten the scan ring + build the grid
    double icp_us = 0.0;       ///< the correspondence/solve iterations
    double total_us = 0.0;
  };

  // Two constructors rather than a defaulted argument: a nested type's default
  // member initializers are not available until the enclosing class is
  // complete, so `= Config{}` in this declaration does not compile.
  LocalPoseTracker();
  explicit LocalPoseTracker(const Config & config);

  void reset();

  /// Advance one control tick.
  ///
  /// `gyro_delta` is the measured yaw increment over the tick and `speed` the
  /// body-frame forward velocity. When `prior_delta_body` is null the prior is
  /// integrated from those two directly -- the self-contained "command-gyro"
  /// arm, blind to wheel slip in the same way a wheel encoder is, with
  /// `motion_scale` the only thing that recovers it. When it is supplied
  /// (pointing at three doubles: dx, dy, dyaw in the body frame) that
  /// increment is used instead, which is the Create 3 odometry path.
  const LocalizerEstimate & update(const PolarSweep & sweep, double gyro_delta,
                                   double speed, double dt = spec::kDt,
                                   const double * prior_delta_body = nullptr);

  const LocalizerEstimate & estimate() const { return estimate_; }
  const Timings & timings() const { return timings_; }
  const Config & config() const { return config_; }

  /// Number of points in the current submap. Zero right after a reset, rising
  /// to roughly `max_submap_scans` times the per-scan point count.
  std::size_t submapSize() const { return submap_x_.size(); }

private:
  /// Sweep -> body-frame points, filtered and strided.
  void extractPoints(const PolarSweep & sweep);
  /// Flatten the scan ring into contiguous arrays and (re)build the grid.
  void rebuildSubmap();
  /// Append the current scan, already transformed into the tracker frame.
  void addScan();
  /// Memoized unit normal of the local surface at submap point `index`.
  void normalAt(int index, float & nx, float & ny);

  Config config_;
  LocalizerEstimate estimate_;
  Timings timings_;

  Pose2 pose_;
  double covariance_[9]{};
  double motion_scale_{1.0};
  double gyro_yaw_{0.0};

  // Current scan in the sensor's own frame.
  std::vector<float> scan_x_;
  std::vector<float> scan_y_;

  // Rolling submap: one entry per retained scan, each already transformed into
  // the tracker frame at the pose that observed it.
  struct Scan
  {
    std::vector<float> x;
    std::vector<float> y;
  };
  std::deque<Scan> scans_;

  // Flattened submap and its index, rebuilt each tick.
  std::vector<float> submap_x_;
  std::vector<float> submap_y_;
  PointGrid grid_;

  // Per-tick normal cache, indexed by submap point.
  std::vector<float> normal_x_;
  std::vector<float> normal_y_;
  std::vector<uint8_t> normal_ready_;

  // ICP scratch, reused across ticks so steady state allocates nothing.
  std::vector<float> placed_x_;
  std::vector<float> placed_y_;
  std::vector<float> residual_;
};

}  // namespace sim2edge
