// One control tick of the deployable stack, start to finish.
//
// This class exists so that there is exactly one implementation of the tick.
// The ROS node calls it; the replay parity harness calls it; a benchmark calls
// it. If the node had its own copy of the ordering -- deskew, bin, push
// history, localize, build goal features, infer -- then the parity test would
// prove something about the harness rather than about the robot.
//
// The stack it runs is the `--driver localized` arm from the simulator, which
// is the only configuration that answers "would the robot do this" rather than
// "is the actor good":
//
//     lidar sweep + gyro + base velocity
//       -> deskew into the current body frame
//       -> 180 normalized bins, pushed onto a 4-frame history
//       -> scan-to-submap tracker maintains a relative pose and its covariance
//       -> the goal is carried in the tracker's frame and re-projected to body
//       -> actor (NPU) -> (v, omega)
//
// **No ground truth enters at any point after the goal is set.** Somebody has
// to point at the map once; nothing drives on it afterwards. Arrival is decided
// by the robot's own uncertainty bound, not by distance to a true position.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sim2edge/actor.hpp"
#include "sim2edge/deskew.hpp"
#include "sim2edge/localizer.hpp"
#include "sim2edge/preprocess.hpp"
#include "sim2edge/types.hpp"

namespace sim2edge
{

class Navigator
{
public:
  struct Config
  {
    LocalPoseTracker::Config localizer;

    /// Arrival is declared when the believed distance *plus* its 1.96-sigma
    /// bound falls inside this radius, for `arrival_ticks` consecutive ticks.
    ///
    /// The bound rather than the point estimate is the whole design: a robot
    /// that stops when it merely believes it has arrived declares success from
    /// a confidently wrong pose. Requiring the bound to fit makes the claim
    /// falsifiable by the robot's own covariance.
    double arrival_radius_m = 0.15;
    int arrival_ticks = 3;

    /// A calibrated systematic scale term, kept separate from the scan
    /// matcher's random covariance because it accumulates linearly with travel
    /// rather than as a random walk.
    double travel_sigma_per_m = 0.01;

    /// Feed the localizer a body-frame odometry increment from the base
    /// instead of integrating gyro and velocity ourselves. The self-contained
    /// arm (false) is blind to wheel slip in the same way a wheel encoder is,
    /// and `motion_scale` from the scan match is the only thing that recovers
    /// it -- which puts the whole translation correction on the lidar.
    bool use_odometry_prior = false;
  };

  struct Result
  {
    Command command;
    Pose2 pose;              ///< tracker-frame pose, relative to the last reset
    double goal_body_x = 0.0;
    double goal_body_y = 0.0;
    double belief_distance_m = 0.0;
    double sigma_m = 0.0;
    double travelled_m = 0.0;
    bool observable = false;
    bool arrived = false;
    bool has_goal = false;
    int arrival_streak = 0;
    double motion_scale = 1.0;

    // Per-stage cost, microseconds. Published rather than merely logged: on
    // this robot the interesting number moves between stages depending on the
    // scene, and an aggregate hides that.
    double deskew_us = 0.0;
    double bin_us = 0.0;
    double localize_us = 0.0;
    double infer_us = 0.0;
    double total_us = 0.0;
  };

  Navigator();
  explicit Navigator(const Config & config);

  /// Attach a loaded actor. Without one, `tick` runs the whole perception and
  /// localization path and returns a zero command -- which is precisely what
  /// the preprocessing parity test wants, and what a board without the QNN
  /// runtime can still exercise.
  void setActor(std::shared_ptr<Actor> actor) { actor_ = std::move(actor); }
  bool hasActor() const { return actor_ != nullptr; }

  /// Set the goal as an offset in the robot's **current** body frame.
  ///
  /// Stored immediately in the tracker's frame and never revisited. This is the
  /// one and only point at which an external frame touches the stack.
  void setGoalBody(double x, double y);
  void clearGoal();

  /// Drop all state: history, submap, pose, goal, arrival streak. Anything that
  /// teleports or re-localizes the robot must call this.
  void reset();

  /// Replace the command the next tick will report as "previous".
  ///
  /// **For the replay parity harness only.** The observation's action block
  /// carries the previous command, so replaying against a recorded reference
  /// requires feeding the recorded command rather than whatever this instance
  /// last produced -- otherwise the two trajectories separate after one tick
  /// and the comparison measures closed-loop divergence instead of agreement.
  /// The robot never calls this: on hardware the previous command genuinely is
  /// the one this node issued.
  void overrideLastCommand(const Command & command) { last_command_ = command; }

  /// Run one control tick.
  ///
  /// `ranges` / `bearings` / `ages` describe the newest revolution, with
  /// bearings already in base_link (see `beamGeometry`). `gyro_delta` is the
  /// measured yaw increment over the tick and `speed` the base's forward
  /// velocity. `prior_delta_body`, when the configuration asks for it, is the
  /// base's own body-frame odometry increment (dx, dy, dyaw).
  const Result & tick(const float * ranges, const float * bearings,
                      const float * ages, std::size_t count, double gyro_delta,
                      double speed, double dt = spec::kDt,
                      const double * prior_delta_body = nullptr);

  const PolarSweep & sweep() const { return sweep_; }
  const std::vector<float> & bins() const { return bins_; }
  const Observation & observation() const { return observation_; }
  const LocalPoseTracker & localizer() const { return tracker_; }
  const Config & config() const { return config_; }

private:
  Config config_;
  std::shared_ptr<Actor> actor_;

  SweepDeskewer deskewer_;
  LocalPoseTracker tracker_;
  ObservationHistory history_;

  PolarSweep sweep_;
  std::vector<float> bins_;
  Observation observation_;
  Result result_;

  bool has_goal_{false};
  double goal_tracker_x_{0.0};
  double goal_tracker_y_{0.0};
  double travelled_m_{0.0};
  int arrival_streak_{0};
  Command last_command_;
  bool first_tick_{true};
};

}  // namespace sim2edge
