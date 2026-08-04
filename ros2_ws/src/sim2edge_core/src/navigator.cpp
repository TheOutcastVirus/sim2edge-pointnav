#include "sim2edge/navigator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <Eigen/Dense>

namespace sim2edge
{
namespace
{
using Clock = std::chrono::steady_clock;

double micros(Clock::time_point a, Clock::time_point b)
{
  return std::chrono::duration<double, std::micro>(b - a).count();
}
}  // namespace

Navigator::Navigator()
: Navigator(Config())
{
}

Navigator::Navigator(const Config & config)
: config_(config), tracker_(config.localizer)
{
  bins_.assign(static_cast<std::size_t>(spec::kLidarBins), spec::kInvalidFill);
}

void Navigator::reset()
{
  deskewer_.reset();
  tracker_.reset();
  history_.reset();
  has_goal_ = false;
  goal_tracker_x_ = goal_tracker_y_ = 0.0;
  travelled_m_ = 0.0;
  arrival_streak_ = 0;
  last_command_ = Command{};
  first_tick_ = true;
  result_ = Result{};
}

void Navigator::setGoalBody(double x, double y)
{
  // Rotate the body-frame offset into the tracker frame at the pose we believe
  // we are in right now, and store it there. From this instant the goal is a
  // fixed point in the robot's own coordinate system and drifts with it -- the
  // localization error is exactly the gap between where the goal is and where
  // the robot thinks it is.
  const Pose2 & pose = tracker_.estimate().pose;
  const double c = std::cos(pose.yaw);
  const double s = std::sin(pose.yaw);
  goal_tracker_x_ = pose.x + c * x - s * y;
  goal_tracker_y_ = pose.y + s * x + c * y;
  has_goal_ = true;
  arrival_streak_ = 0;
}

void Navigator::clearGoal()
{
  has_goal_ = false;
  arrival_streak_ = 0;
}

const Navigator::Result & Navigator::tick(
  const float * ranges, const float * bearings, const float * ages,
  std::size_t count, double gyro_delta, double speed, double dt,
  const double * prior_delta_body)
{
  const Clock::time_point t0 = Clock::now();

  // The very first tick has no motion history and no previous pose, so it must
  // not apply one: a fabricated first increment shows up as a permanent offset
  // in a tracker whose whole output is an integral.
  const double tick_gyro = first_tick_ ? 0.0 : gyro_delta;
  const double tick_speed = first_tick_ ? 0.0 : speed;

  deskewer_.pushMotion(tick_gyro, tick_speed);
  deskewer_.deskew(ranges, bearings, ages, count, sweep_, /*subtick=*/true);
  const Clock::time_point t1 = Clock::now();

  binLidar(sweep_, bins_);
  // The action block carries the previous *command*, normalized the way the
  // trainer stored it. Pushed before inference, so the history the actor reads
  // ends at the command it last issued.
  history_.push(bins_, last_command_.v / spec::kVMax,
                last_command_.omega / spec::kOmegaMax);
  const Clock::time_point t2 = Clock::now();

  const double * prior = (config_.use_odometry_prior && !first_tick_)
                           ? prior_delta_body
                           : nullptr;
  const LocalizerEstimate & estimate =
    tracker_.update(sweep_, tick_gyro, tick_speed, dt, prior);
  const Clock::time_point t3 = Clock::now();

  travelled_m_ += std::abs(tick_speed) * dt;

  // ── Where the robot believes the goal is, in its own body frame ──────────
  float goal3[3] = {1.0f, 0.0f, 1.0f};
  double goal_body_x = 0.0;
  double goal_body_y = 0.0;
  if (has_goal_) {
    toBodyFrame(goal_tracker_x_ - estimate.pose.x,
                goal_tracker_y_ - estimate.pose.y, estimate.pose.yaw,
                goal_body_x, goal_body_y);
    relativeGoalFeatures(static_cast<float>(goal_body_x),
                         static_cast<float>(goal_body_y), goal3);
  }
  history_.fill(goal3, observation_);

  // ── Arrival, decided by the robot's own uncertainty ──────────────────────
  const double belief_distance =
    std::sqrt(goal_body_x * goal_body_x + goal_body_y * goal_body_y);

  Eigen::Matrix2d planar;
  planar << estimate.covariance[0], estimate.covariance[1],
            estimate.covariance[3], estimate.covariance[4];
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(
    planar, Eigen::EigenvaluesOnly);
  const double largest = solver.eigenvalues()(1);
  const double travel_term = config_.travel_sigma_per_m * travelled_m_;
  const double sigma =
    std::sqrt(std::max(0.0, largest) + travel_term * travel_term);

  bool arrived = false;
  if (has_goal_) {
    const bool candidate =
      belief_distance + 1.96 * sigma <= config_.arrival_radius_m;
    arrival_streak_ = candidate ? arrival_streak_ + 1 : 0;
    arrived = arrival_streak_ >= config_.arrival_ticks;
  }

  // ── Actor ────────────────────────────────────────────────────────────────
  Command command;
  Clock::time_point t4 = t3;
  if (actor_ && has_goal_ && !arrived) {
    const Clock::time_point started = Clock::now();
    command = actor_->run(observation_);
    t4 = Clock::now();
    result_.infer_us = micros(started, t4);
  } else {
    result_.infer_us = 0.0;
  }
  last_command_ = command;
  first_tick_ = false;

  const Clock::time_point tend = Clock::now();

  result_.command = command;
  result_.pose = estimate.pose;
  result_.goal_body_x = goal_body_x;
  result_.goal_body_y = goal_body_y;
  result_.belief_distance_m = belief_distance;
  result_.sigma_m = sigma;
  result_.travelled_m = travelled_m_;
  result_.observable = estimate.observable;
  result_.arrived = arrived;
  result_.has_goal = has_goal_;
  result_.arrival_streak = arrival_streak_;
  result_.motion_scale = estimate.motion_scale;
  result_.deskew_us = micros(t0, t1);
  result_.bin_us = micros(t1, t2);
  result_.localize_us = micros(t2, t3);
  result_.total_us = micros(t0, tend);
  return result_;
}

}  // namespace sim2edge
