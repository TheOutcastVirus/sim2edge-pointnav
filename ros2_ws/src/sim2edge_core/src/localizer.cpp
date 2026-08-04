#include "sim2edge/localizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <Eigen/Dense>

#include "sim2edge/preprocess.hpp"

namespace sim2edge
{
namespace
{

using Clock = std::chrono::steady_clock;

double micros(Clock::time_point a, Clock::time_point b)
{
  return std::chrono::duration<double, std::micro>(b - a).count();
}

/// Process noise added to the covariance every tick, matching the Python
/// tracker's diag([2.5e-5, 2.5e-5, 1.0e-5]).
constexpr double kProcessXY = 2.5e-5;
constexpr double kProcessYaw = 1.0e-5;

/// Cauchy-like robust weight scale. Residuals much larger than this are
/// suppressed rather than allowed to relocate the minimum -- the moving and
/// occluding edges that make least squares useless on cluttered scans.
constexpr double kRobustScale = 0.04;

/// Per-iteration step clamps. A large jump is an association failure, and it
/// must not be allowed to drag the submap with it.
constexpr double kMaxStepXY = 0.04;
constexpr double kMaxStepYaw = 0.017453292519943295;  // 1 degree

constexpr int kNormalNeighbours = 7;

}  // namespace

LocalPoseTracker::LocalPoseTracker()
: LocalPoseTracker(Config())
{
}

LocalPoseTracker::LocalPoseTracker(const Config & config)
: config_(config)
{
  if (config_.point_stride < 1) {
    config_.point_stride = 1;
  }
  if (config_.max_submap_scans < 1) {
    config_.max_submap_scans = 1;
  }
  reset();
}

void LocalPoseTracker::reset()
{
  pose_ = Pose2{};
  std::fill(std::begin(covariance_), std::end(covariance_), 0.0);
  covariance_[0] = covariance_[4] = covariance_[8] = 1.0e-6;
  motion_scale_ = 1.0;
  gyro_yaw_ = 0.0;
  scans_.clear();
  submap_x_.clear();
  submap_y_.clear();
  estimate_ = LocalizerEstimate{};
}

void LocalPoseTracker::extractPoints(const PolarSweep & sweep)
{
  scan_x_.clear();
  scan_y_.clear();
  const std::size_t n = sweep.size();
  scan_x_.reserve(n / static_cast<std::size_t>(config_.point_stride) + 1);
  scan_y_.reserve(n / static_cast<std::size_t>(config_.point_stride) + 1);

  // The stride is applied to the *surviving* returns, not to the raw beam
  // index, exactly as the Python `valid` mask then `[::stride]` does. Striding
  // the raw array instead would thin dense regions and sparse ones by different
  // factors, which changes which geometry the match sees.
  int kept = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float r = sweep.ranges[i];
    const float b = sweep.bearings[i];
    if (!sweep.valid[i] || !std::isfinite(r) || !std::isfinite(b)) {
      continue;
    }
    if (r < config_.point_min_range_m || r > config_.point_max_range_m) {
      continue;
    }
    if (kept++ % config_.point_stride != 0) {
      continue;
    }
    scan_x_.push_back(r * std::cos(b));
    scan_y_.push_back(r * std::sin(b));
  }
}

void LocalPoseTracker::addScan()
{
  if (scan_x_.empty()) {
    return;
  }
  const double c = std::cos(pose_.yaw);
  const double s = std::sin(pose_.yaw);

  Scan scan;
  scan.x.resize(scan_x_.size());
  scan.y.resize(scan_x_.size());
  for (std::size_t i = 0; i < scan_x_.size(); ++i) {
    const double px = scan_x_[i];
    const double py = scan_y_[i];
    scan.x[i] = static_cast<float>(c * px - s * py + pose_.x);
    scan.y[i] = static_cast<float>(s * px + c * py + pose_.y);
  }

  scans_.push_back(std::move(scan));
  while (static_cast<int>(scans_.size()) > config_.max_submap_scans) {
    scans_.pop_front();
  }
}

void LocalPoseTracker::rebuildSubmap()
{
  std::size_t total = 0;
  for (const Scan & scan : scans_) {
    total += scan.x.size();
  }
  submap_x_.resize(total);
  submap_y_.resize(total);

  std::size_t offset = 0;
  for (const Scan & scan : scans_) {
    std::copy(scan.x.begin(), scan.x.end(), submap_x_.begin() +
              static_cast<std::ptrdiff_t>(offset));
    std::copy(scan.y.begin(), scan.y.end(), submap_y_.begin() +
              static_cast<std::ptrdiff_t>(offset));
    offset += scan.x.size();
  }

  grid_.build(submap_x_.data(), submap_y_.data(), total,
              config_.max_correspondence_m);

  normal_x_.resize(total);
  normal_y_.resize(total);
  normal_ready_.assign(total, 0);
}

void LocalPoseTracker::normalAt(int index, float & nx, float & ny)
{
  const std::size_t i = static_cast<std::size_t>(index);
  if (normal_ready_[i]) {
    nx = normal_x_[i];
    ny = normal_y_[i];
    return;
  }

  // Local PCA over the k nearest submap points. This is what turns the ICP
  // system into point-to-plane, and it correctly leaves travel along a
  // featureless corridor unobserved rather than inventing a constraint for it.
  int neighbours[kNormalNeighbours];
  const int found = grid_.knn(submap_x_[i], submap_y_[i], kNormalNeighbours,
                              neighbours);

  double mean_x = 0.0;
  double mean_y = 0.0;
  for (int j = 0; j < found; ++j) {
    mean_x += submap_x_[static_cast<std::size_t>(neighbours[j])];
    mean_y += submap_y_[static_cast<std::size_t>(neighbours[j])];
  }
  const double inv = found > 0 ? 1.0 / static_cast<double>(found) : 0.0;
  mean_x *= inv;
  mean_y *= inv;

  double a = 0.0;  // scatter xx
  double b = 0.0;  // scatter xy
  double c = 0.0;  // scatter yy
  for (int j = 0; j < found; ++j) {
    const double dx = submap_x_[static_cast<std::size_t>(neighbours[j])] - mean_x;
    const double dy = submap_y_[static_cast<std::size_t>(neighbours[j])] - mean_y;
    a += dx * dx;
    b += dx * dy;
    c += dy * dy;
  }

  // Closed-form eigenvector of the smaller eigenvalue of [[a, b], [b, c]].
  // The normal is the direction of least spread, i.e. across the surface.
  //
  // The sign convention does not matter and is not matched to LAPACK's: the
  // residual and the Jacobian row both flip with the normal, so both the
  // information matrix (J^T W J) and the gradient (J^T w r) are invariant.
  const double diff = a - c;
  const double root = std::sqrt(diff * diff + 4.0 * b * b);
  const double lambda_min = 0.5 * (a + c - root);

  double vx = b;
  double vy = lambda_min - a;
  double norm = std::sqrt(vx * vx + vy * vy);
  if (!(norm > 1.0e-12)) {
    // b is ~0 and the matrix is already diagonal: the normal is whichever axis
    // carries the smaller variance.
    if (a <= c) {
      vx = 1.0;
      vy = 0.0;
    } else {
      vx = 0.0;
      vy = 1.0;
    }
    norm = 1.0;
  }

  nx = static_cast<float>(vx / norm);
  ny = static_cast<float>(vy / norm);
  normal_x_[i] = nx;
  normal_y_[i] = ny;
  normal_ready_[i] = 1;
}

const LocalizerEstimate & LocalPoseTracker::update(
  const PolarSweep & sweep, double gyro_delta, double speed, double dt,
  const double * prior_delta_body)
{
  const Clock::time_point t0 = Clock::now();

  extractPoints(sweep);
  const Clock::time_point t1 = Clock::now();

  gyro_yaw_ = wrapAngle(gyro_yaw_ + gyro_delta);
  const Pose2 old_pose = pose_;

  // ── Prior ────────────────────────────────────────────────────────────────
  Pose2 prior = pose_;
  double heading = pose_.yaw;
  if (prior_delta_body == nullptr) {
    heading = pose_.yaw + 0.5 * gyro_delta;   // midpoint of the tick's arc
    const double travel = speed * dt * motion_scale_;
    prior.x += travel * std::cos(heading);
    prior.y += travel * std::sin(heading);
    prior.yaw = wrapAngle(pose_.yaw + gyro_delta);
  } else {
    const double c = std::cos(heading);
    const double s = std::sin(heading);
    prior.x += c * prior_delta_body[0] - s * prior_delta_body[1];
    prior.y += s * prior_delta_body[0] + c * prior_delta_body[1];
    prior.yaw = wrapAngle(pose_.yaw + prior_delta_body[2]);
  }
  pose_ = prior;

  auto inflate = [](const double * in, double * out) {
    std::copy(in, in + 9, out);
    out[0] += kProcessXY;
    out[4] += kProcessXY;
    out[8] += kProcessYaw;
  };

  // ── Too little to match against ──────────────────────────────────────────
  if (static_cast<int>(scan_x_.size()) < config_.min_points || scans_.empty()) {
    double inflated[9];
    inflate(covariance_, inflated);
    std::copy(inflated, inflated + 9, covariance_);
    addScan();

    estimate_.pose = pose_;
    std::copy(covariance_, covariance_ + 9, estimate_.covariance);
    estimate_.motion_scale = motion_scale_;
    estimate_.observable = false;
    estimate_.correction = Pose2{};
    estimate_.points_used = 0;
    estimate_.iterations = 0;
    estimate_.motion_prior_source =
      prior_delta_body == nullptr ? spec::kMotionPriorSource : "create3-odom";

    const Clock::time_point tend = Clock::now();
    timings_.extract_us = micros(t0, t1);
    timings_.submap_us = 0.0;
    timings_.icp_us = 0.0;
    timings_.total_us = micros(t0, tend);
    return estimate_;
  }

  rebuildSubmap();
  const Clock::time_point t2 = Clock::now();

  // ── Robust point-to-plane ICP against the submap ─────────────────────────
  Pose2 solved = prior;
  Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
  bool have_system = false;
  int used = 0;
  int iterations = 0;

  const std::size_t point_count = scan_x_.size();
  placed_x_.resize(point_count);
  placed_y_.resize(point_count);
  residual_.reserve(point_count);

  const float radius = config_.max_correspondence_m;

  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    const double c = std::cos(solved.yaw);
    const double s = std::sin(solved.yaw);
    for (std::size_t i = 0; i < point_count; ++i) {
      placed_x_[i] = static_cast<float>(c * scan_x_[i] - s * scan_y_[i] +
                                        solved.x);
      placed_y_[i] = static_cast<float>(s * scan_x_[i] + c * scan_y_[i] +
                                        solved.y);
    }

    Eigen::Matrix3d iteration_information = Eigen::Matrix3d::Zero();
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
    residual_.clear();
    int kept = 0;

    for (std::size_t i = 0; i < point_count; ++i) {
      float d2 = 0.0f;
      const int match = grid_.nearestWithin(placed_x_[i], placed_y_[i], radius,
                                            d2);
      if (match < 0) {
        continue;
      }
      ++kept;

      float nx = 0.0f;
      float ny = 0.0f;
      normalAt(match, nx, ny);

      const double tx = submap_x_[static_cast<std::size_t>(match)];
      const double ty = submap_y_[static_cast<std::size_t>(match)];
      const double sx = placed_x_[i];
      const double sy = placed_y_[i];

      const double r = nx * (sx - tx) + ny * (sy - ty);
      const double weight = 1.0 / (1.0 + (r / kRobustScale) * (r / kRobustScale));

      // d/dyaw of the placed point is the lever arm rotated a quarter turn.
      const double lever_x = sx - solved.x;
      const double lever_y = sy - solved.y;
      const double j2 = nx * (-lever_y) + ny * lever_x;

      const double j0 = nx;
      const double j1 = ny;

      const double w0 = weight * j0;
      const double w1 = weight * j1;
      const double w2 = weight * j2;

      iteration_information(0, 0) += w0 * j0;
      iteration_information(0, 1) += w0 * j1;
      iteration_information(0, 2) += w0 * j2;
      iteration_information(1, 1) += w1 * j1;
      iteration_information(1, 2) += w1 * j2;
      iteration_information(2, 2) += w2 * j2;

      gradient(0) += w0 * r;
      gradient(1) += w1 * r;
      gradient(2) += w2 * r;

      residual_.push_back(static_cast<float>(r));
    }

    if (kept < config_.min_points) {
      break;
    }

    iteration_information(1, 0) = iteration_information(0, 1);
    iteration_information(2, 0) = iteration_information(0, 2);
    iteration_information(2, 1) = iteration_information(1, 2);

    // The system is recorded before the solve, matching the Python ordering:
    // a solve failure leaves this iteration's information standing but does not
    // advance `used`, so the observability gate still needs a prior iteration
    // to have accepted enough points.
    information = iteration_information;
    have_system = true;

    Eigen::Matrix3d damped = iteration_information;
    damped.diagonal().array() += 1.0e-6;
    const Eigen::Vector3d step = -damped.ldlt().solve(gradient);
    if (!step.allFinite()) {
      break;
    }

    used = kept;
    iterations = iteration + 1;

    const double sx = std::clamp(step(0), -kMaxStepXY, kMaxStepXY);
    const double sy = std::clamp(step(1), -kMaxStepXY, kMaxStepXY);
    const double syaw = std::clamp(step(2), -kMaxStepYaw, kMaxStepYaw);
    solved.x += sx;
    solved.y += sy;
    solved.yaw = wrapAngle(solved.yaw + syaw);

    if (std::sqrt(step(0) * step(0) + step(1) * step(1) + step(2) * step(2)) <
        1.0e-5)
    {
      break;
    }
  }
  const Clock::time_point t3 = Clock::now();

  // ── Observability, then the correction ───────────────────────────────────
  bool observable = false;
  double covariance[9];
  inflate(covariance_, covariance);

  if (have_system && used >= config_.min_points && !residual_.empty()) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(
      information, Eigen::EigenvaluesOnly);
    const Eigen::Vector3d eigen = solver.eigenvalues();
    const double smallest = eigen(0);
    const double condition = eigen(2) / std::max(smallest, 1.0e-12);

    const double dx = solved.x - prior.x;
    const double dy = solved.y - prior.y;
    observable = smallest >= config_.min_information &&
                 condition <= config_.max_condition &&
                 std::sqrt(dx * dx + dy * dy) <= config_.max_correction_m &&
                 std::abs(wrapAngle(solved.yaw - prior.yaw)) <=
                   config_.max_correction_yaw_rad;

    if (observable) {
      // Median of squared residuals: a robust scale for the match quality,
      // which is what the covariance should reflect rather than the count.
      const std::size_t mid = residual_.size() / 2;
      std::vector<float> squares(residual_.size());
      for (std::size_t i = 0; i < residual_.size(); ++i) {
        squares[i] = residual_[i] * residual_[i];
      }
      std::nth_element(squares.begin(), squares.begin() +
                       static_cast<std::ptrdiff_t>(mid), squares.end());
      double median = squares[mid];
      if (squares.size() % 2 == 0 && mid > 0) {
        // numpy's median averages the two central order statistics on even
        // counts; reproduce that rather than being off by half a sample.
        const float upper = squares[mid];
        std::nth_element(squares.begin(), squares.begin() +
                         static_cast<std::ptrdiff_t>(mid - 1), squares.end());
        median = 0.5 * (static_cast<double>(squares[mid - 1]) + upper);
      }
      const double variance = std::max(median, 1.0e-6);

      Eigen::Matrix3d damped = information;
      damped.diagonal().array() += 1.0e-6;
      const Eigen::Matrix3d inverse = damped.inverse() * variance;
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          covariance[r * 3 + c] = inverse(r, c);
        }
      }
    }
  }

  Pose2 correction;
  if (observable) {
    const double dx = solved.x - prior.x;
    const double dy = solved.y - prior.y;
    const double dyaw = wrapAngle(solved.yaw - prior.yaw);

    pose_.x = prior.x + dx * config_.correction_gain;
    pose_.y = prior.y + dy * config_.correction_gain;
    pose_.yaw = wrapAngle(prior.yaw + dyaw * config_.yaw_correction_gain);
    std::copy(covariance, covariance + 9, covariance_);

    correction.x = pose_.x - prior.x;
    correction.y = pose_.y - prior.y;
    correction.yaw = wrapAngle(pose_.yaw - prior.yaw);

    const double commanded = speed * dt;
    if (std::abs(commanded) > 0.005) {
      // Estimate the slowly varying drivetrain scale from the *full* scan
      // solution, not the blended output: feeding the complementary-filter gain
      // back into the physical scale makes convergence needlessly slow.
      //
      // Only forward displacement identifies scale. Using the vector norm makes
      // lateral ICP noise strictly positive and therefore biases the scale
      // high -- the worst possible direction when the wheels are slipping.
      const double dir_x = std::cos(heading);
      const double dir_y = std::sin(heading);
      const double achieved = (solved.x - old_pose.x) * dir_x +
                              (solved.y - old_pose.y) * dir_y;
      const double ratio = std::clamp(achieved / commanded, 0.75, 1.10);
      const double rate = config_.motion_scale_update_rate;
      motion_scale_ = (1.0 - rate) * motion_scale_ + rate * ratio;
    }
  } else {
    pose_ = prior;
    std::copy(covariance, covariance + 9, covariance_);
  }

  // The gyro is the authority on heading over short horizons. If the scan match
  // has walked the yaw away from it by more than the allowance, pull it back --
  // off by default, and a diagnostic when it is not.
  const double departure = wrapAngle(pose_.yaw - gyro_yaw_);
  if (std::abs(departure) > config_.max_yaw_departure_rad) {
    pose_.yaw = wrapAngle(gyro_yaw_ + std::clamp(departure,
                                                 -config_.max_yaw_departure_rad,
                                                 config_.max_yaw_departure_rad));
  }

  addScan();

  estimate_.pose = pose_;
  std::copy(covariance_, covariance_ + 9, estimate_.covariance);
  estimate_.motion_scale = motion_scale_;
  estimate_.observable = observable;
  estimate_.correction = correction;
  estimate_.points_used = used;
  estimate_.iterations = iterations;
  // Stamped from what the prior actually used, so an odometry-assisted run can
  // never be reported as lidar/IMU-only evidence.
  estimate_.motion_prior_source =
    prior_delta_body == nullptr ? spec::kMotionPriorSource : "create3-odom";

  const Clock::time_point tend = Clock::now();
  timings_.extract_us = micros(t0, t1);
  timings_.submap_us = micros(t1, t2);
  timings_.icp_us = micros(t2, t3);
  timings_.total_us = micros(t0, tend);
  return estimate_;
}

}  // namespace sim2edge
