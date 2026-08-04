#include "sim2edge/preprocess.hpp"

#include <algorithm>
#include <cmath>

namespace sim2edge
{
namespace
{
constexpr double kTwoPi = 6.283185307179586;
constexpr double kPi = 3.141592653589793;
}  // namespace

float wrapAngle(float a)
{
  // Deliberately computed in double and narrowed once. In float, (a + pi) for
  // a near -pi loses the low bits that decide which side of a bin edge the beam
  // falls on, and torch does this arithmetic in float32 from float32 inputs but
  // with a fused path that keeps more precision. Doing it in double and
  // rounding once is at least as accurate as either, and it is deterministic.
  const double d = static_cast<double>(a);
  return static_cast<float>(d - kTwoPi * std::floor((d + kPi) / kTwoPi));
}

double wrapAngle(double a)
{
  return a - kTwoPi * std::floor((a + kPi) / kTwoPi);
}

void binLidar(const float * ranges, const float * bearings, std::size_t count,
              std::vector<float> & out)
{
  out.assign(static_cast<std::size_t>(spec::kLidarBins), spec::kInvalidFill);

  const float denom = spec::kLidarMaxM - spec::kLidarMinM;
  for (std::size_t i = 0; i < count; ++i) {
    const float r = ranges[i];
    // Normalize first: near -> 0, far -> 1. Anything unusable is "far", which
    // is why there is no mask to keep in sync with the values.
    float norm = spec::kInvalidFill;
    if (std::isfinite(r) && r >= spec::kLidarMinM && r <= spec::kLidarMaxM) {
      norm = (r - spec::kLidarMinM) / denom;
      norm = std::clamp(norm, 0.0f, 1.0f);
    }
    if (norm >= spec::kInvalidFill) {
      continue;  // cannot lower any bin's minimum; skip the index arithmetic
    }

    // `+ pi` moves [-pi, pi) to [0, 2pi) so bin 0 starts dead astern and the
    // index increases counter-clockwise.
    const double wrapped = wrapAngle(static_cast<double>(bearings[i]));
    int idx = static_cast<int>(std::floor((wrapped + kPi) / spec::kBinWidthRad));
    idx = std::clamp(idx, 0, spec::kLidarBins - 1);  // guards the +pi boundary

    float & slot = out[static_cast<std::size_t>(idx)];
    if (norm < slot) {
      slot = norm;
    }
  }
}

void binLidar(const PolarSweep & sweep, std::vector<float> & out)
{
  out.assign(static_cast<std::size_t>(spec::kLidarBins), spec::kInvalidFill);

  const float denom = spec::kLidarMaxM - spec::kLidarMinM;
  const std::size_t count = sweep.size();
  for (std::size_t i = 0; i < count; ++i) {
    const float r = sweep.ranges[i];
    // An explicitly invalid return and an out-of-range one are the same
    // statement -- kInvalidFill -- and neither can lower a bin, so both are
    // simply skipped.
    if (!sweep.valid[i] || !std::isfinite(r) || r < spec::kLidarMinM ||
        r > spec::kLidarMaxM)
    {
      continue;
    }
    const float norm = std::clamp((r - spec::kLidarMinM) / denom, 0.0f, 1.0f);
    if (norm >= spec::kInvalidFill) {
      continue;
    }

    const double wrapped = wrapAngle(static_cast<double>(sweep.bearings[i]));
    int idx = static_cast<int>(std::floor((wrapped + kPi) / spec::kBinWidthRad));
    idx = std::clamp(idx, 0, spec::kLidarBins - 1);

    float & slot = out[static_cast<std::size_t>(idx)];
    if (norm < slot) {
      slot = norm;
    }
  }
}

void relativeGoalFeatures(float goal_x, float goal_y, float * out3)
{
  const double x = goal_x;
  const double y = goal_y;
  const double dist = std::sqrt(x * x + y * y);
  const double safe = std::max(dist, 1.0e-6);
  out3[0] = static_cast<float>(std::min(dist / spec::kGoalMaxM, 1.0));
  out3[1] = static_cast<float>(y / safe);
  out3[2] = static_cast<float>(x / safe);
}

void toBodyFrame(double dx, double dy, double yaw, double & out_x,
                 double & out_y)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  out_x = c * dx + s * dy;
  out_y = -s * dx + c * dy;
}

void goalFeatures(double goal_x, double goal_y, const Pose2 & robot,
                  float * out3)
{
  double bx = 0.0;
  double by = 0.0;
  toBodyFrame(goal_x - robot.x, goal_y - robot.y, robot.yaw, bx, by);
  relativeGoalFeatures(static_cast<float>(bx), static_cast<float>(by), out3);
}

ObservationHistory::ObservationHistory()
: lidar_(static_cast<std::size_t>(spec::kHistoryFrames) * spec::kLidarBins,
         spec::kInvalidFill),
  action_(static_cast<std::size_t>(spec::kHistoryFrames) * spec::kActionDim,
          0.0f)
{
}

void ObservationHistory::push(const std::vector<float> & lidar_bins,
                              float norm_v, float norm_omega)
{
  const std::size_t bins = static_cast<std::size_t>(spec::kLidarBins);
  const int frames = spec::kHistoryFrames;

  if (!initialized_) {
    // Repeat the first frame across the whole history. A zero-filled history is
    // not neutral: zero denormalizes to an obstacle 15 cm dead ahead in every
    // bin, and the policy correctly refuses to move.
    for (int f = 0; f < frames; ++f) {
      std::copy(lidar_bins.begin(), lidar_bins.end(),
                lidar_.begin() + static_cast<std::ptrdiff_t>(f * bins));
      action_[static_cast<std::size_t>(f) * 2 + 0] = norm_v;
      action_[static_cast<std::size_t>(f) * 2 + 1] = norm_omega;
    }
    head_ = 0;
    initialized_ = true;
    return;
  }

  // Overwrite the oldest slot and advance the head: the ring makes this O(bins)
  // instead of the O(frames * bins) memmove a shifting buffer would cost.
  const std::size_t slot = static_cast<std::size_t>(head_);
  std::copy(lidar_bins.begin(), lidar_bins.end(),
            lidar_.begin() + static_cast<std::ptrdiff_t>(slot * bins));
  action_[slot * 2 + 0] = norm_v;
  action_[slot * 2 + 1] = norm_omega;
  head_ = (head_ + 1) % frames;
}

void ObservationHistory::fill(const float * goal3, Observation & obs) const
{
  const std::size_t bins = static_cast<std::size_t>(spec::kLidarBins);
  const int frames = spec::kHistoryFrames;

  float * lidar_out = obs.lidar();
  float * action_out = obs.action();
  for (int k = 0; k < frames; ++k) {
    // Flat layout is oldest to newest; the oldest live frame sits at head_.
    const std::size_t src = static_cast<std::size_t>((head_ + k) % frames);
    std::copy_n(lidar_.begin() + static_cast<std::ptrdiff_t>(src * bins), bins,
                lidar_out + static_cast<std::size_t>(k) * bins);
    action_out[static_cast<std::size_t>(k) * 2 + 0] = action_[src * 2 + 0];
    action_out[static_cast<std::size_t>(k) * 2 + 1] = action_[src * 2 + 1];
  }

  float * goal_out = obs.goal();
  goal_out[0] = goal3[0];
  goal_out[1] = goal3[1];
  goal_out[2] = goal3[2];
}

Command physicalAction(float raw_v, float raw_omega)
{
  Command out;
  out.v = 0.5f * spec::kVMax * (std::tanh(raw_v) + 1.0f);
  out.omega = spec::kOmegaMax * std::tanh(raw_omega);
  return out;
}

}  // namespace sim2edge
