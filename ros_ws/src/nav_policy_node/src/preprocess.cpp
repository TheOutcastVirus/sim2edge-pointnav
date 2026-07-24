#include "nav_policy_node/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nav_policy_node
{

namespace
{
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

inline bool isValidDepth(float d)
{
  return std::isfinite(d) && d > 0.0f;
}
}  // namespace

float wrapAngle(float a)
{
  // Python's `%` returns a result with the sign of the divisor; C's fmod
  // returns one with the sign of the dividend. Reproducing Python's behaviour
  // matters because the bin index is derived from this and an off-by-one
  // hemisphere is a silent, catastrophic mismatch.
  float x = std::fmod(a + kPi, kTwoPi);
  if (x < 0.0f) {
    x += kTwoPi;
  }
  return x - kPi;
}

void blockBounds(int n_in, int n_out, std::vector<int> & starts,
                 std::vector<int> & ends)
{
  starts.resize(n_out);
  ends.resize(n_out);
  for (int i = 0; i < n_out; ++i) {
    const int start = static_cast<int>(
      (static_cast<long long>(i) * n_in) / n_out);
    int end = static_cast<int>(
      (static_cast<long long>(i + 1) * n_in) / n_out);
    end = std::max(end, start + 1);
    end = std::min(end, n_in);
    starts[i] = std::min(start, n_in - 1);
    ends[i] = end;
  }
}

// ── Depth ───────────────────────────────────────────────────────────────────

namespace
{
/// Shared body of both depth entry points. `fetch` returns metric depth for a
/// linear pixel index, which lets the uint16 path avoid materializing an
/// intermediate float image.
template<typename Fetch>
void preprocessDepthImpl(Fetch fetch, int height, int width,
                         std::vector<float> & out)
{
  out.assign(spec::kDepthNumel, spec::kDepthInvalidFill);
  if (height <= 0 || width <= 0) {
    return;
  }

  std::vector<int> row_start, row_end, col_start, col_end;
  blockBounds(height, spec::kDepthHeight, row_start, row_end);
  blockBounds(width, spec::kDepthWidth, col_start, col_end);

  constexpr float span = spec::kDepthMaxM - spec::kDepthMinM;

  for (int r = 0; r < spec::kDepthHeight; ++r) {
    for (int c = 0; c < spec::kDepthWidth; ++c) {
      float best = spec::kDepthInvalidFill;
      for (int sr = row_start[r]; sr < row_end[r]; ++sr) {
        const std::size_t base = static_cast<std::size_t>(sr) * width;
        for (int sc = col_start[c]; sc < col_end[c]; ++sc) {
          const float d = fetch(base + static_cast<std::size_t>(sc));
          // Normalize first, then pool: invalid and far both become 1.0, so
          // the min-pool needs no mask and matches the numpy implementation
          // exactly.
          float value = spec::kDepthInvalidFill;
          if (isValidDepth(d)) {
            const float clipped =
              std::min(std::max(d, spec::kDepthMinM), spec::kDepthMaxM);
            value = (clipped - spec::kDepthMinM) / span;
          }
          best = std::min(best, value);
        }
      }
      out[static_cast<std::size_t>(r) * spec::kDepthWidth + c] = best;
    }
  }
}
}  // namespace

void preprocessDepth(const float * depth_m, int height, int width,
                     std::vector<float> & out)
{
  preprocessDepthImpl(
    [depth_m](std::size_t i) { return depth_m[i]; }, height, width, out);
}

void preprocessDepthMillimetres(const uint16_t * depth_mm, int height, int width,
                                std::vector<float> & out)
{
  preprocessDepthImpl(
    [depth_mm](std::size_t i) {
      return static_cast<float>(depth_mm[i]) * spec::kDepthRawScaleToM;
    }, height, width, out);
}

// ── Lidar ───────────────────────────────────────────────────────────────────

void preprocessLidar(const float * ranges, std::size_t count, float angle_min,
                     float angle_increment, float range_min, float range_max,
                     std::vector<float> & out)
{
  out.assign(spec::kLidarBins, spec::kLidarInvalidFill);
  if (count == 0) {
    return;
  }

  const float lo = std::max(spec::kLidarMinM, range_min);
  const float hi = std::min(spec::kLidarMaxM, range_max);
  const float span_angle = spec::kLidarAngleMax - spec::kLidarAngleMin;
  const float span_range = spec::kLidarMaxM - spec::kLidarMinM;

  for (std::size_t i = 0; i < count; ++i) {
    const float angle =
      angle_min + angle_increment * static_cast<float>(i);
    const float wrapped = wrapAngle(angle);

    int idx = static_cast<int>(std::floor(
      (wrapped - spec::kLidarAngleMin) / span_angle * spec::kLidarBins));
    idx = std::min(std::max(idx, 0), spec::kLidarBins - 1);

    const float r = ranges[i];
    float value = spec::kLidarInvalidFill;
    if (std::isfinite(r) && r >= lo && r <= hi) {
      const float clipped =
        std::min(std::max(r, spec::kLidarMinM), spec::kLidarMaxM);
      value = (clipped - spec::kLidarMinM) / span_range;
    }
    out[idx] = std::min(out[idx], value);
  }
}

// ── Goal and action ─────────────────────────────────────────────────────────

void preprocessGoal(float goal_x, float goal_y, std::vector<float> & out)
{
  out.resize(spec::kGoalDim);
  const float dist = std::hypot(goal_x, goal_y);
  out[0] = std::min(dist, spec::kGoalMaxDistanceM) / spec::kGoalMaxDistanceM;
  if (dist < 1e-6f) {
    out[1] = 0.0f;  // sin
    out[2] = 1.0f;  // cos -- "straight ahead" is the neutral bearing
  } else {
    out[1] = goal_y / dist;
    out[2] = goal_x / dist;
  }
}

void preprocessLastAction(float v, float omega, std::vector<float> & out)
{
  out.resize(spec::kActionDim);
  const float v_n =
    2.0f * (v - spec::kVMin) / (spec::kVMax - spec::kVMin) - 1.0f;
  const float w_n = omega / spec::kOmegaMax;
  out[0] = std::min(std::max(v_n, -1.0f), 1.0f);
  out[1] = std::min(std::max(w_n, -1.0f), 1.0f);
}

void worldGoalToRobotFrame(float goal_x, float goal_y, float robot_x,
                           float robot_y, float robot_yaw,
                           float & out_x, float & out_y)
{
  const float dx = goal_x - robot_x;
  const float dy = goal_y - robot_y;
  const float c = std::cos(-robot_yaw);
  const float s = std::sin(-robot_yaw);
  out_x = c * dx - s * dy;
  out_y = s * dx + c * dy;
}

}  // namespace nav_policy_node
