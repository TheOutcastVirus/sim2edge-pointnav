#include "sim2edge/deskew.hpp"

#include <algorithm>
#include <cmath>

#include "sim2edge/preprocess.hpp"

namespace sim2edge
{

SweepDeskewer::SweepDeskewer()
{
  reset();
}

void SweepDeskewer::reset()
{
  std::fill(std::begin(gyro_), std::end(gyro_), 0.0);
  std::fill(std::begin(speed_), std::end(speed_), 0.0);
  head_ = 0;
  filled_ = 0;
}

void SweepDeskewer::pushMotion(double gyro_increment_rad,
                               double forward_speed_mps)
{
  gyro_[head_] = gyro_increment_rad;
  speed_[head_] = forward_speed_mps;
  head_ = (head_ + 1) % spec::kSpinTicks;
  if (filled_ < spec::kSpinTicks) {
    ++filled_;
  }
}

SweepDeskewer::Transform SweepDeskewer::buildTransform() const
{
  Transform t;
  // Slot `back` holds the pose of the *current* frame expressed in the frame
  // that existed `back` control ticks ago: integrate forward from there.
  //
  // Indexing: head_ points at the slot the next push will overwrite, so the
  // newest sample sits at head_ - 1 and the sample `j` ticks before the newest
  // at head_ - 1 - j, all modulo kSpinTicks.
  for (int back = 1; back < spec::kSpinTicks; ++back) {
    double x = 0.0;
    double y = 0.0;
    double heading = 0.0;
    for (int j = back - 1; j >= 0; --j) {
      const int slot = ((head_ - 1 - j) % spec::kSpinTicks + spec::kSpinTicks) %
                       spec::kSpinTicks;
      const double distance = speed_[slot] * spec::kDt;
      x += distance * std::cos(heading);
      y += distance * std::sin(heading);
      heading += gyro_[slot];
    }
    t.dx[back] = x;
    t.dy[back] = y;
    t.yaw[back] = heading;
  }
  // back == 0 is the identity and stays zero: a beam acquired this instant
  // needs no correction.
  return t;
}

void SweepDeskewer::deskew(const float * ranges, const float * bearings,
                           const float * ages, std::size_t count,
                           PolarSweep & out, bool subtick) const
{
  const Transform t = buildTransform();
  out.resize(count);

  const double max_back = static_cast<double>(spec::kSpinTicks - 1);

  for (std::size_t i = 0; i < count; ++i) {
    const float r = ranges[i];
    const double phi = static_cast<double>(bearings[i]);

    double beam_dx = 0.0;
    double beam_dy = 0.0;
    double beam_yaw = 0.0;

    const double exact =
      std::clamp(static_cast<double>(ages[i]) / spec::kDt, 0.0, max_back);
    if (subtick) {
      const double lower = std::floor(exact);
      const double fraction = exact - lower;
      const int low = static_cast<int>(lower);
      const int high = std::min(low + 1, spec::kSpinTicks - 1);
      beam_dx = t.dx[low] + fraction * (t.dx[high] - t.dx[low]);
      beam_dy = t.dy[low] + fraction * (t.dy[high] - t.dy[low]);
      beam_yaw = t.yaw[low] + fraction * (t.yaw[high] - t.yaw[low]);
    } else {
      const int back = std::clamp(static_cast<int>(std::lround(exact)), 0,
                                  spec::kSpinTicks - 1);
      beam_dx = t.dx[back];
      beam_dy = t.dy[back];
      beam_yaw = t.yaw[back];
    }

    const bool finite = std::isfinite(r);
    const double safe = finite ? static_cast<double>(r) : 0.0;

    // Re-express the return as a point in the current body frame. The point
    // sits at (safe*cos, safe*sin) in the acquiring frame; the current frame's
    // origin moved to (beam_dx, beam_dy) and rotated by beam_yaw within it, so
    // subtract the translation and rotate by -beam_yaw.
    const double px = safe * std::cos(phi) - beam_dx;
    const double py = safe * std::sin(phi) - beam_dy;
    const double c = std::cos(beam_yaw);
    const double s = std::sin(beam_yaw);
    const double cx = c * px + s * py;
    const double cy = -s * px + c * py;

    if (finite) {
      out.ranges[i] = static_cast<float>(std::sqrt(cx * cx + cy * cy));
      out.bearings[i] = static_cast<float>(std::atan2(cy, cx));
    } else {
      // A non-finite range carries no position, so only the bearing can be
      // corrected. Keeping the range as-is preserves "no return here" rather
      // than inventing a distance for it.
      out.ranges[i] = r;
      out.bearings[i] = static_cast<float>(wrapAngle(phi - beam_yaw));
    }

    const float corrected = out.ranges[i];
    out.ages[i] = ages[i];
    out.valid[i] = (std::isfinite(corrected) && corrected >= spec::kLidarMinM &&
                    corrected <= spec::kLidarMaxM)
                     ? uint8_t{1}
                     : uint8_t{0};
  }
}

void beamGeometry(std::size_t count, float angle_min, float angle_increment,
                  float time_increment, double trailing_age_s,
                  std::vector<float> & bearings, std::vector<float> & ages)
{
  bearings.resize(count);
  ages.resize(count);
  if (count == 0) {
    return;
  }

  // Some A1 firmware/driver combinations report time_increment as zero. A
  // constant-rate motor spreads one revolution evenly over the beams, so that
  // is the right fallback -- and it is a fallback rather than an error because
  // an uncorrected sweep is much worse than one corrected with the nominal
  // spin rate.
  double per_beam = static_cast<double>(time_increment);
  if (!(per_beam > 0.0) || !std::isfinite(per_beam)) {
    per_beam = spec::kSpinPeriodS / static_cast<double>(count);
  }

  const double last = static_cast<double>(count - 1);
  for (std::size_t i = 0; i < count; ++i) {
    const double angle = static_cast<double>(angle_min) +
                         static_cast<double>(i) *
                           static_cast<double>(angle_increment);
    // The quarter-turn mount rotation, applied exactly once, here.
    bearings[i] = static_cast<float>(wrapAngle(angle + spec::kLidarMountYaw));
    ages[i] = static_cast<float>(
      (last - static_cast<double>(i)) * per_beam + trailing_age_s);
  }
}

}  // namespace sim2edge
