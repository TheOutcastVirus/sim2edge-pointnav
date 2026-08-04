// Deskew is the correction whose absence is an action-correlated sim-to-real
// gap rather than noise, so these tests pin its geometry against cases where
// the right answer is known by construction -- not against a recorded corpus,
// which the replay parity gate already covers.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "sim2edge/deskew.hpp"
#include "sim2edge/preprocess.hpp"
#include "sim2edge/spec_generated.hpp"

using namespace sim2edge;

namespace
{

/// A revolution of `beams` returns from a wall at constant range.
void makeSweep(std::size_t beams, float range, std::vector<float> & ranges,
               std::vector<float> & bearings, std::vector<float> & ages)
{
  ranges.assign(beams, range);
  bearings.resize(beams);
  ages.resize(beams);
  const double last = static_cast<double>(beams - 1);
  for (std::size_t i = 0; i < beams; ++i) {
    bearings[i] = static_cast<float>(
      -M_PI + 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(beams));
    // Oldest beam first: age falls to zero at the beam the sweep just crossed.
    ages[i] = static_cast<float>(
      (last - static_cast<double>(i)) * spec::kSpinPeriodS /
      static_cast<double>(beams));
  }
}

}  // namespace

TEST(Deskew, StationaryRobotIsAnIdentity)
{
  // No motion, no correction. If this ever fails, deskew is injecting an error
  // rather than removing one -- and it would do so on every tick.
  SweepDeskewer deskewer;
  for (int i = 0; i < spec::kSpinTicks; ++i) {
    deskewer.pushMotion(0.0, 0.0);
  }

  std::vector<float> ranges;
  std::vector<float> bearings;
  std::vector<float> ages;
  makeSweep(720, 3.0f, ranges, bearings, ages);

  PolarSweep out;
  deskewer.deskew(ranges.data(), bearings.data(), ages.data(), ranges.size(),
                  out);

  for (std::size_t i = 0; i < ranges.size(); ++i) {
    EXPECT_NEAR(out.ranges[i], ranges[i], 1e-5f) << "beam " << i;
    // Compared as an angular difference, not a numeric one. The corrected
    // bearing comes back through atan2, which returns +pi where the input was
    // -pi -- the same direction. Python's deskew returns atan2 as well, so the
    // two implementations agree; only a naive subtraction would disagree.
    const double difference = std::abs(wrapAngle(
      static_cast<double>(out.bearings[i]) - static_cast<double>(bearings[i])));
    EXPECT_LT(difference, 1e-5) << "beam " << i;
    EXPECT_TRUE(out.valid[i]);
  }
}

TEST(Deskew, PureRotationRotatesStaleBeamsAndLeavesTheFreshestAlone)
{
  // Spinning in place: a beam acquired one tick ago sits one tick of yaw away
  // from where it is being reported, and the correction is exactly that.
  const double rate = 1.0;                       // rad/s
  const double increment = rate * spec::kDt;     // rad per control tick

  SweepDeskewer deskewer;
  for (int i = 0; i < spec::kSpinTicks; ++i) {
    deskewer.pushMotion(increment, 0.0);
  }

  std::vector<float> ranges;
  std::vector<float> bearings;
  std::vector<float> ages;
  makeSweep(720, 3.0f, ranges, bearings, ages);

  PolarSweep out;
  deskewer.deskew(ranges.data(), bearings.data(), ages.data(), ranges.size(),
                  out);

  for (std::size_t i = 0; i < ranges.size(); ++i) {
    // Range is unaffected by rotation about the sensor.
    EXPECT_NEAR(out.ranges[i], 3.0f, 1e-4f) << "beam " << i;

    const double age = ages[i];
    const double expected = wrapAngle(static_cast<double>(bearings[i]) -
                                      rate * age);
    const double difference =
      std::abs(wrapAngle(static_cast<double>(out.bearings[i]) - expected));
    EXPECT_LT(difference, 2.0e-3) << "beam " << i << " age " << age;
  }

  // The newest beam is the last one and needs no correction at all.
  const std::size_t newest = ranges.size() - 1;
  EXPECT_NEAR(out.bearings[newest], bearings[newest], 1e-5f);
}

TEST(Deskew, ForwardMotionMovesTheWallCloserForStaleBeams)
{
  // Driving straight at a wall: a beam acquired a full revolution ago was
  // measured from further back, so once re-expressed in the current frame the
  // wall must be *nearer* than the raw range says.
  const double speed = 0.4;

  SweepDeskewer deskewer;
  for (int i = 0; i < spec::kSpinTicks; ++i) {
    deskewer.pushMotion(0.0, speed);
  }

  std::vector<float> ranges;
  std::vector<float> bearings;
  std::vector<float> ages;
  makeSweep(720, 3.0f, ranges, bearings, ages);

  PolarSweep out;
  deskewer.deskew(ranges.data(), bearings.data(), ages.data(), ranges.size(),
                  out);

  // Beam 0 is the stalest and points dead astern (-pi), so moving forward takes
  // us *away* from it: its range grows.
  const double oldest_age = ages[0];
  const double travel = speed * oldest_age;
  EXPECT_NEAR(out.ranges[0], 3.0f + travel, 0.02f);

  // The beam pointing dead ahead: the robot closed on it.
  std::size_t ahead = 0;
  double best = 1e9;
  for (std::size_t i = 0; i < bearings.size(); ++i) {
    const double d = std::abs(static_cast<double>(bearings[i]));
    if (d < best) {
      best = d;
      ahead = i;
    }
  }
  EXPECT_LT(out.ranges[ahead], 3.0f);
}

TEST(Deskew, NonFiniteRangesKeepNoReturnRatherThanInventingOne)
{
  SweepDeskewer deskewer;
  for (int i = 0; i < spec::kSpinTicks; ++i) {
    deskewer.pushMotion(0.02, 0.3);
  }

  std::vector<float> ranges;
  std::vector<float> bearings;
  std::vector<float> ages;
  makeSweep(360, 3.0f, ranges, bearings, ages);
  ranges[10] = std::numeric_limits<float>::infinity();
  ranges[11] = std::numeric_limits<float>::quiet_NaN();

  PolarSweep out;
  deskewer.deskew(ranges.data(), bearings.data(), ages.data(), ranges.size(),
                  out);

  EXPECT_FALSE(std::isfinite(out.ranges[10]));
  EXPECT_FALSE(std::isfinite(out.ranges[11]));
  EXPECT_FALSE(out.valid[10]);
  EXPECT_FALSE(out.valid[11]);
  // The bearing is still corrected: rotation is known even where range is not.
  EXPECT_TRUE(std::isfinite(out.bearings[10]));
}

TEST(BeamGeometry, AppliesTheMountRotationExactlyOnce)
{
  // The sensor is bolted on rotated a quarter turn. A scan left in the lidar
  // frame puts every obstacle at right angles to where the policy believes it
  // is -- and applying the rotation twice is just as wrong.
  std::vector<float> bearings;
  std::vector<float> ages;
  beamGeometry(4, 0.0f, static_cast<float>(M_PI_2), 0.001f, 0.0, bearings, ages);

  ASSERT_EQ(bearings.size(), 4u);
  EXPECT_NEAR(bearings[0], static_cast<float>(spec::kLidarMountYaw), 1e-5f);
  EXPECT_NEAR(std::abs(wrapAngle(static_cast<double>(bearings[1]) - M_PI)), 0.0,
              1e-5);
}

TEST(BeamGeometry, AgesDecreaseTowardTheBeamTheSweepJustCrossed)
{
  std::vector<float> bearings;
  std::vector<float> ages;
  beamGeometry(100, -3.14f, 0.0628f, 0.001f, 0.005, bearings, ages);

  ASSERT_EQ(ages.size(), 100u);
  EXPECT_NEAR(ages[99], 0.005f, 1e-6f) << "newest beam carries only the "
                                          "trailing delay";
  EXPECT_NEAR(ages[0], 99 * 0.001f + 0.005f, 1e-5f);
  for (std::size_t i = 1; i < ages.size(); ++i) {
    EXPECT_LT(ages[i], ages[i - 1]);
  }
}

TEST(BeamGeometry, ZeroTimeIncrementFallsBackToTheNominalSpinRate)
{
  // Some A1 driver/firmware combinations report time_increment as zero. An
  // uncorrected sweep is much worse than one corrected with the nominal rate,
  // so this must be a fallback rather than an error.
  std::vector<float> bearings;
  std::vector<float> ages;
  beamGeometry(360, -3.14f, 0.0174f, 0.0f, 0.0, bearings, ages);

  EXPECT_NEAR(ages[0], static_cast<float>(spec::kSpinPeriodS * 359.0 / 360.0),
              1e-4f);
  EXPECT_NEAR(ages[359], 0.0f, 1e-6f);
}
