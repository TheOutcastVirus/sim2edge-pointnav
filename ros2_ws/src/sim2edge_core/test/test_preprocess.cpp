// Properties of the observation contract that must hold whatever the corpus
// says. The replay parity gate compares against recorded Python output; these
// tests pin the invariants that a replay could satisfy while still being wrong.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "sim2edge/preprocess.hpp"
#include "sim2edge/spec_generated.hpp"

using namespace sim2edge;

TEST(WrapAngle, MatchesPythonModuloNotFmod)
{
  // The whole point of writing this with floor: C's fmod keeps the sign of the
  // dividend, Python's % keeps the sign of the divisor. Getting it wrong is an
  // off-by-one-hemisphere error that presents as a badly trained policy.
  EXPECT_NEAR(wrapAngle(0.0), 0.0, 1e-12);
  EXPECT_NEAR(wrapAngle(M_PI - 1e-9), M_PI - 1e-9, 1e-9);
  EXPECT_NEAR(wrapAngle(-M_PI), -M_PI, 1e-12);
  // +pi wraps to -pi, and that is the boundary the bin clamp guards.
  EXPECT_NEAR(wrapAngle(M_PI), -M_PI, 1e-12);
  EXPECT_NEAR(wrapAngle(3.0 * M_PI), -M_PI, 1e-9);
  EXPECT_NEAR(wrapAngle(-3.0 * M_PI), -M_PI, 1e-9);
  EXPECT_NEAR(wrapAngle(1.5 * M_PI), -0.5 * M_PI, 1e-9);
}

TEST(BinLidar, EmptyScanIsAllInvalidFillNotZero)
{
  // Zero is not a neutral value: it denormalizes to an obstacle 15 cm dead
  // ahead in every bin, and the policy correctly refuses to move. Any path that
  // produces a zeroed observation is a stopped robot.
  std::vector<float> out;
  binLidar(nullptr, nullptr, 0, out);
  ASSERT_EQ(out.size(), static_cast<std::size_t>(spec::kLidarBins));
  for (float value : out) {
    EXPECT_FLOAT_EQ(value, spec::kInvalidFill);
  }
}

TEST(BinLidar, ReductionIsMinimumSoThinObstaclesSurvive)
{
  // A table leg covering one beam in ten must reach the policy. Averaging
  // would blend it into the wall behind it; a mean would delete a chair.
  std::vector<float> ranges;
  std::vector<float> bearings;
  const float bearing = 0.0f;  // all in one bin
  for (int i = 0; i < 10; ++i) {
    ranges.push_back(i == 3 ? 0.5f : 6.0f);
    bearings.push_back(bearing);
  }
  std::vector<float> out;
  binLidar(ranges.data(), bearings.data(), ranges.size(), out);

  const int bin = static_cast<int>(
    std::floor((0.0 + M_PI) / spec::kBinWidthRad));
  const float expected =
    (0.5f - spec::kLidarMinM) / (spec::kLidarMaxM - spec::kLidarMinM);
  EXPECT_NEAR(out[static_cast<std::size_t>(bin)], expected, 1e-6f);
}

TEST(BinLidar, InvalidAndFarAreTheSameStatement)
{
  // "Absent" and "distant" both normalize to kInvalidFill, which is exactly
  // why the pipeline needs no companion mask to keep in sync.
  std::vector<float> ranges = {std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN(),
                               spec::kLidarMaxM + 1.0f,
                               spec::kLidarMinM - 0.01f};
  std::vector<float> bearings(ranges.size(), 0.0f);
  std::vector<float> out;
  binLidar(ranges.data(), bearings.data(), ranges.size(), out);
  for (float value : out) {
    EXPECT_FLOAT_EQ(value, spec::kInvalidFill);
  }
}

TEST(BinLidar, BinsByBearingNotIndex)
{
  // The RPLIDAR's start angle is not stable across driver versions. Reversing
  // the beam order must not rotate the policy's view of the world.
  std::vector<float> ranges;
  std::vector<float> bearings;
  for (int i = 0; i < 180; ++i) {
    bearings.push_back(static_cast<float>(-M_PI + i * spec::kBinWidthRad +
                                          0.5 * spec::kBinWidthRad));
    ranges.push_back(1.0f + 0.01f * static_cast<float>(i));
  }
  std::vector<float> forward;
  binLidar(ranges.data(), bearings.data(), ranges.size(), forward);

  std::reverse(ranges.begin(), ranges.end());
  std::reverse(bearings.begin(), bearings.end());
  std::vector<float> reversed;
  binLidar(ranges.data(), bearings.data(), ranges.size(), reversed);

  EXPECT_EQ(forward, reversed);
}

TEST(GoalFeatures, StandingOnTheGoalIsFiniteNotNaN)
{
  float out[3];
  relativeGoalFeatures(0.0f, 0.0f, out);
  for (float value : out) {
    EXPECT_TRUE(std::isfinite(value));
  }
  EXPECT_FLOAT_EQ(out[0], 0.0f);
}

TEST(GoalFeatures, DistanceSaturatesAndBearingIsContinuous)
{
  float far[3];
  relativeGoalFeatures(1000.0f, 0.0f, far);
  EXPECT_FLOAT_EQ(far[0], 1.0f);   // clamped, not unbounded
  EXPECT_NEAR(far[1], 0.0f, 1e-6);
  EXPECT_NEAR(far[2], 1.0f, 1e-6);

  // Either side of the +/-pi wrap the features must be adjacent, which is the
  // reason sin/cos is used instead of a raw angle.
  float behind_a[3];
  float behind_b[3];
  relativeGoalFeatures(-1.0f, 1e-4f, behind_a);
  relativeGoalFeatures(-1.0f, -1e-4f, behind_b);
  EXPECT_NEAR(behind_a[1], behind_b[1], 1e-3);
  EXPECT_NEAR(behind_a[2], behind_b[2], 1e-6);
}

TEST(ObservationHistory, FirstFrameRepeatsAcrossTheWholeHistory)
{
  ObservationHistory history;
  std::vector<float> bins(static_cast<std::size_t>(spec::kLidarBins), 0.25f);
  history.push(bins, 0.5f, -0.5f);

  Observation obs;
  const float goal[3] = {0.1f, 0.2f, 0.3f};
  history.fill(goal, obs);

  for (int f = 0; f < spec::kHistoryFrames; ++f) {
    for (int b = 0; b < spec::kLidarBins; ++b) {
      EXPECT_FLOAT_EQ(obs.lidar()[f * spec::kLidarBins + b], 0.25f);
    }
    EXPECT_FLOAT_EQ(obs.action()[f * 2 + 0], 0.5f);
    EXPECT_FLOAT_EQ(obs.action()[f * 2 + 1], -0.5f);
  }
  EXPECT_FLOAT_EQ(obs.goal()[0], 0.1f);
  EXPECT_FLOAT_EQ(obs.goal()[2], 0.3f);
}

TEST(ObservationHistory, OrdersOldestToNewest)
{
  ObservationHistory history;
  for (int step = 0; step < spec::kHistoryFrames + 3; ++step) {
    std::vector<float> bins(static_cast<std::size_t>(spec::kLidarBins),
                            static_cast<float>(step) * 0.01f);
    history.push(bins, static_cast<float>(step), 0.0f);
  }

  Observation obs;
  const float goal[3] = {0.0f, 0.0f, 1.0f};
  history.fill(goal, obs);

  // Seven pushes, four kept: steps 3, 4, 5, 6 oldest to newest.
  for (int f = 0; f < spec::kHistoryFrames; ++f) {
    const float expected = static_cast<float>(f + 3);
    EXPECT_FLOAT_EQ(obs.action()[f * 2], expected) << "frame " << f;
    EXPECT_NEAR(obs.lidar()[f * spec::kLidarBins], expected * 0.01f, 1e-6f);
  }
}

TEST(PhysicalAction, RespectsTheCreate3Envelope)
{
  // v is a half-range: a differential base cannot be commanded backwards here,
  // and omega is symmetric. These two constants are shared with the simulator's
  // denormalize_action and must not drift apart.
  const Command forward = physicalAction(20.0f, 20.0f);
  EXPECT_NEAR(forward.v, spec::kVMax, 1e-5);
  EXPECT_NEAR(forward.omega, spec::kOmegaMax, 1e-5);

  const Command reverse = physicalAction(-20.0f, -20.0f);
  EXPECT_NEAR(reverse.v, 0.0f, 1e-5);
  EXPECT_NEAR(reverse.omega, -spec::kOmegaMax, 1e-5);

  const Command neutral = physicalAction(0.0f, 0.0f);
  EXPECT_NEAR(neutral.v, 0.5f * spec::kVMax, 1e-6);
  EXPECT_NEAR(neutral.omega, 0.0f, 1e-6);
}

TEST(SpecContract, ObservationLayoutIsTheOneTheActorWasExportedAgainst)
{
  // 4 frames * 180 bins + 3 goal + 4 * 2 actions = 731. If this ever changes,
  // every checkpoint and every .pte is invalidated, so it is asserted rather
  // than assumed.
  EXPECT_EQ(spec::kObsDim, 731);
  EXPECT_EQ(spec::kGoalOffset, 720u);
  EXPECT_EQ(spec::kActionOffset, 723u);
}
