// The grid replaces scipy's cKDTree in the localizer's inner loop, so it has
// to be *exact* rather than merely fast. These tests compare it against brute
// force on adversarial layouts -- clustered, collinear, and empty regions --
// because an approximate nearest neighbour would show up as a slow drift in the
// pose estimate and nowhere else.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sim2edge/point_grid.hpp"

using namespace sim2edge;

namespace
{

int bruteNearest(const std::vector<float> & xs, const std::vector<float> & ys,
                 float qx, float qy, float radius, float & d2)
{
  int best = -1;
  float best_d2 = radius * radius;
  for (std::size_t i = 0; i < xs.size(); ++i) {
    const float dx = xs[i] - qx;
    const float dy = ys[i] - qy;
    const float candidate = dx * dx + dy * dy;
    if (candidate < best_d2) {
      best_d2 = candidate;
      best = static_cast<int>(i);
    }
  }
  if (best >= 0) {
    d2 = best_d2;
  }
  return best;
}

std::vector<int> bruteKnn(const std::vector<float> & xs,
                          const std::vector<float> & ys, float qx, float qy,
                          int k)
{
  std::vector<std::pair<float, int>> all;
  all.reserve(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) {
    const float dx = xs[i] - qx;
    const float dy = ys[i] - qy;
    all.emplace_back(dx * dx + dy * dy, static_cast<int>(i));
  }
  std::sort(all.begin(), all.end());
  std::vector<int> out;
  for (int i = 0; i < k && i < static_cast<int>(all.size()); ++i) {
    out.push_back(all[static_cast<std::size_t>(i)].second);
  }
  return out;
}

}  // namespace

TEST(PointGrid, EmptyGridFindsNothingAndDoesNotCrash)
{
  PointGrid grid;
  grid.build(nullptr, nullptr, 0, 0.2f);
  float d2 = 0.0f;
  EXPECT_EQ(grid.nearestWithin(0.0f, 0.0f, 0.2f, d2), -1);
  int out[7];
  EXPECT_EQ(grid.knn(0.0f, 0.0f, 7, out), 0);
}

TEST(PointGrid, NearestMatchesBruteForceOnRandomClouds)
{
  std::mt19937 rng(20260803);
  std::uniform_real_distribution<float> coord(-8.0f, 8.0f);

  std::vector<float> xs(4000);
  std::vector<float> ys(4000);
  for (std::size_t i = 0; i < xs.size(); ++i) {
    xs[i] = coord(rng);
    ys[i] = coord(rng);
  }

  PointGrid grid;
  const float radius = 0.20f;
  grid.build(xs.data(), ys.data(), xs.size(), radius);

  for (int trial = 0; trial < 2000; ++trial) {
    const float qx = coord(rng);
    const float qy = coord(rng);

    float grid_d2 = 0.0f;
    float brute_d2 = 0.0f;
    const int a = grid.nearestWithin(qx, qy, radius, grid_d2);
    const int b = bruteNearest(xs, ys, qx, qy, radius, brute_d2);

    ASSERT_EQ(a < 0, b < 0) << "disagreement on whether a match exists";
    if (a >= 0) {
      // Ties are possible in principle; compare the distance, not the index.
      EXPECT_NEAR(grid_d2, brute_d2, 1e-9f);
    }
  }
}

TEST(PointGrid, KnnMatchesBruteForceIncludingTheQueryPointItself)
{
  // The localizer queries normals at points that ARE submap points, so the
  // query's own zero-distance self-match must be returned -- that is what the
  // Python does via cKDTree and it changes the PCA neighbourhood.
  std::mt19937 rng(4711);
  std::uniform_real_distribution<float> coord(-3.0f, 3.0f);

  std::vector<float> xs(1500);
  std::vector<float> ys(1500);
  for (std::size_t i = 0; i < xs.size(); ++i) {
    xs[i] = coord(rng);
    ys[i] = coord(rng);
  }

  PointGrid grid;
  grid.build(xs.data(), ys.data(), xs.size(), 0.20f);

  for (int trial = 0; trial < 400; ++trial) {
    const std::size_t index = static_cast<std::size_t>(rng() % xs.size());
    int found[7];
    const int count = grid.knn(xs[index], ys[index], 7, found);
    ASSERT_EQ(count, 7);

    const std::vector<int> expected = bruteKnn(xs, ys, xs[index], ys[index], 7);
    std::vector<float> got_d2;
    std::vector<float> want_d2;
    for (int i = 0; i < 7; ++i) {
      const float gdx = xs[static_cast<std::size_t>(found[i])] - xs[index];
      const float gdy = ys[static_cast<std::size_t>(found[i])] - ys[index];
      got_d2.push_back(gdx * gdx + gdy * gdy);
      const float wdx = xs[static_cast<std::size_t>(expected[static_cast<std::size_t>(i)])] - xs[index];
      const float wdy = ys[static_cast<std::size_t>(expected[static_cast<std::size_t>(i)])] - ys[index];
      want_d2.push_back(wdx * wdx + wdy * wdy);
    }
    for (int i = 0; i < 7; ++i) {
      EXPECT_NEAR(got_d2[static_cast<std::size_t>(i)],
                  want_d2[static_cast<std::size_t>(i)], 1e-9f)
        << "neighbour " << i;
    }
    EXPECT_NEAR(got_d2[0], 0.0f, 1e-12f) << "self-match must be first";
  }
}

TEST(PointGrid, HandlesSparseCloudsWhereRingsMustExpand)
{
  // Four points, far apart: a 3x3 block around most queries is empty, so knn
  // has to keep expanding rather than return short.
  std::vector<float> xs = {-5.0f, 5.0f, -5.0f, 5.0f};
  std::vector<float> ys = {-5.0f, -5.0f, 5.0f, 5.0f};

  PointGrid grid;
  grid.build(xs.data(), ys.data(), xs.size(), 0.20f);

  int found[7];
  const int count = grid.knn(0.0f, 0.0f, 7, found);
  EXPECT_EQ(count, 4) << "must return everything available, not give up early";

  float d2 = 0.0f;
  EXPECT_EQ(grid.nearestWithin(0.0f, 0.0f, 0.2f, d2), -1)
    << "nothing is within the correspondence radius";
  EXPECT_GE(grid.nearestWithin(-5.0f, -5.0f, 0.2f, d2), 0);
}

TEST(PointGrid, HandlesCollinearAndCoincidentPoints)
{
  // A flat wall is the common case and produces exactly collinear points; the
  // PCA normal downstream depends on this neighbourhood being right.
  std::vector<float> xs;
  std::vector<float> ys;
  for (int i = 0; i < 200; ++i) {
    xs.push_back(0.01f * static_cast<float>(i));
    ys.push_back(2.0f);
  }
  xs.push_back(0.5f);
  ys.push_back(2.0f);  // duplicate of an existing coordinate

  PointGrid grid;
  grid.build(xs.data(), ys.data(), xs.size(), 0.20f);

  int found[7];
  EXPECT_EQ(grid.knn(0.5f, 2.0f, 7, found), 7);

  float d2 = 0.0f;
  const int nearest = grid.nearestWithin(0.5f, 2.0f, 0.2f, d2);
  ASSERT_GE(nearest, 0);
  EXPECT_NEAR(d2, 0.0f, 1e-12f);
}
