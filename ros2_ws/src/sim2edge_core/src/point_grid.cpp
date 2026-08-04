#include "sim2edge/point_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sim2edge
{
namespace
{
// A submap that somehow spans a huge area would allocate a huge grid. Cap the
// cell count and coarsen instead: a coarser grid is slower, a 4 GB allocation
// is a crash. The tracker is local and this should never trigger, which is
// exactly why it must not be an assert.
constexpr int kMaxCells = 1 << 20;
}  // namespace

void PointGrid::build(const float * xs, const float * ys, std::size_t n,
                      float cell)
{
  xs_ = xs;
  ys_ = ys;
  count_ = n;
  nx_ = ny_ = 0;
  starts_.clear();
  order_.clear();

  if (n == 0) {
    return;
  }

  float min_x = xs[0];
  float max_x = xs[0];
  float min_y = ys[0];
  float max_y = ys[0];
  for (std::size_t i = 1; i < n; ++i) {
    min_x = std::min(min_x, xs[i]);
    max_x = std::max(max_x, xs[i]);
    min_y = std::min(min_y, ys[i]);
    max_y = std::max(max_y, ys[i]);
  }

  cell_ = std::max(cell, 1.0e-3f);
  for (;;) {
    const double span_x = static_cast<double>(max_x - min_x) / cell_;
    const double span_y = static_cast<double>(max_y - min_y) / cell_;
    const double cells_x = std::floor(span_x) + 3.0;  // one guard ring each side
    const double cells_y = std::floor(span_y) + 3.0;
    if (cells_x * cells_y <= kMaxCells) {
      nx_ = static_cast<int>(cells_x);
      ny_ = static_cast<int>(cells_y);
      break;
    }
    cell_ *= 2.0f;
  }
  inv_cell_ = 1.0f / cell_;
  // Shift the origin one cell out so the guard ring keeps every 3x3 lookup in
  // bounds without a per-query clamp.
  min_x_ = min_x - cell_;
  min_y_ = min_y - cell_;

  const int cells = nx_ * ny_;
  counter_.assign(static_cast<std::size_t>(cells) + 1, 0);
  order_.resize(n);

  // Pass 1: count. Pass 2: prefix sum. Pass 3: scatter.
  for (std::size_t i = 0; i < n; ++i) {
    const int ix = std::clamp(
      static_cast<int>((xs[i] - min_x_) * inv_cell_), 0, nx_ - 1);
    const int iy = std::clamp(
      static_cast<int>((ys[i] - min_y_) * inv_cell_), 0, ny_ - 1);
    ++counter_[static_cast<std::size_t>(cellIndex(ix, iy)) + 1];
  }
  for (int c = 0; c < cells; ++c) {
    counter_[static_cast<std::size_t>(c) + 1] +=
      counter_[static_cast<std::size_t>(c)];
  }
  starts_ = counter_;  // immutable copy; counter_ is consumed as a cursor
  for (std::size_t i = 0; i < n; ++i) {
    const int ix = std::clamp(
      static_cast<int>((xs[i] - min_x_) * inv_cell_), 0, nx_ - 1);
    const int iy = std::clamp(
      static_cast<int>((ys[i] - min_y_) * inv_cell_), 0, ny_ - 1);
    order_[static_cast<std::size_t>(counter_[
      static_cast<std::size_t>(cellIndex(ix, iy))]++)] = static_cast<int>(i);
  }
}

int PointGrid::nearestWithin(float x, float y, float radius,
                             float & out_d2) const
{
  if (count_ == 0) {
    return -1;
  }
  const int cx = std::clamp(static_cast<int>((x - min_x_) * inv_cell_), 0,
                            nx_ - 1);
  const int cy = std::clamp(static_cast<int>((y - min_y_) * inv_cell_), 0,
                            ny_ - 1);

  // Exact for radius <= cell_: anything nearer than one cell edge is inside the
  // 3x3 block. `rings` generalizes that if a caller ever passes a larger radius.
  const int rings = std::max(1, static_cast<int>(std::ceil(radius * inv_cell_)));
  const int lo_x = std::max(cx - rings, 0);
  const int hi_x = std::min(cx + rings, nx_ - 1);
  const int lo_y = std::max(cy - rings, 0);
  const int hi_y = std::min(cy + rings, ny_ - 1);

  float best = radius * radius;
  int best_index = -1;
  for (int iy = lo_y; iy <= hi_y; ++iy) {
    for (int ix = lo_x; ix <= hi_x; ++ix) {
      const int c = cellIndex(ix, iy);
      const int begin = starts_[static_cast<std::size_t>(c)];
      const int end = starts_[static_cast<std::size_t>(c) + 1];
      for (int s = begin; s < end; ++s) {
        const int p = order_[static_cast<std::size_t>(s)];
        const float dx = xs_[p] - x;
        const float dy = ys_[p] - y;
        const float d2 = dx * dx + dy * dy;
        if (d2 < best) {
          best = d2;
          best_index = p;
        }
      }
    }
  }
  if (best_index >= 0) {
    out_d2 = best;
  }
  return best_index;
}

int PointGrid::knn(float x, float y, int k, int * out) const
{
  if (count_ == 0 || k <= 0) {
    return 0;
  }
  k = std::min(k, static_cast<int>(count_));

  const int cx = std::clamp(static_cast<int>((x - min_x_) * inv_cell_), 0,
                            nx_ - 1);
  const int cy = std::clamp(static_cast<int>((y - min_y_) * inv_cell_), 0,
                            ny_ - 1);

  // Small insertion-sorted candidate list. k is 7 here, so a heap would cost
  // more in branches than the linear insert saves.
  constexpr int kMaxK = 16;
  float best_d2[kMaxK];
  int best_i[kMaxK];
  int found = 0;
  k = std::min(k, kMaxK);

  const int max_ring = std::max(nx_, ny_);
  for (int ring = 0; ring <= max_ring; ++ring) {
    // Everything inside ring-1 is already searched, so once the k-th candidate
    // is closer than the nearest point this ring could possibly hold, stop.
    // The guaranteed-covered distance after searching rings [0, ring-1] is
    // (ring-1) cell edges, hence the (ring - 1) here rather than ring.
    if (found == k && ring > 0) {
      const float covered = static_cast<float>(ring - 1) * cell_;
      if (covered > 0.0f && best_d2[k - 1] <= covered * covered) {
        break;
      }
    }

    const int lo_x = cx - ring;
    const int hi_x = cx + ring;
    const int lo_y = cy - ring;
    const int hi_y = cy + ring;
    if (lo_x < 0 && hi_x >= nx_ && lo_y < 0 && hi_y >= ny_ && ring > 0 &&
        found == k)
    {
      break;  // the ring lies entirely outside the grid; nothing further exists
    }

    bool any_cell = false;
    for (int iy = std::max(lo_y, 0); iy <= std::min(hi_y, ny_ - 1); ++iy) {
      const bool edge_row = (iy == lo_y || iy == hi_y);
      for (int ix = std::max(lo_x, 0); ix <= std::min(hi_x, nx_ - 1); ++ix) {
        // Only the perimeter of the ring is new; the interior was covered by
        // earlier rings.
        if (ring > 0 && !edge_row && ix != lo_x && ix != hi_x) {
          continue;
        }
        any_cell = true;
        const int c = cellIndex(ix, iy);
        const int begin = starts_[static_cast<std::size_t>(c)];
        const int end = starts_[static_cast<std::size_t>(c) + 1];
        for (int s = begin; s < end; ++s) {
          const int p = order_[static_cast<std::size_t>(s)];
          const float dx = xs_[p] - x;
          const float dy = ys_[p] - y;
          const float d2 = dx * dx + dy * dy;
          if (found < k) {
            int j = found++;
            while (j > 0 && best_d2[j - 1] > d2) {
              best_d2[j] = best_d2[j - 1];
              best_i[j] = best_i[j - 1];
              --j;
            }
            best_d2[j] = d2;
            best_i[j] = p;
          } else if (d2 < best_d2[k - 1]) {
            int j = k - 1;
            while (j > 0 && best_d2[j - 1] > d2) {
              best_d2[j] = best_d2[j - 1];
              best_i[j] = best_i[j - 1];
              --j;
            }
            best_d2[j] = d2;
            best_i[j] = p;
          }
        }
      }
    }
    if (!any_cell && ring > 0) {
      break;  // walked off the grid entirely
    }
  }

  for (int i = 0; i < found; ++i) {
    out[i] = best_i[i];
  }
  return found;
}

}  // namespace sim2edge
