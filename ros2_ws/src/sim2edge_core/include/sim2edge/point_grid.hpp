// A uniform bucket grid over the submap, replacing scipy's cKDTree.
//
// This is the single change that moves the localizer off the critical path.
// The measured board cost of the Python tracker is 21-27 ms of a 50 ms control
// period (docs/daily/2026_08_04_log.md), and the dominant term is rebuilding a
// KD-tree over ~46,000 submap points every tick -- O(N log N) with a large
// constant, paid in full even though the geometry only changed by one scan.
//
// Two properties of this specific problem make a grid strictly better here,
// rather than merely faster:
//
// 1. **The search is radius-limited by definition.** ICP rejects any
//    correspondence beyond `max_correspondence_m` (0.20 m), so a neighbour
//    further away than that is not "a worse answer", it is *no answer*. Sizing
//    the cell at the rejection radius makes a 3x3 cell scan an exact search
//    over precisely the region that can produce a usable match.
// 2. **The submap is local and bounded.** It is a short-lived scan-to-submap
//    tracker over a few metres, not a global map, so a dense grid over the
//    bounding box is a small array rather than a sparse-hash problem.
//
// Construction is a counting sort: count per cell, prefix-sum, scatter. O(N),
// contiguous, and it reuses its buffers across ticks so a steady-state control
// loop performs no allocation.
//
// The results are *exact*, not approximate. `nearestWithin` returns the true
// nearest neighbour whenever one exists inside the radius, and `knn` returns
// the true k nearest by expanding rings until the ring distance exceeds the
// current worst candidate. Nothing here trades accuracy for speed.

#pragma once

#include <cstddef>
#include <vector>

namespace sim2edge
{

class PointGrid
{
public:
  /// Rebuild over `n` points. `cell` is the bucket edge length in metres and
  /// should be the correspondence rejection radius.
  void build(const float * xs, const float * ys, std::size_t n, float cell);

  /// Nearest neighbour within `radius`, or -1 if the cell neighbourhood is
  /// empty. `out_d2` receives the squared distance when one is found.
  ///
  /// Only the 3x3 cell block around the query is scanned, which is exact for
  /// `radius <= cell`: any point closer than one cell edge must lie in it.
  int nearestWithin(float x, float y, float radius, float & out_d2) const;

  /// The `k` nearest neighbours, written to `out` (capacity >= k), returning
  /// how many were found. Rings are expanded until the guaranteed-covered
  /// distance exceeds the current k-th best, so the result is exact.
  int knn(float x, float y, int k, int * out) const;

  std::size_t size() const { return count_; }
  bool empty() const { return count_ == 0; }

private:
  int cellIndex(int ix, int iy) const { return iy * nx_ + ix; }

  const float * xs_{nullptr};
  const float * ys_{nullptr};
  std::size_t count_{0};

  float cell_{0.2f};
  float inv_cell_{5.0f};
  float min_x_{0.0f};
  float min_y_{0.0f};
  int nx_{0};
  int ny_{0};

  // Counting-sort storage: `starts_` has nx_*ny_+1 entries, `order_` holds
  // point indices grouped by cell.
  std::vector<int> starts_;
  std::vector<int> order_;
  std::vector<int> counter_;
};

}  // namespace sim2edge
