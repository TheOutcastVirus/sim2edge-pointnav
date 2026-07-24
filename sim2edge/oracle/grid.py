"""Occupancy grids, inflation, and geodesic distance fields.

The geodesic distance field is the backbone of the whole training pipeline. It
is used three times over, and it is important that it is the *same* field each
time:

1. the oracle expert descends it to produce demonstration paths,
2. the PPO progress reward is measured along it (Euclidean progress pays the
   robot to drive into the concave side of an obstacle),
3. episode feasibility is checked by asking whether it is finite at the start.

Grids are indexed ``[row, col]`` = ``[y, x]``, with cell centres at
``(col + 0.5) * res`` and origin at the arena's lower-left corner.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# SciPy is optional. It is a large speedup where it works, but it is caught
# broadly rather than on ImportError alone: a SciPy built against a different
# NumPy ABI raises AttributeError at import time, and dataset generation must
# degrade to the numpy fallbacks instead of dying. The fallbacks below are
# exact, not approximations, so results do not depend on which path ran.
try:  # C-speed shortest paths.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra as _scipy_dijkstra

    _HAVE_SCIPY_CSGRAPH = True
except Exception:  # noqa: BLE001 - see comment above
    _HAVE_SCIPY_CSGRAPH = False

try:
    from scipy.ndimage import distance_transform_edt as _edt

    _HAVE_SCIPY_NDIMAGE = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY_NDIMAGE = False

__all__ = ["OccupancyGrid", "distance_field", "obstacle_distance", "inflate"]

#: Sentinel for unreachable cells. Finite so it survives arithmetic and
#: comparisons without producing NaNs in reward differences.
UNREACHABLE = 1.0e6


@dataclass
class OccupancyGrid:
    """Boolean occupancy over a rectangular arena.

    ``occupied[r, c]`` is True where the cell is blocked. ``resolution`` is the
    cell size in metres; the arena's lower-left corner is world ``(0, 0)``.
    """

    occupied: np.ndarray  # (H, W) bool
    resolution: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.occupied.shape

    @property
    def height_m(self) -> float:
        return self.occupied.shape[0] * self.resolution

    @property
    def width_m(self) -> float:
        return self.occupied.shape[1] * self.resolution

    # ── Coordinate transforms ────────────────────────────────────────────────

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """World metres -> ``(row, col)``, clamped to the grid."""
        c = int(x / self.resolution)
        r = int(y / self.resolution)
        h, w = self.occupied.shape
        return (min(max(r, 0), h - 1), min(max(c, 0), w - 1))

    def cell_to_world(self, r: int, c: int) -> Tuple[float, float]:
        """``(row, col)`` -> world metres at the cell centre."""
        return ((c + 0.5) * self.resolution, (r + 0.5) * self.resolution)

    def world_to_cell_array(
        self, xy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized :meth:`world_to_cell` for an ``(N, 2)`` array."""
        h, w = self.occupied.shape
        c = np.clip((xy[..., 0] / self.resolution).astype(np.int64), 0, w - 1)
        r = np.clip((xy[..., 1] / self.resolution).astype(np.int64), 0, h - 1)
        return r, c

    def in_bounds(self, x: float, y: float) -> bool:
        return 0.0 <= x < self.width_m and 0.0 <= y < self.height_m

    def is_occupied(self, x: float, y: float) -> bool:
        if not self.in_bounds(x, y):
            return True  # outside the arena counts as blocked
        r, c = self.world_to_cell(x, y)
        return bool(self.occupied[r, c])

    def inflate(self, radius_m: float) -> "OccupancyGrid":
        """Grow obstacles by ``radius_m`` so the robot can be planned as a point."""
        return OccupancyGrid(inflate(self.occupied, radius_m, self.resolution),
                             self.resolution)


# ─────────────────────────────────────────────────────────────────────────────
# Inflation and clearance
# ─────────────────────────────────────────────────────────────────────────────


#: Distances beyond this are never used: the critic input saturates at 5 m and
#: inflation radii are centimetres. Capping bounds the fallback EDT's work.
_MAX_CLEARANCE_M = 6.0


def inflate(occupied: np.ndarray, radius_m: float, resolution: float) -> np.ndarray:
    """Dilate an occupancy mask by a metric radius (circular structuring element)."""
    if radius_m <= 0.0:
        return occupied.copy()
    # Distance to the nearest obstacle cell; anything within the radius becomes
    # occupied. Exact, and far cheaper than repeated dilation by a disc.
    dist = obstacle_distance(occupied, resolution,
                             max_dist_m=radius_m + 2.0 * resolution)
    return dist <= radius_m


def obstacle_distance(
    occupied: np.ndarray,
    resolution: float,
    max_dist_m: float = _MAX_CLEARANCE_M,
) -> np.ndarray:
    """Euclidean distance from each cell to the nearest occupied cell, metres.

    Feeds the critic's privileged "nearest obstacle distance" observation, the
    oracle's clearance-aware speed profile, and obstacle inflation. Distances
    are exact up to ``max_dist_m`` and saturate above it.
    """
    if _HAVE_SCIPY_NDIMAGE:
        return _edt(~occupied, sampling=resolution).astype(np.float32)
    return _edt_separable(occupied, resolution, max_dist_m)


def _edt_separable(
    occupied: np.ndarray, resolution: float, max_dist_m: float
) -> np.ndarray:
    """Exact bounded Euclidean distance transform, numpy-vectorized.

    Standard separable EDT: a 1D distance along columns, then a windowed minimum
    of ``dy^2 + dx^2`` along rows. The window is bounded by ``max_dist_m``, which
    is what makes the second pass a fixed number of whole-array operations
    instead of a per-row parabola envelope. Vectorized across the non-scanned
    axis, so the Python-level loop count is ``H + 2k`` rather than ``H * W``.
    """
    h, w = occupied.shape
    k = int(math.ceil(max_dist_m / resolution))
    big = np.float32(k + 1)

    # ── Pass 1: distance in cells to the nearest occupied cell in the column ──
    g = np.where(occupied, np.float32(0.0), big)
    for r in range(1, h):  # sweep down
        np.minimum(g[r], g[r - 1] + 1.0, out=g[r])
    for r in range(h - 2, -1, -1):  # sweep up
        np.minimum(g[r], g[r + 1] + 1.0, out=g[r])
    np.minimum(g, big, out=g)

    # ── Pass 2: combine across the row within the bounded window ─────────────
    g2 = g * g
    padded = np.full((h, w + 2 * k), big * big, dtype=np.float32)
    padded[:, k:k + w] = g2
    best = padded[:, k:k + w].copy()
    for dx in range(1, k + 1):
        cand = np.minimum(padded[:, k - dx:k - dx + w],
                          padded[:, k + dx:k + dx + w]) + np.float32(dx * dx)
        np.minimum(best, cand, out=best)

    out = np.sqrt(best) * np.float32(resolution)
    return np.minimum(out, np.float32(max_dist_m)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Geodesic distance field
# ─────────────────────────────────────────────────────────────────────────────


def distance_field(
    free: np.ndarray,
    goal_rc: Tuple[int, int],
    resolution: float,
) -> np.ndarray:
    """8-connected geodesic distance (metres) from ``goal_rc`` over ``free``.

    Unreachable and occupied cells are :data:`UNREACHABLE`. Diagonal moves cost
    ``sqrt(2) * resolution``, so the field approximates true Euclidean geodesic
    distance to within the usual ~8% octile-metric error — irrelevant for a
    reward shaping term and for gradient descent along the field.
    """
    free = np.ascontiguousarray(free, dtype=bool)
    h, w = free.shape
    gr, gc = goal_rc
    if not free[gr, gc]:
        # The goal cell was inflated away (goal tight against an obstacle).
        # Snap to the nearest free cell so the episode stays solvable.
        snapped = _nearest_free(free, gr, gc)
        if snapped is None:
            return np.full((h, w), UNREACHABLE, dtype=np.float32)
        gr, gc = snapped

    if _HAVE_SCIPY_CSGRAPH:
        return _distance_field_scipy(free, (gr, gc), resolution)
    return _distance_field_heapq(free, (gr, gc), resolution)


def _neighbor_offsets(resolution: float):
    s, d = resolution, resolution * math.sqrt(2.0)
    return (
        (-1, 0, s), (1, 0, s), (0, -1, s), (0, 1, s),
        (-1, -1, d), (-1, 1, d), (1, -1, d), (1, 1, d),
    )


def _distance_field_scipy(
    free: np.ndarray, goal_rc: Tuple[int, int], resolution: float
) -> np.ndarray:
    h, w = free.shape
    n = h * w
    idx = np.arange(n, dtype=np.int64).reshape(h, w)

    rows, cols, vals = [], [], []
    for dr, dc, cost in _neighbor_offsets(resolution):
        # Overlapping slices of the index grid give every (cell, neighbour) pair
        # in one shot, with no Python-level loop over cells.
        r0, r1 = max(0, -dr), h - max(0, dr)
        c0, c1 = max(0, -dc), w - max(0, dc)
        src = idx[r0:r1, c0:c1]
        dst = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        ok = free[r0:r1, c0:c1] & free[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        rows.append(src[ok])
        cols.append(dst[ok])
        vals.append(np.full(int(ok.sum()), cost, dtype=np.float64))

    graph = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()

    dist = _scipy_dijkstra(graph, directed=False,
                           indices=goal_rc[0] * w + goal_rc[1])
    out = dist.reshape(h, w).astype(np.float32)
    out[~np.isfinite(out)] = UNREACHABLE
    out[~free] = UNREACHABLE
    return out


def _distance_field_heapq(
    free: np.ndarray, goal_rc: Tuple[int, int], resolution: float
) -> np.ndarray:
    h, w = free.shape
    dist = np.full((h, w), UNREACHABLE, dtype=np.float32)
    gr, gc = goal_rc
    dist[gr, gc] = 0.0
    offsets = _neighbor_offsets(resolution)
    pq = [(0.0, gr, gc)]
    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        for dr, dc, cost in offsets:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w or not free[nr, nc]:
                continue
            nd = d + cost
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(pq, (nd, nr, nc))
    dist[~free] = UNREACHABLE
    return dist


def _nearest_free(
    free: np.ndarray, r: int, c: int, max_radius: int = 40
) -> Optional[Tuple[int, int]]:
    """Nearest free cell to ``(r, c)`` by expanding ring search."""
    h, w = free.shape
    for rad in range(1, max_radius + 1):
        r0, r1 = max(0, r - rad), min(h, r + rad + 1)
        c0, c1 = max(0, c - rad), min(w, c + rad + 1)
        window = free[r0:r1, c0:c1]
        if not window.any():
            continue
        rr, cc = np.nonzero(window)
        rr, cc = rr + r0, cc + c0
        best = np.argmin((rr - r) ** 2 + (cc - c) ** 2)
        return (int(rr[best]), int(cc[best]))
    return None


def sample_field(field: np.ndarray, grid: OccupancyGrid,
                 x: float, y: float) -> float:
    """Look up a grid field at a world position (nearest cell)."""
    r, c = grid.world_to_cell(x, y)
    return float(field[r, c])
