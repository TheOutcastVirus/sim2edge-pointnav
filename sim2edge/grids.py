"""Occupancy, clearance, and geodesic distance on a uniform grid.

CPU numpy, run **once per layout** when the pool is built -- never in the
training loop. The training loop only ever indexes into the arrays produced
here, which is why they are worth computing exactly rather than approximately.

Grid convention: ``a[r, c]``, with ``r`` along +y and ``c`` along +x, cell
centres at ``((c + 0.5) * res, (r + 0.5) * res)``. The origin is the arena's
lower-left corner, so all world coordinates in a room are non-negative.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

__all__ = ["INF", "UNREACHABLE", "world_to_cell", "cell_to_world", "rasterize",
           "clearance", "geodesic", "flow_field"]

#: Sentinel for "no path exists". Chosen so that INF + a few thousand step
#: costs is still representable in float32 without changing value.
INF = np.float32(1e9)

#: Anything at or above this is unreachable. Compare against this, not INF,
#: because relaxation adds step costs to the sentinel.
UNREACHABLE = np.float32(1e8)

# 8-connected neighbourhood as (dr, dc, cost_in_cells).
_NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, np.sqrt(2.0)), (-1, 1, np.sqrt(2.0)),
    (1, -1, np.sqrt(2.0)), (1, 1, np.sqrt(2.0)),
]


def world_to_cell(x, y, res: float):
    """World metres -> (row, col). Vectorized; accepts scalars or arrays."""
    return (np.floor(np.asarray(y) / res).astype(np.int32),
            np.floor(np.asarray(x) / res).astype(np.int32))


def cell_to_world(r, c, res: float):
    """(row, col) -> world metres at the *cell centre*."""
    return ((np.asarray(c) + 0.5) * res, (np.asarray(r) + 0.5) * res)


def _shift(a: np.ndarray, dr: int, dc: int, fill) -> np.ndarray:
    """``out[r, c] = a[r + dr, c + dc]``, with ``fill`` outside the array.

    Written as slice arithmetic rather than ``np.roll`` because roll wraps, and
    a wrapped geodesic field would let the planner tunnel through the arena
    wall and out the other side.
    """
    out = np.full_like(a, fill)
    h, w = a.shape
    dst_r = slice(max(0, -dr), min(h, h - dr))
    src_r = slice(max(0, dr), min(h, h + dr))
    dst_c = slice(max(0, -dc), min(w, w - dc))
    src_c = slice(max(0, dc), min(w, w + dc))
    out[dst_r, dst_c] = a[src_r, src_c]
    return out


def rasterize(kind: np.ndarray, pos: np.ndarray, half: np.ndarray,
              yaw: np.ndarray, valid: np.ndarray, size_xy, res: float,
              wall_thickness: float) -> np.ndarray:
    """Fixed-length obstacle arrays -> boolean occupancy, walls included.

    ``kind`` is 0 for an axis-aligned box and 1 for a cylinder. For a cylinder
    ``half[:, 0]`` is the radius and ``half[:, 1]`` is unused. Invalid slots
    (``valid == False``) are skipped -- they exist so Isaac can pre-allocate a
    fixed number of prims per environment and simply park the unused ones.

    The loop runs over at most a couple of dozen obstacles while each body is a
    full-grid vectorized op, so this is not the place to optimize further.
    """
    w_m, h_m = float(size_xy[0]), float(size_xy[1])
    nc, nr = int(round(w_m / res)), int(round(h_m / res))
    xs = (np.arange(nc, dtype=np.float32) + 0.5) * res
    ys = (np.arange(nr, dtype=np.float32) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)                       # both (nr, nc)

    occ = np.zeros((nr, nc), dtype=bool)

    # Boundary walls. A robot that can leave the arena makes the geodesic field
    # meaningless, so these are obstacles like any other.
    t = wall_thickness
    occ |= (X < t) | (X > w_m - t) | (Y < t) | (Y > h_m - t)

    for i in np.flatnonzero(valid):
        px, py = float(pos[i, 0]), float(pos[i, 1])
        if kind[i] == 0:
            # Rotate the query points into the box frame rather than rotating
            # the box: an axis-aligned test in that frame is exact, and it is
            # the same arithmetic Isaac applies to the prim's orientation.
            ca, sa = np.cos(-yaw[i]), np.sin(-yaw[i])
            dx, dy = X - px, Y - py
            occ |= ((np.abs(dx * ca - dy * sa) <= half[i, 0])
                    & (np.abs(dx * sa + dy * ca) <= half[i, 1]))
        else:
            occ |= ((X - px) ** 2 + (Y - py) ** 2) <= half[i, 0] ** 2
    return occ


def clearance(occ: np.ndarray, res: float) -> np.ndarray:
    """Metres from each cell centre to the nearest occupied cell.

    Exact Euclidean, via scipy's separable transform. Used for two things:
    inflating obstacles by the robot radius (a robot centre is legal exactly
    where ``clearance >= radius``), and rejecting spawn poses jammed against a
    wall, which are technically valid and pedagogically useless.

    Sim-side only -- this never has to be reimplemented in C++.
    """
    return (distance_transform_edt(~occ, sampling=res)).astype(np.float32)


def geodesic(free: np.ndarray, goal_rc, res: float,
             max_iters: int = 10000,
             cost_scale: np.ndarray | None = None) -> np.ndarray:
    """Shortest-path distance in metres from every free cell to ``goal_rc``.

    Octile metric on the 8-connected free space. Diagonal moves are forbidden
    when either shared cardinal neighbour is blocked, so a path cannot squeeze
    through the corner contact between two obstacles -- a gap of exactly zero
    width that the robot obviously cannot use.

    Implementation is vectorized Bellman-Ford: every sweep relaxes the whole
    grid against its 8 neighbours at once, and information travels one cell per
    sweep, so it converges in roughly the longest path length in cells. For a
    10 m arena at 5 cm that is a few hundred sweeps of a 200x200 array -- well
    under a second, and it happens once per layout at pool-build time. A heap
    based Dijkstra would need fewer operations but would run them in Python,
    which is strictly slower at this size.

    Returns ``INF``-valued cells where no path exists, including all blocked
    cells. Compare against :data:`UNREACHABLE`, not equality with ``INF``.
    """
    scale = (np.ones_like(free, dtype=np.float32) if cost_scale is None
             else cost_scale.astype(np.float32))

    gr, gc = int(goal_rc[0]), int(goal_rc[1])
    D = np.full(free.shape, INF, dtype=np.float32)
    if not free[gr, gc]:
        return D                       # goal inside an obstacle: nothing reachable
    D[gr, gc] = 0.0

    # Precompute the diagonal corner-cut masks once; they depend only on `free`.
    masks = []
    for dr, dc, cost in _NEIGHBOURS:
        if dr != 0 and dc != 0:
            passable = _shift(free, dr, 0, False) & _shift(free, 0, dc, False)
        else:
            passable = None
        # Step costs carry `res` so the field comes out in metres directly;
        # scaling afterwards would also scale the INF sentinel.
        masks.append((dr, dc, np.float32(cost * res), passable))

    for _ in range(max_iters):
        best = D
        for dr, dc, cost, passable in masks:
            cand = _shift(D, dr, dc, INF) + cost * scale
            if passable is not None:
                cand = np.where(passable, cand, INF)
            best = np.minimum(best, cand)
        best = np.where(free, np.minimum(best, INF), INF)
        if np.array_equal(best, D):
            return D
        D = best
    raise RuntimeError("geodesic() did not converge; grid larger than expected")


def flow_field(free: np.ndarray, D: np.ndarray, res: float,
               cost_scale: np.ndarray | None = None) -> np.ndarray:
    """For every cell, the flat index of the next cell on the shortest path.

    This is the whole reason the oracle is cheap. Following the gradient of
    ``D`` on the GPU would mean an 8-way comparison plus the corner-cut test at
    every step, for every environment, every control tick. Precomputing the
    answer once per layout turns the oracle's path walk into a chain of gathers:
    ``idx = flow[idx]``, repeated. The geometry stays here in numpy where it is
    tested, and the device side has no branches at all.

    The goal cell and every unreachable cell point at themselves, so walking is
    idempotent at the ends and needs no termination check on device.
    """
    h, w = free.shape
    flat = np.arange(h * w, dtype=np.int32).reshape(h, w)

    # Minimum over *neighbours*, excluding the cell itself. Seeding this with
    # D would make every comparison a tie -- by construction the best neighbour
    # realizes D exactly, since that is how D was computed -- and every cell
    # would self-loop.
    scale = (np.ones_like(D, dtype=np.float32) if cost_scale is None
             else cost_scale.astype(np.float32))

    best = np.full_like(D, INF)
    nxt = flat.copy()

    for dr, dc, cost in _NEIGHBOURS:
        cand = _shift(D, dr, dc, INF) + np.float32(cost * res) * scale
        if dr != 0 and dc != 0:
            passable = _shift(free, dr, 0, False) & _shift(free, 0, dc, False)
            cand = np.where(passable, cand, INF)
        # Strictly-less keeps the first neighbour on ties, so the field is
        # deterministic for a given neighbour ordering.
        take = cand < best
        best = np.where(take, cand, best)
        nxt = np.where(take, _shift(flat, dr, dc, 0), nxt)

    # Self-loop wherever no neighbour improves on the cell's own distance: the
    # goal (where D == 0), and anything walled off. `best > D` is an exact
    # comparison because `best` recomputes the same float32 expression
    # `geodesic` reached its fixed point on.
    stuck = ~free | (D >= UNREACHABLE) | (best > D)
    nxt = np.where(stuck, flat, nxt)

    # Recovery. A robot only ever *starts* on the free grid; tracking error,
    # physics, and deliberate action perturbation all push it off. Leaving
    # those cells self-looping means the oracle's lookahead point becomes the
    # robot's own position, which yields a meaningless bearing and, in practice,
    # a robot that spins in place until the episode times out.
    #
    # So every non-free cell points at the nearest free one: the first hop of
    # the walk climbs back onto the path and the remaining hops descend it
    # normally. One EDT call, and the device side stays a plain gather chain.
    _, (near_r, near_c) = distance_transform_edt(~free, return_indices=True)
    nxt = np.where(free, nxt, near_r * w + near_c)
    return nxt.astype(np.int32)
