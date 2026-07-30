"""Procedural room generation. Pure numpy -- no simulator import, by design.

Isaac consumes the output of this module; it does not produce it. The reason is
the oracle: planning needs an occupancy grid and a geodesic field, and deriving
those from spawned USD prims is a lossy round-trip through geometry we already
had in analytic form. Generating the grid first and the prims second is exact
and free.

Obstacles come out as **fixed-length arrays plus a validity mask**, not a
variable-length list. Isaac pre-allocates :data:`MAX_OBSTACLES` prims per
environment once at startup and repositions them at reset; creating and
destroying prims at runtime is slow enough to dominate step time.

Room parameters live here rather than in ``spec.py`` because this is their only
consumer -- see the rule at the top of ``spec.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import grids
from .spec import BODY_RADIUS_M, GOAL_MAX_M, ROBOT_RADIUS_M

__all__ = ["MAX_OBSTACLES", "Room", "sample_room"]

#: Sampled obstacle slots.
N_OBSTACLE_SLOTS = 24

#: Arena boundary slots. Always present, never sampled.
#:
#: The walls have to be real prims, not just occupied cells in the grid. When
#: they existed only in the grid, the oracle planned against a boundary the
#: lidar could not see: rays left the arena and returned max range, the robot
#: could drive straight out, and those demonstrations were not merely hard but
#: unlearnable.
#:
#: They are fixed-length slabs that get *positioned*, never scaled, because
#: MultiMeshRayCaster bakes scale at spawn. A slab longer than the arena simply
#: protrudes outward, which costs nothing.
N_WALL_SLOTS = 4

#: Total prims allocated per environment.
MAX_OBSTACLES = N_OBSTACLE_SLOTS + N_WALL_SLOTS

ARENA_MIN_M, ARENA_MAX_M = 4.0, 10.0
N_OBSTACLES_MIN, N_OBSTACLES_MAX = 5, 20
WALL_THICKNESS_M = 0.10
GRID_RES_M = 0.05

# Obstacle sizes are a discrete ladder, not a continuous range, and that is a
# simulator constraint rather than a modelling choice.
#
# Isaac's MultiMeshRayCaster tracks each obstacle prim's position and
# orientation every step, but **scale is baked into the warp mesh once at
# spawn** (`mesh.apply_scale` during parsing). A continuously-sized obstacle
# would therefore have to be respawned at reset -- runtime USD prim authoring,
# the exact cost we pre-allocate to avoid.
#
# So each environment gets one prim per entry below, spawned once at the listed
# size. A layout selects a subset and places them; the rest are parked under the
# floor. Diversity comes from which slots are used, where they go, and their
# yaw -- all of which are tracked per step and therefore free.
BOX_HALF_EXTENTS = [
    (0.25, 0.25), (0.25, 0.60), (0.60, 0.25),
    (0.50, 0.50), (0.40, 0.80), (0.80, 0.40),
]
CYLINDER_RADII = [0.25, 0.40, 0.55]

#: How many prims of each prototype exist per environment. The totals here are
#: what Isaac allocates, and MAX_OBSTACLES is their sum.
BOX_SLOTS_PER_SHAPE = 3        # 6 shapes x 3 = 18 box prims
CYLINDER_SLOTS_PER_SHAPE = 2   # 3 radii  x 2 =  6 cylinder prims

#: Wall slab half-extents. Length must exceed ARENA_MAX_M so one slab spans any
#: arena; thickness matches the grid border that rasterize() draws.
WALL_HALF = (6.0, WALL_THICKNESS_M / 2.0)

#: Largest half-extent in the ladder, used for wall margins.
OBSTACLE_MAX_M = max(max(max(b) for b in BOX_HALF_EXTENTS), max(CYLINDER_RADII))


def _slot_table():
    """The fixed per-environment prim inventory: (kind, half_x, half_y) per slot.

    Index into this is a *prim identity*, not a description -- slot ``i`` is
    always the same USD prim at the same baked scale, for the whole run.
    """
    slots = []
    for hx, hy in BOX_HALF_EXTENTS:
        slots += [(KIND_BOX, hx, hy)] * BOX_SLOTS_PER_SHAPE
    for r in CYLINDER_RADII:
        slots += [(KIND_CYLINDER, r, r)] * CYLINDER_SLOTS_PER_SHAPE
    assert len(slots) == N_OBSTACLE_SLOTS, (len(slots), N_OBSTACLE_SLOTS)
    # Walls last, so obstacle slot indices are unaffected by their presence.
    slots += [(KIND_BOX, *WALL_HALF)] * N_WALL_SLOTS
    return slots


def wall_placement(size_xy):
    """(centre, yaw) for the four boundary slabs, in wall-slot order.

    Placed to coincide with the grid border ``rasterize`` draws, so the geometry
    the lidar sees and the boundary the oracle plans against are the same
    surface. ``test_wall_prims_cover_the_grid_border`` pins that.
    """
    w, h = float(size_xy[0]), float(size_xy[1])
    t = WALL_THICKNESS_M / 2.0
    quarter = math.pi / 2.0
    return [
        ((w / 2.0, t), 0.0),            # south
        ((w / 2.0, h - t), 0.0),        # north
        ((t, h / 2.0), quarter),        # west
        ((w - t, h / 2.0), quarter),    # east
    ]

#: Geodesic (not straight-line) distance from start to goal. Using path length
#: makes difficulty mean what it should: a goal 3 m away through a wall is a
#: harder episode than one 6 m away down an open corridor.
GOAL_MIN_DIST_M = 2.0

#: Extra room beyond the robot radius required at the start and goal poses.
#: Spawning wedged against a wall is legal and teaches nothing.
SPAWN_CLEARANCE_M = 0.10

#: Path following prefers this much room on either side. Paths that hug the
#: 0.18 m planning boundary put the robot one tracking error away from the
#: inflated band, where the flow field's recovery pushes it back while the
#: lookahead pulls it forward -- a limit cycle that shows up as an episode which
#: both grazes geometry and never finishes.
CLEARANCE_PREFERRED_M = 0.45

#: How much more a wall-hugging cell costs than an open one. Only the *route*
#: is biased; ``Room.geodesic`` stays a true metric distance.
CLEARANCE_PENALTY = 3.0

KIND_BOX, KIND_CYLINDER = 0, 1


@dataclass(frozen=True)
class Room:
    """One layout. Analytic obstacles for the simulator, grids for the oracle."""

    size_xy: np.ndarray        # (2,)  arena extent in metres
    res: float

    # Fixed-length obstacle description -> USD prims.
    kind: np.ndarray           # (MAX_OBSTACLES,) int8   0=box 1=cylinder
    pos: np.ndarray            # (MAX_OBSTACLES, 2) float32  centre, world
    half: np.ndarray           # (MAX_OBSTACLES, 2) float32  half-extent / radius
    yaw: np.ndarray            # (MAX_OBSTACLES,) float32   box rotation, radians
    valid: np.ndarray          # (MAX_OBSTACLES,) bool

    # Derived grids -> oracle, reward, spawn sampling.
    occ: np.ndarray            # (H, W) bool     obstacles and walls as drawn
    clearance: np.ndarray      # (H, W) float32  metres to nearest obstacle
    free: np.ndarray           # (H, W) bool     legal robot-centre cells
    geodesic: np.ndarray       # (H, W) float32  TRUE metres to goal (difficulty)
    flow: np.ndarray           # (H, W) int32    next cell on the *preferred* route
    body_free: np.ndarray      # (H, W) bool     not in collision (body radius)

    start: np.ndarray          # (3,) float32  x, y, yaw
    goal: np.ndarray           # (2,) float32  x, y

    @property
    def path_length_m(self) -> float:
        """Oracle path length for this episode -- the denominator for SPL."""
        r, c = grids.world_to_cell(self.start[0], self.start[1], self.res)
        return float(self.geodesic[r, c])


def _sample_obstacles(rng: np.random.Generator, size_xy) -> tuple:
    """Choose which prim slots to use, and where to put them.

    Arrays are indexed by **slot**, not compacted: entry ``i`` always describes
    prim ``i``, whose baked size never changes. ``valid`` marks the subset in
    play, and unused slots keep their true size so the caller can never
    accidentally believe an idle prim is a different shape than it is.

    Overlap is rejected rather than resolved. Two fused boxes are a legal shape,
    but they can seal a region without any single obstacle looking wrong, and
    the pair-gap padding below guarantees the robot can always drive between any
    two obstacles rather than facing a gap it can see through but not fit
    through.
    """
    slots = _slot_table()
    kind = np.array([s[0] for s in slots], dtype=np.int8)
    half = np.array([[s[1], s[2]] for s in slots], dtype=np.float32)
    pos = np.zeros((MAX_OBSTACLES, 2), dtype=np.float32)
    yaw = np.zeros(MAX_OBSTACLES, dtype=np.float32)
    valid = np.zeros(MAX_OBSTACLES, dtype=bool)

    target = min(int(rng.integers(N_OBSTACLES_MIN, N_OBSTACLES_MAX + 1)),
                 N_OBSTACLE_SLOTS)
    gap = 2.0 * (ROBOT_RADIUS_M + SPAWN_CLEARANCE_M)
    placed: list[int] = []

    # Shuffle the slot order so a layout is not biased toward the small shapes
    # that happen to come first in the ladder.
    for slot in rng.permutation(N_OBSTACLE_SLOTS):
        if len(placed) >= target:
            break
        hx, hy = float(half[slot, 0]), float(half[slot, 1])
        # Yaw is free diversity: MultiMeshRayCaster tracks orientation every
        # step even though it bakes scale. A rotated box is effectively a shape
        # the ladder does not contain.
        a = rng.uniform(0.0, 2.0 * np.pi) if kind[slot] == KIND_BOX else 0.0
        reach = math.hypot(hx, hy)                 # bounding circle once rotated
        margin = WALL_THICKNESS_M + reach
        if 2.0 * margin >= min(size_xy):
            continue                               # too big for this arena

        for _ in range(20):
            px = rng.uniform(margin, size_xy[0] - margin)
            py = rng.uniform(margin, size_xy[1] - margin)
            if placed:
                # Separating-axis test on bounding circles-as-boxes: conservative
                # for both kinds and for any rotation.
                P, H = pos[placed], half[placed]
                other = np.hypot(H[:, 0], H[:, 1])
                dx = np.abs(P[:, 0] - px) - (other + reach + gap)
                dy = np.abs(P[:, 1] - py) - (other + reach + gap)
                if np.any((dx < 0) & (dy < 0)):
                    continue
            pos[slot], yaw[slot], valid[slot] = (px, py), a, True
            placed.append(int(slot))
            break

    # Walls are unconditional: every arena is closed.
    for k, ((cx, cy), a) in enumerate(wall_placement(size_xy)):
        slot = N_OBSTACLE_SLOTS + k
        pos[slot], yaw[slot], valid[slot] = (cx, cy), a, True

    return kind, pos, half, yaw, valid


def sample_room(rng: np.random.Generator, max_attempts: int = 20) -> Room:
    """Draw one solvable layout.

    Feasibility is guaranteed by *construction*, not by rejection after the
    fact: the goal is placed first, one geodesic solve labels every cell with
    its true path length to that goal, and the start is then drawn from the
    cells whose label already lies in the required range. A single field solve
    yields reachability and difficulty together.
    """
    for _ in range(max_attempts):
        size_xy = np.asarray(rng.uniform(ARENA_MIN_M, ARENA_MAX_M, size=2),
                             dtype=np.float32)
        kind, pos, half, yaw, valid = _sample_obstacles(rng, size_xy)

        occ = grids.rasterize(kind, pos, half, yaw, valid, size_xy,
                              GRID_RES_M, WALL_THICKNESS_M)
        clear = grids.clearance(occ, GRID_RES_M)

        # A robot centre is legal exactly where it has its own radius of room.
        free = clear >= ROBOT_RADIUS_M
        # Collision is a different question from planning margin: the body is
        # smaller than the radius paths are inflated by.
        body_free = clear >= BODY_RADIUS_M
        spawnable = clear >= (ROBOT_RADIUS_M + SPAWN_CLEARANCE_M)
        if not spawnable.any():
            continue

        gr, gc = _choose(rng, spawnable)
        field = grids.geodesic(free, (gr, gc), GRID_RES_M)

        ok = spawnable & (field >= GOAL_MIN_DIST_M) & (field <= GOAL_MAX_M)
        if not ok.any():
            continue                     # this goal has no well-placed start
        sr, sc = _choose(rng, ok)

        # Two fields, because they answer different questions. `field` is true
        # metric distance and drives difficulty sampling, SPL, and later the
        # progress reward. The route the oracle *follows* comes from a
        # clearance-penalised cost, so it runs down the middle of gaps instead of
        # scraping the planning boundary. One extra solve per layout, ~20 ms.
        penalty = 1.0 + CLEARANCE_PENALTY * np.clip(
            (CLEARANCE_PREFERRED_M - clear) / CLEARANCE_PREFERRED_M, 0.0, 1.0)
        cost = grids.geodesic(free, (gr, gc), GRID_RES_M, cost_scale=penalty)
        flow = grids.flow_field(free, cost, GRID_RES_M, cost_scale=penalty)

        gx, gy = grids.cell_to_world(gr, gc, GRID_RES_M)
        sx, sy = grids.cell_to_world(sr, sc, GRID_RES_M)
        return Room(
            size_xy=size_xy, res=GRID_RES_M,
            kind=kind, pos=pos, half=half, yaw=yaw, valid=valid,
            occ=occ, clearance=clear, free=free, geodesic=field, flow=flow,
            body_free=body_free,
            start=np.array([sx, sy, rng.uniform(0, 2 * np.pi)], dtype=np.float32),
            goal=np.array([gx, gy], dtype=np.float32),
        )

    raise RuntimeError(
        f"no solvable layout in {max_attempts} attempts; the obstacle density "
        f"or the {GOAL_MIN_DIST_M}-{GOAL_MAX_M} m range is too aggressive"
    )


def _choose(rng: np.random.Generator, mask: np.ndarray) -> tuple:
    """Uniformly pick one True cell of ``mask``, returned as (row, col)."""
    idx = int(rng.choice(np.flatnonzero(mask)))
    return divmod(idx, mask.shape[1])
