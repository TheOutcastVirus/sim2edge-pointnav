"""Procedural room generation — pure numpy, shared by every simulator.

This module is the shared half of the Isaac Lab / ManiSkill interop story: both
adapters call :func:`sample_room` and instantiate the *same* layout, so a policy
can be trained on one and evaluated on the other without the layout distribution
silently changing underneath it.

Nothing here imports a simulator. The output is a :class:`Room` — an analytic
obstacle list (for a simulator to turn into prims) plus rasterized grids (for
the oracle, the reward, and the reference env's ray casting).

**Obstacles have a vertical extent, and that is the point.** Three classes exist
in the distribution:

* *tall* obstacles, which both the lidar and the camera see;
* *low* obstacles below the lidar plane, which the lidar cannot see at all;
* *overhanging* obstacles whose underside is above the robot, which are visible
  to both sensors but are not obstacles.

A lidar-only policy is structurally unable to solve the first two cases
correctly, which is the whole reason depth is in the observation. If the layout
generator only produced tall boxes, the depth channel would look useless in
evaluation and the Gate C comparison would be meaningless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .oracle.grid import (
    UNREACHABLE,
    OccupancyGrid,
    distance_field,
    obstacle_distance,
)
from .spec import DEFAULT_SPEC, NavSpec

__all__ = ["Obstacle", "Room", "sample_room"]

#: Thickness of the arena boundary wall, metres.
WALL_THICKNESS = 0.10


@dataclass
class Obstacle:
    """An axis-oriented box or a cylinder, with a vertical extent.

    ``z_min`` is the underside height and ``z_max`` the top. A normal box on the
    floor has ``z_min = 0``; a table top has ``z_min > 0``.
    """

    kind: str  # "box" | "cylinder"
    x: float
    y: float
    #: Box: half-extents. Cylinder: ``sx`` is the radius and ``sy`` is unused.
    sx: float
    sy: float
    yaw: float
    z_min: float
    z_max: float

    def spans(self, z_lo: float, z_hi: float) -> bool:
        """True if this obstacle occupies any of the height band [z_lo, z_hi]."""
        return self.z_min < z_hi and self.z_max > z_lo


@dataclass
class Room:
    """A generated layout plus everything derived from it."""

    arena_w: float
    arena_h: float
    obstacles: List[Obstacle]
    start: Tuple[float, float, float]  # x, y, yaw
    goal: Tuple[float, float]
    resolution: float
    spec: NavSpec = field(default=DEFAULT_SPEC, repr=False)

    # Rasterized layers, filled by _rasterize.
    collision: OccupancyGrid = field(init=False, repr=False)
    lidar_occ: OccupancyGrid = field(init=False, repr=False)
    #: Per-cell obstacle top height, metres; 0 where free. Used by the 2.5D
    #: depth renderer to decide how tall a hit surface is.
    z_max_map: np.ndarray = field(init=False, repr=False)
    z_min_map: np.ndarray = field(init=False, repr=False)

    #: Euclidean clearance to the nearest collidable obstacle, metres.
    clearance: np.ndarray = field(init=False, repr=False)
    inflated: OccupancyGrid = field(init=False, repr=False)
    _geodesic: Optional[np.ndarray] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._rasterize()
        self._build_fields()

    @property
    def geodesic(self) -> np.ndarray:
        """Geodesic distance to the goal over the inflated free space, metres.

        Computed lazily and cached: it is the most expensive part of building a
        Room, and layout sampling constructs throwaway probe Rooms purely to
        rejection-sample start/goal positions, which never need it.
        """
        if self._geodesic is None:
            free = ~self.inflated.occupied
            goal_rc = self.collision.world_to_cell(*self.goal)
            self._geodesic = distance_field(free, goal_rc, self.resolution)
        return self._geodesic

    # ── Rasterization ────────────────────────────────────────────────────────

    def _grid_shape(self) -> Tuple[int, int]:
        return (int(round(self.arena_h / self.resolution)),
                int(round(self.arena_w / self.resolution)))

    def _cell_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        h, w = self._grid_shape()
        xs = (np.arange(w) + 0.5) * self.resolution
        ys = (np.arange(h) + 0.5) * self.resolution
        return np.meshgrid(xs, ys)  # (H, W) each

    def _mask(self, obs: Obstacle, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        dx, dy = xx - obs.x, yy - obs.y
        if obs.kind == "cylinder":
            return dx * dx + dy * dy <= obs.sx * obs.sx
        c, s = math.cos(-obs.yaw), math.sin(-obs.yaw)
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        return (np.abs(lx) <= obs.sx) & (np.abs(ly) <= obs.sy)

    def _rasterize(self) -> None:
        h, w = self._grid_shape()
        xx, yy = self._cell_centres()
        robot_h = self.spec.robot.height_m
        lidar_z = self.spec.robot.lidar_height_m

        collision = np.zeros((h, w), dtype=bool)
        lidar_occ = np.zeros((h, w), dtype=bool)
        z_max_map = np.zeros((h, w), dtype=np.float32)
        z_min_map = np.full((h, w), np.inf, dtype=np.float32)

        for obs in self.obstacles:
            m = self._mask(obs, xx, yy)
            if not m.any():
                continue
            # Collidable only if it intrudes into the robot's vertical envelope.
            if obs.spans(0.0, robot_h):
                collision |= m
            # Visible to the lidar only if it crosses the scan plane exactly.
            if obs.z_min <= lidar_z <= obs.z_max:
                lidar_occ |= m
            # The camera sees everything, so the depth layers take all of it.
            np.maximum(z_max_map, np.float32(obs.z_max), out=z_max_map, where=m)
            np.minimum(z_min_map, np.float32(obs.z_min), out=z_min_map, where=m)

        # Arena boundary walls: collidable, lidar-visible, full height.
        wall = int(max(1, round(WALL_THICKNESS / self.resolution)))
        for layer in (collision, lidar_occ):
            layer[:wall, :] = True
            layer[-wall:, :] = True
            layer[:, :wall] = True
            layer[:, -wall:] = True
        for layer, value in ((z_max_map, np.float32(1.5)),
                             (z_min_map, np.float32(0.0))):
            layer[:wall, :] = value
            layer[-wall:, :] = value
            layer[:, :wall] = value
            layer[:, -wall:] = value

        z_min_map[~np.isfinite(z_min_map)] = 0.0

        self.collision = OccupancyGrid(collision, self.resolution)
        self.lidar_occ = OccupancyGrid(lidar_occ, self.resolution)
        self.z_max_map = z_max_map
        self.z_min_map = z_min_map

    def _build_fields(self) -> None:
        self.clearance = obstacle_distance(self.collision.occupied,
                                           self.resolution)
        # Plan for a point robot in inflated space. The inflation must exceed
        # the collision radius, not merely equal it: planning at exactly the
        # radius puts every path tangent to the collision boundary, and the
        # pure-pursuit tracking error then turns most episodes into collisions.
        pad = self.spec.robot.radius_m + self.spec.room.planning_margin_m
        self.inflated = self.collision.inflate(pad)

    # ── Queries ──────────────────────────────────────────────────────────────

    def geodesic_at(self, x: float, y: float) -> float:
        r, c = self.collision.world_to_cell(x, y)
        return float(self.geodesic[r, c])

    def clearance_at(self, x: float, y: float) -> float:
        r, c = self.collision.world_to_cell(x, y)
        return float(self.clearance[r, c])

    def in_collision(self, x: float, y: float,
                     radius: Optional[float] = None) -> bool:
        """True if a disc robot at ``(x, y)`` overlaps an obstacle or a wall."""
        if radius is None:
            radius = self.spec.robot.radius_m
        if not (radius <= x <= self.arena_w - radius
                and radius <= y <= self.arena_h - radius):
            return True
        return self.clearance_at(x, y) < radius

    @property
    def is_feasible(self) -> bool:
        sx, sy, _ = self.start
        return self.geodesic_at(sx, sy) < UNREACHABLE * 0.5

    @property
    def optimal_path_length(self) -> float:
        """Geodesic start-to-goal distance, the denominator of SPL."""
        sx, sy, _ = self.start
        return self.geodesic_at(sx, sy)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────


def _sample_obstacle(rng: np.random.Generator, spec: NavSpec,
                     arena_w: float, arena_h: float) -> Obstacle:
    room = spec.room
    lo, hi = room.obstacle_size_m
    kind = "cylinder" if rng.random() < 0.35 else "box"

    margin = WALL_THICKNESS + 0.2
    x = rng.uniform(margin, arena_w - margin)
    y = rng.uniform(margin, arena_h - margin)

    # Height class. The proportions matter: too many invisible-to-lidar
    # obstacles and the lidar-only baseline (the Day-5 guaranteed demo) becomes
    # unsolvable rather than merely worse.
    roll = rng.random()
    if roll < 0.70:  # tall: both sensors see it
        z_min, z_max = 0.0, rng.uniform(0.4, 1.4)
    elif roll < 0.90:  # low: under the lidar plane, only depth sees it
        z_min, z_max = 0.0, rng.uniform(0.05, spec.robot.lidar_height_m - 0.02)
    else:  # overhanging: visible but not an obstacle, the robot drives under
        z_min = rng.uniform(spec.robot.height_m + 0.05,
                            spec.robot.height_m + 0.35)
        z_max = z_min + rng.uniform(0.1, 0.5)

    if kind == "cylinder":
        radius = rng.uniform(lo, hi) * 0.5
        return Obstacle("cylinder", x, y, radius, radius, 0.0, z_min, z_max)

    # Vary aspect ratio so the distribution includes wall-like slabs, not just
    # near-square boxes.
    sx = rng.uniform(lo, hi) * 0.5
    aspect = rng.uniform(0.25, 4.0)
    sy = float(np.clip(sx * aspect, lo * 0.5, hi * 1.5))
    return Obstacle("box", x, y, sx, sy, rng.uniform(0.0, math.pi), z_min, z_max)


def _sample_free_xy(
    rng: np.random.Generator,
    grid: OccupancyGrid,
    clearance: np.ndarray,
    min_clearance: float,
    tries: int = 200,
) -> Optional[Tuple[float, float]]:
    """Rejection-sample a world point with at least ``min_clearance``."""
    ok = clearance >= min_clearance
    if not ok.any():
        return None
    rows, cols = np.nonzero(ok)
    for _ in range(tries):
        i = rng.integers(len(rows))
        x, y = grid.cell_to_world(int(rows[i]), int(cols[i]))
        # Jitter within the cell so positions are not grid-quantized.
        x += rng.uniform(-0.5, 0.5) * grid.resolution
        y += rng.uniform(-0.5, 0.5) * grid.resolution
        if clearance[grid.world_to_cell(x, y)] >= min_clearance:
            return (float(x), float(y))
    return None


def sample_room(
    rng: np.random.Generator | int | None = None,
    spec: NavSpec = DEFAULT_SPEC,
    max_attempts: int = 12,
) -> Room:
    """Sample a feasible procedurally generated room.

    Feasibility is checked against the *same* geodesic field the oracle and the
    reward use, so an episode can never be generated that the expert cannot
    solve. Retries on infeasible draws (a goal walled off by a chance
    arrangement of obstacles) and falls back to an empty arena in the pathological
    case, which keeps dataset generation from ever hanging.
    """
    if rng is None or isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)

    room_spec = spec.room
    min_clear = spec.robot.radius_m + room_spec.spawn_clearance_m
    g_lo, g_hi = room_spec.goal_distance_m

    for attempt in range(max_attempts):
        arena_w = rng.uniform(room_spec.arena_min_m, room_spec.arena_max_m)
        arena_h = rng.uniform(room_spec.arena_min_m, room_spec.arena_max_m)
        # Snap to the grid so rasterization is exact.
        arena_w = round(arena_w / room_spec.grid_resolution_m) * room_spec.grid_resolution_m
        arena_h = round(arena_h / room_spec.grid_resolution_m) * room_spec.grid_resolution_m

        n_obs = int(rng.integers(room_spec.n_obstacles[0],
                                 room_spec.n_obstacles[1] + 1))
        # Keep obstacle density roughly constant across arena sizes; otherwise
        # small arenas get 20 obstacles and are trivially infeasible.
        density_scale = (arena_w * arena_h) / (7.0 * 7.0)
        n_obs = max(2, int(round(n_obs * min(1.5, max(0.35, density_scale)))))
        obstacles = [_sample_obstacle(rng, spec, arena_w, arena_h)
                     for _ in range(n_obs)]

        probe = Room(arena_w, arena_h, obstacles, (0.0, 0.0, 0.0),
                     (arena_w * 0.5, arena_h * 0.5),
                     room_spec.grid_resolution_m, spec)

        start_xy = _sample_free_xy(rng, probe.collision, probe.clearance,
                                   min_clear)
        if start_xy is None:
            continue

        # Sample a goal at an acceptable straight-line distance, then confirm it
        # is actually reachable via the geodesic field below.
        goal_xy = None
        max_reach = min(g_hi, math.hypot(arena_w, arena_h) - 1.0)
        want = rng.uniform(g_lo, max(g_lo + 0.5, max_reach))
        candidates = _free_points(probe.collision, probe.clearance, min_clear)
        if candidates.size == 0:
            continue
        d = np.hypot(candidates[:, 0] - start_xy[0], candidates[:, 1] - start_xy[1])
        near = np.nonzero(np.abs(d - want) < 0.5)[0]
        if near.size == 0:
            near = np.nonzero(d >= g_lo)[0]
        if near.size == 0:
            continue
        pick = candidates[near[rng.integers(near.size)]]
        goal_xy = (float(pick[0]), float(pick[1]))

        room = Room(arena_w, arena_h, obstacles,
                    (start_xy[0], start_xy[1], float(rng.uniform(-math.pi, math.pi))),
                    goal_xy, room_spec.grid_resolution_m, spec)
        if room.is_feasible and room.optimal_path_length >= g_lo * 0.75:
            return room

    # Degenerate fallback: an empty arena is always solvable. Reaching this is
    # rare enough to be worth knowing about, so it is not silent.
    arena = 6.0
    return Room(arena, arena, [], (1.0, 1.0, 0.0), (arena - 1.0, arena - 1.0),
                room_spec.grid_resolution_m, spec)


def _free_points(grid: OccupancyGrid, clearance: np.ndarray,
                 min_clearance: float) -> np.ndarray:
    """World coordinates of every cell with at least ``min_clearance``."""
    rows, cols = np.nonzero(clearance >= min_clearance)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    xs = (cols + 0.5) * grid.resolution
    ys = (rows + 0.5) * grid.resolution
    return np.stack([xs, ys], axis=1).astype(np.float32)
