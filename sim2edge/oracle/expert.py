"""The oracle expert: geodesic planner + pure-pursuit controller.

Produces the ``(v, omega)`` demonstrations behaviour cloning learns from.

**The expert is constrained to the robot's real limits.** Its speed, turn rate,
and accelerations are read from the same :class:`~sim2edge.spec.ActionSpec` the
simulator and the C++ node use. An oracle that demonstrates commands the Create3
cannot track produces a policy that faithfully imitates commands the Create3
cannot track, and the resulting sim-to-real gap looks like a perception problem
while actually being a dynamics problem.

Planning descends the geodesic distance field rather than running a separate A*.
The field already exists — it is needed for the PPO progress reward — and
steepest descent on it *is* the shortest path under the field's octile metric.
Running an independent A* would introduce a second notion of "optimal path" that
could disagree with the reward, which is a subtle way to make PPO fight BC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..rooms import Room
from ..spec import DEFAULT_SPEC, NavSpec
from .grid import UNREACHABLE, OccupancyGrid

__all__ = ["plan_path", "smooth_path", "PurePursuitExpert", "ExpertCommand"]


# ─────────────────────────────────────────────────────────────────────────────
# Planning
# ─────────────────────────────────────────────────────────────────────────────

_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]


def plan_path(room: Room, start_xy: Optional[Tuple[float, float]] = None,
              max_steps: int = 20000) -> np.ndarray:
    """Steepest descent on the geodesic field: start -> goal.

    Returns an ``(N, 2)`` array of world waypoints, or an empty array when the
    goal is unreachable from the start.
    """
    if start_xy is None:
        start_xy = (room.start[0], room.start[1])

    field = room.geodesic
    grid = room.collision
    h, w = field.shape

    r, c = grid.world_to_cell(*start_xy)
    if field[r, c] >= UNREACHABLE * 0.5:
        return np.empty((0, 2), dtype=np.float32)

    path_cells: List[Tuple[int, int]] = [(r, c)]
    seen = {(r, c)}
    for _ in range(max_steps):
        if field[r, c] <= 0.0:
            break
        best, best_val = None, field[r, c]
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and field[nr, nc] < best_val:
                best, best_val = (nr, nc), field[nr, nc]
        if best is None or best in seen:
            break  # local minimum (should not happen on an exact field)
        r, c = best
        seen.add(best)
        path_cells.append(best)

    pts = np.array([grid.cell_to_world(rr, cc) for rr, cc in path_cells],
                   dtype=np.float32)
    if len(pts):
        # Descent ends at the goal *cell* centre; finish at the true goal.
        pts = np.vstack([pts, np.array(room.goal, dtype=np.float32)])
    return pts


def _line_of_sight(grid: OccupancyGrid, a: np.ndarray, b: np.ndarray) -> bool:
    """True if the straight segment ``a -> b`` stays in free space."""
    dist = float(np.hypot(*(b - a)))
    n = max(2, int(dist / (grid.resolution * 0.5)) + 1)
    ts = np.linspace(0.0, 1.0, n)[:, None]
    pts = a[None, :] * (1.0 - ts) + b[None, :] * ts
    rows, cols = grid.world_to_cell_array(pts)
    return not bool(grid.occupied[rows, cols].any())


def smooth_path(room: Room, path: np.ndarray,
                iterations: int = 2) -> np.ndarray:
    """Shortcut ("string-pull") then corner-round a grid-descent path.

    A raw descent path is staircased at the grid resolution. Pure-pursuit over a
    staircase produces a visibly wobbling ``omega`` command, and behaviour
    cloning would learn to reproduce the wobble — a smoothness penalty in PPO
    then has to unlearn what BC was explicitly taught.

    Shortcutting is checked against the *inflated* grid, so a shortcut can never
    cut a corner the robot's footprint would clip.
    """
    if len(path) < 3:
        return path

    grid = room.inflated
    pts = path
    for _ in range(iterations):
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            # Greedily jump to the furthest visible waypoint.
            j = len(pts) - 1
            while j > i + 1 and not _line_of_sight(grid, pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        pts = np.array(out, dtype=np.float32)
        if len(pts) < 3:
            break

    return _densify(pts, spacing=0.05)


def _densify(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline at roughly uniform spacing.

    Pure pursuit interpolates along the path by arc length, so a polyline with
    metre-long segments and a 0.5 m lookahead would give a coarse, jumpy target.
    """
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-6:
        return pts
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(total / spacing) + 1)
    want = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=np.float32)
    out[:, 0] = np.interp(want, cum, pts[:, 0])
    out[:, 1] = np.interp(want, cum, pts[:, 1])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Control
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExpertCommand:
    v: float
    omega: float
    #: Index of the pursued waypoint; useful for debugging and for detecting a
    #: stuck expert during dataset generation.
    target_index: int
    done: bool


class PurePursuitExpert:
    """Pure-pursuit path follower with a clearance- and curvature-aware profile.

    Stateful only in the previous command, which is what the acceleration limits
    are applied against. Construct one per episode.
    """

    def __init__(
        self,
        room: Room,
        spec: NavSpec = DEFAULT_SPEC,
        lookahead_m: float = 0.45,
        min_lookahead_m: float = 0.25,
        rotate_in_place_rad: float = 0.7,
    ):
        self.room = room
        self.spec = spec
        self.lookahead_m = lookahead_m
        self.min_lookahead_m = min_lookahead_m
        self.rotate_in_place_rad = rotate_in_place_rad

        self.path = smooth_path(room, plan_path(room))
        self._i = 0
        self._prev = (0.0, 0.0)

    @property
    def has_path(self) -> bool:
        return len(self.path) >= 2

    def reset(self) -> None:
        self._i = 0
        self._prev = (0.0, 0.0)

    def __call__(self, x: float, y: float, yaw: float) -> ExpertCommand:
        return self.act(x, y, yaw)

    def act(self, x: float, y: float, yaw: float) -> ExpertCommand:
        a = self.spec.action
        gx, gy = self.room.goal
        goal_dist = math.hypot(gx - x, gy - y)
        if goal_dist <= self.spec.reward.goal_radius_m or not self.has_path:
            return self._limit(0.0, 0.0, self._i, done=True)

        # ── Advance along the path ───────────────────────────────────────────
        # Only ever move forward through the path, so a robot pushed sideways
        # rejoins ahead of where it was rather than doubling back.
        pos = np.array([x, y], dtype=np.float32)
        d = np.linalg.norm(self.path[self._i:] - pos[None, :], axis=1)
        self._i += int(np.argmin(d))

        clearance = self.room.clearance_at(x, y)
        # Shorten the lookahead in tight spaces: a long lookahead in a doorway
        # cuts the corner, and the inflated-grid shortcut guarantee does not
        # extend to the controller's tracking error.
        lookahead = float(np.clip(
            self.min_lookahead_m + 0.6 * (clearance - self.spec.robot.radius_m),
            self.min_lookahead_m, self.lookahead_m))

        target_i = self._i
        while (target_i < len(self.path) - 1
               and np.linalg.norm(self.path[target_i] - pos) < lookahead):
            target_i += 1
        target = self.path[target_i]

        # ── Pure pursuit ─────────────────────────────────────────────────────
        dx, dy = float(target[0] - x), float(target[1] - y)
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx = c * dx - s * dy  # forward component in the robot frame
        ly = s * dx + c * dy  # left component
        heading_err = math.atan2(ly, lx)

        if abs(heading_err) > self.rotate_in_place_rad:
            # Turn on the spot. Driving forward through a large heading error
            # wastes distance and, near obstacles, sweeps the footprint into
            # them; the real Create3 turns in place too.
            omega = math.copysign(a.omega_max * 0.8, heading_err)
            return self._limit(0.0, omega, target_i, done=False)

        dist_sq = lx * lx + ly * ly
        curvature = 0.0 if dist_sq < 1e-9 else (2.0 * ly / dist_sq)

        # ── Speed profile ────────────────────────────────────────────────────
        v = a.v_max
        v *= max(0.15, 1.0 - abs(heading_err) / self.rotate_in_place_rad * 0.7)
        v *= float(np.clip(0.35 + 0.65 * (clearance - self.spec.robot.radius_m)
                           / 0.4, 0.35, 1.0))
        # Decelerate into the goal so the episode ends stopped rather than
        # overshooting and oscillating.
        v = min(v, math.sqrt(max(0.0, 2.0 * a.accel_max * max(0.0, goal_dist - 0.05))))
        # Respect the turn-rate limit rather than letting omega saturate, which
        # would silently break the pure-pursuit geometry.
        if abs(curvature) > 1e-6:
            v = min(v, a.omega_max / abs(curvature))
        v = float(np.clip(v, 0.0, a.v_max))

        return self._limit(v, v * curvature, target_i, done=False)

    def _limit(self, v: float, omega: float, target_i: int,
               done: bool) -> ExpertCommand:
        """Clamp to the velocity and acceleration limits of the real base."""
        a = self.spec.action
        dt = a.dt
        pv, pw = self._prev
        v = float(np.clip(v, max(a.v_min, pv - a.accel_max * dt),
                          min(a.v_max, pv + a.accel_max * dt)))
        omega = float(np.clip(omega,
                              max(-a.omega_max, pw - a.alpha_max * dt),
                              min(a.omega_max, pw + a.alpha_max * dt)))
        self._prev = (v, omega)
        return ExpertCommand(v, omega, target_i, done)
