"""One environment, driven by a person: click a goal, watch the policy take it.

This is the viewer's half of the adapter. Everything else in the project wants
thousands of environments that reset themselves and report aggregates; a person
watching one robot wants the opposite, and the three differences below are the
whole of it.

1. **Nothing auto-resets.** :meth:`Sim2EdgeNavEnv._get_dones` returns collision
   and arrival to the base class, which resets the environment on either -- the
   right behaviour for collection and RL, and useless here: the robot would
   teleport to a fresh room at the instant the thing worth looking at happened.
   The outcome flags are still computed and published; only the termination is
   withheld, so the viewer decides what a collision means.

2. **Layout, start and yaw can be pinned** through ``_sample_task``, which is
   why that seam exists in the task. "Reset" puts the robot back where this
   episode began, "randomize" moves it inside the same room, and "new layout"
   draws fresh geometry. All three go through the task's own ``_reset_idx``, so
   none of them can forget to reseed a noise model or clear a scan history.

3. **The goal is an input, not a property of the layout.** The policy's three
   goal features are a *relative* bearing and distance (``spec.GOAL_MAX_M``), so
   any reachable point is a legal goal and none of the layout's precomputed
   fields are needed to drive to it. The geodesic distance field still belongs
   to the sampled goal and is *not* recomputed on a click: the progress reward
   it feeds is meaningless in a viewer, and recomputing it would imply the
   oracle can be asked about a clicked goal when it cannot.
"""

from __future__ import annotations

import math

import numpy as np
import torch

try:
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    _HAVE_ISAACLAB = True
except ImportError:                                  # pragma: no cover
    _HAVE_ISAACLAB = False

from isaaclab_env.nav_task import Sim2EdgeNavEnv
from sim2edge.spec import GOAL_TOLERANCE_M

__all__ = ["InteractiveNavEnv", "MARKER_PATH", "frame_camera", "snap_goal"]

#: Markers live outside ``/World/envs`` so environment cloning never sees them,
#: and outside the lidar's target expression so no ray can return off one. A
#: goal the robot could *measure* would be a different task.
MARKER_PATH = "/Visuals/sim2edge_interactive"

#: Marker prototype indices, in the order they are declared below.
MARKER_GOAL, MARKER_BELIEF = 0, 1

#: Flat discs, sitting just above the floor. Tall markers hide the robot from a
#: top-down camera, which is the view a person clicks a floor plan in.
MARKER_HEIGHT_M = 0.02
MARKER_Z_M = 0.011


def frame_camera(size_xy, *, tilt: float = 0.55) -> tuple[tuple, tuple]:
    """``(eye, lookat)`` framing one arena, in that arena's own coordinates.

    Rooms span ``(0, 0)`` to ``size_xy`` from their environment origin, so a
    default camera aimed at the origin looks at a *corner* -- the documented
    reason the viewport reads as empty. Height is proportional to the arena so a
    4 m room and a 10 m room are framed the same way, and the eye is pulled back
    along -y rather than placed straight overhead: a pure plan view loses the
    robot's heading, which is most of what there is to watch.
    """
    width, height = float(size_xy[0]), float(size_xy[1])
    centre = (width / 2.0, height / 2.0)
    span = max(width, height)
    return ((centre[0], centre[1] - tilt * span, 1.15 * span),
            (centre[0], centre[1], 0.0))


def snap_goal(point_xy, free: np.ndarray, res: float,
              max_search_m: float = 0.60) -> np.ndarray | None:
    """Nearest legal robot centre to a clicked point, or ``None``.

    A click is a couple of centimetres of pointing accuracy against a 5 cm grid,
    so demanding that it land exactly in free space would reject perfectly clear
    intentions next to a wall. Snapping to the nearest free cell within
    ``max_search_m`` accepts those and still refuses a click well inside an
    obstacle -- which must be refused, because the policy would drive at it
    until something stopped it and the viewer would look broken rather than
    misused.

    ``free`` is the *planning* occupancy (clearance >= ROBOT_RADIUS_M), not the
    body-radius collision grid: a goal the robot can only reach by scraping is
    not a goal worth offering.
    """
    if free.ndim != 2:
        raise ValueError("free must be a 2-D occupancy grid")
    rows, cols = free.shape
    col = int(math.floor(float(point_xy[0]) / res))
    row = int(math.floor(float(point_xy[1]) / res))
    radius = int(math.ceil(max_search_m / res))

    best, best_distance = None, math.inf
    for dr in range(-radius, radius + 1):
        r = row + dr
        if not 0 <= r < rows:
            continue
        for dc in range(-radius, radius + 1):
            c = col + dc
            if not 0 <= c < cols or not free[r, c]:
                continue
            # Compare in cells; the grid is square so the ordering is the same
            # as in metres and this stays exact.
            distance = dr * dr + dc * dc
            if distance < best_distance:
                best, best_distance = (r, c), distance
    if best is None or best_distance * res * res > max_search_m ** 2:
        return None
    return np.array([(best[1] + 0.5) * res, (best[0] + 0.5) * res],
                    dtype=np.float32)


class InteractiveNavEnv(Sim2EdgeNavEnv):
    """The navigation task with termination and task sampling under the caller."""

    def __init__(self, cfg, render_mode: str | None = None, **kw):
        # Set before super().__init__, which reaches _setup_scene before this
        # constructor body resumes.
        self._pinned_task = None
        super().__init__(cfg, render_mode, **kw)

    # ── scene ────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        super()._setup_scene()
        disc = lambda colour: sim_utils.CylinderCfg(
            radius=GOAL_TOLERANCE_M, height=MARKER_HEIGHT_M,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=colour, emissive_color=colour, opacity=0.85))
        self._markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path=MARKER_PATH,
            markers={
                # The commanded goal: where the operator clicked.
                "goal": disc((0.10, 0.80, 0.35)),
                # Where the robot *believes* that goal is. Under a pose-fed
                # driver the two coincide exactly and the second is hidden; with
                # the localizer running, the gap between them is the
                # localization error, drawn at the scale a person can see.
                "belief": disc((0.95, 0.55, 0.10)),
            },
        ))

    def show_markers(self, goal_w, belief_w=None) -> None:
        """Place the goal disc, and the believed-goal disc when there is one."""
        points = [(float(goal_w[0]), float(goal_w[1]), MARKER_Z_M)]
        indices = [MARKER_GOAL]
        if belief_w is not None:
            points.append((float(belief_w[0]), float(belief_w[1]), MARKER_Z_M))
            indices.append(MARKER_BELIEF)
        self._markers.visualize(
            translations=torch.tensor(points, device=self.device),
            marker_indices=indices)

    # ── task sampling ────────────────────────────────────────────────────────

    def _sample_task(self, n: int):
        layout, start_id, yaw = super()._sample_task(n)
        if self._pinned_task is None:
            return layout, start_id, yaw
        pinned_layout, pinned_start, pinned_yaw = self._pinned_task
        if pinned_layout is not None:
            layout = torch.full_like(layout, int(pinned_layout))
        if pinned_start is not None:
            start_id = torch.full_like(start_id, int(pinned_start))
        if pinned_yaw is not None:
            yaw = torch.full_like(yaw, float(pinned_yaw))
        return layout, start_id, yaw

    def respawn(self, *, layout: int | None = None, start_id: int | None = None,
                yaw: float | None = None) -> None:
        """Reset the single environment, holding whatever is passed fixed.

        ``None`` means "draw it", so ``respawn()`` is a fresh task and
        ``respawn(layout=current)`` is a new pose in the same room.
        """
        self._pinned_task = (layout, start_id, yaw)
        try:
            self.reset()
        finally:
            self._pinned_task = None

    # ── termination ──────────────────────────────────────────────────────────

    def _get_dones(self):
        """Publish the outcome flags; never terminate.

        The parent's return value is exactly what triggers ``_reset_idx`` inside
        ``step``. Discarding it is the whole mechanism: ``_reached``,
        ``_collided`` and ``_pose`` are still computed by the same code the
        evaluators read, and the viewer decides what to do about them.
        """
        super()._get_dones()
        idle = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return idle, idle.clone()

    # ── goal ─────────────────────────────────────────────────────────────────

    @property
    def room(self):
        """The numpy layout the single environment is currently running."""
        return self._rooms[int(self._layout[0])]

    def env_origin(self) -> np.ndarray:
        return self.scene.env_origins[0, :2].detach().cpu().numpy()

    def set_goal(self, xy_local) -> None:
        """Move the goal to a point in arena coordinates.

        The environment stores goals in world coordinates because the batch path
        has one arena per environment origin; a viewer has exactly one, and
        every other quantity a person reads off the screen is arena-local, so
        the conversion happens here and once.
        """
        origin = self.scene.env_origins[0, :2]
        self._goal_w[0] = torch.as_tensor(
            xy_local, device=self.device, dtype=self._goal_w.dtype) + origin

    def goal_local(self) -> np.ndarray:
        origin = self.scene.env_origins[0, :2]
        return (self._goal_w[0] - origin).detach().cpu().numpy()

    def pose_local(self) -> np.ndarray:
        """``(3,)`` x, y, yaw of the robot in arena coordinates."""
        return self._robot_pose()[0].detach().cpu().numpy()
