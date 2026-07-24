"""ManiSkill adapter for the sim2edge navigation task.

**Role and scope.** This is the secondary simulator (§4 of the implementation
plan). Its purposes are iteration on the AMD 6900 XT via ROCm PyTorch, and the
cross-simulator interoperability claim for the writeup. It is explicitly *not*
on the critical path — the deployed policy trains in Isaac Lab — and it is first
on the cut list if the schedule slips.

**What makes the interop claim real** is that this file, like the Isaac Lab
adapter, contains no task definition of its own. Observation shapes,
normalization, action limits, rewards and layouts all come from
``sim2edge.spec`` and ``sim2edge.rooms``. The three simulators are three
renderings of one task, not three similar tasks — which is the only version of
that claim worth making.

Requires ManiSkill; imports cleanly degrade elsewhere.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import sapien
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose

    _HAVE_MANISKILL = True
except ImportError as exc:  # pragma: no cover - depends on the host
    _HAVE_MANISKILL = False
    _IMPORT_ERROR = exc
    BaseEnv = object  # type: ignore[assignment,misc]

    def register_env(*_a, **_k):  # type: ignore[misc]
        return lambda cls: cls

from sim2edge.oracle.grid import UNREACHABLE
from sim2edge.preprocess import (
    preprocess_goal,
    preprocess_last_action,
    world_goal_to_robot_frame,
)
from sim2edge.rooms import Room, sample_room
from sim2edge.spec import DEFAULT_SPEC, NavSpec

SPEC: NavSpec = DEFAULT_SPEC


@register_env("Sim2Edge-Nav-v0", max_episode_steps=SPEC.reward.max_episode_steps)
class Sim2EdgeNavEnv(BaseEnv):
    """Goal-conditioned navigation over the shared spec, in SAPIEN/ManiSkill."""

    SUPPORTED_ROBOTS = ["turtlebot4"]

    def __init__(self, *args, use_depth: bool = True,
                 spec: NavSpec = SPEC, **kwargs):
        if not _HAVE_MANISKILL:
            raise ImportError(
                "ManiSkill is not installed. This adapter is the secondary "
                "simulator; use sim2edge.envs.reference_env for the portable "
                f"path. Original error: {_IMPORT_ERROR}")
        self.spec = spec
        self.use_depth = use_depth
        self._room: Optional[Room] = None
        self._last_action = np.zeros(2, dtype=np.float32)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_geodesic = 0.0
        super().__init__(*args, **kwargs)

    # ── Scene ────────────────────────────────────────────────────────────────

    def _load_scene(self, options: dict) -> None:
        """Build the layout from the SHARED numpy generator."""
        seed = options.get("seed", None) if options else None
        self._room = sample_room(np.random.default_rng(seed), self.spec)

        builder = self.scene.create_actor_builder()
        for obs in self._room.obstacles:
            height = obs.z_max - obs.z_min
            pose = sapien.Pose(p=[obs.x, obs.y, obs.z_min + height * 0.5])
            if obs.kind == "cylinder":
                builder.add_cylinder_collision(pose, radius=obs.sx,
                                               half_length=height * 0.5)
                builder.add_cylinder_visual(pose, radius=obs.sx,
                                            half_length=height * 0.5)
            else:
                half = [obs.sx, obs.sy, height * 0.5]
                builder.add_box_collision(pose, half_size=half)
                builder.add_box_visual(pose, half_size=half)
        builder.build_static(name="obstacles")

    # ── Episode ──────────────────────────────────────────────────────────────

    def _initialize_episode(self, env_idx, options: dict) -> None:
        assert self._room is not None
        x, y, yaw = self._room.start
        self.agent.robot.set_pose(
            Pose.create_from_pq(p=[x, y, 0.0],
                                q=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        self._last_action[:] = 0.0
        self._prev_action[:] = 0.0
        self._prev_geodesic = self._room.geodesic_at(x, y)

    # ── Observation ──────────────────────────────────────────────────────────

    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        assert self._room is not None
        x, y, yaw = self._pose()
        goal_robot = world_goal_to_robot_frame(self._room.goal, (x, y), yaw)
        return {
            "goal": preprocess_goal(*goal_robot, spec=self.spec),
            "last_action": preprocess_last_action(*self._last_action,
                                                  spec=self.spec),
        }

    # ── Reward ───────────────────────────────────────────────────────────────

    def compute_dense_reward(self, obs, action, info) -> torch.Tensor:
        """Identical formula to the reference env and the Isaac Lab task.

        Geodesic progress, never Euclidean: Euclidean progress pays the robot to
        drive into the concave side of an obstacle.
        """
        assert self._room is not None
        rw = self.spec.reward
        x, y, _ = self._pose()

        geodesic = self._room.geodesic_at(x, y)
        if geodesic >= UNREACHABLE * 0.5 or self._prev_geodesic >= UNREACHABLE * 0.5:
            progress = 0.0
        else:
            progress = self._prev_geodesic - geodesic
        self._prev_geodesic = geodesic

        a = self.spec.action
        norm = np.array(a.normalize(*self._last_action))
        prev = np.array(a.normalize(*self._prev_action))
        smooth = float(np.sum((norm - prev) ** 2))

        reward = (rw.progress * progress
                  - rw.smoothness * smooth
                  - rw.time_penalty)
        if info.get("success", False):
            reward += rw.goal_bonus
        if info.get("collision", False):
            reward -= rw.collision_penalty
        return torch.tensor(reward, dtype=torch.float32)

    def evaluate(self) -> dict:
        assert self._room is not None
        x, y, _ = self._pose()
        goal_dist = float(np.hypot(self._room.goal[0] - x,
                                   self._room.goal[1] - y))
        return {
            "success": goal_dist <= self.spec.reward.goal_radius_m,
            "collision": self._room.in_collision(x, y),
            "goal_distance": goal_dist,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _pose(self):
        pose = self.agent.robot.get_pose()
        p = pose.p.cpu().numpy().reshape(-1)
        q = pose.q.cpu().numpy().reshape(-1)
        yaw = float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                               1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))
        return float(p[0]), float(p[1]), yaw
