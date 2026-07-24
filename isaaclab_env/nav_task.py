"""Isaac Lab task: goal-conditioned navigation on a TB4 Lite / Create3.

This is the **primary training environment** for the deployed policy. It runs on
the 4090 with 2-4k parallel environments for the lidar-only configuration, and
a few hundred with tiled-rendered depth.

The adapter contract (§4 of the implementation plan) is what keeps this honest:
this file may import Isaac Lab freely, but it owns **no task definition**. Every
observation shape, normalization constant, action limit, reward coefficient and
randomization range is read from ``sim2edge.spec``, and the layouts come from
``sim2edge.rooms``, which is pure numpy and shared with the reference env and
ManiSkill. A change to the task is a change to the spec, and it lands in all
three simulators plus the C++ node at once.

Requires Isaac Lab; import fails cleanly elsewhere, so the rest of the repo
(tests, export, ROS) never depends on a 40 GB install being present.

Run:
    ./isaaclab.sh -p isaaclab_env/train_rsl_rl.py --task Sim2Edge-Nav-v0 \\
        --num_envs 2048 --headless
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import (
        Camera,
        CameraCfg,
        RayCaster,
        RayCasterCfg,
        patterns,
    )
    from isaaclab.utils import configclass

    _HAVE_ISAACLAB = True
except ImportError as exc:  # pragma: no cover - depends on the training host
    _HAVE_ISAACLAB = False
    _IMPORT_ERROR = exc

    def configclass(cls):  # type: ignore[misc]
        return cls

    DirectRLEnv = object  # type: ignore[assignment,misc]
    DirectRLEnvCfg = object  # type: ignore[assignment,misc]

from sim2edge.oracle.grid import UNREACHABLE
from sim2edge.rooms import Room, sample_room
from sim2edge.spec import DEFAULT_SPEC, NavSpec

SPEC: NavSpec = DEFAULT_SPEC


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class NavEnvCfg(DirectRLEnvCfg):
    """Direct-workflow config. Sizes come from the shared spec, not literals."""

    decimation = 10  # physics at 200 Hz, control at the spec's 20 Hz
    episode_length_s = SPEC.reward.max_episode_steps * SPEC.action.dt

    action_space = SPEC.action.shape[0]
    # Flattened observation. rsl_rl consumes a flat vector; the policy's
    # unflattening back into the four named tensors happens in
    # `Sim2EdgePolicyWrapper` below, using the same order the exported graph
    # binds -- so a reordering bug cannot survive to deployment unnoticed.
    observation_space = int(
        np.prod(SPEC.obs.depth.shape) + SPEC.obs.lidar.bins
        + SPEC.obs.goal.shape[0] + SPEC.obs.last_action.shape[0])
    state_space = SPEC.obs.privileged.dim

    use_depth: bool = True

    if _HAVE_ISAACLAB:
        sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
            dt=SPEC.action.dt / decimation,
            render_interval=decimation,
        )
        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=2048, env_spacing=12.0, replicate_physics=False)

        robot: ArticulationCfg = MISSING  # set from the Create3 asset

        lidar: RayCasterCfg = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=RayCasterCfg.OffsetCfg(
                pos=(0.0, 0.0, SPEC.robot.lidar_height_m)),
            attach_yaw_only=True,
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=(0.0, 0.0),
                horizontal_fov_range=(-180.0, 180.0),
                # Simulate the real beam count and let preprocessing min-pool to
                # the spec's bins, exactly as on the robot. Ray-casting straight
                # to 180 bins would skip the pooling step that the deployed
                # pipeline actually performs.
                horizontal_res=1.0,
            ),
            max_distance=SPEC.obs.lidar.max_m,
            mesh_prim_paths=["/World/ground"],
        )

        camera: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/oak_d",
            offset=CameraCfg.OffsetCfg(
                pos=(0.10, 0.0, SPEC.robot.camera_height_m),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
            data_types=["distance_to_image_plane"],  # forward depth, as OAK-D
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.93,
                clipping_range=(0.1, SPEC.obs.depth.max_m * 2.0),
            ),
            # Render at the policy resolution: tiled rendering at higher
            # resolution is the dominant throughput cost, and the policy input
            # is deliberately tiny.
            width=SPEC.obs.depth.width,
            height=SPEC.obs.depth.height,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────


class Sim2EdgeNavEnv(DirectRLEnv):
    """Isaac Lab implementation of the shared navigation spec."""

    cfg: "NavEnvCfg"

    def __init__(self, cfg: "NavEnvCfg", render_mode: str | None = None, **kw):
        if not _HAVE_ISAACLAB:
            raise ImportError(
                "Isaac Lab is not available in this environment. This task is "
                "for the training host; use sim2edge.envs.reference_env "
                f"elsewhere. Original error: {_IMPORT_ERROR}")
        super().__init__(cfg, render_mode, **kw)

        self.spec = SPEC
        n = self.num_envs
        dev = self.device

        self._last_action = torch.zeros((n, 2), device=dev)
        self._prev_action = torch.zeros((n, 2), device=dev)
        self._prev_geodesic = torch.zeros(n, device=dev)
        self._goal_world = torch.zeros((n, 2), device=dev)

        # Per-env layouts, generated by the SHARED numpy generator. Their
        # geodesic fields are uploaded to the GPU so the progress reward can be
        # looked up without a host round-trip every step.
        self._rooms: list[Room] = []
        self._geodesic: torch.Tensor | None = None
        self._clearance: torch.Tensor | None = None

        # Action delay queue: the single most important randomization for
        # sim-to-real on a real ROS stack.
        max_delay = int(self.spec.randomization.action_delay_steps[1])
        self._action_queue = torch.zeros((max_delay + 1, n, 2), device=dev)
        self._delay_steps = torch.zeros(n, dtype=torch.long, device=dev)

    # ── Setup ────────────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self.lidar = RayCaster(self.cfg.lidar)
        self.scene.sensors["lidar"] = self.lidar
        if self.cfg.use_depth:
            self.camera = Camera(self.cfg.camera)
            self.scene.sensors["camera"] = self.camera

        self._spawn_layouts()
        self.scene.clone_environments(copy_from_source=False)

    def _spawn_layouts(self) -> None:
        """Instantiate one procedurally generated room per environment.

        Obstacles come from ``sim2edge.rooms.sample_room``, the same generator
        the reference env and ManiSkill use, so the three simulators train on
        the same layout distribution rather than three similar-looking ones.
        """
        rng = np.random.default_rng(self.cfg.seed if hasattr(self.cfg, "seed")
                                    else 0)
        for env_id in range(self.num_envs):
            room = sample_room(rng, self.spec)
            self._rooms.append(room)
            for i, obs in enumerate(room.obstacles):
                prim = f"/World/envs/env_{env_id}/obstacle_{i}"
                height = obs.z_max - obs.z_min
                if obs.kind == "cylinder":
                    shape = sim_utils.CylinderCfg(radius=obs.sx, height=height)
                else:
                    shape = sim_utils.CuboidCfg(
                        size=(obs.sx * 2.0, obs.sy * 2.0, height))
                shape.func(prim, shape, translation=(
                    obs.x, obs.y, obs.z_min + height * 0.5))

        self._upload_layout_fields()

    def _upload_layout_fields(self) -> None:
        """Stack per-env geodesic and clearance grids onto the GPU.

        Rooms have different arena sizes, so the grids are zero-padded to the
        largest. Padding uses the UNREACHABLE sentinel for the geodesic field
        and zero clearance, so a robot that somehow leaves its arena reads as
        "unreachable and in collision" rather than as a free shortcut.
        """
        h = max(r.geodesic.shape[0] for r in self._rooms)
        w = max(r.geodesic.shape[1] for r in self._rooms)
        geo = np.full((len(self._rooms), h, w), UNREACHABLE, dtype=np.float32)
        clear = np.zeros((len(self._rooms), h, w), dtype=np.float32)
        for i, room in enumerate(self._rooms):
            rh, rw = room.geodesic.shape
            geo[i, :rh, :rw] = room.geodesic
            clear[i, :rh, :rw] = room.clearance
            self._goal_world[i, 0] = room.goal[0]
            self._goal_world[i, 1] = room.goal[1]
        self._geodesic = torch.from_numpy(geo).to(self.device)
        self._clearance = torch.from_numpy(clear).to(self.device)

    # ── Stepping ─────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        a = self.spec.action
        # The policy's tanh squash is in-graph, so actions arrive in SI units.
        v = actions[:, 0].clamp(a.v_min, a.v_max)
        omega = actions[:, 1].clamp(-a.omega_max, a.omega_max)
        self._prev_action = self._last_action.clone()
        self._last_action = torch.stack([v, omega], dim=1)

        # Latency: pop the command issued `delay` steps ago.
        self._action_queue = torch.roll(self._action_queue, shifts=-1, dims=0)
        self._action_queue[-1] = self._last_action
        idx = (self._action_queue.shape[0] - 1 - self._delay_steps)
        delayed = self._action_queue[idx, torch.arange(self.num_envs)]
        self._apply_wheel_velocities(delayed)

    def _apply_wheel_velocities(self, cmd: torch.Tensor) -> None:
        """Differential-drive inverse kinematics onto the Create3's wheels."""
        half_base = self.spec.robot.wheel_base_m * 0.5
        wheel_radius = 0.036  # Create3
        v, omega = cmd[:, 0], cmd[:, 1]
        left = (v - omega * half_base) / wheel_radius
        right = (v + omega * half_base) / wheel_radius
        self.robot.set_joint_velocity_target(torch.stack([left, right], dim=1))

    def _get_observations(self) -> dict:
        """Assemble the flat observation, in ``actor_inputs`` order.

        Normalization is done here rather than by calling
        ``sim2edge.preprocess`` because that module is numpy and this data is a
        GPU tensor batch. The arithmetic is intentionally identical, and
        ``tests/test_isaac_obs_parity.py`` checks these formulas against the
        numpy reference on the training host. Any divergence would mean the
        policy trains on one normalization and deploys under another.
        """
        d, l = self.spec.obs.depth, self.spec.obs.lidar
        n = self.num_envs

        # ── Lidar: hit distances -> normalized, invalid -> "far" ─────────────
        hits = self.lidar.data.ray_hits_w  # (N, B, 3)
        origin = self.robot.data.root_pos_w.unsqueeze(1)
        ranges = torch.norm(hits[..., :2] - origin[..., :2], dim=-1)
        ranges = torch.nan_to_num(ranges, nan=l.max_m, posinf=l.max_m)
        lidar = (ranges.clamp(l.min_m, l.max_m) - l.min_m) / (l.max_m - l.min_m)
        # Min-pool the simulated beams down to the spec's bins, matching what
        # preprocessing does to the real scan.
        lidar = lidar.view(n, l.bins, -1).amin(dim=-1)

        # ── Depth ────────────────────────────────────────────────────────────
        if self.cfg.use_depth:
            raw = self.camera.data.output["distance_to_image_plane"]
            raw = raw.view(n, 1, d.height, d.width)
            valid = torch.isfinite(raw) & (raw > 0.0)
            depth = (raw.clamp(d.min_m, d.max_m) - d.min_m) / (d.max_m - d.min_m)
            depth = torch.where(valid, depth,
                                torch.full_like(depth, d.invalid_fill))
        else:
            depth = torch.zeros((n, 1, d.height, d.width), device=self.device)

        # ── Goal, in the robot frame ─────────────────────────────────────────
        pos = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        yaw = self.robot.data.heading_w
        delta = self._goal_world - pos
        cos_y, sin_y = torch.cos(-yaw), torch.sin(-yaw)
        rel_x = cos_y * delta[:, 0] - sin_y * delta[:, 1]
        rel_y = sin_y * delta[:, 0] + cos_y * delta[:, 1]
        dist = torch.sqrt(rel_x ** 2 + rel_y ** 2).clamp_min(1e-6)
        goal = torch.stack([
            dist.clamp(max=self.spec.obs.goal.max_distance_m)
            / self.spec.obs.goal.max_distance_m,
            rel_y / dist,
            rel_x / dist,
        ], dim=1)

        a = self.spec.action
        last = torch.stack([
            (2.0 * (self._last_action[:, 0] - a.v_min) / (a.v_max - a.v_min)
             - 1.0).clamp(-1.0, 1.0),
            (self._last_action[:, 1] / a.omega_max).clamp(-1.0, 1.0),
        ], dim=1)

        obs = torch.cat([depth.flatten(1), lidar, goal, last], dim=1)
        return {"policy": obs, "critic": self._privileged()}

    def _privileged(self) -> torch.Tensor:
        """Ground-truth critic input, matching PrivilegedSpec's layout."""
        p, a = self.spec.obs.privileged, self.spec.action
        pos = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        yaw = self.robot.data.heading_w
        vel = self.robot.data.root_lin_vel_b[:, 0]
        ang = self.robot.data.root_ang_vel_w[:, 2]
        arena = torch.tensor(
            [[r.arena_w, r.arena_h] for r in self._rooms], device=self.device)
        return torch.stack([
            self._lookup_geodesic().clamp(max=p.max_geodesic_m) / p.max_geodesic_m,
            self._lookup_clearance().clamp(max=p.max_obstacle_m) / p.max_obstacle_m,
            pos[:, 0] / arena[:, 0],
            pos[:, 1] / arena[:, 1],
            torch.sin(yaw),
            torch.cos(yaw),
            vel / a.v_max,
            ang / a.omega_max,
        ], dim=1)

    def _get_rewards(self) -> torch.Tensor:
        rw = self.spec.reward
        geodesic = self._lookup_geodesic()
        progress = self._prev_geodesic - geodesic
        # Guard the unreachable sentinel; differencing it against a finite value
        # would produce a six-figure reward spike.
        progress = torch.where(
            (geodesic >= UNREACHABLE * 0.5) | (self._prev_geodesic >= UNREACHABLE * 0.5),
            torch.zeros_like(progress), progress)
        self._prev_geodesic = geodesic

        a = self.spec.action
        norm = torch.stack([
            2.0 * (self._last_action[:, 0] - a.v_min) / (a.v_max - a.v_min) - 1.0,
            self._last_action[:, 1] / a.omega_max], dim=1)
        prev_norm = torch.stack([
            2.0 * (self._prev_action[:, 0] - a.v_min) / (a.v_max - a.v_min) - 1.0,
            self._prev_action[:, 1] / a.omega_max], dim=1)
        smooth = ((norm - prev_norm) ** 2).sum(dim=1)

        reward = (rw.progress * progress
                  - rw.smoothness * smooth
                  - rw.time_penalty)
        reward = reward + rw.goal_bonus * self._reached().float()
        reward = reward - rw.collision_penalty * self._collided().float()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._reached() | self._collided()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ── Helpers (implemented against the Isaac Lab buffers) ──────────────────

    def _reached(self) -> torch.Tensor:
        pos = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        return (torch.norm(self._goal_world - pos, dim=1)
                <= self.spec.reward.goal_radius_m)

    def _collided(self) -> torch.Tensor:
        return self._lookup_clearance() < self.spec.robot.radius_m

    def _lookup_geodesic(self) -> torch.Tensor:
        return _sample_field(self._geodesic, self._robot_cells())

    def _lookup_clearance(self) -> torch.Tensor:
        return _sample_field(self._clearance, self._robot_cells())

    def _robot_cells(self) -> torch.Tensor:
        pos = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        return (pos / self.spec.room.grid_resolution_m).long()


def _sample_field(field: torch.Tensor | None, cells: torch.Tensor) -> torch.Tensor:
    """Gather per-env values from a stacked (N, H, W) field at integer cells."""
    if field is None:
        raise RuntimeError("layout fields were not uploaded; call _spawn_layouts")
    n, h, w = field.shape
    r = cells[:, 1].clamp(0, h - 1)
    c = cells[:, 0].clamp(0, w - 1)
    return field[torch.arange(n, device=field.device), r, c]
