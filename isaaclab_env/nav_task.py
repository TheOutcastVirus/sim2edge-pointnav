"""Isaac Lab task: goal-conditioned navigation on a TurtleBot 4 Lite.

This file is an **adapter**, not a task definition. Every observation shape,
normalization constant, action limit and sensor mount pose is read from
``sim2edge.spec``; every layout comes from ``sim2edge.rooms``. A change to the
task is a change to those, and it lands here, in the dataset loader, and in the
C++ node at once.

Isaac Lab imports are guarded so the rest of the repo -- tests, dataset, export
-- never depends on a 40 GB install being present.

Three constraints from Isaac Lab 0.54.4 shape everything below; each is load
bearing and none is obvious from the outside:

1. **Plain ``RayCaster`` cannot see moving obstacles.** It accepts exactly one
   mesh prim, takes the first ``Mesh`` child it finds, and bakes the world
   transform into a static warp mesh at init. Obstacles repositioned at reset
   would be invisible to the lidar. ``MultiMeshRayCaster`` is the one that
   tracks transforms, so it is what we use.

2. **It tracks pose but bakes scale.** ``mesh.apply_scale`` runs once during
   parsing; only ``_mesh_positions_w`` / ``_mesh_orientations_w`` update per
   step. Obstacle prims therefore spawn once at fixed sizes drawn from
   ``rooms._slot_table()``, and a layout chooses *which* slots to use and where
   to put them. Slot index is a prim identity, not a description.

3. **The RPLIDAR is mounted yaw-rotated by pi/2.** Beam bearings arrive a
   quarter turn off the robot's forward axis, so they are rotated into
   base_link before binning. The goal bearing is in the robot frame; a scan
   left in the sensor frame would put every obstacle at right angles to where
   the policy believes it is.
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch

try:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import (Articulation, ArticulationCfg, RigidObjectCfg,
                                 RigidObjectCollection,
                                 RigidObjectCollectionCfg)
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import MultiMeshRayCaster, MultiMeshRayCasterCfg
    from isaaclab.sensors.ray_caster import patterns
    from isaaclab.utils import configclass

    _HAVE_ISAACLAB = True
except ImportError as exc:                       # pragma: no cover
    _HAVE_ISAACLAB = False
    _IMPORT_ERROR = exc

    def configclass(cls):                        # type: ignore[misc]
        return cls

    DirectRLEnv = object                         # type: ignore[assignment,misc]
    DirectRLEnvCfg = object                      # type: ignore[assignment,misc]

from sim2edge import rooms as rooms_mod
from sim2edge.oracle import LayoutPool, oracle_action
from sim2edge.preprocess import bin_lidar, build_observation, goal_features
from sim2edge.rooms import MAX_OBSTACLES, sample_room
from sim2edge.spec import (DT, GOAL_MAX_M, LIDAR_MAX_M, LIDAR_MOUNT_XYZ,
                           LIDAR_MOUNT_YAW, OBS_DIM, OMEGA_MAX, ROBOT_RADIUS_M,
                           V_MAX)

__all__ = ["NavEnvCfg", "Sim2EdgeNavEnv", "normalize_action",
           "denormalize_action"]

# ─────────────────────────────────────────────────────────────────────────────
# Robot geometry, from irobot_create_description. These have one reader -- this
# file -- so per the rule in spec.py they live here.
# ─────────────────────────────────────────────────────────────────────────────

WHEEL_RADIUS_M = 0.03575        # wheel.urdf.xacro: radius = 3.575 cm
WHEEL_SEPARATION_M = 0.233      # create3.urdf.xacro: distance_between_wheels

#: Where unused obstacle prims go. Far enough below the floor that no ray can
#: reach them, close enough that the physics broadphase does not care.
PARKED_Z = -10.0

#: Obstacles are tall enough that the lidar plane always intersects them. Height
#: classes (low obstacles the scan misses, overhangs it wrongly sees) belong to
#: the depth work and are deliberately absent from the lidar-only task.
OBSTACLE_HEIGHT_M = 0.6

#: Beams per scan. The A1 delivers a variable count near 720 at 10 Hz; binning
#: is by angle, so the exact number only has to be dense enough to fill 180 bins.
LIDAR_BEAMS = 720


def _obstacle_object_cfgs() -> dict:
    """One RigidObjectCfg per slot, in slot order.

    Collection ordering follows this dict's insertion order, so slot index is
    the object index -- no prim-path parsing, and none of the lexicographic
    ordering hazard that comes with regex-matched views (`env_10` sorts before
    `env_2`).
    """
    out = {}
    for slot, (kind, hx, hy) in enumerate(rooms_mod._slot_table()):
        common = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        if kind == rooms_mod.KIND_BOX:
            spawn = sim_utils.CuboidCfg(
                size=(2 * hx, 2 * hy, OBSTACLE_HEIGHT_M),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.75, 0.72, 0.65)),
                **common)
        else:
            spawn = sim_utils.CylinderCfg(
                radius=hx, height=OBSTACLE_HEIGHT_M,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.80, 0.45, 0.35)),
                **common)
        out[f"Obstacle_{slot:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Obstacle_{slot:02d}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, PARKED_Z)),
        )
    return out


@configclass
class NavEnvCfg(DirectRLEnvCfg):
    """Sizes come from the shared spec, never from literals."""

    # Physics at 200 Hz, control at the spec's 20 Hz.
    decimation = int(round(DT / 0.005))
    # Sized from the observed expert speed, not guessed. Real collection shows
    # ~0.29 m/s effective progress, and goals are sampled by geodesic distance
    # out to GOAL_MAX_M = 8 m, so the longest legitimate episode needs ~28 s.
    # At 25 s those episodes timed out and the success filter silently discarded
    # them -- biasing the set away from exactly the hard layouts the policy needs.
    # 45 s leaves ~1.6x headroom for turning overhead.
    episode_length_s = 45.0

    action_space = 2
    observation_space = OBS_DIM
    state_space = 0

    # Aim the viewport at environment 0's arena. `eye` and `lookat` are offsets
    # from env_origins[env_index], and a room spans (0,0)->(size_x, size_y) from
    # its own origin -- so the default lookat of (0,0,0) points at the arena's
    # *corner*, which is why the scene looks empty. Look at the middle instead,
    # from high enough to frame a 10 m arena.
    viewer: ViewerCfg = ViewerCfg(
        origin_type="env", env_index=0,
        eye=(3.5, -10.0, 13.0), lookat=(3.5, 3.5, 0.0),
    ) if _HAVE_ISAACLAB else None

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005, render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            # PhysX's GPU broadphase pair buffer. The default 2**21 overflows at
            # a few thousand environments, and the failure mode is quiet and
            # bad: "the simulation will miss interactions" means dropped
            # contacts -- robots passing through geometry, and potentially
            # missed wheel-ground contact, so the dynamics stop being
            # trustworthy while the run appears to continue normally.
            #
            # 2**22 = 4.19M, against the 3.33M PhysX asked for at 4096 envs.
            # Raise further if it complains again; the message states the
            # number it wants.
            gpu_found_lost_pairs_capacity=2 ** 22,
        ),
    ) if _HAVE_ISAACLAB else None

    # Spacing is set by the *wall slabs*, not by the arena.
    #
    # A slab is 2 * WALL_HALF[0] = 12 m long and sits centred on the arena's
    # mid-line, so relative to the env origin it spans w/2 +/- 6: for a 4 m
    # arena that is [-4, +8], for a 10 m arena [-1, +11]. The union across
    # layouts is a 15 m footprint, which the old 14 m spacing did not clear --
    # so neighbouring environments' slabs overlapped in the broadphase, and
    # every such pair was work PhysX had to track. That is a large part of why
    # the found/lost pair buffer overflowed at 4096 envs.
    #
    # Overlap needs S > 12 + (w_a - w_b)/2, worst case 15 m. 18 m clears it with
    # margin. Wider spacing costs only a larger (cheap) floor.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=2.0 * rooms_mod.WALL_HALF[0] + 6.0,
        replicate_physics=True,
    ) if _HAVE_ISAACLAB else None

    #: USD produced by ``tools/convert_tb4.py`` from the ROS Jazzy description.
    robot_usd: str = "assets/turtlebot4_lite.usd"

    robot_cfg: ArticulationCfg = ArticulationCfg(
        # The Direct workflow does not substitute {ENV_REGEX_NS}; that is a
        # manager-workflow token InteractiveScene expands. Direct tasks write
        # the regex out, and every shipped one does.
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_linear_velocity=2.0,
                max_angular_velocity=8.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
        actuators={
            # Velocity control: the task's action is a wheel speed target, and
            # first-order motor lag is what the real base does anyway.
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel_joint"],
                velocity_limit=V_MAX / WHEEL_RADIUS_M * 1.5,
                effort_limit=10.0, stiffness=0.0, damping=800.0,
            ),
        },
    ) if _HAVE_ISAACLAB else None

    #: Obstacles as kinematic rigid bodies, one per slot in rooms._slot_table().
    #:
    #: They must be *physics* prims, not bare Xforms. MultiMeshRayCaster warns
    #: and falls back to per-prim USD reads otherwise: "position lookups will be
    #: much slower and the pose will most likely not be updated correctly when
    #: running in headless mode". That fallback is both a hang and a silent
    #: correctness bug -- obstacles the lidar never sees move.
    #:
    #: Kinematic so they hold their pose: they are scenery, not physics. A
    #: dynamic obstacle would be shoved by the robot and fall out of agreement
    #: with the occupancy grid the oracle plans on.
    obstacles: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects=_obstacle_object_cfgs()
    ) if _HAVE_ISAACLAB else None

    lidar_cfg: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        # Mounted exactly where the hardware puts it, quarter-turn yaw included.
        # The beam bearings are rotated back into base_link in __init__, so the
        # rotation is applied once here and undone once there -- never twice.
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=LIDAR_MOUNT_XYZ,
            rot=(math.cos(LIDAR_MOUNT_YAW / 2), 0.0, 0.0,
                 math.sin(LIDAR_MOUNT_YAW / 2)),
        ),
        # Yaw-only alignment: the scan plane stays level when the base pitches
        # over a threshold, which is what a real 2D lidar on a rigid mount does.
        ray_alignment="yaw",
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Obstacle_.*",
                # The point of this sensor: obstacles move at reset and the
                # rays must follow them.
                track_mesh_transforms=True,
                # 24 prims x num_envs, but only a handful of distinct shapes.
                is_shared=False,
            ),
            # The ground is deliberately NOT a target. This is a horizontal 2D
            # scan (vertical_fov_range = (0, 0)) at a fixed height, so a
            # horizontal floor is unreachable by every ray. Including it only
            # adds a mesh to parse -- and GroundPlaneCfg's asset comes from
            # ISAAC_NUCLEUS_DIR, which may be a remote fetch.
        ],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=360.0 / LIDAR_BEAMS,
        ),
        max_distance=LIDAR_MAX_M,
        debug_vis=False,
    ) if _HAVE_ISAACLAB else None

    #: How many distinct layouts to build. Diversity is pool size times
    #: start/goal sampling, which is cheap; unique geometry per environment is
    #: not, because it is a proportional number of USD prims.
    num_layouts: int = 512
    layout_seed: int = 0


class Sim2EdgeNavEnv(DirectRLEnv):
    """Direct-workflow environment. Produces observations and oracle labels."""

    cfg: NavEnvCfg

    def __init__(self, cfg: NavEnvCfg, render_mode: str | None = None, **kw):
        if not _HAVE_ISAACLAB:                   # pragma: no cover
            raise ImportError(
                "isaaclab is required to run this task"
            ) from _IMPORT_ERROR
        super().__init__(cfg, render_mode, **kw)

        # Resolve wheel joints once, by explicit name and in a guaranteed
        # order. A regex like ".*wheel_joint" matches both, but the order it
        # returns them in is not specified -- and swapping left for right
        # produces a robot that turns the wrong way, which reads as a broken
        # controller rather than a lookup bug. preserve_order pins it.
        self._wheel_ids, names = self.robot.find_joints(
            ["left_wheel_joint", "right_wheel_joint"], preserve_order=True)
        assert names == ["left_wheel_joint", "right_wheel_joint"], names

        self._build_layout_pool()

        # Per-env episode state.
        n = self.num_envs
        self._layout = torch.zeros(n, dtype=torch.long, device=self.device)
        self._goal_w = torch.zeros(n, 2, device=self.device)
        # The last physical command issued. One buffer serves two readers: the
        # observation's last-action input, and the oracle's rate limiter. They
        # want the same quantity, and keeping two copies is how they drift.
        # Allocated here because reset() builds an observation before any
        # action has been applied.
        self._action = torch.zeros(n, 2, device=self.device)

        # Beam bearings, rotated from the sensor frame into base_link once.
        # Doing it here rather than per step keeps the hot path a single add.
        beams = torch.arange(LIDAR_BEAMS, device=self.device, dtype=torch.float32)
        self._beam_angles = (beams * (2.0 * math.pi / LIDAR_BEAMS)
                             + LIDAR_MOUNT_YAW)

    # ── scene ────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # A locally-generated static box, not GroundPlaneCfg: that config pulls
        # default_environment.usd from ISAAC_NUCLEUS_DIR, so it depends on a
        # reachable Omniverse asset root. Nothing here needs the grid texture,
        # and the floor is never ray-cast against (see lidar_cfg).
        # Sized to the environment grid rather than a fixed 200 m: a floor far
        # larger than the arenas is the grey mass that fills the viewport and
        # makes the scene unreadable at any sane camera distance.
        import math as _math
        side = _math.ceil(_math.sqrt(max(1, self.num_envs)))
        extent = side * self.cfg.scene.env_spacing + 4.0
        floor = sim_utils.CuboidCfg(
            size=(extent, extent, 0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.10, 0.12)),
        )
        floor.func("/World/ground", floor, translation=(0.0, 0.0, -0.1))

        # RigidObjectCollection spawns one prim per env from its regex path,
        # so no manual authoring and no clone-order interaction.
        self.obstacles = RigidObjectCollection(self.cfg.obstacles)
        self.scene.rigid_object_collections["obstacles"] = self.obstacles

        self.scene.clone_environments(copy_from_source=False)

        self.lidar = MultiMeshRayCaster(self.cfg.lidar_cfg)
        self.scene.sensors["lidar"] = self.lidar

        light = sim_utils.DomeLightCfg(intensity=800.0)
        light.func("/World/Light", light)

    # ── layouts ──────────────────────────────────────────────────────────────

    def _build_layout_pool(self):
        """Build the pool on CPU, then upload. Runs once, costs ~20 ms/layout."""
        rng = np.random.default_rng(self.cfg.layout_seed)
        self._rooms = [sample_room(rng) for _ in range(self.cfg.num_layouts)]
        self.pool = LayoutPool(self._rooms, device=self.device)

        stack = lambda attr, dtype: torch.tensor(
            np.stack([getattr(r, attr) for r in self._rooms]),
            device=self.device, dtype=dtype)

        self._obstacle_pos = stack("pos", torch.float32)     # (P, S, 2)
        self._obstacle_yaw = stack("yaw", torch.float32)     # (P, S)
        self._obstacle_valid = stack("valid", torch.bool)    # (P, S)
        self._room_start = stack("start", torch.float32)     # (P, 3)
        self._room_goal = stack("goal", torch.float32)       # (P, 2)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # New layout per environment. Reassigning here rather than pinning one
        # layout per env for the run means an env sees the whole pool over
        # training, so pool size sets diversity rather than num_envs.
        layout = torch.randint(0, self.cfg.num_layouts, (n,), device=self.device)
        self._layout[env_ids] = layout

        origins = self.scene.env_origins[env_ids]            # (n, 3)

        # Move this env's obstacle prims. Unused slots go under the floor
        # instead of being destroyed -- the ray caster follows position, and a
        # parked prim is simply out of every beam's reach.
        pos = self._obstacle_pos[layout]                     # (n, S, 2)
        yaw = self._obstacle_yaw[layout]                     # (n, S)
        valid = self._obstacle_valid[layout]                 # (n, S)

        world = torch.zeros(n, MAX_OBSTACLES, 3, device=self.device)
        world[:, :, :2] = pos + origins[:, None, :2]
        world[:, :, 2] = torch.where(
            valid,
            torch.full_like(yaw, OBSTACLE_HEIGHT_M / 2.0),
            torch.full_like(yaw, PARKED_Z),
        )
        self._write_obstacle_poses(env_ids, world, yaw)

        # Robot pose. The start is guaranteed to have robot-radius clearance by
        # construction in rooms.sample_room, so no rejection is needed here.
        start = self._room_start[layout]
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :2] = start[:, :2] + origins[:, :2]
        root[:, 2] = origins[:, 2] + 0.05
        root[:, 3:7] = _yaw_to_quat(start[:, 2])
        root[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root, env_ids)

        self._goal_w[env_ids] = self._room_goal[layout] + origins[:, :2]
        self._action[env_ids] = 0.0

    def _write_obstacle_poses(self, env_ids, world_xyz, yaw):
        """One batched PhysX write for every obstacle in the given envs.

        Object index is slot index by construction, so there is no permutation
        to maintain and nothing to get out of order.
        """
        n = world_xyz.shape[0]
        quat = _yaw_to_quat(yaw.reshape(-1)).reshape(n, MAX_OBSTACLES, 4)
        pose = torch.cat([world_xyz, quat], dim=-1)          # (n, S, 7)
        self.obstacles.write_object_pose_to_sim(pose, env_ids=env_ids)

    # ── actions ──────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        """Actions arrive in [-1, 1]; the spec's limits make them physical.

        The policy's exported graph squashes and rescales in-graph so the C++
        node emits SI units directly. Applying the same scaling here keeps sim
        and robot on one convention.
        """
        self._action = denormalize_action(actions)

    def _apply_action(self):
        """Differential-drive inverse kinematics: (v, omega) -> wheel speeds."""
        v, w = self._action[:, 0], self._action[:, 1]
        half = 0.5 * WHEEL_SEPARATION_M
        left = (v - w * half) / WHEEL_RADIUS_M
        right = (v + w * half) / WHEEL_RADIUS_M
        self.robot.set_joint_velocity_target(
            torch.stack([left, right], dim=1), joint_ids=self._wheel_ids)

    # ── observation ──────────────────────────────────────────────────────────

    def _robot_pose(self) -> torch.Tensor:
        """(N, 3) world x, y, yaw, with env origins removed."""
        root = self.robot.data.root_state_w
        xy = root[:, :2] - self.scene.env_origins[:, :2]
        return torch.cat([xy, _quat_to_yaw(root[:, 3:7]).unsqueeze(1)], dim=1)

    def _lidar_ranges(self) -> torch.Tensor:
        """(N, B) metres. Rays that hit nothing come back as non-finite, which
        ``bin_lidar`` maps to INVALID_FILL -- the identical value a real
        no-return produces."""
        hits = self.lidar.data.ray_hits_w                    # (N, B, 3)
        origin = self.lidar.data.pos_w.unsqueeze(1)          # (N, 1, 3)
        return torch.linalg.vector_norm(hits[..., :2] - origin[..., :2], dim=-1)

    def _get_observations(self) -> dict:
        pose = self._robot_pose()
        obs = build_observation(
            bin_lidar(self._lidar_ranges(), self._beam_angles),
            goal_features(pose, self._goal_w - self.scene.env_origins[:, :2]),
            torch.stack([self._action[:, 0] / V_MAX,
                         self._action[:, 1] / OMEGA_MAX], dim=1),
        )
        return {"policy": obs}

    # ── the oracle, for BC collection ────────────────────────────────────────

    def oracle_command(self) -> torch.Tensor:
        """(N, 2) the expert's *commanded* (v, omega) for the current state.

        This is the BC label. It is the command, never the achieved velocity
        read back from physics: training on achieved velocity teaches the policy
        to emit a value the actuator will lag a second time, a systematic
        underactuation that is invisible in sim and obvious on hardware.
        """
        return oracle_action(self.pool, self._layout, self._robot_pose(),
                             self._action)

    def raw_scan(self):
        """What the dataset stores: raw ranges and base_link beam bearings.

        Deliberately *not* the observation. Noise and binning happen in the
        loader so retuning either does not invalidate a collection run.
        """
        return self._lidar_ranges(), self._beam_angles

    # ── termination and reward ───────────────────────────────────────────────

    def _get_dones(self):
        pose = self._robot_pose()
        goal = self._goal_w - self.scene.env_origins[:, :2]
        reached = torch.linalg.vector_norm(pose[:, :2] - goal, dim=1) < 0.25
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return reached, timeout

    def _get_rewards(self) -> torch.Tensor:
        """Placeholder. BC does not use it; PPO will replace this with the
        geodesic-progress reward once the demonstration set exists."""
        return torch.zeros(self.num_envs, device=self.device)


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion helpers. Isaac uses (w, x, y, z).
# ─────────────────────────────────────────────────────────────────────────────

def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    out = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
    out[:, 0] = torch.cos(half)
    out[:, 3] = torch.sin(half)
    return out


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def denormalize_action(a: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> physical (v, omega). Forward-only: v maps to [0, V_MAX].

    Paired with :func:`normalize_action`; collection needs the inverse to feed
    an oracle command back through ``env.step``, and deriving it separately
    there is how the two silently stop agreeing.
    """
    return torch.stack([
        (a[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5 * V_MAX,
        a[:, 1].clamp(-1.0, 1.0) * OMEGA_MAX,
    ], dim=1)


def normalize_action(cmd: torch.Tensor) -> torch.Tensor:
    """Physical (v, omega) -> [-1, 1]. Exact inverse of the above."""
    return torch.stack([
        (cmd[:, 0] / V_MAX) * 2.0 - 1.0,
        cmd[:, 1] / OMEGA_MAX,
    ], dim=1)
