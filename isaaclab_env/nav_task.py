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
from sim2edge.noise import (BEARING_BIAS_RAD, OdometryDrift,
                            WHEEL_SLIP_SCALE, corrupt_goal, corrupt_lidar,
                            wheel_motion_factor)
from sim2edge.oracle import LayoutPool, oracle_action
from sim2edge.preprocess import (ObservationHistory, bin_lidar, goal_features,
                                 rasterize_lidar, wrap_angle)
from sim2edge.rooms import MAX_OBSTACLES, sample_room, sample_start_positions
from sim2edge.sensors import (GyroModel, PolarSweep, advance_phase,
                              deskew_history, sweep_scan)
from sim2edge.spec import (DESKEW_WITH_GYRO, DT, GOAL_MAX_M, GOAL_TOLERANCE_M,
                           HISTORY_FRAMES, LIDAR_MAX_M, LIDAR_MOUNT_XYZ,
                           LIDAR_MIN_M, LIDAR_MOUNT_YAW, OBS_DIM, OMEGA_MAX, ROBOT_RADIUS_M,
                           MOTION_HISTORY_FRAMES, SPIN_TICKS, V_MAX)

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

#: Largest geodesic change a single control step may be paid for, in metres.
#:
#: This is a guard against grid artifacts, not a shaping term, so it sits far
#: above both one grid cell (5 cm) and one step of travel (V_MAX * DT = 2.3 cm).
#: Clamping tightly would be worse than not clamping at all: consecutive
#: distances differ by whole cells, and the per-episode sum of unclamped
#: differences telescopes to the true distance the robot closed.
PROGRESS_CLAMP_M = 0.5


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
            # 2**23 = 8.39M, against the 6.67M PhysX requested at 16,384 envs.
            # Raise further if it complains again; the message states the
            # number it wants.
            gpu_found_lost_pairs_capacity=2 ** 23,
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
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05),
            # Wheel-drop travel is irrelevant to planar navigation and its
            # Gazebo-only spring model is not preserved faithfully in USD.
            # Hold both carriers at the nominal chassis height: a compliant
            # spring unloaded one wheel under differential torque, so straight
            # commands worked while in-place turns barely moved the base.
            joint_pos={".*wheel_drop.*": 0.0, ".*wheel_joint": 0.0},
        ),
        actuators={
            # Track the wheel speeds implied by the planar Twist for coherent
            # joint state and visualization. Base propulsion is applied below;
            # these imported cylinder contacts are not a usable tire model.
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel_joint"],
                velocity_limit_sim=V_MAX / WHEEL_RADIUS_M * 1.5,
                effort_limit_sim=10.0, stiffness=0.0, damping=800.0,
            ),
            # These joints must be held, not left free and not approximated as
            # soft suspension. Either failure unloads a wheel under opposing
            # motor torque and destroys commanded yaw response.
            "suspension": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel_drop.*"],
                velocity_limit_sim=0.2, effort_limit_sim=1000.0,
                stiffness=10_000.0, damping=1_000.0,
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
        # Base alignment is the sensor-contract-v2 fix. In Isaac Lab's local
        # yaw path origins follow base yaw but directions remain world-fixed.
        # The planar controller keeps the base level, so full base alignment is
        # both physically correct and free of pitch/roll ambiguity here.
        ray_alignment="base",
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

    #: How many distinct obstacle/goal/flow-field layouts to build. Each layout
    #: has several feasible starts and reset samples yaw independently.
    num_layouts: int = 512
    layout_seed: int = 0
    starts_per_layout: int = 32

    #: Collection consumes raw scans and oracle labels, not policy
    #: observations.  Avoid binning 720 rays into an observation that the
    #: caller immediately discards.  PPO leaves this false and sees the full
    #: task API.
    collection_mode: bool = False

    #: Sensor and odometry corruption applied to the environment's own
    #: observation, as a multiple of the ranges in ``sim2edge.noise``.
    #:
    #: Zero for collection, because a stored shard must keep its clean ranges
    #: and let the loader draw fresh corruption every epoch. On-policy RL has no
    #: shard to protect -- the observation is consumed once and discarded -- so
    #: the noise has to be injected here or the fine-tuned policy would learn on
    #: inputs cleaner than the ones BC trained on and evaluation measures.
    observation_noise_scale: float = 0.0

    #: Whether the base slips relative to its command. On by default because
    #: the alternative is a robot that tracks to 0.22 mm, which flatters every
    #: odometry result this project produces. Turn it off only to make a run
    #: bit-repeatable while debugging something else.
    wheel_slip: bool = True

    #: End an episode when simulator truth enters the goal tolerance. This is
    #: correct for ordinary BC/PPO, but hardware-style belief evaluation must
    #: leave it false: the robot has no truth signal and needs several sensor
    #: ticks to confirm its own arrival. Collision and timeout still terminate.
    terminate_on_goal: bool = True

    # ── reward ───────────────────────────────────────────────────────────────
    # Per step:
    #   progress * (d_{t-1} - d_t)  +  goal * reached  -  collision * collided
    #   -  time  -  smoothness * ||a_t - a_{t-1}||^2      (normalized axes)
    #
    # Scale check, which is the only way these numbers mean anything: an 8 m
    # episode earns 3.0 * 8 = 24 of progress plus a 10 goal bonus against about
    # 1.8 of accumulated time cost, while colliding 2 m in earns 6 and pays 10.
    # Reaching the goal is worth more than any route to it, and no amount of
    # loitering is worth a collision.

    #: Per metre of geodesic progress. The term that does the work; everything
    #: else only breaks ties it cannot see.
    progress_reward: float = 3.0
    goal_reward: float = 10.0
    collision_penalty: float = 10.0
    #: Terminal cost for consuming the entire horizon. Without this, the
    #: accumulated 0.005 step cost is only 4.5, so once collision avoidance gets
    #: difficult PPO correctly prefers to stop in free space instead of
    #: finishing the route. Keep timeout at least as costly as collision after
    #: the accumulated step cost, while successful arrival retains its bonus.
    timeout_penalty: float = 10.0
    #: Small per-step cost. Without it, safe-but-stationary loitering is an
    #: optimal policy for any episode the robot expects to fail.
    time_penalty: float = 0.005
    #: On the *normalized* command delta, so one unit means the same fraction of
    #: each axis limit. This is the term that keeps the real base from jerking.
    smoothness_penalty: float = 0.05


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
        self._suspension_ids, suspension_names = self.robot.find_joints(
            ["wheel_drop_left_joint", "wheel_drop_right_joint"],
            preserve_order=True)
        assert suspension_names == ["wheel_drop_left_joint",
                                    "wheel_drop_right_joint"], suspension_names
        self._build_layout_pool()

        # Per-env episode state.
        n = self.num_envs
        self._layout = torch.zeros(n, dtype=torch.long, device=self.device)
        self._start_id = torch.zeros(n, dtype=torch.long, device=self.device)
        self._optimal_path_m = torch.zeros(n, device=self.device)
        self._goal_w = torch.zeros(n, 2, device=self.device)
        # The last physical command issued. One buffer serves two readers: the
        # observation's last-action input, and the oracle's rate limiter. They
        # want the same quantity, and keeping two copies is how they drift.
        # Allocated here because reset() builds an observation before any
        # action has been applied.
        self._action = torch.zeros(n, 2, device=self.device)
        # The command before that, for the reward's smoothness term only.
        self._previous_action = torch.zeros(n, 2, device=self.device)
        # Geodesic metres to the goal at the previous control step. The reward
        # is a difference of consecutive values, so this must be re-seeded on
        # reset -- otherwise the first step of every episode is paid for a
        # distance change that belongs to two different layouts.
        self._previous_distance = torch.zeros(n, device=self.device)
        # Pose at the current control step, published by _get_dones.
        self._pose = torch.zeros(n, 3, device=self.device)
        # Allocated once: the reward divides by this every step, and a fresh
        # host tensor per step is a per-step host-to-device copy.
        self._action_limits = torch.tensor([V_MAX, OMEGA_MAX],
                                           device=self.device)
        # Planar command integration is kept separate from the contact-solver
        # state so tire scrub cannot erase commanded yaw between substeps.
        self._planar_pose_target = torch.zeros(n, 3, device=self.device)

        # Observation corruption is drawn from a stream seeded off the task
        # seed, so a rollout is reproducible without being identical between
        # environments.
        self._noise_generator = torch.Generator(
            device=torch.device(self.device).type)
        self._noise_generator.manual_seed(int(self.cfg.seed or 0) + 977)

        # Isaac's LidarPattern starts at -pi.  Metadata must describe those
        # exact rays in base_link; omitting the origin labels every beam pi away
        # from the surface that produced it.
        beams = torch.arange(LIDAR_BEAMS, device=self.device, dtype=torch.float32)
        self._beam_angles = (-math.pi
                             + beams * (2.0 * math.pi / LIDAR_BEAMS)
                             + LIDAR_MOUNT_YAW)
        self._observation_history = ObservationHistory(
            n, device=self.device)
        self._history_reset = torch.ones(
            n, dtype=torch.bool, device=self.device)

        # ── spinning sensor state ────────────────────────────────────────────
        # A revolution spans SPIN_TICKS control periods, so the environment has
        # to remember that many raw scans to hand `sweep_scan` a full sweep.
        # Without this the env would keep producing instantaneous scans while
        # the dataset loader produced swept ones -- PPO and BC would then be
        # training on two different sensors under one fingerprint, which is
        # precisely the silent-disagreement failure `spec.py` exists to prevent.
        self._scan_history = torch.zeros(
            n, SPIN_TICKS, LIDAR_BEAMS, device=self.device)
        # Phase is per-environment and free-running: the motor does not restart
        # when an episode does, and two envs are not in step with each other.
        self._spin_phase = torch.rand(n, device=self.device)
        self._gyro = GyroModel(n, scale=float(self.cfg.observation_noise_scale),
                               device=self.device,
                               generator=self._noise_generator)
        self._gyro_history = torch.zeros(
            n, MOTION_HISTORY_FRAMES, device=self.device)
        self._motion_action_history = torch.zeros(
            n, MOTION_HISTORY_FRAMES, 2, device=self.device)
        # Deskew spans a full lidar revolution, which is longer than the
        # estimator's four-frame feature history. Keep the two contracts
        # separate so changing actor history cannot silently shorten deskew.
        self._deskew_gyro_history = torch.zeros(
            n, SPIN_TICKS, device=self.device)
        self._deskew_speed_history = torch.zeros(
            n, SPIN_TICKS, device=self.device)
        self._motion_scan_history = torch.full(
            (n, MOTION_HISTORY_FRAMES, LIDAR_BEAMS), LIDAR_MAX_M,
            device=self.device)
        self._motion_valid_history = torch.zeros(
            n, MOTION_HISTORY_FRAMES, LIDAR_BEAMS, dtype=torch.bool,
            device=self.device)
        self._canonical_ranges = torch.full(
            (n, LIDAR_BEAMS), LIDAR_MAX_M, device=self.device)
        self._canonical_bearings = self._beam_angles[None].expand(n, -1).clone()
        self._canonical_ages = torch.zeros(
            n, LIDAR_BEAMS, device=self.device)
        self._canonical_valid = torch.zeros(
            n, LIDAR_BEAMS, dtype=torch.bool, device=self.device)
        self._odometry_drift = OdometryDrift(
            n, scale=float(self.cfg.observation_noise_scale),
            device=self.device, generator=self._noise_generator)
        # Yaw at the previous control step, for the gyro's true input.
        self._previous_yaw = torch.zeros(n, device=self.device)
        # Per-episode drivetrain calibration error. Redrawn at reset.
        self._wheel_calibration = torch.zeros(n, device=self.device)
        # Sensor mounting error is systematic within a trajectory.  Drawing it
        # afresh every control tick turns a fixed calibration offset into lidar
        # jitter and breaks the sequence-training contract.
        self._lidar_bearing_bias = torch.zeros(n, device=self.device)
        # A deployable belief policy may terminate on its own confirmed arrival
        # decision. Keeping this explicit avoids collecting a stationary
        # false-arrival tail until the simulator timeout.
        self._external_done = torch.zeros(
            n, dtype=torch.bool, device=self.device)

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
        self._room_goal = stack("goal", torch.float32)       # (P, 2)
        # Geometry, goal and flow are the expensive objects. Starts are cheap:
        # every reachable cell already has a route to this layout's goal.
        start_rng = np.random.default_rng(self.cfg.layout_seed ^ 0x5EED5EED)
        starts = [sample_start_positions(
            room, start_rng, self.cfg.starts_per_layout) for room in self._rooms]
        self._room_starts = torch.tensor(
            np.stack(starts), device=self.device, dtype=torch.float32)
        path_lengths = [[room.path_length_from(xy) for xy in candidates]
                        for room, candidates in zip(self._rooms, starts)]
        self._room_path_length = torch.tensor(
            path_lengths, device=self.device, dtype=torch.float32)

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

        # Robot pose. Position is sampled from a reusable feasible table; yaw is
        # fresh every episode. The old fixed start/yaw turned 500k episodes into
        # repetitions of only 512 task templates and produced 95% memorized / 37%
        # held-out success.
        start_id = torch.randint(
            0, self.cfg.starts_per_layout, (n,), device=self.device)
        self._start_id[env_ids] = start_id
        start = torch.empty(n, 3, device=self.device)
        start[:, :2] = self._room_starts[layout, start_id]
        start[:, 2] = torch.rand(n, device=self.device) * (2.0 * math.pi)
        self._optimal_path_m[env_ids] = self._room_path_length[layout, start_id]
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :2] = start[:, :2] + origins[:, :2]
        root[:, 2] = origins[:, 2] + 0.05
        root[:, 3:7] = _yaw_to_quat(start[:, 2])
        root[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root, env_ids)

        # Reset internal articulation state as well as the root.  Without this,
        # wheel spin and suspension displacement leaked across episodes even
        # though the robot pose jumped back to its start.
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel,
                                            env_ids=env_ids)
        self.robot.set_joint_position_target(
            joint_pos[:, self._suspension_ids],
            joint_ids=self._suspension_ids, env_ids=env_ids)
        self.robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)

        self._goal_w[env_ids] = self._room_goal[layout] + origins[:, :2]
        self._action[env_ids] = 0.0
        self._previous_action[env_ids] = 0.0
        self._previous_distance[env_ids] = self.pool.distance_to_goal(
            layout, start[:, :2])
        self._history_reset[env_ids] = True
        self._planar_pose_target[env_ids, :2] = root[:, :2]
        self._planar_pose_target[env_ids, 2] = start[:, 2]

        # Sensor state. The scan history is filled on the next observation --
        # marking it stale via _history_reset is enough -- but the error models
        # carry per-episode constants (gyro bias, wheel calibration) that have
        # to be redrawn, or one unlucky draw would follow an environment for the
        # whole run and read as a systematically bad policy.
        self._gyro.reset(env_ids)
        self._odometry_drift.reset(env_ids)
        self._gyro_history[env_ids] = 0.0
        self._motion_action_history[env_ids] = 0.0
        self._deskew_gyro_history[env_ids] = 0.0
        self._deskew_speed_history[env_ids] = 0.0
        self._motion_scan_history[env_ids] = LIDAR_MAX_M
        self._motion_valid_history[env_ids] = False
        self._previous_yaw[env_ids] = start[:, 2]
        self._wheel_calibration[env_ids] = torch.randn(
            n, device=self.device,
            generator=self._noise_generator) * WHEEL_SLIP_SCALE
        self._lidar_bearing_bias[env_ids] = torch.randn(
            n, device=self.device,
            generator=self._noise_generator) * BEARING_BIAS_RAD
        self._external_done[env_ids] = False

    def set_external_done(self, done: torch.Tensor) -> None:
        """Request termination on the next step for model arrival logic."""
        if done.shape != self._external_done.shape:
            raise ValueError(
                f"expected external done shape {tuple(self._external_done.shape)}, "
                f"got {tuple(done.shape)}")
        self._external_done |= done.to(device=self.device, dtype=torch.bool)

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
        self._previous_action = self._action
        self._action = denormalize_action(actions)

    def _apply_action(self):
        """Apply the planar Twist and animate its differential wheel speeds.

        The imported finite-width cylinder contacts do not implement a tire
        model: opposing wheel speeds freely scrub while same-sign speeds drive
        correctly. Base yaw was therefore unrelated to commanded yaw even with
        held suspension and exact wheel-speed tracking. The task's action is a
        base Twist, so impose that contract directly and leave PhysX to resolve
        contacts during the following substeps.
        """
        v, w = self._action[:, 0], self._action[:, 1]
        # Wheel slip: what the base does is not what was asked for.
        #
        # Measured on the collected corpus, achieved displacement tracked the
        # command to a standard deviation of 0.22 mm per tick. That is not a
        # robot, and it is not harmless: it makes dead reckoning exact, which
        # leaves the belief policy's odometry head with a target ~90x below the
        # lidar's own range noise and therefore nothing to learn. A design whose
        # whole purpose is to correct dead reckoning cannot be validated in a
        # world where dead reckoning is already perfect.
        #
        # Applied to the pose integration only. The commanded value is what the
        # policy reads back as its previous action -- a robot knows what it
        # asked for, and only the floor knows what it got.
        v = v * self._slip_factor(w)
        target = self._planar_pose_target
        yaw = target[:, 2]
        target[:, 0] += v * torch.cos(yaw) * self.physics_dt
        target[:, 1] += v * torch.sin(yaw) * self.physics_dt
        target[:, 2] = torch.atan2(torch.sin(yaw + w * self.physics_dt),
                                   torch.cos(yaw + w * self.physics_dt))

        root_pose = self.robot.data.root_pose_w.clone()
        root_pose[:, :2] = target[:, :2]
        root_pose[:, 3:7] = _yaw_to_quat(target[:, 2])
        self.robot.write_root_pose_to_sim(root_pose)

        root_velocity = self.robot.data.root_vel_w.clone()
        root_velocity[:, 0] = v * torch.cos(target[:, 2])
        root_velocity[:, 1] = v * torch.sin(target[:, 2])
        root_velocity[:, 5] = w
        self.robot.write_root_velocity_to_sim(root_velocity)

        half = 0.5 * WHEEL_SEPARATION_M
        left = (v - w * half) / WHEEL_RADIUS_M
        right = (v + w * half) / WHEEL_RADIUS_M
        self.robot.set_joint_velocity_target(
            torch.stack([left, right], dim=1), joint_ids=self._wheel_ids)

    def _slip_factor(self, yaw_rate: torch.Tensor) -> torch.Tensor:
        """(N,) multiplier on commanded forward speed, in ``(0, 1]``.

        Per-episode systematic component (wheel radius, tyre wear, floor) plus a
        per-substep random component plus a turn-dependent term, because
        skid-steer scrubs hardest when the wheels fight each other. Gated on
        ``wheel_slip`` so a run can be made exactly repeatable for debugging.
        """
        if not self.cfg.wheel_slip:
            return torch.ones_like(yaw_rate)
        return wheel_motion_factor(
            self._wheel_calibration, yaw_rate,
            generator=self._noise_generator)

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
        if self.cfg.collection_mode:
            # DirectRLEnv.step() always asks for an observation.  A single
            # cached tensor makes that request O(1) during BC collection while
            # preserving the regular environment path for PPO and checks.
            if not hasattr(self, "_collection_obs"):
                self._collection_obs = torch.empty(
                    self.num_envs, 0, device=self.device)
            return {"policy": self._collection_obs}

        pose = self._robot_pose()
        goal = self._goal_w - self.scene.env_origins[:, :2]
        ranges, angles = self._lidar_ranges(), self._beam_angles

        # The same corruption functions the dataset loader applies, in the same
        # order, so a policy fine-tuned here sees the input distribution it was
        # cloned on. Applied to the raw scan, before binning -- corrupting bins
        # instead would put noise on a quantity the sensor does not produce.
        scale = float(self.cfg.observation_noise_scale)
        if scale > 0.0:
            ranges, angles = corrupt_lidar(ranges, angles, scale,
                                           self._noise_generator,
                                           bearing_bias=(
                                               self._lidar_bearing_bias[:, None]
                                               * scale))

        # ── the spinning sensor ──────────────────────────────────────────────
        # Push this instant's scan, then hand the last SPIN_TICKS of them to the
        # sweep so each beam is drawn from whichever one is nearest its own
        # acquisition time. A freshly reset environment has no history, so its
        # scan is replicated across the buffer -- the same convention the
        # loader's episode-boundary handling uses.
        reset = self._history_reset
        self._scan_history[:, :-1] = self._scan_history[:, 1:].clone()
        self._scan_history[:, -1] = ranges
        self._scan_history[reset] = ranges[reset, None, :]

        measured_rate = self._gyro.step(self._true_yaw_rate(pose))
        # A reset observation has no preceding control interval. The offline
        # gyro synthesizer and evaluator both publish zero for this sample.
        measured_rate[reset] = 0.0
        increment = measured_rate * self.cfg.sim.dt * self.cfg.decimation
        self._gyro_history[:, :-1] = self._gyro_history[:, 1:].clone()
        self._gyro_history[:, -1] = increment
        self._gyro_history[reset] = increment[reset, None]
        normalized_action = torch.stack([
            self._action[:, 0] / V_MAX, self._action[:, 1] / OMEGA_MAX
        ], dim=1)
        self._motion_action_history[:, :-1] = (
            self._motion_action_history[:, 1:].clone())
        self._motion_action_history[:, -1] = normalized_action
        self._motion_action_history[reset] = normalized_action[reset, None]
        self._deskew_gyro_history[:, :-1] = (
            self._deskew_gyro_history[:, 1:].clone())
        self._deskew_gyro_history[:, -1] = increment
        self._deskew_speed_history[:, :-1] = (
            self._deskew_speed_history[:, 1:].clone())
        self._deskew_speed_history[:, -1] = self._action[:, 0]
        self._deskew_gyro_history[reset] = increment[reset, None]
        self._deskew_speed_history[reset] = self._action[reset, 0, None]

        swept, ages = sweep_scan(self._scan_history, self._spin_phase,
                                 subtick=True)
        if DESKEW_WITH_GYRO:
            swept, angles = deskew_history(
                swept, angles, ages, self._deskew_gyro_history,
                self._deskew_speed_history, subtick=True)
        canonical_angles = (angles if angles.dim() == 2
                            else angles[None].expand_as(swept))
        canonical_valid = (torch.isfinite(swept)
                           & (swept >= LIDAR_MIN_M) & (swept <= LIDAR_MAX_M))
        self._canonical_ranges = swept
        self._canonical_bearings = canonical_angles
        self._canonical_ages = ages
        self._canonical_valid = canonical_valid
        metric, metric_valid = rasterize_lidar(swept, angles, LIDAR_BEAMS)
        self._motion_scan_history[:, :-1] = self._motion_scan_history[:, 1:].clone()
        self._motion_valid_history[:, :-1] = self._motion_valid_history[:, 1:].clone()
        self._motion_scan_history[:, -1] = metric
        self._motion_valid_history[:, -1] = metric_valid
        self._motion_scan_history[reset] = metric[reset, None]
        self._motion_valid_history[reset] = metric_valid[reset, None]
        self._spin_phase = advance_phase(self._spin_phase)

        # Odometry drift, for the ground-truth-goal observation only. The belief
        # policy never reads this path -- it has no pose input at all -- but PPO
        # and the legacy evaluation do, and they need the integrated walk rather
        # than the white noise that used to stand in for it.
        if scale > 0.0:
            pose = pose + self._odometry_drift.step(
                self._action[:, 1] * self.cfg.sim.dt * self.cfg.decimation)
            pose, goal = corrupt_goal(pose, goal, scale, self._noise_generator)

        self._observation_history.push(
            bin_lidar(swept, angles),
            torch.stack([self._action[:, 0] / V_MAX,
                         self._action[:, 1] / OMEGA_MAX], dim=1),
            self._history_reset)
        self._history_reset.zero_()
        obs = self._observation_history.observation(goal_features(pose, goal))
        return {"policy": obs}

    def _true_yaw_rate(self, pose: torch.Tensor) -> torch.Tensor:
        """(N,) rad/s over the control period just ended, from the true pose.

        Differenced from the pose rather than read off ``root_ang_vel_w``
        because the gyro integrates *attitude change*, and the two disagree
        whenever the contact solver has nudged the base between substeps. The
        pose difference is what actually happened.
        """
        rate = wrap_angle(pose[:, 2] - self._previous_yaw) / (
            self.cfg.sim.dt * self.cfg.decimation)
        self._previous_yaw = pose[:, 2].clone()
        return rate

    def gyro_increment(self) -> torch.Tensor:
        """(N, HISTORY_FRAMES) measured yaw increments, oldest to newest.

        The belief policy's only proprioceptive input. Published as its own
        accessor rather than folded into the observation vector so that the
        actor's 731-number layout -- and therefore every pretrained
        checkpoint -- stays exactly as it was. See ``spec.GYRO_CHANNELS``.
        """
        return self._gyro_history

    def motion_scan(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Deskewed 720-bin metric scan history and valid-return mask.

        This is the high-resolution input for the internal motion estimator. It
        is deliberately separate from the actor's 180-bin observation so actor
        checkpoints do not silently change meaning.
        """
        return self._motion_scan_history, self._motion_valid_history

    def canonical_sweep(self) -> PolarSweep:
        """Newest exact-bearing metric sweep for the CPU localizer."""
        return PolarSweep(self._canonical_ranges, self._canonical_bearings,
                          self._canonical_ages, self._canonical_valid)

    def motion_action(self) -> torch.Tensor:
        """Normalized command history spanning the motion-estimation window."""
        return self._motion_action_history

    # ── the oracle, for BC collection ────────────────────────────────────────

    def oracle_command(self, pose: torch.Tensor | None = None) -> torch.Tensor:
        """(N, 2) the expert's *commanded* (v, omega) for the current state.

        This is the BC label. It is the command, never the achieved velocity
        read back from physics: training on achieved velocity teaches the policy
        to emit a value the actuator will lag a second time, a systematic
        underactuation that is invisible in sim and obvious on hardware.
        """
        if pose is None:
            pose = self._robot_pose()
        return oracle_action(self.pool, self._layout, pose, self._action)

    def raw_scan(self):
        """What the dataset stores: raw ranges and base_link beam bearings.

        Deliberately *not* the observation. Noise and binning happen in the
        loader so retuning either does not invalidate a collection run.
        """
        return self._lidar_ranges(), self._beam_angles

    # ── termination and reward ───────────────────────────────────────────────

    def _get_dones(self):
        # Cached for _get_rewards, which DirectRLEnv.step calls immediately
        # after this on the same physics state and before any reset. Reading
        # the pose again there would be equivalent but would leave which pose
        # it meant -- pre- or post-reset -- resting on call ordering.
        pose = self._pose = self._robot_pose()
        goal = self._goal_w - self.scene.env_origins[:, :2]
        self._reached = (
            torch.linalg.vector_norm(pose[:, :2] - goal, dim=1)
            < GOAL_TOLERANCE_M)
        self._collided = ~self.pool.is_free(self._layout, pose[:, :2])
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        # There is no useful post-collision trajectory: collection truncates
        # before the colliding state, and an RL episode should also end on a
        # collision.  Resetting here avoids simulating as many as 899 steps
        # that the collector is guaranteed to throw away.
        reached_terminal = (self._reached if self.cfg.terminate_on_goal
                            else torch.zeros_like(self._reached))
        external_done = self._external_done.clone()
        self._external_done.zero_()
        return reached_terminal | self._collided | external_done, timeout

    def _get_rewards(self) -> torch.Tensor:
        """Dense geodesic progress, with terminal outcomes and two small costs.

        **Geodesic, never Euclidean.** Straight-line progress pays the robot to
        drive into the concave face of an obstacle and then charges it for the
        detour back out; around a U-shaped obstacle that is a local optimum the
        policy can and does get stuck in. The field this reads is ``Room
        .geodesic`` -- true metric path length, the same field that decides
        which starts are sampled and supplies the SPL denominator, not the
        clearance-penalised cost the oracle steers by. So "closer to the goal"
        means one thing across the whole project.

        The progress term telescopes: over an episode the discounted-free sum
        of ``d_{t-1} - d_t`` is the total distance closed, whatever route was
        taken. Shaping like this changes *when* reward arrives, not which
        trajectory is optimal.
        """
        if self.cfg.collection_mode:
            # Collection and evaluation never read this, and the gathers below
            # are not free at 16k environments.
            return torch.zeros(self.num_envs, device=self.device)

        distance = self.pool.distance_to_goal(self._layout, self._pose[:, :2])
        progress = (self._previous_distance - distance).clamp(
            -PROGRESS_CLAMP_M, PROGRESS_CLAMP_M)
        self._previous_distance = distance

        delta = (self._action - self._previous_action) / self._action_limits

        return (self.cfg.progress_reward * progress
                + self.cfg.goal_reward * self._reached.float()
                - self.cfg.collision_penalty * self._collided.float()
                - self.cfg.timeout_penalty * self.reset_time_outs.float()
                - self.cfg.time_penalty
                - self.cfg.smoothness_penalty * delta.square().sum(dim=1))


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
