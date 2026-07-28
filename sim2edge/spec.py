"""Single source of truth for the sim2edge navigation policy contract.

Everything that must agree between simulation, training, quantization
calibration, and the on-robot C++ node lives here: observation shapes,
preprocessing constants, action limits, reward coefficients, and domain
randomization ranges.

**This module is deliberately stdlib-only.** No numpy, no torch, no simulator
imports. It has to be importable from:

* the Isaac Lab training environment (heavy CUDA stack),
* the board's ExecuTorch export venv (pinned torch, no torchvision, minimal deps),
* ``tools/gen_spec_header.py``, which emits the C++ header the ROS node compiles
  against.

The #1 documented failure mode in both reference projects was a calibration /
runtime preprocessing mismatch, which produces silent garbage rather than an
error. The structural mitigation is that there is exactly one place these
constants are written down (here), one Python implementation
(``sim2edge.preprocess``), one C++ implementation generated from this file, and
a parity test that runs both on the same raw messages.

``NavSpec.fingerprint()`` returns a short hash of the whole contract. It is
written into the ``.pte`` sidecar at export time and checked by the ROS node at
load time, so a policy exported under one preprocessing contract cannot silently
be run under another.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Tuple, get_origin, get_type_hints

SPEC_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Observations
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DepthSpec:
    """Depth channel: the primary visual input.

    Geometry transfers from sim to real for free; textures and lighting do not.
    Deliberately small — the policy needs obstacle geometry, not detail, and a
    tiny input keeps both camera-env training throughput and NPU latency cheap.
    """

    height: int = 96
    width: int = 160

    #: Metric clip range. Below ``min_m`` the OAK-D Lite stereo is unreliable;
    #: beyond ``max_m`` the policy does not need the detail.
    min_m: float = 0.3
    max_m: float = 6.0

    #: Invalid pixels (stereo holes, out of range, zero) map to this normalized
    #: value. 1.0 == "far", i.e. holes are optimistically treated as free space.
    #: The lidar and the classical safety bubble cover the pessimistic case.
    invalid_fill: float = 1.0

    #: Raw device stream is 16UC1 millimetres; this converts to metres.
    raw_scale_to_m: float = 0.001

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (1, self.height, self.width)

    @property
    def numel(self) -> int:
        return self.height * self.width


@dataclass(frozen=True)
class LidarSpec:
    """2D lidar channel, min-pooled from the raw RPLIDAR scan.

    Min-pooling (not averaging or subsampling) is deliberate: it is the only
    reduction that cannot delete a thin obstacle from the observation.
    """

    bins: int = 180

    min_m: float = 0.15
    max_m: float = 8.0

    #: Invalid returns (inf, nan, out of range) normalize to this. 1.0 == "far".
    invalid_fill: float = 1.0

    #: Scan sector kept, radians, robot frame, 0 = forward, CCW positive.
    #: Full 360 degrees: the TB4 Lite's RPLIDAR sees behind the robot and
    #: reversing away from a trap needs that.
    angle_min: float = -math.pi
    angle_max: float = math.pi

    @property
    def shape(self) -> Tuple[int]:
        return (self.bins,)


@dataclass(frozen=True)
class GoalSpec:
    """Relative goal vector: (normalized distance, sin bearing, cos bearing).

    The angle is encoded as sin/cos rather than a raw angle so the input is
    continuous across the +/-pi wrap. A wrap discontinuity in an INT8-quantized
    input is a nasty, hard-to-debug failure right where the robot most needs to
    turn around.
    """

    #: Distance is clipped here before normalizing. Also the honest range limit
    #: of the demo: Create3 odometry drifts ~1-2%/m, so goals stay <= 8 m.
    max_distance_m: float = 8.0

    @property
    def shape(self) -> Tuple[int]:
        return (3,)


@dataclass(frozen=True)
class LastActionSpec:
    """Previous command, normalized to [-1, 1].

    Required for the smoothness penalty to be learnable at all: without it the
    penalty depends on information not in the observation, so the policy cannot
    represent the optimal action.
    """

    @property
    def shape(self) -> Tuple[int]:
        return (2,)


@dataclass(frozen=True)
class PrivilegedSpec:
    """Critic-only observations. Never exported, never needed on the robot.

    Asymmetric actor-critic: the critic gets ground truth the actor cannot have,
    which lowers value-function variance for free because the critic never ships.

    Layout: [geodesic distance to goal (normalized), nearest obstacle distance
    (normalized), x, y (arena-normalized), sin yaw, cos yaw, v, omega (normalized)]
    """

    dim: int = 8

    max_geodesic_m: float = 20.0
    max_obstacle_m: float = 5.0

    @property
    def shape(self) -> Tuple[int]:
        return (self.dim,)


@dataclass(frozen=True)
class ObservationSpec:
    depth: DepthSpec = field(default_factory=DepthSpec)
    lidar: LidarSpec = field(default_factory=LidarSpec)
    goal: GoalSpec = field(default_factory=GoalSpec)
    last_action: LastActionSpec = field(default_factory=LastActionSpec)
    privileged: PrivilegedSpec = field(default_factory=PrivilegedSpec)

    #: Actor input order. This is also the positional order of the exported
    #: graph's forward() arguments, and therefore the order the C++ node must
    #: bind input tensors in. Changing it changes the fingerprint.
    actor_inputs: Tuple[str, ...] = ("depth", "lidar", "goal", "last_action")

    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "depth": self.depth.shape,
            "lidar": self.lidar.shape,
            "goal": self.goal.shape,
            "last_action": self.last_action.shape,
            "privileged": self.privileged.shape,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActionSpec:
    """(v, omega) velocity commands at the deployment control rate.

    Limits are the real Create3's, minus a margin. The oracle, the simulator,
    the policy's in-graph squash, and the C++ clamp all read these same numbers:
    if the expert demonstrates commands the robot cannot track, behaviour
    cloning faithfully learns to issue commands the robot cannot track.
    """

    v_max: float = 0.30  # m/s   (Create3 hardware max ~0.306)
    v_min: float = 0.0  # no reverse; the policy turns instead
    omega_max: float = 1.50  # rad/s (Create3 hardware max ~1.90)

    #: Rate limits, applied in sim (first-order lag + clamp) and in C++.
    accel_max: float = 0.60  # m/s^2
    alpha_max: float = 3.00  # rad/s^2

    control_rate_hz: float = 20.0

    @property
    def dt(self) -> float:
        return 1.0 / self.control_rate_hz

    @property
    def shape(self) -> Tuple[int]:
        return (2,)

    def normalize(self, v: float, omega: float) -> Tuple[float, float]:
        """Physical (v, omega) -> [-1, 1]^2, the ``last_action`` encoding."""
        v_n = 2.0 * (v - self.v_min) / (self.v_max - self.v_min) - 1.0
        w_n = omega / self.omega_max
        return (_clip(v_n, -1.0, 1.0), _clip(w_n, -1.0, 1.0))

    def denormalize(self, v_n: float, w_n: float) -> Tuple[float, float]:
        """[-1, 1]^2 -> physical (v, omega). Inverse of :meth:`normalize`."""
        v = self.v_min + 0.5 * (v_n + 1.0) * (self.v_max - self.v_min)
        return (_clip(v, self.v_min, self.v_max),
                _clip(w_n * self.omega_max, -self.omega_max, self.omega_max))


# ─────────────────────────────────────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RewardSpec:
    """Per-step reward coefficients.

    Progress uses **geodesic** distance from the oracle's distance field, never
    Euclidean: Euclidean progress pays the robot to drive into the concave side
    of an obstacle, which is exactly where local navigation policies get stuck.
    """

    goal_bonus: float = 10.0  # terminal, on reaching the goal
    progress: float = 3.0  # per metre of geodesic progress
    collision_penalty: float = 10.0  # terminal
    smoothness: float = 0.05  # * ||a_t - a_{t-1}||^2, normalized actions
    time_penalty: float = 0.005  # per step; 0.1/s at 20 Hz

    #: Success radius used by the reward and by the sim success criterion.
    #: The on-robot protocol uses a laxer 0.5 m measured externally, because
    #: drifting odometry cannot be trusted to judge itself.
    goal_radius_m: float = 0.35

    #: Episode length cap, in control steps (60 s at 20 Hz).
    max_episode_steps: int = 1200


# ─────────────────────────────────────────────────────────────────────────────
# Domain randomization
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RandomizationSpec:
    """Ranges are ``(low, high)``, sampled per episode unless noted.

    Set from measured rosbags, not guessed — see ``docs/domain_randomization.md``.
    ``scale`` multiplies every range, so the eval harness can run the same
    policy at clean (0.0), deployment (1.0) and 2x-deployment noise tiers.
    """

    enabled: bool = True
    scale: float = 1.0

    # ── Lidar ────────────────────────────────────────────────────────────────
    lidar_range_noise_m: Tuple[float, float] = (0.0, 0.03)  # per-beam sigma
    lidar_dropout: Tuple[float, float] = (0.0, 0.05)  # fraction of beams
    lidar_angular_offset_rad: Tuple[float, float] = (-0.0175, 0.0175)  # +/-1 deg

    # ── Depth ────────────────────────────────────────────────────────────────
    #: Stereo error grows with distance squared: sigma = coeff * d^2.
    depth_noise_coeff: Tuple[float, float] = (0.0, 0.01)
    depth_hole_fraction: Tuple[float, float] = (0.0, 0.12)
    depth_speckle_fraction: Tuple[float, float] = (0.0, 0.02)
    depth_scale_jitter: Tuple[float, float] = (0.97, 1.03)
    #: Extrinsic perturbation of the camera mount (pitch rad, height m).
    depth_pitch_noise_rad: Tuple[float, float] = (-0.035, 0.035)
    depth_height_noise_m: Tuple[float, float] = (-0.015, 0.015)

    # ── Dynamics ─────────────────────────────────────────────────────────────
    velocity_tracking_gain: Tuple[float, float] = (0.90, 1.10)
    mass_scale: Tuple[float, float] = (0.80, 1.20)
    #: First-order motor lag time constant.
    motor_tau_s: Tuple[float, float] = (0.05, 0.15)
    wheel_slip: Tuple[float, float] = (0.0, 0.05)

    # ── Latency ──────────────────────────────────────────────────────────────
    #: Action delay in whole control steps, sampled per episode. The single most
    #: important randomization for sim-to-real on a real ROS stack: sensor ->
    #: cmd_vel latency on the robot is never less than one control period.
    action_delay_steps: Tuple[int, int] = (1, 3)

    # ── Odometry ─────────────────────────────────────────────────────────────
    #: The real goal vector comes from drifting Create3 odometry, so the goal
    #: input is corrupted with a slow random walk plus white noise.
    odom_drift_rate: Tuple[float, float] = (0.0, 0.02)  # m of drift per m driven
    odom_yaw_drift_rate: Tuple[float, float] = (0.0, 0.02)  # rad per rad turned
    odom_noise_m: Tuple[float, float] = (0.0, 0.02)

    # ── Layout ───────────────────────────────────────────────────────────────
    friction: Tuple[float, float] = (0.6, 1.2)

    def at_scale(self, scale: float) -> "RandomizationSpec":
        """Return a copy with all ranges scaled about their low end.

        ``scale=0`` collapses every range to a point (clean), ``1`` is
        deployment level, ``2`` is the robustness-margin tier.
        """
        return RandomizationSpec(
            enabled=self.enabled and scale > 0.0,
            scale=scale,
            **{
                f.name: _scale_range(getattr(self, f.name), scale)
                for f in fields(self)
                if f.name not in ("enabled", "scale")
            },
        )


def _scale_range(rng: Tuple[float, float], scale: float) -> Tuple[float, float]:
    lo, hi = rng
    # Ranges centred on zero (noise, offsets) scale about zero; one-sided ranges
    # (fractions, gains) scale about their nominal value.
    if lo < 0.0 < hi:
        out = (lo * scale, hi * scale)
    elif lo == 0.0:
        out = (0.0, hi * scale)
    else:
        mid = 0.5 * (lo + hi)
        out = (mid + (lo - mid) * scale, mid + (hi - mid) * scale)
    if isinstance(lo, int) and isinstance(hi, int):
        return (int(round(out[0])), int(round(out[1])))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Robot / sensor geometry
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RobotSpec:
    """TB4 Lite geometry. Validate against the real robot before trusting sim."""

    radius_m: float = 0.175  # Create3 footprint radius
    wheel_base_m: float = 0.235

    #: Top of the robot including the TB4 Lite standoffs and the lidar. An
    #: obstacle collides only if it occupies space below this; an obstacle whose
    #: underside is above it (a table top, a shelf) is driven under, and must
    #: NOT be rasterized into the collision grid or the oracle refuses to plan
    #: through gaps the robot actually fits under.
    height_m: float = 0.31

    #: OAK-D Lite mount on the TB4 Lite standoffs.
    camera_height_m: float = 0.19
    camera_pitch_rad: float = 0.0  # +down
    camera_hfov_rad: float = 1.204  # 69 deg
    camera_vfov_rad: float = 0.767  # 44 deg

    #: RPLIDAR scan plane height.
    lidar_height_m: float = 0.175


@dataclass(frozen=True)
class RoomSpec:
    """Procedural layout generator ranges."""

    arena_min_m: float = 4.0
    arena_max_m: float = 10.0
    n_obstacles: Tuple[int, int] = (5, 20)
    obstacle_size_m: Tuple[float, float] = (0.15, 1.20)
    goal_distance_m: Tuple[float, float] = (2.0, 8.0)

    #: Occupancy grid resolution used by the oracle and the distance field.
    grid_resolution_m: float = 0.05

    #: Clearance required around start/goal when sampling, on top of the robot
    #: radius, so episodes do not begin in an unrecoverable state.
    spawn_clearance_m: float = 0.15

    #: Extra inflation on top of the robot radius when planning. Must be > 0:
    #: planning at exactly the collision radius makes every path tangent to the
    #: collision boundary, so any tracking error at all becomes a collision, and
    #: the oracle's demonstrations become a dataset of near-misses.
    planning_margin_m: float = 0.06


# ─────────────────────────────────────────────────────────────────────────────
# Safety layer (classical, outside the network)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SafetySpec:
    """Classical guards that run *outside* the policy, on the robot.

    Deliberately part of the shared spec so the simulator can model the same
    intervention behaviour the robot will apply, and so intervention counts are
    comparable between sim and hardware.
    """

    #: Forward sector half-angle over which the lidar bubble looks.
    forward_sector_rad: float = 0.61  # +/-35 deg

    stop_distance_m: float = 0.30  # v := 0 (rotation still allowed)
    slow_distance_m: float = 0.60  # scale v down linearly

    #: Watchdog: no policy output or no fresh sensors for this long -> zero Twist.
    policy_timeout_s: float = 0.5
    lidar_max_age_s: float = 0.15
    depth_max_age_s: float = 0.20
    odom_max_age_s: float = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NavSpec:
    version: str = SPEC_VERSION
    obs: ObservationSpec = field(default_factory=ObservationSpec)
    action: ActionSpec = field(default_factory=ActionSpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    randomization: RandomizationSpec = field(default_factory=RandomizationSpec)
    robot: RobotSpec = field(default_factory=RobotSpec)
    room: RoomSpec = field(default_factory=RoomSpec)
    safety: SafetySpec = field(default_factory=SafetySpec)

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NavSpec":
        return _from_dict(cls, data)

    @classmethod
    def from_json(cls, text: str) -> "NavSpec":
        return cls.from_dict(json.loads(text))

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            fh.write(self.to_json() + "\n")

    @classmethod
    def load(cls, path: str) -> "NavSpec":
        with open(path) as fh:
            return cls.from_json(fh.read())

    # ── Contract fingerprint ─────────────────────────────────────────────────

    def fingerprint(self) -> str:
        """Short stable hash of the full contract.

        Written into the ``.pte`` sidecar at export time and verified by the ROS
        node at load. A policy quantized under one preprocessing contract then
        run under another produces plausible-looking garbage rather than an
        error, so this check is the difference between a five-minute fix and a
        two-day debugging session.

        Randomization ranges are excluded: they affect how the policy was
        trained, not how its inputs are constructed at runtime, and the eval
        harness legitimately varies them.
        """
        payload = self.to_dict()
        payload.pop("randomization", None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def actor_input_shapes(self) -> Tuple[Tuple[int, ...], ...]:
        """Shapes of the exported graph's inputs, batch dim excluded."""
        shapes = self.obs.shapes()
        return tuple(shapes[name] for name in self.obs.actor_inputs)


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _from_dict(cls, data):
    """Rebuild a nested frozen dataclass from ``asdict`` output.

    ``from __future__ import annotations`` makes ``field.type`` a string, so the
    real types are resolved with ``get_type_hints``. JSON has no tuple type, so
    declared tuple fields are coerced back from lists — otherwise a round-tripped
    spec would compare unequal to the original despite being identical.
    """
    if not is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        hint, value = hints[f.name], data[f.name]
        if is_dataclass(hint):
            kwargs[f.name] = _from_dict(hint, value)
        elif get_origin(hint) is tuple and isinstance(value, list):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


#: The contract. Import this, do not construct your own.
DEFAULT_SPEC = NavSpec()


__all__ = [
    "SPEC_VERSION",
    "DEFAULT_SPEC",
    "NavSpec",
    "ObservationSpec",
    "DepthSpec",
    "LidarSpec",
    "GoalSpec",
    "LastActionSpec",
    "PrivilegedSpec",
    "ActionSpec",
    "RewardSpec",
    "RandomizationSpec",
    "RobotSpec",
    "RoomSpec",
    "SafetySpec",
]
