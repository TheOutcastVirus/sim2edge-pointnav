"""Reference navigation environment — pure numpy, no simulator dependency.

This is the third adapter of the spec, alongside Isaac Lab and ManiSkill, and it
exists for reasons the other two cannot serve:

* it runs anywhere (CI, a laptop, the board) with no GPU and no simulator
  install, so the *pipeline* — spec, preprocessing, oracle, BC, PPO, eval,
  export, parity — is testable end to end without a 40 GB dependency;
* it is the reference the other two adapters are validated against: same rooms,
  same reward, same observation construction, so a disagreement is a bug in an
  adapter rather than a property of a simulator.

It is deliberately **not** the training environment for the deployed policy —
Isaac Lab is (§3, §4 of the implementation plan). Physics here is a
kinematic differential-drive model with first-order motor lag, which is a good
match for a Create3 on flat floor but is not a rigid-body simulation.

Conventions: world frame x-right/y-up, robot frame x-forward/y-left, yaw CCW.
Actions are **physical** ``(v, omega)`` in m/s and rad/s, matching the exported
policy's output and the ``Twist`` the C++ node publishes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..preprocess import Preprocessor, world_goal_to_robot_frame, wrap_angle
from ..randomization import (
    EpisodeRandomization,
    corrupt_depth,
    corrupt_lidar,
    sample_randomization,
)
from ..rooms import Room, sample_room
from ..oracle.grid import UNREACHABLE
from ..safety import SafetyLayer
from ..sensors import render_depth, simulate_lidar
from ..spec import DEFAULT_SPEC, NavSpec, RandomizationSpec

__all__ = ["NavEnv", "EnvConfig", "StepResult"]


@dataclass
class EnvConfig:
    """Knobs that are properties of *this run*, not of the task contract."""

    spec: NavSpec = field(default_factory=lambda: DEFAULT_SPEC)

    #: Skip depth rendering entirely. The lidar-only configuration is the
    #: Day-5 guaranteed demo and trains far faster, so it is a first-class mode
    #: rather than an ablation: the depth tensor is still produced (the graph
    #: shape must not change) but filled with the "no return" value.
    use_depth: bool = True

    #: Override the randomization ranges, e.g. for the eval noise tiers.
    randomization: Optional[RandomizationSpec] = None

    #: Run the classical safety layer inside the env. Off during training (a
    #: policy trained behind a safety net learns to rely on it); on for
    #: deployment-parity evaluation, where intervention count is a metric.
    apply_safety: bool = False

    max_episode_steps: Optional[int] = None

    #: Terminate on collision. Some BC/DAgger collection modes prefer to keep
    #: rolling so the expert can demonstrate the recovery.
    terminate_on_collision: bool = True


@dataclass
class StepResult:
    obs: Dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class NavEnv:
    """Single-robot goal-conditioned navigation environment."""

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None):
        self.config = config or EnvConfig()
        self.spec = self.config.spec
        self.pre = Preprocessor(self.spec)
        self.rng = np.random.default_rng(seed)
        self.safety = SafetyLayer(self.spec)

        self.room: Optional[Room] = None
        self.dr = EpisodeRandomization.none()

        # Pose and velocity state.
        self.x = self.y = self.yaw = 0.0
        self.v = self.omega = 0.0
        self.step_count = 0

        self._action_queue: list[Tuple[float, float]] = []
        self._last_action = (0.0, 0.0)
        self._prev_geodesic = 0.0
        self._path_length = 0.0
        self._odom_offset = np.zeros(2, dtype=np.float64)
        self._odom_yaw_offset = 0.0
        self._intervention_count = 0
        self._last_raw = {}

    # ── Episode lifecycle ────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        room: Optional[Room] = None,
    ) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.room = room if room is not None else sample_room(self.rng, self.spec)
        self.dr = sample_randomization(self.rng, self.spec,
                                       self.config.randomization)

        self.x, self.y, self.yaw = self.room.start
        self.v = self.omega = 0.0
        self.step_count = 0
        self._intervention_count = 0
        self._path_length = 0.0
        self._last_action = (0.0, 0.0)
        self._odom_offset[:] = 0.0
        self._odom_yaw_offset = 0.0

        # Pre-fill the delay queue with zero commands: the real robot is
        # stationary and has issued nothing when the policy's first tick lands.
        self._action_queue = [(0.0, 0.0)] * self.dr.action_delay_steps

        self._prev_geodesic = self._geodesic_now()
        self.safety.reset()
        return self._observe()

    # ── Stepping ─────────────────────────────────────────────────────────────

    def step(self, action) -> StepResult:
        """Advance one control step with a physical ``(v, omega)`` command."""
        a = self.spec.action
        v_cmd = float(np.clip(action[0], a.v_min, a.v_max))
        w_cmd = float(np.clip(action[1], -a.omega_max, a.omega_max))
        raw_cmd = (v_cmd, w_cmd)

        intervened = False
        if self.config.apply_safety:
            ranges, angle_min, angle_inc = self._last_raw["scan"]
            v_cmd, w_cmd, intervened = self.safety.filter(
                v_cmd, w_cmd, ranges, angle_min, angle_inc)
            self._intervention_count += int(intervened)

        # ── Latency ──────────────────────────────────────────────────────────
        self._action_queue.append((v_cmd, w_cmd))
        v_applied, w_applied = self._action_queue.pop(0) if self._action_queue \
            else (v_cmd, w_cmd)

        self._integrate(v_applied, w_applied)

        # `last_action` is what the *policy* commanded, not what the safety layer
        # or the motor lag produced: it is the policy's own previous output, and
        # the smoothness penalty is defined on the policy's action sequence.
        prev_action = self._last_action
        self._last_action = raw_cmd

        self.step_count += 1
        return self._finish_step(raw_cmd, prev_action, intervened)

    def _integrate(self, v_cmd: float, w_cmd: float) -> None:
        """First-order motor lag, then unicycle kinematics."""
        dt = self.spec.action.dt
        dr = self.dr

        # Heavier robots respond more sluggishly; mass_scale stretches the lag.
        tau = max(1e-3, dr.motor_tau_s * dr.mass_scale)
        alpha = 1.0 - math.exp(-dt / tau)
        self.v += (v_cmd * dr.velocity_gain - self.v) * alpha
        self.omega += (w_cmd * dr.velocity_gain - self.omega) * alpha

        # Slip removes commanded translation without removing rotation, which is
        # what a wheel losing traction on a smooth floor actually does.
        v_eff = self.v * (1.0 - dr.wheel_slip)

        x0, y0 = self.x, self.y
        if abs(self.omega) < 1e-6:
            self.x += v_eff * math.cos(self.yaw) * dt
            self.y += v_eff * math.sin(self.yaw) * dt
        else:
            # Exact arc integration; Euler integration at 20 Hz visibly
            # under-turns at omega_max and would misattribute the resulting
            # tracking error to the policy.
            r = v_eff / self.omega
            yaw2 = self.yaw + self.omega * dt
            self.x += r * (math.sin(yaw2) - math.sin(self.yaw))
            self.y -= r * (math.cos(yaw2) - math.cos(self.yaw))
        self.yaw = wrap_angle(self.yaw + self.omega * dt)

        moved = math.hypot(self.x - x0, self.y - y0)
        self._path_length += moved

        # ── Odometry drift ───────────────────────────────────────────────────
        # Drift accumulates with distance travelled and with rotation, the way
        # wheel-encoder odometry actually degrades. It corrupts only the goal
        # vector the policy sees; ground truth is untouched.
        if dr.odom_drift_rate > 0.0:
            self._odom_offset += moved * dr.odom_drift_rate * np.array(
                [math.cos(dr.odom_drift_dir), math.sin(dr.odom_drift_dir)])
        if dr.odom_yaw_drift_rate > 0.0:
            self._odom_yaw_offset += abs(self.omega * dt) * dr.odom_yaw_drift_rate

    def _finish_step(self, action: Tuple[float, float],
                     prev_action: Tuple[float, float],
                     intervened: bool) -> StepResult:
        assert self.room is not None
        rw = self.spec.reward

        collided = self.room.in_collision(self.x, self.y)
        goal_dist = math.hypot(self.room.goal[0] - self.x,
                               self.room.goal[1] - self.y)
        reached = goal_dist <= rw.goal_radius_m

        geodesic = self._geodesic_now()
        # Guard the unreachable sentinel: differencing it against a finite value
        # would produce a six-figure reward spike.
        if geodesic >= UNREACHABLE * 0.5 or self._prev_geodesic >= UNREACHABLE * 0.5:
            progress = 0.0
        else:
            progress = self._prev_geodesic - geodesic
        self._prev_geodesic = geodesic

        a = self.spec.action
        an = np.array(a.normalize(*action))
        pn = np.array(a.normalize(*prev_action))
        smooth = float(np.sum((an - pn) ** 2))

        reward = (rw.progress * progress
                  - rw.smoothness * smooth
                  - rw.time_penalty)
        if reached:
            reward += rw.goal_bonus
        if collided:
            reward -= rw.collision_penalty

        max_steps = self.config.max_episode_steps or rw.max_episode_steps
        terminated = bool(reached or (collided and self.config.terminate_on_collision))
        truncated = bool(self.step_count >= max_steps and not terminated)

        obs = self._observe()
        info = {
            "success": bool(reached),
            "collision": bool(collided),
            "goal_distance": goal_dist,
            "geodesic": geodesic,
            "path_length": self._path_length,
            "optimal_path_length": self.room.optimal_path_length,
            "interventions": self._intervention_count,
            "intervened": bool(intervened),
            "steps": self.step_count,
            "pose": (self.x, self.y, self.yaw),
        }
        if reached:
            info["spl"] = _spl(self.room.optimal_path_length, self._path_length)
        return StepResult(obs, float(reward), terminated, truncated, info)

    # ── Observation ──────────────────────────────────────────────────────────

    def _geodesic_now(self) -> float:
        assert self.room is not None
        return self.room.geodesic_at(self.x, self.y)

    def odom_pose(self) -> Tuple[float, float, float]:
        """Pose as the robot's (drifting) odometry believes it to be."""
        return (self.x + float(self._odom_offset[0]),
                self.y + float(self._odom_offset[1]),
                wrap_angle(self.yaw + self._odom_yaw_offset))

    def _observe(self) -> Dict[str, np.ndarray]:
        assert self.room is not None
        cfg, spec = self.config, self.spec

        # ── Lidar ────────────────────────────────────────────────────────────
        ranges, angle_min, angle_inc = simulate_lidar(
            self.room, self.x, self.y, self.yaw, spec)
        ranges = corrupt_lidar(ranges, self.dr, self.rng)
        # An angular calibration offset is a property of the mount, so it is
        # applied to the scan's declared start angle, exactly as a miscalibrated
        # real lidar would report it.
        self._last_raw["scan"] = (ranges, angle_min + self.dr.lidar_angle_offset,
                                  angle_inc)

        # ── Depth ────────────────────────────────────────────────────────────
        if cfg.use_depth:
            depth = render_depth(
                self.room, self.x, self.y, self.yaw, spec,
                camera_height=spec.robot.camera_height_m + self.dr.depth_height_offset,
                camera_pitch=spec.robot.camera_pitch_rad + self.dr.depth_pitch_offset,
            )
            depth = corrupt_depth(depth, self.dr, self.rng)
        else:
            # Lidar-only mode: a constant "no return" image. The tensor must
            # still exist so the exported graph's signature never changes
            # between configurations.
            depth = np.zeros((spec.obs.depth.height, spec.obs.depth.width),
                             dtype=np.float32)
        self._last_raw["depth"] = depth

        # ── Goal, from drifting odometry ─────────────────────────────────────
        ox, oy, oyaw = self.odom_pose()
        noise = self.dr.odom_noise_m
        if noise > 0.0:
            ox += float(self.rng.normal(0.0, noise))
            oy += float(self.rng.normal(0.0, noise))
        goal_robot = world_goal_to_robot_frame(self.room.goal, (ox, oy), oyaw)

        obs = self.pre.build(
            depth_m=depth,
            scan_ranges=self._last_raw["scan"][0],
            scan_angle_min=self._last_raw["scan"][1],
            scan_angle_increment=angle_inc,
            goal_xy_robot=goal_robot,
            last_action=self._last_action,
        )
        obs["privileged"] = self._privileged()
        return obs

    def _privileged(self) -> np.ndarray:
        """Ground-truth critic input. Never exported, never available on robot."""
        assert self.room is not None
        p, a = self.spec.obs.privileged, self.spec.action
        geo = min(self._geodesic_now(), p.max_geodesic_m) / p.max_geodesic_m
        clear = min(self.room.clearance_at(self.x, self.y),
                    p.max_obstacle_m) / p.max_obstacle_m
        return np.array([
            geo,
            clear,
            self.x / max(self.room.arena_w, 1e-6),
            self.y / max(self.room.arena_h, 1e-6),
            math.sin(self.yaw),
            math.cos(self.yaw),
            self.v / a.v_max,
            self.omega / a.omega_max,
        ], dtype=np.float32)

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def pose(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.yaw)

    def raw_sensors(self) -> Dict[str, Any]:
        """Latest *raw* sensor output, pre-preprocessing.

        Used by the calibration-set builder so quantization sees data that has
        been through the identical preprocessing path the robot uses, and by the
        parity test as synthetic input when no rosbag is available.
        """
        return dict(self._last_raw)


def _spl(optimal: float, actual: float) -> float:
    """Success weighted by (normalized inverse) Path Length, for one episode."""
    if actual <= 1e-6:
        return 1.0 if optimal <= 1e-6 else 0.0
    return float(min(1.0, optimal / max(actual, optimal)))
