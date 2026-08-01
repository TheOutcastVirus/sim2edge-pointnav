"""Sensor and odometry corruption, applied at **load time** rather than in sim.

This is the change that makes the dataset durable. If the simulator injected
noise, each stored state would carry exactly one corruption sample forever;
retuning the noise model would mean recollecting everything. Applied here, every
epoch draws a fresh corruption of the same clean scan -- strictly better
regularization, at no cost, and the ranges become a training knob instead of a
dataset property.

What this cannot model is corruption that is not a function of the clean ranges:
multipath, and reflectivity-dependent dropout. Isaac's ray caster does not
simulate those either, so nothing is lost by moving noise out of it.

Ranges live here rather than in ``spec.py`` because this module is their only
consumer -- and because they are deliberately excluded from the fingerprint:
noise describes how a policy was *trained*, not how its inputs are *built*.
"""

from __future__ import annotations

import math

import torch

__all__ = ["RANGE_STD_M", "DROPOUT_PROB", "BEARING_BIAS_RAD", "ODOM_XY_STD_M",
           "ODOM_YAW_STD_RAD", "ODOM_XY_WALK_M", "ODOM_YAW_WALK_RAD",
           "ODOM_YAW_RATE_BIAS_RAD", "corrupt_lidar", "corrupt_goal",
           "odometry_drift", "OdometryDrift"]

#: Gaussian range error, metres. Roughly constant over the A1's usable range.
RANGE_STD_M = 0.02
#: Per-beam probability of no return: dark, specular, or glancing surfaces.
DROPOUT_PROB = 0.03
#: Whole-scan bearing offset, radians. The mount yaw is never exactly nominal,
#: and unlike per-beam noise this one is *systematic within an episode*.
BEARING_BIAS_RAD = 0.02

# ── Odometry drift ───────────────────────────────────────────────────────────
# Drift is a **random walk**, not white noise, and getting that wrong made every
# odometry number this project has produced optimistic.
#
# The original model drew a fresh i.i.d. Gaussian per sample. That is trivially
# rejectable: it has zero autocorrelation, so it averages out across the
# four-frame history and the policy can low-pass it away for free. Real wheel
# odometry error *integrates* -- a wheel that slipped 2 cm ago is still 2 cm
# wrong now, and stays wrong for the rest of the episode. A policy trained
# against white noise has never been asked to cope with an error that does not
# come back, which is the only kind that actually threatens it.
#
# Three terms, because they dominate at different timescales:
#   * per-step process noise (below), integrated -- slip and quantization;
#   * a per-episode yaw-*rate* bias -- a miscalibrated wheel base, which turns
#     into a heading error growing linearly in *distance travelled*, and is the
#     single largest contributor over a 10 s episode;
#   * the legacy instantaneous terms, retained at reduced weight as measurement
#     noise on top of the walk.
#
# These are the per-step standard deviations of the walk increments.
ODOM_XY_WALK_M = 0.004          # ~1.8 cm over a 190-step episode (sqrt-t growth)
ODOM_YAW_WALK_RAD = 0.0015      # ~1.2 deg over the same episode
#: Systematic yaw-rate error, constant within an episode. Scales with turning,
#: not with time, so it punishes exactly the manoeuvres that need heading.
ODOM_YAW_RATE_BIAS_RAD = 0.01
#: Residual instantaneous pose noise. Small now that the walk carries the real
#: error -- these numbers used to carry all of it, which is why they were large.
ODOM_XY_STD_M = 0.01
ODOM_YAW_STD_RAD = 0.005

# ── Gyro ─────────────────────────────────────────────────────────────────────
# Rate-gyro error for the TB4's MEMS IMU, as the three terms that actually
# matter over a ten-second episode. They live here rather than in ``sensors.py``
# so that this module remains the single answer to "what sensor error does this
# project simulate"; ``sensors.py`` is their only consumer.
#
# Sized so integrated heading error lands around 1 degree per episode, which is
# what makes the gyro worth having: the wheel-odometry yaw it replaces is worth
# several degrees, and the goal tolerance is 0.25 m at up to 8 m of range.
#
#: Angle random walk: white rate noise, in rad/sqrt(s). Contributes ~0.03 deg
#: over an episode -- negligible, and included only so its negligibility is
#: demonstrated rather than assumed.
GYRO_ARW_RAD_PER_SQRT_S = 1.75e-4
#: Residual bias after power-on calibration, constant within an episode. The
#: dominant term at short timescales: 1e-3 rad/s over 9.5 s is 0.55 degrees.
GYRO_BIAS_RAD_S = 1.0e-3
#: Rate random walk -- how fast the bias itself wanders, in rad/s per sqrt(s).
GYRO_BIAS_WALK_RAD_S = 1.0e-4
#: Scale-factor error, constant within an episode. Proportional to rotation
#: rather than to time, so it is the term that punishes cornering.
GYRO_SCALE_ERROR = 5.0e-3

# ── Wheel slip ───────────────────────────────────────────────────────────────
# The gap between what was commanded and what the base actually did.
#
# This exists because measuring the collected corpus showed it did not: achieved
# forward displacement tracked the command to a standard deviation of 0.22 mm
# per tick, a correlation of 0.9996. That is not a robot, it is a scripted
# animation -- ``_apply_action`` writes the integrated command straight to the
# root pose, so PhysX never gets a say.
#
# The consequence is specific and severe for this design: the odometry head
# exists to correct dead reckoning, dead reckoning was already exact, and the
# residual it was being asked to predict sat ~90x below the lidar's own 20 mm
# range noise. The head had nothing to learn, would have trained to zero, and
# would have looked perfect in simulation while contributing nothing on a robot
# whose wheels really do slip on carpet.
#
# Real differential-drive slip on a Create3: roughly 1-3% on hard floor, up to
# 5-10% on carpet or during hard acceleration, plus a systematic component from
# wheel-radius and tyre-wear calibration that is constant for a whole run.
#
#: Per-episode systematic scale error -- wheel radius, tyre wear, floor type.
#: Constant within an episode, so it integrates into a distance-proportional
#: error rather than averaging out. This is the term that matters.
WHEEL_SLIP_SCALE = 0.03
#: Per-step random slip, as a fraction of the commanded step.
WHEEL_SLIP_STD = 0.02
#: Extra slip while turning, as a fraction per rad/s of commanded yaw rate.
#: Skid-steer scrubs, and it scrubs most when the wheels fight each other.
WHEEL_SLIP_TURN = 0.02


def corrupt_lidar(ranges: torch.Tensor, angles: torch.Tensor, scale: float = 1.0,
                  generator: torch.Generator | None = None, *,
                  bearing_bias: torch.Tensor | None = None):
    """``(N, B)`` clean ranges -> noisy ranges and per-scan angles.

    Returns ``(ranges, angles)`` where ``angles`` is now ``(N, B)`` because the
    bearing bias differs per scan. Dropped beams become ``inf``, which
    :func:`~sim2edge.preprocess.bin_lidar` maps to ``INVALID_FILL`` -- the same
    value the live path produces for a beam that got no return. That agreement
    is the point; a placeholder that differs from what the real path emits is
    the single most expensive bug this project has had.

    ``scale`` multiplies every term, so evaluation can sweep clean (0.0) through
    beyond-training (>1.0) without a second config object.

    ``bearing_bias`` lets a caller supply the ``(N, 1)`` mount error instead of
    having one drawn here. Sequence training needs that: the bias is a property
    of how the sensor is bolted on, so it must be constant down a trajectory,
    and a fresh draw per frame would present it as shakeable noise rather than
    the systematic offset the policy has to live with for a whole episode.
    """
    kw = {"generator": generator, "device": ranges.device, "dtype": ranges.dtype}

    noisy = ranges + torch.randn(ranges.shape, **kw) * (RANGE_STD_M * scale)

    keep = torch.rand(ranges.shape, **kw) >= (DROPOUT_PROB * scale)
    noisy = torch.where(keep, noisy, torch.full_like(noisy, math.inf))

    if angles.dim() == 1:
        angles = angles.unsqueeze(0).expand(ranges.shape[0], -1)
    if bearing_bias is None:
        # One bias per scan, not per beam: a mount error rotates the whole sensor.
        bearing_bias = torch.randn((ranges.shape[0], 1), **kw) * (
            BEARING_BIAS_RAD * scale)
    return noisy, angles + bearing_bias


def corrupt_goal(robot_pose: torch.Tensor, goal_xy: torch.Tensor,
                 scale: float = 1.0, generator: torch.Generator | None = None):
    """Perturb the robot's belief about where it is, which moves the goal.

    Drift is applied to the *pose*, not the goal, because that is where it
    physically originates -- and because a pose error rotates the goal bearing
    as well as translating it, which a goal-only perturbation would miss.

    This is the **instantaneous residual only**. The integrated part lives in
    :func:`odometry_drift`, because it needs episode structure that a batch of
    independently sampled rows does not have. Callers that hold whole episodes
    must apply both; applying only this one reproduces the original white-noise
    model and the optimistic numbers that came with it.
    """
    kw = {"generator": generator, "device": robot_pose.device,
          "dtype": robot_pose.dtype}
    drift = torch.empty_like(robot_pose)
    drift[:, :2] = torch.randn((robot_pose.shape[0], 2), **kw) * (ODOM_XY_STD_M * scale)
    drift[:, 2] = torch.randn((robot_pose.shape[0],), **kw) * (ODOM_YAW_STD_RAD * scale)
    return robot_pose + drift, goal_xy


def _episode_starts(episode: torch.Tensor) -> torch.Tensor:
    """(T,) bool, true at the first sample of each contiguous episode."""
    first = torch.ones_like(episode, dtype=torch.bool)
    first[1:] = episode[1:] != episode[:-1]
    return first


def _segmented_cumsum(values: torch.Tensor,
                      episode: torch.Tensor) -> torch.Tensor:
    """Cumulative sum along dim 0 that restarts at every episode boundary.

    Done as one global cumsum minus the running total at each episode's first
    sample, rather than a Python loop over episodes: a 200k-sample shard holds
    on the order of 800 episodes, and 800 kernel launches per epoch to compute
    a noise term is not a trade worth making.
    """
    total = torch.cumsum(values, dim=0)
    exclusive = total - values
    first = _episode_starts(episode)
    ordinal = torch.cumsum(first.long(), dim=0) - 1      # (T,) episode index
    starts = torch.nonzero(first, as_tuple=False).squeeze(1)
    return total - exclusive[starts][ordinal]


def odometry_drift(episode: torch.Tensor, scale: float = 1.0,
                   generator: torch.Generator | None = None, *,
                   yaw_increment: torch.Tensor | None = None) -> torch.Tensor:
    """(T, 3) integrated pose error for a whole shard, restarting per episode.

    ``yaw_increment`` is the per-step commanded rotation in radians (omega * DT).
    When supplied it drives the scale-factor term, which is the one that grows
    with *how much the robot turned* rather than with elapsed time. That term
    dominates over a ten-second episode and it is the reason a policy can look
    fine on a straight run and lose the goal completely after two corners.

    Returned drift is added to the true pose; the caller still applies
    :func:`corrupt_goal` on top for the instantaneous measurement noise.
    """
    device, dtype = episode.device, torch.float32
    steps = episode.shape[0]
    kw = {"generator": generator, "device": device, "dtype": dtype}

    increments = torch.empty((steps, 3), **{k: v for k, v in kw.items()
                                            if k != "generator"})
    increments[:, :2] = torch.randn((steps, 2), **kw) * (ODOM_XY_WALK_M * scale)
    increments[:, 2] = torch.randn((steps,), **kw) * (ODOM_YAW_WALK_RAD * scale)
    drift = _segmented_cumsum(increments, episode)

    if yaw_increment is not None:
        # One bias per episode, broadcast back over that episode's samples. The
        # walk above is zero-mean and forgets; this one does not.
        first = _episode_starts(episode)
        ordinal = torch.cumsum(first.long(), dim=0) - 1
        count = int(ordinal[-1].item()) + 1 if steps else 0
        bias = torch.randn((count,), **kw) * (ODOM_YAW_RATE_BIAS_RAD * scale)
        rotated = _segmented_cumsum(
            yaw_increment.to(dtype).reshape(-1, 1), episode).squeeze(1)
        drift[:, 2] = drift[:, 2] + bias[ordinal] * rotated
    return drift


class OdometryDrift:
    """The same walk, stepped one control tick at a time for live rollouts.

    The offline path can afford a cumulative sum over a whole shard. A running
    environment cannot -- it does not know the future -- so the integration
    happens incrementally here. Both must describe the same process, and
    ``test_noise.py`` pins their statistics against each other.
    """

    def __init__(self, batch_size: int, *, scale: float = 1.0, device=None,
                 generator: torch.Generator | None = None):
        self.scale = float(scale)
        self.generator = generator
        self.device = device
        self.drift = torch.zeros(batch_size, 3, device=device)
        self.yaw_rate_bias = torch.zeros(batch_size, device=device)
        self.reset(torch.arange(batch_size, device=device))

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.drift[env_ids] = 0.0
        self.yaw_rate_bias[env_ids] = torch.randn(
            (env_ids.numel(),), generator=self.generator, device=self.device
        ) * (ODOM_YAW_RATE_BIAS_RAD * self.scale)

    def step(self, yaw_increment: torch.Tensor) -> torch.Tensor:
        """Advance one tick and return the current (N, 3) pose error."""
        n = self.drift.shape[0]
        kw = {"generator": self.generator, "device": self.device}
        self.drift[:, :2] += torch.randn((n, 2), **kw) * (ODOM_XY_WALK_M * self.scale)
        self.drift[:, 2] += (torch.randn((n,), **kw) * (ODOM_YAW_WALK_RAD * self.scale)
                             + self.yaw_rate_bias * yaw_increment)
        return self.drift
