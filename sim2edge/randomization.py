"""Per-episode domain randomization.

Sampled once per episode from :class:`~sim2edge.spec.RandomizationSpec` and held
fixed for its duration, which is what makes it learnable: a policy can adapt to a
robot that is consistently sluggish this episode, but not to one whose latency
changes every step.

**Noise is applied to raw sensor output, before preprocessing.** Injecting it
after preprocessing would be a different — and wrong — distribution: the
min-pooling in preprocessing interacts strongly with per-beam noise and dropout,
biasing pooled values downward, and that interaction is exactly what the real
robot will do too.

The ranges themselves must be measured, not guessed. ``tools/fit_noise_from_bag.py``
derives them from recorded rosbags; see ``docs/domain_randomization.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .spec import DEFAULT_SPEC, NavSpec, RandomizationSpec

__all__ = ["EpisodeRandomization", "sample_randomization"]


@dataclass
class EpisodeRandomization:
    """One episode's worth of sampled perturbations."""

    # Lidar
    lidar_sigma_m: float = 0.0
    lidar_dropout: float = 0.0
    lidar_angle_offset: float = 0.0

    # Depth
    depth_noise_coeff: float = 0.0
    depth_hole_fraction: float = 0.0
    depth_speckle_fraction: float = 0.0
    depth_scale: float = 1.0
    depth_pitch_offset: float = 0.0
    depth_height_offset: float = 0.0

    # Dynamics
    velocity_gain: float = 1.0
    motor_tau_s: float = 0.05
    wheel_slip: float = 0.0
    mass_scale: float = 1.0

    # Latency
    action_delay_steps: int = 0

    # Odometry
    odom_drift_rate: float = 0.0
    odom_yaw_drift_rate: float = 0.0
    odom_noise_m: float = 0.0
    #: Fixed random direction for the slow translational drift.
    odom_drift_dir: float = 0.0

    @classmethod
    def none(cls) -> "EpisodeRandomization":
        """A perfect robot with perfect sensors and no latency."""
        return cls(motor_tau_s=1e-3)


def _u(rng: np.random.Generator, rng_range: Tuple[float, float]) -> float:
    lo, hi = rng_range
    return float(rng.uniform(min(lo, hi), max(lo, hi)))


def sample_randomization(
    rng: np.random.Generator,
    spec: NavSpec = DEFAULT_SPEC,
    override: RandomizationSpec | None = None,
) -> EpisodeRandomization:
    """Draw one episode's randomization."""
    r = override if override is not None else spec.randomization
    if not r.enabled:
        return EpisodeRandomization.none()

    lo, hi = r.action_delay_steps
    delay = int(rng.integers(min(lo, hi), max(lo, hi) + 1))

    return EpisodeRandomization(
        lidar_sigma_m=_u(rng, r.lidar_range_noise_m),
        lidar_dropout=_u(rng, r.lidar_dropout),
        lidar_angle_offset=_u(rng, r.lidar_angular_offset_rad),
        depth_noise_coeff=_u(rng, r.depth_noise_coeff),
        depth_hole_fraction=_u(rng, r.depth_hole_fraction),
        depth_speckle_fraction=_u(rng, r.depth_speckle_fraction),
        depth_scale=_u(rng, r.depth_scale_jitter),
        depth_pitch_offset=_u(rng, r.depth_pitch_noise_rad),
        depth_height_offset=_u(rng, r.depth_height_noise_m),
        velocity_gain=_u(rng, r.velocity_tracking_gain),
        motor_tau_s=max(1e-3, _u(rng, r.motor_tau_s)),
        wheel_slip=_u(rng, r.wheel_slip),
        mass_scale=_u(rng, r.mass_scale),
        action_delay_steps=max(0, delay),
        odom_drift_rate=_u(rng, r.odom_drift_rate),
        odom_yaw_drift_rate=_u(rng, r.odom_yaw_drift_rate),
        odom_noise_m=_u(rng, r.odom_noise_m),
        odom_drift_dir=float(rng.uniform(-math.pi, math.pi)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sensor corruption
# ─────────────────────────────────────────────────────────────────────────────


def corrupt_lidar(
    ranges: np.ndarray,
    dr: EpisodeRandomization,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply range noise and beam dropout to a raw scan.

    Dropped beams become ``inf``, which is what the driver reports and what
    preprocessing maps to "far". Dropping to a *near* value instead would be a
    much more dangerous default, teaching the policy to brake at random.
    """
    out = ranges.astype(np.float32, copy=True)
    finite = np.isfinite(out)
    if dr.lidar_sigma_m > 0.0:
        noise = rng.normal(0.0, dr.lidar_sigma_m, size=out.shape).astype(np.float32)
        out[finite] += noise[finite]
        np.maximum(out, 0.0, out=out)
    if dr.lidar_dropout > 0.0:
        drop = rng.random(out.shape) < dr.lidar_dropout
        out[drop] = np.inf
    return out


def corrupt_depth(
    depth_m: np.ndarray,
    dr: EpisodeRandomization,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply stereo-like corruption to a rendered metric depth image.

    Models the three failure modes that actually dominate an OAK-D Lite indoors:

    * **range-dependent noise** growing as distance squared (stereo disparity
      quantization),
    * **holes at depth discontinuities**, because block matching fails where
      the two views disagree — this is why holes are drawn along edges rather
      than uniformly,
    * **speckle**, isolated invalid pixels.

    Invalid pixels are written as 0.0, the device's own invalid encoding.
    """
    out = depth_m.astype(np.float32, copy=True)
    valid = out > 0.0

    if dr.depth_scale != 1.0:
        out[valid] *= dr.depth_scale

    if dr.depth_noise_coeff > 0.0:
        sigma = dr.depth_noise_coeff * out * out
        out[valid] += (rng.normal(0.0, 1.0, size=out.shape).astype(np.float32)
                       * sigma)[valid]
        np.maximum(out, 0.0, out=out)

    if dr.depth_hole_fraction > 0.0:
        # Edge-correlated: probability of a hole rises with the local depth
        # gradient, so holes cluster on object boundaries like real ones do.
        gy = np.abs(np.diff(out, axis=0, prepend=out[:1]))
        gx = np.abs(np.diff(out, axis=1, prepend=out[:, :1]))
        edge = gx + gy
        denom = float(edge.mean()) + 1e-6
        p = np.clip(dr.depth_hole_fraction * (0.3 + 1.4 * edge / denom), 0.0, 1.0)
        out[rng.random(out.shape) < p] = 0.0

    if dr.depth_speckle_fraction > 0.0:
        out[rng.random(out.shape) < dr.depth_speckle_fraction] = 0.0

    return out
