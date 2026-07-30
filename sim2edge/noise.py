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
           "ODOM_YAW_STD_RAD", "corrupt_lidar", "corrupt_goal"]

#: Gaussian range error, metres. Roughly constant over the A1's usable range.
RANGE_STD_M = 0.02
#: Per-beam probability of no return: dark, specular, or glancing surfaces.
DROPOUT_PROB = 0.03
#: Whole-scan bearing offset, radians. The mount yaw is never exactly nominal,
#: and unlike per-beam noise this one is *systematic within an episode*.
BEARING_BIAS_RAD = 0.02
#: Goal-pose error from odometry drift. There is no SLAM here, so the goal is
#: anchored in a drifting frame; this is that drift.
ODOM_XY_STD_M = 0.05
ODOM_YAW_STD_RAD = 0.03


def corrupt_lidar(ranges: torch.Tensor, angles: torch.Tensor, scale: float = 1.0,
                  generator: torch.Generator | None = None):
    """``(N, B)`` clean ranges -> noisy ranges and per-scan angles.

    Returns ``(ranges, angles)`` where ``angles`` is now ``(N, B)`` because the
    bearing bias differs per scan. Dropped beams become ``inf``, which
    :func:`~sim2edge.preprocess.bin_lidar` maps to ``INVALID_FILL`` -- the same
    value the live path produces for a beam that got no return. That agreement
    is the point; a placeholder that differs from what the real path emits is
    the single most expensive bug this project has had.

    ``scale`` multiplies every term, so evaluation can sweep clean (0.0) through
    beyond-training (>1.0) without a second config object.
    """
    kw = {"generator": generator, "device": ranges.device, "dtype": ranges.dtype}

    noisy = ranges + torch.randn(ranges.shape, **kw) * (RANGE_STD_M * scale)

    keep = torch.rand(ranges.shape, **kw) >= (DROPOUT_PROB * scale)
    noisy = torch.where(keep, noisy, torch.full_like(noisy, math.inf))

    if angles.dim() == 1:
        angles = angles.unsqueeze(0).expand(ranges.shape[0], -1)
    # One bias per scan, not per beam: a mount error rotates the whole sensor.
    bias = torch.randn((ranges.shape[0], 1), **kw) * (BEARING_BIAS_RAD * scale)
    return noisy, angles + bias


def corrupt_goal(robot_pose: torch.Tensor, goal_xy: torch.Tensor,
                 scale: float = 1.0, generator: torch.Generator | None = None):
    """Perturb the robot's belief about where it is, which moves the goal.

    Drift is applied to the *pose*, not the goal, because that is where it
    physically originates -- and because a pose error rotates the goal bearing
    as well as translating it, which a goal-only perturbation would miss.
    """
    kw = {"generator": generator, "device": robot_pose.device,
          "dtype": robot_pose.dtype}
    drift = torch.empty_like(robot_pose)
    drift[:, :2] = torch.randn((robot_pose.shape[0], 2), **kw) * (ODOM_XY_STD_M * scale)
    drift[:, 2] = torch.randn((robot_pose.shape[0],), **kw) * (ODOM_YAW_STD_RAD * scale)
    return robot_pose + drift, goal_xy
