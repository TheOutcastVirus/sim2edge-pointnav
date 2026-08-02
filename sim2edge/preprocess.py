"""Raw sensor readings -> the policy's input vector. Batched torch, on device.

This runs in two places that must agree exactly: here, on a GPU batch during
training, and in C++ on the board. It is written to be transcribable literally
-- explicit index arithmetic, no library-behaviour dependence, no op whose
semantics differ between torch and the C++ standard library.

**It is deliberately not in C++ on this side.** Isaac hands us lidar as a
``(num_envs, beams)`` tensor already resident on the GPU; routing that through
C++ would mean a device->host->device round trip per step, which costs far more
than the microsecond of arithmetic it would save. The C++ implementation exists
because the *robot* needs it, and a parity test pins the two together.

Two choices worth stating, because both look arbitrary and neither is:

**Beams are binned by wrapped angle, never by array index.** The RPLIDAR's
start angle is not stable across driver versions and its beam count varies with
spin rate. Index binning would silently rotate the policy's entire view of the
world when the driver changed.

**Ranges are normalized before pooling, not after.** "Invalid" and "far" both
map to 1.0, so pooling needs no companion mask -- there is no second data
structure to keep in sync between the two implementations.
"""

from __future__ import annotations

import math

import torch

from .spec import (BIN_WIDTH_RAD, GOAL_MAX_M, HISTORY_FRAMES, INVALID_FILL,
                   LIDAR_BINS, LIDAR_MAX_M, LIDAR_MIN_M)

__all__ = ["ObservationHistory", "wrap_angle", "bin_lidar", "rasterize_lidar",
           "goal_features",
           "relative_goal_features", "build_observation"]


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap to [-pi, pi).

    Written with floor rather than fmod on purpose: C's ``fmod`` keeps the sign
    of the dividend, Python's and torch's ``%`` keep the sign of the divisor,
    and the bin index depends on which. The C++ transcription must reproduce
    *this* behaviour; getting it wrong is an off-by-one-hemisphere error that
    looks like a badly trained policy.
    """
    return a - 2.0 * math.pi * torch.floor((a + math.pi) / (2.0 * math.pi))


def bin_lidar(ranges: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """``(N, B)`` raw ranges + angles -> ``(N, LIDAR_BINS)`` in [0, 1].

    ``angles`` may be ``(B,)`` if every scan shares a beam layout, or ``(N, B)``
    when a per-scan bearing bias has been applied.

    Reduction is **minimum**, which is the only choice that cannot delete a thin
    obstacle: a table leg covering one beam out of ten in a bin survives to the
    policy. Averaging would blend it into the wall behind it and mean would
    make a chair disappear.

    Bins with no beam in them come out as :data:`INVALID_FILL` -- "nothing out
    to max range" -- which is also what an out-of-range or non-finite return
    becomes. Absent data and distant data are the same statement to the policy.
    """
    if angles.dim() == 1:
        angles = angles.unsqueeze(0).expand_as(ranges)

    # Normalize first: near -> 0, far -> 1. Anything unusable is "far".
    denom = LIDAR_MAX_M - LIDAR_MIN_M
    norm = ((ranges - LIDAR_MIN_M) / denom).clamp(0.0, 1.0)
    usable = torch.isfinite(ranges) & (ranges >= LIDAR_MIN_M) & (ranges <= LIDAR_MAX_M)
    norm = torch.where(usable, norm, torch.full_like(norm, INVALID_FILL))

    # Bin by wrapped bearing. `+ pi` moves [-pi, pi) to [0, 2pi) so bin 0 starts
    # dead astern and bin index increases counter-clockwise.
    idx = torch.floor((wrap_angle(angles) + math.pi) / BIN_WIDTH_RAD).long()
    idx = idx.clamp(0, LIDAR_BINS - 1)          # guards the +pi boundary only

    out = torch.full((ranges.shape[0], LIDAR_BINS), float(INVALID_FILL),
                     dtype=norm.dtype, device=norm.device)
    # scatter_reduce with "amin" is a single fused kernel; a Python loop over
    # bins would be 180 launches per step.
    out.scatter_reduce_(1, idx, norm, reduce="amin", include_self=True)
    return out


def rasterize_lidar(ranges: torch.Tensor, angles: torch.Tensor,
                    bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Angle-bin raw ranges without normalization, returning a validity mask.

    The navigation actor intentionally collapses the A1 to 180 normalized bins.
    Scan motion cannot afford that loss of angular detail, so the estimator uses
    this companion operation to retain 720 uniformly-spaced metric ranges. Empty
    bins are filled with ``LIDAR_MAX_M`` and are distinguished by the returned
    boolean mask.
    """
    if bins < 1:
        raise ValueError("bins must be positive")
    if angles.dim() == 1:
        angles = angles.unsqueeze(0).expand_as(ranges)
    usable = (torch.isfinite(ranges) & (ranges >= LIDAR_MIN_M)
              & (ranges <= LIDAR_MAX_M))
    values = torch.where(usable, ranges, torch.full_like(ranges, math.inf))
    width = 2.0 * math.pi / bins
    idx = torch.floor((wrap_angle(angles) + math.pi) / width).long()
    idx = idx.clamp(0, bins - 1)
    out = torch.full((ranges.shape[0], bins), math.inf,
                     dtype=ranges.dtype, device=ranges.device)
    out.scatter_reduce_(1, idx, values, reduce="amin", include_self=True)
    valid = torch.isfinite(out)
    return torch.where(valid, out, torch.full_like(out, LIDAR_MAX_M)), valid


def goal_features(robot_pose: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
    """``(N, 3)`` = (distance / GOAL_MAX_M, sin bearing, cos bearing).

    ``robot_pose`` is ``(N, 3)`` of x, y, yaw in the world frame. The output is
    in the *robot* frame, which is what makes the policy translation- and
    rotation-invariant: the same relative goal produces the same input anywhere
    in any room.

    Distance is clamped at 1.0 rather than left to grow. A goal beyond
    GOAL_MAX_M is "far away in that direction"; the exact number carries no
    information the policy can act on, and an unbounded input quantizes badly.
    """
    d = goal_xy - robot_pose[:, :2]
    yaw = robot_pose[:, 2]
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    # Rotate the world-frame offset into the robot frame here rather than
    # subtracting yaw from a bearing later: it lets both callers share one
    # encoder, and the belief policy never has a world frame to subtract from.
    relative = torch.stack([cos * d[:, 0] + sin * d[:, 1],
                            -sin * d[:, 0] + cos * d[:, 1]], dim=1)
    return relative_goal_features(relative)


def relative_goal_features(goal_body: torch.Tensor) -> torch.Tensor:
    """``(N, 2)`` goal offset already in the robot frame -> the same ``(N, 3)``.

    Split out from :func:`goal_features` because the goal-belief policy carries
    exactly this quantity as its recurrent state and never possesses the world
    pose the other entry point subtracts. Both paths must produce bit-identical
    features for a given relative goal, or a checkpoint trained through one and
    evaluated through the other would be measuring two different contracts --
    so there is one implementation and two ways in.

    The epsilon guards ``atan2(0, 0)`` at the instant the robot stands on the
    goal. Direction is meaningless there, and the distance feature is already
    zero, so any finite value is as good as another; NaN is not.
    """
    dist = torch.linalg.vector_norm(goal_body, dim=1)
    safe = dist.clamp_min(1.0e-6)
    return torch.stack([
        (dist / GOAL_MAX_M).clamp(max=1.0),
        goal_body[:, 1] / safe,
        goal_body[:, 0] / safe,
    ], dim=1)


def build_observation(lidar: torch.Tensor, goal: torch.Tensor,
                      last_action: torch.Tensor) -> torch.Tensor:
    """Concatenate history in the agreed order: lidar history|goal|action history.

    History tensors are ``(N, H, B)`` / ``(N, H, 2)`` and ordered oldest to
    newest. A two-dimensional input is accepted as a one-frame legacy layout.
    ``last_action`` contains previous *commands*, normalized by the caller.
    """
    return torch.cat([lidar.flatten(1), goal, last_action.flatten(1)], dim=1)


class ObservationHistory:
    """Small per-environment FIFO shared by live collection and evaluation."""

    def __init__(self, batch_size: int, *, frames: int = HISTORY_FRAMES,
                 lidar_bins: int = LIDAR_BINS, device=None):
        if frames < 1:
            raise ValueError("history frames must be positive")
        self.frames = frames
        self.lidar = torch.empty(batch_size, frames, lidar_bins, device=device)
        self.action = torch.empty(batch_size, frames, 2, device=device)
        self.initialized = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def push(self, lidar: torch.Tensor, action: torch.Tensor,
             reset: torch.Tensor | None = None) -> None:
        """Append one frame; reset environments repeat it across the history."""
        if reset is None:
            reset = ~self.initialized
        else:
            reset = reset | ~self.initialized
        if self.frames > 1:
            self.lidar[:, :-1] = self.lidar[:, 1:].clone()
            self.action[:, :-1] = self.action[:, 1:].clone()
        self.lidar[:, -1] = lidar
        self.action[:, -1] = action
        # Empty advanced indexing is a no-op and avoids a GPU->CPU sync on the
        # every-control-tick live path just to ask whether any bit is set.
        self.lidar[reset] = lidar[reset, None, :]
        self.action[reset] = action[reset, None, :]
        self.initialized.fill_(True)

    def observation(self, goal: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized.all()):
            raise RuntimeError("history must receive a frame before observation")
        return build_observation(self.lidar, goal, self.action)
