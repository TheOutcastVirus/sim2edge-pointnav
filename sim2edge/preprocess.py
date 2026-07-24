"""Reference implementation of observation preprocessing.

This is the Python half of a two-implementation contract. The other half is
``ros_ws/src/nav_policy_node/src/preprocess.cpp``, and
``tools/parity_check.py`` asserts they agree on the same raw messages before any
robot trial.

Every algorithm here is deliberately written so it can be transcribed to C++
literally, with no library-behaviour dependence:

* **Downsampling is explicit min-pooling over integer index blocks**, not
  ``cv::resize``. Bilinear or area interpolation across a depth discontinuity
  invents a surface that is in neither the near object nor the far background,
  and INTER_AREA in OpenCV would not match anything reasonable written in numpy.
  Min-pooling is also the only reduction that cannot delete a thin obstacle.
* **Invalid pixels/beams are normalized before pooling**, so "invalid" and "far"
  reduce identically and there is no separate mask to keep in sync.
* No dependence on scan direction, message field order, or OpenCV version.

Units in: metres and radians, robot frame, x forward, y left, yaw CCW.
Units out: float32 arrays normalized to [0, 1] (sensors) or [-1, 1] (actions).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .spec import DEFAULT_SPEC, NavSpec

__all__ = [
    "Preprocessor",
    "block_min_pool",
    "normalize_depth_raw",
    "preprocess_depth",
    "preprocess_lidar",
    "preprocess_goal",
    "preprocess_last_action",
]


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────


def _block_bounds(n_in: int, n_out: int) -> np.ndarray:
    """Index boundaries mapping ``n_in`` samples onto ``n_out`` bins.

    Returns an ``(n_out, 2)`` int array of ``[start, end)`` pairs. ``end`` is
    always at least ``start + 1`` so upsampling degenerates to nearest-neighbour
    rather than producing empty bins. The C++ side computes the same integers
    with the same integer arithmetic.
    """
    edges = (np.arange(n_out + 1, dtype=np.int64) * n_in) // n_out
    starts = edges[:-1]
    ends = np.maximum(edges[1:], starts + 1)
    np.minimum(ends, n_in, out=ends)
    starts = np.minimum(starts, n_in - 1)
    return np.stack([starts, ends], axis=1)


def block_min_pool(src: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Min-pool a 2D array onto an ``(out_h, out_w)`` grid.

    Conservative by construction: the output pixel reports the *nearest* thing
    seen in its block, so downsampling can never make an obstacle disappear.
    """
    src = np.asarray(src, dtype=np.float32)
    in_h, in_w = src.shape
    rows = _block_bounds(in_h, out_h)
    cols = _block_bounds(in_w, out_w)

    # Reduce rows then columns; both reductions are ragged, so they are done as
    # a loop over output bins (out_h + out_w iterations, not out_h * out_w).
    tmp = np.empty((out_h, in_w), dtype=np.float32)
    for r in range(out_h):
        tmp[r] = src[rows[r, 0]:rows[r, 1]].min(axis=0)
    out = np.empty((out_h, out_w), dtype=np.float32)
    for c in range(out_w):
        out[:, c] = tmp[:, cols[c, 0]:cols[c, 1]].min(axis=1)
    return out


def normalize_depth_raw(
    depth_m: np.ndarray,
    spec: NavSpec = DEFAULT_SPEC,
) -> np.ndarray:
    """Metric depth -> [0, 1], invalid -> ``invalid_fill``, at input resolution.

    Invalid means non-positive, NaN, or infinite. Note that *far* and *invalid*
    both map to 1.0: a stereo hole is treated as free space, and the lidar plus
    the classical safety bubble cover the pessimistic case. A policy taught to
    brake for stereo holes would never move in a real room.
    """
    d = np.asarray(depth_m, dtype=np.float32)
    ds = spec.obs.depth
    valid = np.isfinite(d) & (d > 0.0)
    clipped = np.clip(d, ds.min_m, ds.max_m)
    norm = (clipped - ds.min_m) / (ds.max_m - ds.min_m)
    return np.where(valid, norm, np.float32(ds.invalid_fill)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def preprocess_depth(
    depth_m: np.ndarray,
    spec: NavSpec = DEFAULT_SPEC,
) -> np.ndarray:
    """Raw metric depth image -> normalized ``(1, H, W)`` policy input."""
    norm = normalize_depth_raw(depth_m, spec)
    ds = spec.obs.depth
    pooled = block_min_pool(norm, ds.height, ds.width)
    return pooled.reshape(1, ds.height, ds.width)


def preprocess_lidar(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    spec: NavSpec = DEFAULT_SPEC,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
) -> np.ndarray:
    """Raw ``LaserScan.ranges`` -> normalized ``(bins,)`` policy input.

    Beams are binned by their *actual* angle rather than by index, so the result
    does not depend on scan direction, start angle, or beam count. This matters:
    the RPLIDAR's scan start angle is not stable across driver versions, and
    binning by index would silently rotate the policy's world view.

    ``range_min`` / ``range_max`` are the message's own validity limits when
    given; returns outside them are treated as invalid rather than clamped.
    """
    ls = spec.obs.lidar
    r = np.asarray(ranges, dtype=np.float32)
    n = r.shape[0]
    if n == 0:
        return np.full(ls.bins, ls.invalid_fill, dtype=np.float32)

    lo = ls.min_m if range_min is None else max(ls.min_m, float(range_min))
    hi = ls.max_m if range_max is None else min(ls.max_m, float(range_max))

    angles = angle_min + angle_increment * np.arange(n, dtype=np.float32)
    # Wrap to [-pi, pi) so the bin index is well defined regardless of whether
    # the driver reports [0, 2pi) or [-pi, pi).
    wrapped = (angles + np.pi) % (2.0 * np.pi) - np.pi

    span = ls.angle_max - ls.angle_min
    idx = np.floor((wrapped - ls.angle_min) / span * ls.bins).astype(np.int64)
    np.clip(idx, 0, ls.bins - 1, out=idx)

    valid = np.isfinite(r) & (r >= lo) & (r <= hi)
    clipped = np.clip(r, ls.min_m, ls.max_m)
    norm = (clipped - ls.min_m) / (ls.max_m - ls.min_m)
    norm = np.where(valid, norm, np.float32(ls.invalid_fill)).astype(np.float32)

    out = np.full(ls.bins, ls.invalid_fill, dtype=np.float32)
    # Min-pool into bins. np.minimum.at handles the many-to-one scatter; the C++
    # side does the same thing with a plain loop.
    np.minimum.at(out, idx, norm)
    return out


def preprocess_goal(
    goal_x: float,
    goal_y: float,
    spec: NavSpec = DEFAULT_SPEC,
) -> np.ndarray:
    """Goal position in the *robot* frame -> (normalized distance, sin, cos).

    The bearing is encoded as sin/cos rather than an angle so the input stays
    continuous across the +/-pi wrap. A discontinuity there would land exactly
    where the robot needs to turn around, and INT8 quantization of a wrapping
    input is a silent-failure generator.
    """
    dist = math.hypot(goal_x, goal_y)
    max_d = spec.obs.goal.max_distance_m
    d_norm = min(dist, max_d) / max_d
    if dist < 1e-6:
        sin_b, cos_b = 0.0, 1.0
    else:
        sin_b, cos_b = goal_y / dist, goal_x / dist
    return np.array([d_norm, sin_b, cos_b], dtype=np.float32)


def preprocess_last_action(
    v: float,
    omega: float,
    spec: NavSpec = DEFAULT_SPEC,
) -> np.ndarray:
    """Previous physical command -> normalized ``(2,)`` in [-1, 1]."""
    v_n, w_n = spec.action.normalize(float(v), float(omega))
    return np.array([v_n, w_n], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Bundled interface
# ─────────────────────────────────────────────────────────────────────────────


class Preprocessor:
    """Stateless bundle of the four channel preprocessors.

    Used identically by the sim env, the calibration-set builder, and the parity
    test, so calibration data is guaranteed to pass through the same code path
    the deployed policy's inputs do.
    """

    def __init__(self, spec: NavSpec = DEFAULT_SPEC):
        self.spec = spec

    def __call__(self, **kwargs) -> Dict[str, np.ndarray]:
        return self.build(**kwargs)

    def build(
        self,
        depth_m: np.ndarray,
        scan_ranges: np.ndarray,
        scan_angle_min: float,
        scan_angle_increment: float,
        goal_xy_robot: Tuple[float, float],
        last_action: Tuple[float, float],
        scan_range_min: Optional[float] = None,
        scan_range_max: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Build the full actor observation from raw sensor data."""
        return {
            "depth": preprocess_depth(depth_m, self.spec),
            "lidar": preprocess_lidar(
                scan_ranges, scan_angle_min, scan_angle_increment, self.spec,
                scan_range_min, scan_range_max,
            ),
            "goal": preprocess_goal(*goal_xy_robot, spec=self.spec),
            "last_action": preprocess_last_action(*last_action, spec=self.spec),
        }

    def as_tuple(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, ...]:
        """Order a built observation the way the exported graph expects it."""
        return tuple(obs[name] for name in self.spec.obs.actor_inputs)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers shared by sim and deployment
# ─────────────────────────────────────────────────────────────────────────────


def world_goal_to_robot_frame(
    goal_world: Tuple[float, float],
    robot_xy: Tuple[float, float],
    robot_yaw: float,
) -> Tuple[float, float]:
    """Transform a world/odom-frame goal into the robot frame.

    On the robot this runs every tick against the latest odometry; in sim it
    runs against ground truth (optionally corrupted by the odometry-drift
    randomization). Same function both places, so a sign error cannot differ
    between training and deployment.
    """
    dx = goal_world[0] - robot_xy[0]
    dy = goal_world[1] - robot_xy[1]
    c, s = math.cos(-robot_yaw), math.sin(-robot_yaw)
    return (c * dx - s * dy, s * dx + c * dy)


def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi
