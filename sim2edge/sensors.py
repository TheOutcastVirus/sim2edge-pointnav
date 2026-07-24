"""Simulated lidar and depth camera over a :class:`~sim2edge.rooms.Room`.

Both sensors produce **raw** output in the same units and conventions the real
drivers publish — metres, ``LaserScan``-style range arrays, forward-distance
depth images — and are then fed through the *same* ``sim2edge.preprocess``
functions the C++ node uses. Nothing here produces a policy input directly.
That ordering is deliberate: it means the preprocessing code path is exercised
by every single training step, so a preprocessing bug shows up as a training
failure rather than as a surprise on the robot.

The depth renderer is 2.5D: obstacles are extruded prisms with a vertical
extent, and a ray hits one only where it passes through that extent. This is
what lets the simulator represent the two cases the depth channel exists for —
obstacles below the lidar plane, and overhangs the robot can drive under.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .rooms import Room
from .spec import DEFAULT_SPEC, NavSpec

__all__ = ["simulate_lidar", "render_depth", "march_rays", "LIDAR_BEAMS"]

#: Beam count of the real RPLIDAR A1 on the TB4 Lite, near enough. The policy
#: sees 180 min-pooled bins regardless, but simulating the true beam count means
#: the min-pooling in preprocessing does the same work in sim as on the robot.
LIDAR_BEAMS = 360


def march_rays(
    origin: Tuple[float, float],
    angles: np.ndarray,
    max_range: float,
    resolution: float,
    grid_shape: Tuple[int, int],
    step_scale: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample cell indices along a fan of rays.

    Returns ``(rows, cols, distances)``, each ``(n_rays, n_steps)``. Marching
    uses a fixed step of ``step_scale`` cells rather than an exact DDA: it
    vectorizes cleanly over all rays at once, and the resulting range error is
    bounded by half a cell (2.5 cm at the default resolution), which is smaller
    than the lidar noise the domain randomization injects anyway.

    Indices are clamped to the grid, so callers must treat out-of-bounds as a
    hit — the arena is walled, so in practice a ray always terminates on a wall.
    """
    step = resolution * step_scale
    n_steps = int(math.ceil(max_range / step)) + 1
    dists = np.arange(n_steps, dtype=np.float32) * step  # (S,)

    cos_a = np.cos(angles, dtype=np.float64)[:, None]  # (N, 1)
    sin_a = np.sin(angles, dtype=np.float64)[:, None]

    xs = origin[0] + cos_a * dists  # (N, S)
    ys = origin[1] + sin_a * dists

    h, w = grid_shape
    cols = np.clip((xs / resolution).astype(np.int64), 0, w - 1)
    rows = np.clip((ys / resolution).astype(np.int64), 0, h - 1)
    return rows, cols, np.broadcast_to(dists, rows.shape)


def _first_hit(hit_mask: np.ndarray, dists: np.ndarray,
               max_range: float) -> np.ndarray:
    """First True along axis 1 -> distance; no hit -> ``max_range``."""
    any_hit = hit_mask.any(axis=1)
    idx = hit_mask.argmax(axis=1)  # 0 where there is no hit
    out = np.where(any_hit, np.take_along_axis(dists, idx[:, None], 1)[:, 0],
                   np.float32(max_range))
    return np.minimum(out, np.float32(max_range)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Lidar
# ─────────────────────────────────────────────────────────────────────────────


def simulate_lidar(
    room: Room,
    x: float,
    y: float,
    yaw: float,
    spec: NavSpec = DEFAULT_SPEC,
    n_beams: int = LIDAR_BEAMS,
) -> Tuple[np.ndarray, float, float]:
    """Simulate a 360-degree planar scan.

    Returns ``(ranges, angle_min, angle_increment)`` — the three ``LaserScan``
    fields the preprocessing consumes, so sim and robot enter preprocessing
    through an identical interface.

    Ray casting uses the room's ``lidar_occ`` layer, which contains only
    obstacles that actually cross the scan plane. A box 10 cm tall is simply not
    there, exactly as on the real robot.
    """
    angle_min = -math.pi
    angle_increment = 2.0 * math.pi / n_beams
    angles = yaw + angle_min + angle_increment * np.arange(n_beams)

    max_range = spec.obs.lidar.max_m
    occ = room.lidar_occ.occupied
    rows, cols, dists = march_rays((x, y), angles, max_range, room.resolution,
                                   occ.shape)
    ranges = _first_hit(occ[rows, cols], dists, max_range)

    # Returns closer than the device minimum are reported as invalid, which is
    # what the driver does; preprocessing then maps them to "far". The safety
    # bubble, not the policy, is responsible for that regime.
    ranges = np.where(ranges < spec.obs.lidar.min_m, np.float32(np.inf), ranges)
    return ranges.astype(np.float32), angle_min, angle_increment


# ─────────────────────────────────────────────────────────────────────────────
# Depth camera
# ─────────────────────────────────────────────────────────────────────────────


def _pinhole_angles(n: int, fov: float, flip: bool) -> np.ndarray:
    """Per-pixel ray angles for a pinhole camera.

    Angles are ``atan`` of a linear image coordinate, not linear in the pixel
    index. Using linear angles instead would distort the geometry toward the
    image edges by several degrees at a 69-degree FOV, and the policy would
    learn obstacle bearings that do not match the real camera's.
    """
    half = math.tan(fov * 0.5)
    ndc = (np.arange(n, dtype=np.float64) + 0.5) / n * 2.0 - 1.0  # [-1, 1)
    if flip:
        ndc = -ndc
    return np.arctan(ndc * half)


def render_depth(
    room: Room,
    x: float,
    y: float,
    yaw: float,
    spec: NavSpec = DEFAULT_SPEC,
    height: Optional[int] = None,
    width: Optional[int] = None,
    camera_height: Optional[float] = None,
    camera_pitch: Optional[float] = None,
) -> np.ndarray:
    """Render a metric forward-distance depth image, ``(H, W)`` float32.

    Values follow the OAK-D convention: the *forward* (optical-axis) component
    of the range, not the range itself. Pixels with nothing in range return 0.0,
    the same "invalid" encoding the device uses, so the invalid-handling branch
    of preprocessing is exercised in training too.

    ``camera_height`` / ``camera_pitch`` override the nominal extrinsics so the
    domain randomization can perturb the mount pose per episode.
    """
    cam = spec.robot
    h = height if height is not None else spec.obs.depth.height
    w = width if width is not None else spec.obs.depth.width
    z_cam = cam.camera_height_m if camera_height is None else camera_height
    pitch = cam.camera_pitch_rad if camera_pitch is None else camera_pitch
    max_range = spec.obs.depth.max_m

    # Azimuth per column (+ is left), elevation per row (+ is up).
    az = _pinhole_angles(w, cam.camera_hfov_rad, flip=False)[::-1].copy()
    el = _pinhole_angles(h, cam.camera_vfov_rad, flip=True) - pitch

    # ── March one ray per column through the ground plane ────────────────────
    # Ground distance is what the height test needs; the vertical structure is
    # resolved afterwards per row, which keeps this a 2D march rather than 3D.
    occ_any = room.z_max_map > 0.0
    rows_i, cols_i, dists = march_rays((x, y), yaw + az, max_range,
                                       room.resolution, occ_any.shape)
    z_lo = room.z_min_map[rows_i, cols_i]  # (W, S)
    z_hi = room.z_max_map[rows_i, cols_i]
    solid = occ_any[rows_i, cols_i]
    ground_d = dists  # (W, S), identical across columns

    tan_el = np.tan(el).astype(np.float32)[:, None]  # (H, 1)

    # ── Resolve the first hit per pixel ──────────────────────────────────────
    # Loop over march steps rather than materializing an (H, W, S) mask: the
    # full tensor is ~2e6 elements per frame and this is called once per control
    # step of every episode. Early exit once every pixel has resolved, which in
    # a cluttered room happens well before the far plane.
    hit = np.full((h, w), np.float32(np.inf))
    pending = np.ones((h, w), dtype=bool)
    for s in range(ground_d.shape[1]):
        if not pending.any():
            break
        d = ground_d[0, s]
        if d <= 0.0:
            continue
        # Ray height at this ground distance, per row: (H, 1) broadcast over W.
        z_ray = np.float32(z_cam) + tan_el * np.float32(d)
        blocked = solid[:, s][None, :] & (z_ray >= z_lo[:, s][None, :]) \
            & (z_ray <= z_hi[:, s][None, :])
        newly = blocked & pending
        if newly.any():
            hit[newly] = d
            pending &= ~newly

    # ── Floor ────────────────────────────────────────────────────────────────
    # Downward rays that hit no obstacle terminate on the floor plane. Without
    # this the lower half of every image would read "invalid", which is nothing
    # like a real depth camera and would teach the policy to ignore that region.
    with np.errstate(divide="ignore", invalid="ignore"):
        floor_d = np.where(tan_el < -1e-6, -np.float32(z_cam) / tan_el,
                           np.float32(np.inf))
    floor_d = np.broadcast_to(floor_d, (h, w))
    ground = np.minimum(hit, floor_d)

    # Ground distance along the azimuth -> forward component (OAK-D convention).
    forward = ground * np.cos(az).astype(np.float32)[None, :]
    forward[~np.isfinite(forward)] = 0.0
    forward[forward > max_range] = 0.0  # out of range reads as invalid
    return forward.astype(np.float32)
