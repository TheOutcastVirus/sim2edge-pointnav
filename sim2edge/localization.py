"""CPU local pose tracking from exact-bearing lidar sweeps.

The tracker deliberately sits outside the learned/quantized actor.  Gyro and
commanded motion form a prior; robust point-to-plane ICP supplies a correction
only when the local geometry makes all three planar degrees of freedom
observable.  Rejected scans still advance the prior but never masquerade as a
lidar estimate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from .sensors import PolarSweep
from .spec import DT, MOTION_PRIOR_SOURCE

__all__ = ["LocalPoseTracker", "LocalizerEstimate"]


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class LocalizerEstimate:
    relative_pose: np.ndarray
    covariance: np.ndarray
    motion_scale: float
    observable: bool
    correction: np.ndarray | None
    motion_prior_source: str = MOTION_PRIOR_SOURCE


class LocalPoseTracker:
    """Short-lived scan-to-submap SE(2) tracker for a single robot."""

    def __init__(self, *, max_submap_scans: int = 16,
                 max_correspondence_m: float = 0.20,
                 min_points: int = 60, min_information: float = 25.0,
                 max_condition: float = 2.0e4,
                 correction_gain: float = 0.10,
                 yaw_correction_gain: float = 0.05,
                 max_correction_m: float = 0.12,
                 max_correction_yaw_rad: float = math.radians(4.0),
                 motion_scale_update_rate: float = 0.02,
                 max_yaw_departure_rad: float = math.inf,
                 point_stride: int = 2,
                 motion_prior_source: str = MOTION_PRIOR_SOURCE):
        self.max_submap_scans = max_submap_scans
        self.max_correspondence_m = max_correspondence_m
        self.min_points = min_points
        self.min_information = min_information
        self.max_condition = max_condition
        self.correction_gain = correction_gain
        self.yaw_correction_gain = yaw_correction_gain
        self.max_correction_m = max_correction_m
        self.max_correction_yaw_rad = max_correction_yaw_rad
        self.motion_scale_update_rate = motion_scale_update_rate
        self.max_yaw_departure_rad = max_yaw_departure_rad
        if point_stride < 1:
            raise ValueError("point_stride must be positive")
        self.point_stride = point_stride
        self.motion_prior_source = motion_prior_source
        self.reset()

    def reset(self) -> None:
        self.pose = np.zeros(3, dtype=np.float64)
        self.covariance = np.eye(3, dtype=np.float64) * 1.0e-6
        self.motion_scale = 1.0
        self._gyro_yaw = 0.0
        self._scans: deque[np.ndarray] = deque(maxlen=self.max_submap_scans)

    def _points(self, sweep: PolarSweep) -> np.ndarray:
        if sweep.ranges.ndim != 1:
            raise ValueError("LocalPoseTracker consumes one unbatched sweep")
        ranges = sweep.ranges.detach().cpu().float().numpy()
        bearings = sweep.bearings.detach().cpu().float().numpy()
        valid = sweep.valid.detach().cpu().numpy().astype(bool)
        valid &= np.isfinite(ranges) & np.isfinite(bearings)
        # Very distant returns contribute little local constraint and make
        # nearest-neighbour association unnecessarily ambiguous.
        valid &= (ranges >= 0.15) & (ranges <= 8.0)
        r = ranges[valid][::self.point_stride]
        a = bearings[valid][::self.point_stride]
        return np.column_stack((r * np.cos(a), r * np.sin(a)))

    @staticmethod
    def _transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
        rotation = np.array([[c, -s], [s, c]])
        return points @ rotation.T + pose[:2]

    def _add_scan(self, points: np.ndarray) -> None:
        if points.size:
            self._scans.append(self._transform(points, self.pose))

    def update(self, sweep: PolarSweep, gyro_delta: float,
               commanded_speed: float, dt: float = DT,
               prior_delta_body: np.ndarray | None = None) -> LocalizerEstimate:
        points = self._points(sweep)
        self._gyro_yaw = _wrap(self._gyro_yaw + float(gyro_delta))
        old_pose = self.pose.copy()
        prior = self.pose.copy()
        if prior_delta_body is None:
            yaw = _wrap(float(self.pose[2]) + float(gyro_delta))
            heading = float(self.pose[2]) + 0.5 * float(gyro_delta)
            travel = float(commanded_speed) * float(dt) * self.motion_scale
            prior[:2] += travel * np.array([math.cos(heading), math.sin(heading)])
            prior[2] = yaw
        else:
            delta = np.asarray(prior_delta_body, dtype=np.float64)
            if delta.shape != (3,):
                raise ValueError("prior_delta_body must have shape (3,)")
            heading = float(self.pose[2])
            c, s = math.cos(heading), math.sin(heading)
            prior[:2] += np.array([c * delta[0] - s * delta[1],
                                   s * delta[0] + c * delta[1]])
            prior[2] = _wrap(float(self.pose[2]) + float(delta[2]))
        self.pose = prior

        if len(points) < self.min_points or not self._scans:
            self.covariance = self.covariance + np.diag(
                [2.5e-5, 2.5e-5, 1.0e-5])
            self._add_scan(points)
            return LocalizerEstimate(self.pose.copy(), self.covariance.copy(),
                                     self.motion_scale, False, None,
                                     self.motion_prior_source)

        submap = np.concatenate(tuple(self._scans), axis=0)
        tree = cKDTree(submap)
        estimate = prior.copy()
        information = None
        residual = None
        used = 0
        for _ in range(6):
            placed = self._transform(points, estimate)
            distance, index = tree.query(placed, k=1)
            keep = distance < self.max_correspondence_m
            if int(keep.sum()) < self.min_points:
                break
            source, target = placed[keep], submap[index[keep]]
            # Local PCA normals turn the ICP system into point-to-plane. This
            # correctly leaves travel along a featureless corridor unobserved.
            _, neighbours = tree.query(target, k=min(7, len(submap)))
            cloud = submap[neighbours]
            centred = cloud - cloud.mean(axis=1, keepdims=True)
            scatter = np.einsum("nki,nkj->nij", centred, centred)
            _, vectors = np.linalg.eigh(scatter)
            normal = vectors[:, :, 0]
            residual = np.einsum("ni,ni->n", normal, source - target)
            # Cauchy-like robust weights suppress moving/occluding edges.
            weight = 1.0 / (1.0 + (residual / 0.04) ** 2)
            lever = source - estimate[:2]
            d_yaw = np.column_stack((-lever[:, 1], lever[:, 0]))
            jacobian = np.column_stack((normal,
                np.einsum("ni,ni->n", normal, d_yaw)))
            information = jacobian.T @ (weight[:, None] * jacobian)
            gradient = jacobian.T @ (weight * residual)
            try:
                step = -np.linalg.solve(information + np.eye(3) * 1.0e-6,
                                        gradient)
            except np.linalg.LinAlgError:
                break
            # The scan correction is local; a large jump is an association
            # failure and must not drag the submap with it.
            step[:2] = np.clip(step[:2], -0.04, 0.04)
            step[2] = np.clip(step[2], -math.radians(1.0), math.radians(1.0))
            estimate += step
            estimate[2] = _wrap(float(estimate[2]))
            used = int(keep.sum())
            if np.linalg.norm(step) < 1.0e-5:
                break

        observable = False
        covariance = self.covariance + np.diag([2.5e-5, 2.5e-5, 1.0e-5])
        if information is not None and residual is not None and used >= self.min_points:
            eigen = np.linalg.eigvalsh(information)
            condition = float(eigen[-1] / max(eigen[0], 1.0e-12))
            observable = bool(eigen[0] >= self.min_information
                              and condition <= self.max_condition
                              and np.linalg.norm(estimate[:2] - prior[:2])
                                  <= self.max_correction_m
                              and abs(_wrap(float(estimate[2] - prior[2])))
                                  <= self.max_correction_yaw_rad)
            if observable:
                variance = max(float(np.median(residual * residual)), 1.0e-6)
                covariance = np.linalg.inv(information + np.eye(3) * 1.0e-6) * variance

        correction = None
        if observable:
            # A full per-sweep correction turns independent centimetre-scale
            # range noise into a pose random walk.  The submap error we seek is
            # slowly accumulated drivetrain scale, so apply a conservative
            # complementary-filter gain; gyro/command remain the fast prior.
            delta = estimate - prior
            delta[2] = _wrap(float(delta[2]))
            gain = self.correction_gain
            self.pose = prior + delta * np.array(
                [gain, gain, self.yaw_correction_gain])
            self.pose[2] = _wrap(float(self.pose[2]))
            self.covariance = covariance
            correction = self.pose - prior
            correction[2] = _wrap(float(correction[2]))
            commanded = float(commanded_speed) * float(dt)
            if abs(commanded) > 0.005:
                # Estimate the slowly varying drivetrain scale from the full
                # scan solution, then average it heavily.  Using the blended
                # output here would feed the complementary-filter gain back
                # into the physical scale and make convergence unnecessarily
                # slow.
                # Only forward displacement identifies drivetrain scale.
                # Using the vector norm makes lateral ICP noise strictly
                # positive and therefore biases the scale high—the worst
                # possible direction when the wheels are slipping.
                direction = np.array([math.cos(heading), math.sin(heading)])
                achieved = float(np.dot(estimate[:2] - old_pose[:2], direction))
                ratio = np.clip(achieved / commanded, 0.75, 1.10)
                rate = self.motion_scale_update_rate
                self.motion_scale = ((1.0 - rate) * self.motion_scale
                                     + rate * float(ratio))
        else:
            self.pose = prior
            self.covariance = covariance

        departure = _wrap(float(self.pose[2]) - self._gyro_yaw)
        if abs(departure) > self.max_yaw_departure_rad:
            self.pose[2] = _wrap(
                self._gyro_yaw + float(np.clip(
                    departure, -self.max_yaw_departure_rad,
                    self.max_yaw_departure_rad)))

        self._add_scan(points)
        return LocalizerEstimate(self.pose.copy(), self.covariance.copy(),
                                 self.motion_scale, observable, correction,
                                 self.motion_prior_source)
