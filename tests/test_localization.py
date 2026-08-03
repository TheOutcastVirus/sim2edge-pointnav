import math

import numpy as np
import pytest
import torch

from sim2edge.localization import LocalPoseTracker
from sim2edge.rooms import analytic_lidar_ranges, sample_room
from sim2edge.sensors import PolarSweep
from sim2edge.spec import LIDAR_MAX_M


BEARINGS = -math.pi + torch.arange(720) * (2.0 * math.pi / 720)


def _sweep(room, pose, valid=None):
    ranges = torch.from_numpy(analytic_lidar_ranges(
        room, pose, BEARINGS.numpy(), LIDAR_MAX_M))
    if valid is None:
        valid = torch.ones_like(ranges, dtype=torch.bool)
    return PolarSweep(ranges, BEARINGS.clone(), torch.zeros_like(ranges), valid)


def test_polar_sweep_keeps_exact_irregular_bearings_and_ages():
    ranges = torch.tensor([1.0, 2.0, 3.0])
    bearings = torch.tensor([-2.91, 0.17, 2.40])
    ages = torch.tensor([0.18, 0.02, 0.09])
    sweep = PolarSweep(ranges, bearings, ages, torch.tensor([True, False, True]))
    assert torch.equal(sweep.bearings, bearings)
    assert torch.equal(sweep.ages, ages)


def test_polar_sweep_rejects_silent_shape_or_mask_changes():
    with pytest.raises(ValueError, match="identical shapes"):
        PolarSweep(torch.ones(3), torch.ones(2), torch.ones(3),
                   torch.ones(3, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        PolarSweep(torch.ones(3), torch.ones(3), torch.ones(3), torch.ones(3))


def test_tracker_corrects_accumulated_forward_scale_in_asymmetric_room():
    room = sample_room(np.random.default_rng(123))
    x, y = map(float, room.start[:2])
    tracker = LocalPoseTracker(min_information=10.0)
    first = tracker.update(_sweep(room, [x, y, 0.0]), 0.0, 0.0)
    assert not first.observable and first.correction is None
    result = tracker.update(_sweep(room, [x + 0.05, y, 0.0]), 0.0, 1.15)
    assert result.observable and result.correction is not None
    assert result.relative_pose[0] == pytest.approx(0.05, abs=0.008)
    assert result.correction[0] < 0.0
    assert result.motion_scale < 1.0


def test_tracker_emits_no_correction_for_sparse_or_open_space():
    room = sample_room(np.random.default_rng(4))
    tracker = LocalPoseTracker()
    empty = _sweep(room, room.start, torch.zeros(720, dtype=torch.bool))
    assert tracker.update(empty, 0.0, 0.0).correction is None
    result = tracker.update(empty, 0.1, 0.3)
    assert not result.observable and result.correction is None
    assert np.isfinite(result.covariance).all()


def test_tracker_accepts_body_delta_and_bounds_scan_yaw_from_gyro():
    room = sample_room(np.random.default_rng(19))
    x, y = map(float, room.start[:2])
    tracker = LocalPoseTracker(
        min_information=10.0, yaw_correction_gain=1.0,
        max_yaw_departure_rad=math.radians(2.0))
    tracker.update(_sweep(room, [x, y, 0.0]), 0.0, 0.0)
    result = tracker.update(
        _sweep(room, [x + 0.05, y, 0.0]), 0.0, 0.0,
        prior_delta_body=np.array([0.05, 0.0, 0.10]))
    assert result.relative_pose[0] == pytest.approx(0.05, abs=0.01)
    assert abs(result.relative_pose[2]) <= math.radians(2.0) + 1.0e-9
