"""Preprocessing. This is the half of the train/deploy contract that lives in
Python; the C++ node is the other half, and a parity test will pin them together
once it exists. Until then these tests are the specification."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sim2edge.preprocess import (ObservationHistory, bin_lidar,
                                 build_observation, goal_features, wrap_angle)
from sim2edge.spec import (GOAL_MAX_M, INVALID_FILL, LIDAR_BINS, LIDAR_MAX_M,
                           LIDAR_MIN_M, HISTORY_FRAMES, OBS_DIM)


def _beams(n=720):
    """A full sweep, the way a driver hands one over: [0, 2pi)."""
    return torch.linspace(0, 2 * math.pi, n + 1)[:-1]


# ── angle handling ───────────────────────────────────────────────────────────

def test_wrap_angle_lands_in_the_half_open_interval():
    a = torch.tensor([0.0, math.pi - 1e-4, -math.pi, 3 * math.pi, -3.5 * math.pi])
    w = wrap_angle(a)
    # float32 pi is a hair larger in magnitude than float64 pi, so the lower
    # bound needs single-precision slack; the upper bound is genuinely strict.
    eps = 1e-6
    assert torch.all(w >= -math.pi - eps) and torch.all(w < math.pi)


def test_wrap_angle_sends_plus_pi_to_minus_pi():
    """The interval is half-open, so the seam has exactly one representative.
    If both ends were included, one bearing would land in two different bins
    depending on rounding."""
    w = wrap_angle(torch.tensor([math.pi, 3 * math.pi]))
    assert torch.allclose(w, torch.full_like(w, -math.pi), atol=1e-5)


def test_wrap_angle_matches_python_modulo_semantics_for_negatives():
    """C's fmod keeps the dividend's sign and would put these in the wrong
    hemisphere. The C++ transcription has to reproduce *this*."""
    a = torch.tensor([-0.1, -math.pi + 0.1, -2 * math.pi + 0.1])
    expected = torch.tensor([-0.1, -math.pi + 0.1, 0.1])
    assert torch.allclose(wrap_angle(a), expected, atol=1e-5)


# ── binning ──────────────────────────────────────────────────────────────────

def test_output_shape_and_range():
    r = torch.full((4, 720), 5.0)
    out = bin_lidar(r, _beams())
    assert out.shape == (4, LIDAR_BINS)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_binning_is_by_angle_not_index():
    """The A1's start angle is not stable across driver versions. If bins were
    index-based, a driver update would silently rotate the world."""
    n = 720
    r = torch.ones((1, n)) * LIDAR_MAX_M
    r[0, 0] = 1.0                                  # one close return at bearing 0
    a = _beams(n)

    forward = bin_lidar(r, a)
    # Same physical scan, delivered starting a quarter turn later.
    rolled = bin_lidar(torch.roll(r, 180, dims=1), torch.roll(a, 180))
    assert torch.allclose(forward, rolled), "output depends on array order"


def test_binning_is_independent_of_beam_count():
    r_lo = torch.full((1, 360), 3.0)
    r_hi = torch.full((1, 1440), 3.0)
    assert torch.allclose(bin_lidar(r_lo, _beams(360)),
                          bin_lidar(r_hi, _beams(1440)))


def test_reduction_is_min_so_a_thin_obstacle_survives():
    """A table leg covering one beam in a bin must reach the policy. Averaging
    would blend it into the wall behind it."""
    n = 720
    r = torch.full((1, n), LIDAR_MAX_M)
    r[0, 3] = LIDAR_MIN_M                          # single very close return
    out = bin_lidar(r, _beams(n))
    assert out.min() == pytest.approx(0.0, abs=1e-6)
    assert (out < 0.5).sum() == 1, "exactly one bin should be affected"


def test_normalization_maps_near_to_zero_and_far_to_one():
    a = _beams(360)
    near = bin_lidar(torch.full((1, 360), LIDAR_MIN_M), a)
    far = bin_lidar(torch.full((1, 360), LIDAR_MAX_M), a)
    assert torch.allclose(near, torch.zeros_like(near), atol=1e-6)
    assert torch.allclose(far, torch.ones_like(far), atol=1e-6)


# ── the invalid-fill contract ────────────────────────────────────────────────

def test_absent_returns_become_invalid_fill_not_zero():
    """Zero denormalizes to LIDAR_MIN_M -- an obstacle 15 cm ahead in every
    direction. A policy trained on that never moves."""
    r = torch.full((1, 360), float("inf"))
    out = bin_lidar(r, _beams(360))
    assert torch.all(out == INVALID_FILL)


def test_out_of_range_returns_are_treated_as_far_not_near():
    a = _beams(360)
    too_far = bin_lidar(torch.full((1, 360), LIDAR_MAX_M + 5.0), a)
    too_near = bin_lidar(torch.full((1, 360), LIDAR_MIN_M - 0.05), a)
    assert torch.all(too_far == INVALID_FILL)
    assert torch.all(too_near == INVALID_FILL), "sub-minimum is unusable, not close"


def test_nan_does_not_propagate():
    r = torch.full((1, 360), 4.0)
    r[0, :10] = float("nan")
    out = bin_lidar(r, _beams(360))
    assert torch.isfinite(out).all()


def test_bins_with_no_beam_report_no_obstacle():
    """A sparse scan must not claim a wall where it simply did not look.

    Two beams 0.1 rad apart fall in different bins (bins are 2 degrees), so
    exactly two bins carry a reading and the other 178 stay at the fill value.
    """
    a = torch.tensor([0.0, 0.1])
    r = torch.tensor([[1.0, 1.0]])
    out = bin_lidar(r, a)
    assert (out == INVALID_FILL).sum() == LIDAR_BINS - 2
    # Bearing 0 is dead ahead, which is the middle bin: index 90 of 180.
    # 0.1 rad is 2.86 bin widths further round, so it lands in bin 92.
    assert out[0, 90] < INVALID_FILL and out[0, 92] < INVALID_FILL


def test_beams_inside_one_bin_collapse_to_their_minimum():
    a = torch.tensor([0.0, 0.005])                 # 0.29 degrees apart: same bin
    r = torch.tensor([[4.0, 1.0]])
    out = bin_lidar(r, a)
    assert (out == INVALID_FILL).sum() == LIDAR_BINS - 1
    expected = (1.0 - LIDAR_MIN_M) / (LIDAR_MAX_M - LIDAR_MIN_M)
    assert out[0, 90] == pytest.approx(expected, abs=1e-6), "kept the far return"


# ── goal features ────────────────────────────────────────────────────────────

def test_goal_ahead_reads_as_straight_ahead():
    pose = torch.tensor([[0.0, 0.0, 0.0]])
    g = goal_features(pose, torch.tensor([[2.0, 0.0]]))
    assert g[0, 0] == pytest.approx(2.0 / GOAL_MAX_M)
    assert g[0, 1] == pytest.approx(0.0, abs=1e-6)     # sin
    assert g[0, 2] == pytest.approx(1.0, abs=1e-6)     # cos


def test_goal_is_expressed_in_the_robot_frame():
    """Same relative geometry anywhere in any room must give the same input --
    that invariance is most of why the policy transfers at all."""
    a = goal_features(torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0]]))
    b = goal_features(torch.tensor([[5.0, 3.0, math.pi / 2]]),
                      torch.tensor([[5.0 - 1.0, 3.0 + 1.0]]))
    assert torch.allclose(a, b, atol=1e-5)


def test_bearing_is_continuous_across_the_rear():
    """The sin/cos encoding exists so that turning around is not a jump. A raw
    angle would leap by 2pi here, which quantizes catastrophically."""
    pose = torch.tensor([[0.0, 0.0, 0.0]])
    just_left = goal_features(pose, torch.tensor([[-1.0, 1e-4]]))
    just_right = goal_features(pose, torch.tensor([[-1.0, -1e-4]]))
    assert torch.allclose(just_left, just_right, atol=1e-3)


def test_distance_saturates_rather_than_growing_without_bound():
    pose = torch.tensor([[0.0, 0.0, 0.0]])
    g = goal_features(pose, torch.tensor([[100.0, 0.0]]))
    assert g[0, 0] == pytest.approx(1.0)


# ── assembly ─────────────────────────────────────────────────────────────────

def test_observation_layout_matches_the_declared_dimension():
    obs = build_observation(
        torch.zeros(3, HISTORY_FRAMES, LIDAR_BINS), torch.zeros(3, 3),
        torch.zeros(3, HISTORY_FRAMES, 2))
    assert obs.shape == (3, OBS_DIM)


def test_observation_order_is_lidar_goal_action():
    lidar = torch.arange(HISTORY_FRAMES * LIDAR_BINS).reshape(
        1, HISTORY_FRAMES, LIDAR_BINS)
    goal = torch.tensor([[1.0, 2.0, 3.0]])
    last = torch.arange(HISTORY_FRAMES * 2).reshape(1, HISTORY_FRAMES, 2)
    obs = build_observation(lidar, goal, last)
    lidar_width = HISTORY_FRAMES * LIDAR_BINS
    assert torch.equal(obs[0, :lidar_width], lidar.flatten())
    assert torch.equal(obs[0, lidar_width:lidar_width + 3], goal[0])
    assert torch.equal(obs[0, -HISTORY_FRAMES * 2:], last.flatten())


def test_live_history_repeats_first_frame_then_advances_oldest_to_newest():
    history = ObservationHistory(2, frames=3, lidar_bins=2, device="cpu")
    first_lidar = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
    first_action = torch.tensor([[0.1, 0.2], [1.0, 2.0]])
    history.push(first_lidar, first_action)
    assert torch.equal(history.lidar[0], first_lidar[0].expand(3, -1))

    second_lidar = first_lidar + 1
    second_action = first_action + 1
    history.push(second_lidar, second_action, torch.tensor([False, True]))
    assert torch.equal(history.lidar[0, 0], first_lidar[0])
    assert torch.equal(history.lidar[0, -1], second_lidar[0])
    assert torch.equal(history.lidar[1], second_lidar[1].expand(3, -1))
    assert torch.equal(history.action[1], second_action[1].expand(3, -1))
