"""The spinning lidar and the gyro.

These pin the properties that would otherwise fail silently: a sweep that
secretly delivers instantaneous scans, a gyro whose error averages out, and a
deskew that rotates the wrong way. Each of those looks like a training problem
rather than a sensor bug, which is the reason for testing them here.
"""

from __future__ import annotations

import math

import pytest
import torch

from sim2edge.preprocess import wrap_angle
from sim2edge.sensors import (SPIN_PERIOD_S, GyroModel, advance_phase,
                              beam_ages, deskew, sweep_scan, synthesize_gyro)
from sim2edge.spec import DT, LIDAR_SPIN_HZ, SPIN_TICKS

BEAMS = 720


def test_spin_ticks_covers_a_full_revolution():
    """The history must be long enough to hold every beam's acquisition."""
    assert SPIN_TICKS * DT >= SPIN_PERIOD_S


def test_beam_ages_span_exactly_one_revolution():
    ages = beam_ages(torch.tensor([0.0, 0.37, 0.99]), BEAMS)
    assert ages.shape == (3, BEAMS)
    assert float(ages.min()) >= 0.0
    assert float(ages.max()) < SPIN_PERIOD_S
    # Every beam has a distinct age; a constant age would be the old
    # instantaneous model wearing a sweep's clothes.
    assert torch.unique(ages[0]).numel() == BEAMS


def test_the_beam_the_sweep_just_passed_is_the_freshest():
    for phase in (0.0, 0.25, 0.5, 0.9):
        ages = beam_ages(torch.tensor([phase]), BEAMS)
        expected = int(round(phase * BEAMS)) % BEAMS
        assert int(ages[0].argmin()) == expected


def test_phase_advances_by_the_spin_rate_each_tick():
    """One tick is 0.275 of a revolution -- deliberately not a whole fraction.

    A revolution takes 3.64 control periods, so the sweep never lands on the
    same bearing twice and no integer number of ticks returns the phase to zero.
    That non-commensurability is a feature: a sensor that realigned with the
    control loop every few ticks would let the policy learn the alignment
    instead of the geometry.
    """
    phase = torch.zeros(1)
    for step in range(1, 20):
        phase = advance_phase(phase)
        expected = (step * LIDAR_SPIN_HZ * DT) % 1.0
        assert float(phase) == pytest.approx(expected, abs=1e-5)
    assert 1.0 / (LIDAR_SPIN_HZ * DT) != int(1.0 / (LIDAR_SPIN_HZ * DT))


def test_sweep_draws_every_beam_from_its_own_frame():
    """A swept scan is stitched from several ticks, not copied from the newest."""
    history = torch.arange(SPIN_TICKS, dtype=torch.float32)
    history = history[None, :, None].expand(2, SPIN_TICKS, BEAMS).contiguous()
    ranges, _ = sweep_scan(history, torch.tensor([0.0, 0.6]))
    # Frame indices appear as values, so the set of frames actually used is
    # directly observable -- and it must be all of them.
    assert set(ranges[0].tolist()) == set(range(SPIN_TICKS))


def test_sweep_rejects_too_little_history():
    history = torch.zeros(1, SPIN_TICKS - 1, BEAMS)
    with pytest.raises(ValueError, match="scan history"):
        sweep_scan(history, torch.zeros(1))


def test_deskew_removes_a_pure_rotation():
    """Under constant yaw rate and no translation, deskew is exact.

    This is the whole justification for spending a gyro on the problem: the
    correction is not an approximation when the assumption holds.
    """
    ages = beam_ages(torch.tensor([0.3]), BEAMS)
    angles = torch.arange(BEAMS, dtype=torch.float32) * (2 * math.pi / BEAMS)
    rate = torch.tensor([1.4])
    skewed = angles[None, :] + rate[:, None] * ages     # what the sweep produced
    ranges = torch.full((1, BEAMS), 3.0)
    _, corrected = deskew(ranges, skewed, ages, rate)
    assert torch.allclose(wrap_angle(corrected), wrap_angle(angles[None, :]),
                          atol=1e-5)


def test_deskew_translation_moves_a_forward_return_closer():
    """Driving forward for a beam's age shortens the range to something ahead."""
    ages = torch.full((1, 4), 0.1)
    angles = torch.zeros(1, 4)                       # dead ahead
    ranges = torch.full((1, 4), 2.0)
    moved, _ = deskew(ranges, angles, ages, torch.zeros(1),
                      forward_speed=torch.tensor([0.4]))
    assert torch.allclose(moved, torch.full((1, 4), 2.0 - 0.4 * 0.1), atol=1e-5)


def test_deskew_preserves_absent_returns():
    """A dropped beam must stay dropped; ``bin_lidar`` relies on it."""
    ages = torch.full((1, 3), 0.05)
    ranges = torch.tensor([[1.0, math.inf, 2.0]])
    moved, angles = deskew(ranges, torch.zeros(1, 3), ages,
                           torch.tensor([0.5]), torch.tensor([0.3]))
    assert math.isinf(float(moved[0, 1]))
    assert torch.isfinite(angles).all()


def test_gyro_error_is_dominated_by_bias_not_white_noise():
    """Integrated heading error must grow, not average away.

    A gyro modelled as white noise would show error flat in time. The bias and
    scale terms are the ones that matter over an episode, and this is what says
    they are actually present.
    """
    steps, envs = 190, 512
    generator = torch.Generator().manual_seed(3)
    model = GyroModel(envs, generator=generator)
    truth = torch.full((envs,), 0.5)
    error = torch.zeros(envs)
    early = None
    for step in range(steps):
        error += (model.step(truth) - truth) * DT
        if step == 47:
            early = error.abs().mean().clone()
    late = error.abs().mean()
    assert late > 2.0 * early


def _integrated_heading_error(rates, seed):
    """Mean |heading error| in degrees after driving ``rates`` for an episode."""
    generator = torch.Generator().manual_seed(seed)
    envs = rates.shape[1]
    model = GyroModel(envs, generator=generator)
    error = torch.zeros(envs)
    for step in range(rates.shape[0]):
        error += (model.step(rates[step]) - rates[step]) * DT
    return float(error.abs().mean() * 180.0 / math.pi)


def test_gyro_heading_error_stays_under_a_degree_on_a_real_trajectory():
    """The number the whole design rests on: yaw is no longer the constraint.

    One degree at 3 m of remaining travel is 5 cm of lateral error, against a
    0.25 m goal tolerance. Wheel-odometry yaw -- the thing being removed -- is
    worth several degrees, which is what makes the IMU the cheap half of this.
    """
    generator = torch.Generator().manual_seed(11)
    # Yaw rate that varies like a real run, rather than a constant: turns
    # partially cancel, which is why this case is easier than the one below.
    rates = torch.randn(190, 1024, generator=generator) * 0.5
    assert _integrated_heading_error(rates, seed=11) < 1.0


def test_constant_rotation_is_the_worst_case_for_scale_error():
    """Sustained turning in one direction is where the scale term is largest.

    It accumulates with total rotation rather than with time, so a robot that
    spins one way for ten seconds gets no cancellation. Still comfortably inside
    the error budget, but it is the case to quote as the ceiling.
    """
    rates = torch.full((190, 1024), 0.5)
    assert _integrated_heading_error(rates, seed=11) < 2.0


def test_synthesize_gyro_matches_truth_when_noiseless():
    yaw = torch.cumsum(torch.randn(300) * 0.02, dim=0)
    episode = torch.zeros(300, dtype=torch.int32)
    measured = synthesize_gyro(yaw, episode, scale=0.0)
    expected = torch.zeros(300)
    expected[1:] = wrap_angle(yaw[1:] - yaw[:-1])
    assert torch.allclose(measured, expected, atol=1e-6)


def test_synthesize_gyro_zeroes_the_first_sample_of_each_episode():
    """No increment may span an episode boundary -- it would be another room."""
    yaw = torch.randn(60)
    episode = torch.repeat_interleave(torch.arange(3, dtype=torch.int32), 20)
    measured = synthesize_gyro(yaw, episode, scale=1.0,
                               generator=torch.Generator().manual_seed(0))
    assert torch.allclose(measured[torch.tensor([0, 20, 40])], torch.zeros(3))
