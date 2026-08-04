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
from sim2edge.sensors import (EDGE_TOLERANCE_M, SPIN_PERIOD_S, GyroModel,
                              advance_phase, apply_mount_occlusion,
                              beam_ages, deskew, deskew_history,
                              occlusion_mask, sweep_scan,
                              synthesize_gyro)
from sim2edge.spec import (DT, LIDAR_OCCLUSION_ARCS_RAD, LIDAR_SPIN_HZ,
                           SPIN_TICKS)

BEAMS = 720


def test_spin_ticks_covers_a_full_revolution():
    """The history must be long enough to hold every beam's acquisition."""
    assert SPIN_TICKS * DT >= SPIN_PERIOD_S


# ── the camera mount's occlusion ─────────────────────────────────────────────
# Measured, not assumed: 14.51% of bearings blocked in every one of 500 real
# revolutions. The simulator had none of this, so the policy trained believing
# it had 360 degrees of coverage.

def _full_circle(beams: int = 1080) -> torch.Tensor:
    return torch.linspace(-math.pi, math.pi, beams + 1)[:-1]


def test_occlusion_matches_the_measured_fraction():
    """The blocked share must stay what the hardware showed. Drift here is a
    silent sim2real gap, not a test artefact."""
    blocked = occlusion_mask(_full_circle()).float().mean().item()
    assert blocked == pytest.approx(0.1370, abs=0.005)


def test_occlusion_is_behind_the_robot():
    """97% of the blocked sector is astern, and the forward cone the safety
    layer watches is essentially clear. If a change puts it in front, the
    consequence is a policy blind to what it is driving into."""
    angles = _full_circle()
    blocked = occlusion_mask(angles)
    forward = angles.abs() < math.radians(35)
    assert blocked[forward].sum() == 0
    rear = angles.abs() >= math.radians(90)
    assert blocked[rear].sum() >= 0.9 * blocked.sum()


def test_occlusion_arc_crossing_pi_wraps():
    """One measured arc runs from +163 to +185 degrees. Bearings are reported
    in [-pi, pi), so a naive start <= theta <= end would silently miss the half
    of that arc past the seam."""
    crossing = [(s, e) for s, e in LIDAR_OCCLUSION_ARCS_RAD if e > math.pi]
    assert crossing, "expected an arc past +pi; the wrap case is the point"
    start, end = crossing[0]
    assert occlusion_mask(torch.tensor([start + 1e-3])).item()
    assert occlusion_mask(torch.tensor([end - 1e-3 - 2 * math.pi])).item()
    # just outside either end stays clear
    assert not occlusion_mask(torch.tensor([start - 1e-2])).item()
    assert not occlusion_mask(torch.tensor([end + 1e-2 - 2 * math.pi])).item()


def test_occluded_beams_are_the_same_absence_a_no_return_is():
    """An occluded beam must reach the policy as INVALID_FILL, identical to a
    genuine no-return. A distinguishable placeholder is the most expensive bug
    class this project has had."""
    from sim2edge.preprocess import bin_lidar
    from sim2edge.spec import INVALID_FILL
    angles = _full_circle()
    ranges = torch.full((4, angles.shape[0]), 3.0)
    out = apply_mount_occlusion(ranges, angles)
    assert torch.isinf(out).any()
    assert bin_lidar(out, angles.unsqueeze(0).expand(4, -1)).max() == INVALID_FILL


def test_occlusion_does_not_retract_at_zero_noise():
    """Standoffs are geometry. corrupt_lidar's scale=0 is a clean *noise*
    baseline and must stay one, but a noiseless evaluation is still performed
    by a robot with a camera bolted to it."""
    angles = _full_circle()
    ranges = torch.full((2, angles.shape[0]), 3.0)
    quiet, _ = __import__("sim2edge.noise", fromlist=["x"]).corrupt_lidar(
        ranges, angles, scale=0.0)
    assert torch.equal(quiet, ranges)                     # noise really is off
    assert torch.isinf(apply_mount_occlusion(quiet, angles)).any()


def test_occlusion_rejects_mismatched_beam_counts():
    with pytest.raises(ValueError, match="beam count"):
        apply_mount_occlusion(torch.zeros((2, 16)), _full_circle(32))


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
    """One tick is 0.366 of a revolution -- deliberately not a whole fraction.

    A revolution takes 2.73 control periods, so the sweep never lands on the
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


def test_history_deskew_matches_constant_rate_rotation():
    """Piecewise integration reduces to scalar deskew at tick-aligned ages."""
    ages = torch.arange(SPIN_TICKS, dtype=torch.float32)[None, :] * DT
    ranges = torch.full_like(ages, 3.0)
    angles = torch.linspace(-1.0, 1.0, SPIN_TICKS)[None, :]
    rate = torch.tensor([0.7])
    increments = torch.full((1, SPIN_TICKS), float(rate * DT))
    speed = torch.zeros_like(increments)

    old_ranges, old_angles = deskew(ranges, angles, ages, rate)
    new_ranges, new_angles = deskew_history(
        ranges, angles, ages, increments, speed)

    assert torch.allclose(new_ranges, old_ranges, atol=1e-6)
    assert torch.allclose(wrap_angle(new_angles), wrap_angle(old_angles),
                          atol=1e-6)


def test_history_deskew_integrates_varying_motion_per_beam():
    """Older returns use every intervening command and gyro increment."""
    ages = torch.arange(SPIN_TICKS, dtype=torch.float32)[None, :] * DT
    ranges = torch.full_like(ages, 2.0)
    angles = torch.zeros_like(ages)
    speed = torch.arange(1, SPIN_TICKS + 1, dtype=torch.float32)[None, :] * 0.1
    increments = torch.zeros_like(speed)

    moved, bearing = deskew_history(
        ranges, angles, ages, increments, speed)
    expected_distance = torch.tensor([
        speed[0, SPIN_TICKS - back:].sum() * DT
        for back in range(SPIN_TICKS)
    ])

    assert torch.allclose(moved[0], 2.0 - expected_distance, atol=1e-6)
    assert torch.allclose(bearing, torch.zeros_like(bearing), atol=1e-6)


def test_history_deskew_zero_reset_history_is_identity():
    """A new episode cannot inherit motion from the episode before reset."""
    ages = beam_ages(torch.tensor([0.4]), BEAMS)
    ranges = torch.linspace(0.5, 5.0, BEAMS)[None, :]
    angles = torch.linspace(-math.pi, math.pi, BEAMS)[None, :]
    reset = torch.zeros(1, SPIN_TICKS)

    moved, bearing = deskew_history(ranges, angles, ages, reset, reset)

    assert torch.allclose(moved, ranges, atol=1e-6)
    assert torch.allclose(wrap_angle(bearing - angles),
                          torch.zeros_like(angles), atol=1e-6)


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


def test_gyro_heading_error_stays_small_on_a_real_trajectory():
    """The number the whole design rests on: yaw is no longer the constraint.

    This bound was 1.0 degree while the gyro constants were a design target.
    Measuring the actual OAK-D on 2026-08-04 put its residual bias at
    -0.00285 rad/s, 2.8x what had been assumed, and the integrated error over an
    episode moved to ~1.23 degrees. The conclusion survives the correction and
    the claim has been restated rather than the constant retuned to suit it:
    1.23 degrees at 3 m of remaining travel is 6 cm of lateral error against a
    0.25 m goal tolerance, and wheel-odometry yaw -- the thing being removed --
    is worth several degrees. The IMU is still the cheap half of this.

    The bound is 1.5 rather than 1.3 so that ordinary seed variation does not
    fail it, and low enough that another 2x in bias would.
    """
    generator = torch.Generator().manual_seed(11)
    # Yaw rate that varies like a real run, rather than a constant: turns
    # partially cancel, which is why this case is easier than the one below.
    rates = torch.randn(190, 1024, generator=generator) * 0.5
    assert _integrated_heading_error(rates, seed=11) < 1.5


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


# ── sub-tick sweep reconstruction ────────────────────────────────────────────
#
# The acceptance criterion these serve is in
# ``scripts/probe_sensor_consistency.py``: two reconstructed frames placed by
# the *true* relative pose must agree to better than the 0.078 cm/tick slip
# signal the estimator exists to measure. These tests pin the mechanism, not
# that number, which needs a corpus.


def test_subtick_sweep_is_off_by_default_and_reproduces_rounding():
    """Every existing corpus must read back bit-identically."""
    torch.manual_seed(0)
    ranges = torch.rand(3, SPIN_TICKS, 720) * 3.0 + 1.0
    phase = torch.rand(3)
    rounded, ages = sweep_scan(ranges, phase)
    default, _ = sweep_scan(ranges, phase, subtick=False)
    torch.testing.assert_close(default, rounded, atol=0.0, rtol=0.0)


def test_subtick_sweep_agrees_with_rounding_on_tick_boundaries():
    """Where a beam's age lands exactly on a stored tick there is nothing to
    interpolate, so the two paths must give the identical measurement."""
    beams = 720
    # Phase zero puts beam 0 at age 0 and every other beam at a fraction of a
    # revolution; pick the beams whose age is an exact multiple of DT.
    phase = torch.zeros(1)
    ranges = torch.rand(1, SPIN_TICKS, beams) * 3.0 + 1.0
    rounded, ages = sweep_scan(ranges, phase)
    blended, _ = sweep_scan(ranges, phase, subtick=True)
    exact = ages / DT
    on_tick = (exact - exact.round()).abs() < 1e-6
    assert bool(on_tick.any())
    torch.testing.assert_close(blended[on_tick], rounded[on_tick],
                               atol=1e-6, rtol=0.0)


def test_subtick_sweep_interpolates_a_smoothly_approaching_surface():
    """A wall closing at constant rate is exactly what interpolation is for."""
    beams = 8
    # Beam 0 at phase 0.5 is half a revolution old, which is
    # SPIN_PERIOD_S / 2 / DT ticks and is not a whole number at any spin rate
    # this project has used -- so it falls between stored scans and must be
    # blended rather than snapped.
    #
    # The stack is exactly SPIN_TICKS frames, oldest first, and the steps are
    # smaller than EDGE_TOLERANCE_M so nothing here is a depth discontinuity.
    # It used to be five hand-written frames, which silently stopped lining up
    # with the index arithmetic below when the measured spin rate took
    # SPIN_TICKS from 5 to 4.
    near, far = 2.0, 2.4
    ramp = torch.linspace(far, near, SPIN_TICKS)
    assert (ramp[:-1] - ramp[1:]).abs().max() < EDGE_TOLERANCE_M
    ranges = ramp[:, None].expand(SPIN_TICKS, beams).contiguous()[None]
    phase = torch.full((1,), 0.5)
    blended, ages = sweep_scan(ranges, phase, subtick=True)
    rounded, _ = sweep_scan(ranges, phase)
    exact = ages / DT
    between = (exact - exact.round()).abs() > 0.2
    assert bool(between.any())
    # The blended value must lie strictly between the two bracketing samples,
    # which the rounded one cannot do.
    low = exact.floor().long().clamp(0, SPIN_TICKS - 1)
    high = (low + 1).clamp(max=SPIN_TICKS - 1)
    lo = ranges[0, (SPIN_TICKS - 1) - low[0], torch.arange(beams)]
    hi = ranges[0, (SPIN_TICKS - 1) - high[0], torch.arange(beams)]
    inside = ((blended[0] - torch.minimum(lo, hi) >= -1e-6)
              & (torch.maximum(lo, hi) - blended[0] >= -1e-6))
    assert bool(inside[between[0]].all())
    assert not torch.equal(blended[between], rounded[between])


def test_subtick_sweep_refuses_to_blend_across_a_depth_edge():
    """Blending a wall against the corridor behind it invents free space.

    A surface that disappears between two stored ticks is an occlusion change,
    not the same surface seen twice, and the midpoint of 1 m and 6 m is a
    return where nothing exists.
    """
    beams = 8
    close, distant = 1.0, 6.0
    assert distant - close > EDGE_TOLERANCE_M
    ranges = torch.stack([
        torch.full((beams,), distant), torch.full((beams,), distant),
        torch.full((beams,), distant), torch.full((beams,), close),
        torch.full((beams,), close),
    ])[None]
    blended, _ = sweep_scan(ranges, torch.full((1,), 0.5), subtick=True)
    # Every value is one of the two real surfaces, never between them.
    on_a_surface = (((blended - close).abs() < 1e-6)
                    | ((blended - distant).abs() < 1e-6))
    assert bool(on_a_surface.all()), blended


def test_subtick_deskew_reduces_to_rounding_at_integer_ages():
    """The transform interpolation must be consistent with the range sourcing."""
    beams = 720
    torch.manual_seed(1)
    ranges = torch.rand(2, beams) * 3.0 + 1.0
    angles = torch.linspace(-math.pi, math.pi, beams)
    gyro = torch.randn(2, SPIN_TICKS) * 0.02
    speed = torch.rand(2, SPIN_TICKS) * 0.4
    # Ages that are exact multiples of a control period.
    ages = (torch.arange(beams) % SPIN_TICKS).float()[None].repeat(2, 1) * DT
    rounded = deskew_history(ranges, angles, ages, gyro, speed)
    blended = deskew_history(ranges, angles, ages, gyro, speed, subtick=True)
    for want, got in zip(rounded, blended):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0.0)


def test_subtick_deskew_moves_a_beam_between_the_bracketing_corrections():
    """A half-tick-old beam is corrected by about half a tick of motion."""
    beams = 4
    ranges = torch.full((1, beams), 2.0)
    angles = torch.zeros(beams)
    gyro = torch.zeros(1, SPIN_TICKS)
    speed = torch.full((1, SPIN_TICKS), 0.4)
    half = torch.full((1, beams), 1.5 * DT)
    one = torch.full((1, beams), 1.0 * DT)
    two = torch.full((1, beams), 2.0 * DT)
    mid, _ = deskew_history(ranges, angles, half, gyro, speed, subtick=True)
    near, _ = deskew_history(ranges, angles, one, gyro, speed, subtick=True)
    far, _ = deskew_history(ranges, angles, two, gyro, speed, subtick=True)
    assert float(far[0, 0]) < float(mid[0, 0]) < float(near[0, 0])
