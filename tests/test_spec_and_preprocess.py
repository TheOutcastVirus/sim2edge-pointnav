"""Tests for the task contract and the observation pipeline.

These are the highest-value tests in the repo. The spec and preprocessing are
what training, quantization calibration, and the robot all have to agree on, and
a disagreement there fails silently rather than loudly.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim2edge.preprocess import (  # noqa: E402
    Preprocessor,
    block_min_pool,
    preprocess_depth,
    preprocess_goal,
    preprocess_last_action,
    preprocess_lidar,
    world_goal_to_robot_frame,
)
from sim2edge.spec import DEFAULT_SPEC, NavSpec  # noqa: E402

S = DEFAULT_SPEC


# ─────────────────────────────────────────────────────────────────────────────
# Spec
# ─────────────────────────────────────────────────────────────────────────────


def test_spec_round_trips_through_json():
    restored = NavSpec.from_json(S.to_json())
    assert restored == S
    assert restored.fingerprint() == S.fingerprint()


def test_fingerprint_changes_with_the_contract():
    other = NavSpec()
    assert other.fingerprint() == S.fingerprint()

    from dataclasses import replace

    changed = replace(S, obs=replace(S.obs, lidar=replace(S.obs.lidar, bins=90)))
    assert changed.fingerprint() != S.fingerprint()


def test_fingerprint_ignores_randomization():
    """Randomization affects how a policy was trained, not how its inputs are
    built at runtime, and the eval harness legitimately varies it."""
    from dataclasses import replace

    noisy = replace(S, randomization=S.randomization.at_scale(2.0))
    assert noisy.fingerprint() == S.fingerprint()


def test_action_normalize_round_trips():
    a = S.action
    for v, w in [(0.0, 0.0), (a.v_max, a.omega_max), (0.13, -0.7)]:
        assert a.denormalize(*a.normalize(v, w)) == pytest.approx((v, w), abs=1e-6)


def test_noise_tiers_scale_sensibly():
    clean = S.randomization.at_scale(0.0)
    assert clean.lidar_range_noise_m == (0.0, 0.0)
    assert not clean.enabled

    doubled = S.randomization.at_scale(2.0)
    assert doubled.lidar_range_noise_m[1] == pytest.approx(
        S.randomization.lidar_range_noise_m[1] * 2.0)
    # Gain-style ranges scale about their nominal value, not about zero.
    lo, hi = doubled.velocity_tracking_gain
    assert lo < 1.0 < hi


def test_planning_margin_exceeds_nothing_silently():
    """Planning inflation must exceed the collision radius.

    Regression guard: with inflation below the collision radius, every oracle
    path is tangent to the collision boundary and the expert's success rate
    collapses (it measured 42%). The spec must keep the margin positive.
    """
    assert S.room.planning_margin_m > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing: depth
# ─────────────────────────────────────────────────────────────────────────────


def test_depth_normalization_endpoints():
    d = S.obs.depth
    img = np.full((d.height, d.width), d.min_m, dtype=np.float32)
    assert preprocess_depth(img).min() == pytest.approx(0.0)

    img[:] = d.max_m
    assert preprocess_depth(img).max() == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_depth_reads_as_far(bad):
    """Invalid must map to "far", not "near".

    A policy that brakes for every stereo hole never moves in a real room; the
    lidar and the classical bubble cover the pessimistic case.
    """
    img = np.full((S.obs.depth.height, S.obs.depth.width), bad, dtype=np.float32)
    out = preprocess_depth(img)
    assert np.allclose(out, S.obs.depth.invalid_fill)


def test_depth_downsampling_is_conservative():
    """Min-pooling must never let a thin obstacle disappear."""
    img = np.full((480, 640), 5.0, dtype=np.float32)
    img[240, :] = 0.5  # a single-pixel-high bar, far thinner than a pool block
    out = preprocess_depth(img)
    expected = (0.5 - S.obs.depth.min_m) / (S.obs.depth.max_m - S.obs.depth.min_m)
    assert out.min() == pytest.approx(expected, abs=1e-6)


def test_block_min_pool_handles_non_integer_ratios():
    src = np.arange(7 * 11, dtype=np.float32).reshape(7, 11)
    out = block_min_pool(src, 3, 4)
    assert out.shape == (3, 4)
    assert out.min() == src.min()  # the global min must survive


def test_block_min_pool_upsamples_without_empty_bins():
    src = np.array([[1.0, 2.0]], dtype=np.float32)
    out = block_min_pool(src, 2, 4)
    assert out.shape == (2, 4)
    assert np.isfinite(out).all()


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing: lidar
# ─────────────────────────────────────────────────────────────────────────────


def test_lidar_binning_is_independent_of_scan_convention():
    """The same physical scan must give the same observation whether the driver
    reports angles over [-pi, pi) or [0, 2pi).

    The RPLIDAR's start angle is not stable across driver versions, and binning
    by index instead of angle would silently rotate the policy's world view.

    Beams are placed at bin quarter-points rather than on bin edges. Binning is
    inherently boundary-sensitive -- a beam exactly on an edge can land either
    side of it depending on a 1-ULP difference in how its angle was accumulated
    -- and that is a property of any binning scheme, not the convention
    independence this test is about. The Python/C++ parity check
    (tools/parity_check.py) is what pins down edge behaviour, and it agrees
    bit-exactly because both implementations do the same arithmetic.
    """
    n = 360
    rng = np.random.default_rng(0)
    ranges = rng.uniform(0.3, 7.0, n).astype(np.float32)
    inc = 2.0 * math.pi / n

    a = preprocess_lidar(ranges, -math.pi + 0.5 * inc, inc)
    # The same physical beams, relabelled so index 0 sits at angle ~0.
    rolled = np.roll(ranges, -n // 2)
    b = preprocess_lidar(rolled, 0.5 * inc, inc)
    assert np.array_equal(a, b)


def test_lidar_invalid_returns_read_as_far():
    ranges = np.full(360, np.inf, dtype=np.float32)
    out = preprocess_lidar(ranges, -math.pi, 2 * math.pi / 360)
    assert np.allclose(out, S.obs.lidar.invalid_fill)


def test_lidar_respects_message_validity_limits():
    """Returns outside the message's own range limits are invalid, not clamped."""
    ranges = np.full(360, 0.05, dtype=np.float32)  # below any sane range_min
    out = preprocess_lidar(ranges, -math.pi, 2 * math.pi / 360, range_min=0.15)
    assert np.allclose(out, S.obs.lidar.invalid_fill)


def test_lidar_min_pools_within_a_bin():
    n = 360
    ranges = np.full(n, 6.0, dtype=np.float32)
    ranges[0] = 0.5
    out = preprocess_lidar(ranges, -math.pi, 2 * math.pi / n)
    near = (0.5 - S.obs.lidar.min_m) / (S.obs.lidar.max_m - S.obs.lidar.min_m)
    assert out.min() == pytest.approx(near, abs=1e-6)


def test_empty_scan_is_handled():
    out = preprocess_lidar(np.array([], dtype=np.float32), -math.pi, 0.01)
    assert out.shape == (S.obs.lidar.bins,)
    assert np.allclose(out, S.obs.lidar.invalid_fill)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing: goal and action
# ─────────────────────────────────────────────────────────────────────────────


def test_goal_encoding_is_continuous_across_the_pi_wrap():
    """sin/cos encoding must not jump where the bearing wraps.

    A discontinuity there lands exactly where the robot needs to turn around,
    and quantizing a wrapping input to INT8 is a silent-failure generator.
    """
    eps = 1e-4
    just_below = preprocess_goal(-1.0, eps)
    just_above = preprocess_goal(-1.0, -eps)
    assert np.max(np.abs(just_below - just_above)) < 1e-3


def test_goal_distance_saturates():
    far = preprocess_goal(1000.0, 0.0)
    assert far[0] == pytest.approx(1.0)
    assert far[2] == pytest.approx(1.0)  # cos: straight ahead


def test_goal_at_origin_is_well_defined():
    out = preprocess_goal(0.0, 0.0)
    assert np.allclose(out, [0.0, 0.0, 1.0])


def test_last_action_is_bounded():
    out = preprocess_last_action(1e6, -1e6)
    assert np.all(np.abs(out) <= 1.0)


def test_goal_transform_matches_hand_computation():
    # Robot at (1, 1) facing +y; goal at (1, 3) is 2 m straight ahead.
    x, y = world_goal_to_robot_frame((1.0, 3.0), (1.0, 1.0), math.pi / 2)
    assert x == pytest.approx(2.0, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)


def test_preprocessor_orders_tuple_as_the_graph_expects():
    pre = Preprocessor(S)
    obs = pre.build(
        depth_m=np.full((60, 80), 2.0, dtype=np.float32),
        scan_ranges=np.full(360, 3.0, dtype=np.float32),
        scan_angle_min=-math.pi,
        scan_angle_increment=2 * math.pi / 360,
        goal_xy_robot=(1.0, 0.5),
        last_action=(0.1, 0.2),
    )
    shapes = [t.shape for t in pre.as_tuple(obs)]
    assert shapes == [tuple(s) for s in S.actor_input_shapes]


# ─────────────────────────────────────────────────────────────────────────────
# Generated artifacts
# ─────────────────────────────────────────────────────────────────────────────


def test_generated_spec_header_is_current():
    """The C++ header and spec.json must be regenerated when the spec changes.

    Without this, a contract change lands in training and is silently forgotten
    on the robot -- the worst version of the calibration/runtime mismatch.
    """
    result = subprocess.run(
        [sys.executable, str(REPO / "tools/gen_spec_header.py"), "--check"],
        capture_output=True, text=True)
    assert result.returncode == 0, (
        f"{result.stdout}{result.stderr}\n"
        "Run: python3 tools/gen_spec_header.py")


def test_generated_header_carries_the_current_fingerprint():
    header = (REPO / "ros_ws/src/nav_policy_node/include/nav_policy_node"
              / "spec_generated.hpp").read_text()
    assert S.fingerprint() in header
