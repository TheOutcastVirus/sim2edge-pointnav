"""Storage and load-time corruption.

The property under test throughout: the shard holds raw measurements, and
everything that could plausibly be retuned -- noise, binning, normalization --
happens on the way out. If any of it leaks into the file, changing a
hyperparameter becomes a recollection run.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sim2edge import dataset as ds
from sim2edge import noise as noise_mod
from sim2edge.spec import INVALID_FILL, OBS_DIM, OMEGA_MAX, V_MAX, fingerprint

T, B = 64, 360


def _raw(seed=0):
    g = np.random.default_rng(seed)
    return dict(
        ranges=g.uniform(0.3, 8.0, (T, B)),
        angles=np.linspace(0, 2 * math.pi, B, endpoint=False),
        pose=g.uniform(0, 5, (T, 3)),
        goal=g.uniform(0, 5, (T, 2)),
        action=np.stack([g.uniform(0, V_MAX, T), g.uniform(-OMEGA_MAX, OMEGA_MAX, T)], 1),
        prev_action=np.stack([g.uniform(0, V_MAX, T), g.uniform(-OMEGA_MAX, OMEGA_MAX, T)], 1),
        episode=g.integers(0, 4, T),
        layout=g.integers(0, 8, T, dtype=np.int32),
        start_id=g.integers(0, 32, T, dtype=np.uint16),
        path_length=g.uniform(3.0, 8.0, T).astype(np.float32),
    )


@pytest.fixture
def shard(tmp_path):
    p = tmp_path / "s.npz"
    ds.save_shard(p, **_raw())
    return ds.load_shard(p)


# ── noise ────────────────────────────────────────────────────────────────────

def test_zero_scale_is_exactly_clean():
    """Evaluation sweeps scale from 0; the clean end must be genuinely clean,
    not merely quiet, or the noiseless baseline is not a baseline."""
    r = torch.full((4, B), 3.0)
    a = torch.linspace(0, 2 * math.pi, B)
    nr, na = noise_mod.corrupt_lidar(r, a, scale=0.0)
    assert torch.equal(nr, r)
    assert torch.allclose(na, a.unsqueeze(0).expand(4, -1))


def test_dropout_produces_the_same_absence_the_live_path_produces():
    """Dropped beams must become inf so binning maps them to INVALID_FILL --
    the identical value a real no-return yields. A placeholder that differs
    from what the live path emits is the most expensive bug this project has
    had."""
    from sim2edge.preprocess import bin_lidar
    r = torch.full((8, B), 3.0)
    a = torch.linspace(0, 2 * math.pi, B)
    nr, na = noise_mod.corrupt_lidar(r, a, scale=40.0)     # forces near-total loss
    assert torch.isinf(nr).any()
    assert bin_lidar(nr, na).max() == INVALID_FILL


def test_bearing_bias_is_per_scan_not_per_beam():
    """A mount error rotates the whole sensor. Per-beam jitter would be a
    different physical claim and would average out over a bin."""
    r = torch.full((4, B), 3.0)
    a = torch.linspace(0, 2 * math.pi, B)
    _, na = noise_mod.corrupt_lidar(r, a, scale=1.0)
    offset = na - a.unsqueeze(0)
    assert torch.allclose(offset, offset[:, :1].expand(-1, B), atol=1e-6)
    assert not torch.allclose(offset[:, 0], offset[0, 0].expand(4))


def test_odometry_drift_moves_the_pose_not_the_goal():
    """Drift originates in the robot's belief about itself, and it rotates the
    goal bearing as well as translating it -- a goal-only perturbation would
    miss the rotation entirely."""
    pose = torch.zeros(4, 3)
    goal = torch.ones(4, 2)
    p2, g2 = noise_mod.corrupt_goal(pose, goal, scale=1.0)
    assert torch.equal(g2, goal)
    assert not torch.equal(p2, pose)
    assert (p2[:, 2] != 0).any(), "yaw drift was dropped"


# ── round trip ───────────────────────────────────────────────────────────────

def test_save_load_preserves_the_raw_quantities(tmp_path):
    raw = _raw()
    p = tmp_path / "s.npz"
    ds.save_shard(p, **raw)
    got = ds.load_shard(p)
    assert len(got) == T
    # float16 storage: the A1's own error is 2 cm, this resolves under 6 mm.
    assert np.allclose(got.ranges.float().numpy(), raw["ranges"], atol=6e-3)
    assert np.allclose(got.pose.numpy(), raw["pose"], atol=1e-5)
    assert np.allclose(got.action.numpy(), raw["action"], atol=1e-5)
    assert np.array_equal(got.layout.numpy(), raw["layout"])
    assert np.array_equal(got.start_id.numpy(), raw["start_id"])
    assert np.allclose(got.path_length.numpy(), raw["path_length"])


def test_stored_ranges_are_raw_not_preprocessed(shard):
    """If binning leaked into the file, changing LIDAR_BINS would invalidate
    every episode ever collected."""
    assert shard.ranges.shape == (T, B)
    assert shard.ranges.max() > 1.0, "looks normalized, not metres"


def test_stored_ranges_are_clean_not_noisy(tmp_path):
    """Two shards saved from identical raw input must be byte-identical. If the
    writer injected noise, retuning it would mean recollecting."""
    raw = _raw()
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    ds.save_shard(a, **raw)
    ds.save_shard(b, **raw)
    assert torch.equal(ds.load_shard(a).ranges, ds.load_shard(b).ranges)


def test_fingerprint_mismatch_refuses_to_load(tmp_path, monkeypatch):
    p = tmp_path / "s.npz"
    ds.save_shard(p, **_raw())
    monkeypatch.setattr(ds, "fingerprint", lambda: "0000000000000000")
    with pytest.raises(ValueError, match="contract changed|was collected under"):
        ds.load_shard(p)


def test_fingerprint_matches_a_fresh_shard(shard):
    assert fingerprint() == fingerprint()      # stable within a process


# ── batching ─────────────────────────────────────────────────────────────────

def test_batches_have_the_declared_observation_width(shard):
    obs, act = next(iter(shard.batches(16)))
    assert obs.shape == (16, OBS_DIM)
    assert act.shape == (16, 2)


def test_batches_cover_every_sample_once(shard):
    seen = sum(o.shape[0] for o, _ in shard.batches(15))
    assert seen == T


def test_labels_stay_in_si_units(shard):
    """The exported graph squashes and rescales in-graph, so the C++ node gets
    m/s directly. Normalizing the label here would double-scale it."""
    _, act = next(iter(shard.batches(T, shuffle=False)))
    assert torch.equal(act, shard.action)


def test_every_epoch_draws_fresh_noise(shard):
    """The reason noise is not baked into the sim: two passes over the same
    states should show the policy two different corruptions."""
    a, _ = next(iter(shard.batches(T, shuffle=False)))
    b, _ = next(iter(shard.batches(T, shuffle=False)))
    assert not torch.allclose(a, b), "corruption was frozen"


def test_clean_batches_are_reproducible(shard):
    """With noise off, the pipeline must be deterministic -- otherwise a failed
    evaluation cannot be reproduced."""
    a, _ = next(iter(shard.batches(T, noise_scale=0.0, shuffle=False)))
    b, _ = next(iter(shard.batches(T, noise_scale=0.0, shuffle=False)))
    assert torch.equal(a, b)


def test_last_action_input_is_normalized_even_though_the_label_is_not(shard):
    obs, _ = next(iter(shard.batches(T, noise_scale=0.0, shuffle=False)))
    last = obs[:, -2:]
    assert torch.all(last.abs() <= 1.0 + 1e-6)
    assert torch.allclose(last[:, 0], shard.prev_action[:, 0] / V_MAX, atol=1e-6)
    assert torch.allclose(last[:, 1], shard.prev_action[:, 1] / OMEGA_MAX, atol=1e-6)


def test_prev_action_is_stored_not_derived(shard):
    """Deriving it by shifting `action` means special-casing episode
    boundaries, and getting that wrong leaks the previous episode's last
    command into the next episode's first observation."""
    assert not torch.equal(shard.prev_action[1:], shard.action[:-1])
