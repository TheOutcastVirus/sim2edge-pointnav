"""The sequence data path that feeds BPTT.

Windows must never straddle an episode, labels must be truth, and the observed
inputs must never contain a pose. The last of those is the design in one line:
if ground truth reaches the model through anything but a label, the odometry
input has come back in through a side door and every result is meaningless.
"""

from __future__ import annotations

import math

import pytest
import torch

from sim2edge.dataset import Shard
from sim2edge.models import BeliefNavPolicy
from sim2edge.spec import (DT, HISTORY_FRAMES, MOTION_HISTORY_FRAMES,
                           SPIN_TICKS, V_MAX)

BEAMS, LENGTH, EPISODES = 720, 96, 8


def _shard(seed: int = 0) -> Shard:
    """A synthetic run inside a circular room, with exact analytic ranges."""
    generator = torch.Generator().manual_seed(seed)
    total = LENGTH * EPISODES
    episode = torch.arange(EPISODES, dtype=torch.int32).repeat_interleave(LENGTH)
    speed = torch.rand(total, generator=generator) * V_MAX
    yaw_rate = torch.randn(total, generator=generator) * 0.6

    pose = torch.zeros(total, 3)
    for e in range(EPISODES):
        start = e * LENGTH
        pose[start, 2] = float(torch.rand(1, generator=generator)) * 2 * math.pi
        for t in range(start + 1, start + LENGTH):
            yaw = pose[t - 1, 2]
            pose[t, 0] = pose[t - 1, 0] + speed[t - 1] * math.cos(yaw) * DT
            pose[t, 1] = pose[t - 1, 1] + speed[t - 1] * math.sin(yaw) * DT
            pose[t, 2] = yaw + yaw_rate[t - 1] * DT

    angles = torch.arange(BEAMS, dtype=torch.float32) * (2 * math.pi / BEAMS)
    bearing = pose[:, 2:3] + angles[None, :]
    px, py = pose[:, 0:1], pose[:, 1:2]
    b = px * torch.cos(bearing) + py * torch.sin(bearing)
    c = px**2 + py**2 - 25.0
    ranges = -b + torch.sqrt((b**2 - c).clamp_min(0.0))

    goal = torch.randn(EPISODES, 2, generator=generator).repeat_interleave(
        LENGTH, dim=0) * 2.0
    action = torch.stack([speed, yaw_rate], dim=1)
    previous = torch.roll(action, 1, 0)
    previous[::LENGTH] = 0.0
    return Shard(ranges=ranges.half(), angles=angles, pose=pose, goal=goal,
                 action=action, prev_action=previous, episode=episode)


def test_episode_start_is_the_first_sample_of_each_episode():
    shard = _shard()
    assert shard.episode_start[:LENGTH].unique().tolist() == [0]
    assert shard.episode_start[LENGTH:2 * LENGTH].unique().tolist() == [LENGTH]


def test_sequence_windows_never_cross_an_episode_boundary():
    """A belief carried across a reset integrates motion between two rooms."""
    shard = _shard()
    for seq_len in (2, 16, 32):
        starts = shard.sequence_starts(seq_len)
        assert starts.numel() == EPISODES * (LENGTH - seq_len + 1)
        assert torch.equal(shard.episode[starts],
                           shard.episode[starts + seq_len - 1])


def test_sequence_batch_has_the_expected_time_major_shapes():
    shard = _shard()
    generator = torch.Generator().manual_seed(1)
    batch = next(shard.sequence_batches(8, 12, 1.0, generator=generator))
    assert batch.lidar.shape[:2] == (12, 8)
    assert batch.lidar.shape[2] == HISTORY_FRAMES
    assert batch.raw_lidar.shape == (12, 8, MOTION_HISTORY_FRAMES, BEAMS)
    assert batch.raw_valid.shape == batch.raw_lidar.shape
    assert batch.gyro.shape == (12, 8, MOTION_HISTORY_FRAMES)
    assert batch.previous_action.shape == (12, 8, MOTION_HISTORY_FRAMES, 2)
    assert batch.action.shape == (12, 8, 2)
    assert batch.belief.shape == (12, 8, 2)
    assert batch.translation.shape == (12, 8, 2)
    assert batch.initial_belief.shape == (8, 2)
    assert batch.valid_steps.shape == (12, 8)


def test_sequence_batch_is_finite_everywhere():
    """A non-finite bin index is undefined behaviour in ``bin_lidar``."""
    shard = _shard()
    generator = torch.Generator().manual_seed(2)
    batch = next(shard.sequence_batches(8, 16, 2.0, generator=generator))
    for name in ("lidar", "gyro", "previous_action", "action", "belief",
                 "translation", "initial_belief"):
        assert torch.isfinite(getattr(batch, name)).all(), name


def test_binned_lidar_stays_in_range():
    shard = _shard()
    generator = torch.Generator().manual_seed(3)
    batch = next(shard.sequence_batches(8, 16, 1.0, generator=generator))
    assert float(batch.lidar.min()) >= 0.0
    assert float(batch.lidar.max()) <= 1.0


def test_initial_belief_equals_the_first_true_belief():
    shard = _shard()
    generator = torch.Generator().manual_seed(4)
    batch = next(shard.sequence_batches(8, 16, 1.0, generator=generator))
    assert torch.allclose(batch.belief[0], batch.initial_belief)


def test_true_belief_matches_a_direct_computation():
    """The label is the goal in the body frame -- nothing clever, and checked."""
    shard = _shard()
    index = torch.arange(30, 60)
    delta = shard.goal[index] - shard.pose[index, :2]
    yaw = shard.pose[index, 2]
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    expected = torch.stack([cos * delta[:, 0] + sin * delta[:, 1],
                            -sin * delta[:, 0] + cos * delta[:, 1]], dim=1)
    assert torch.allclose(shard._true_belief(index), expected, atol=1e-5)


def test_true_translation_is_zero_at_an_episode_boundary():
    shard = _shard()
    out = torch.tensor([[0, 1, 2], [LENGTH, LENGTH + 1, LENGTH + 2]])
    translation = shard._true_translation(out)
    assert float(translation[:, 0].abs().max()) == 0.0
    assert float(translation[:, 1:].abs().max()) > 0.0


def test_wheel_slip_creates_the_gap_the_odometry_head_lives_on():
    """Without slip, dead reckoning is exact and the head has nothing to learn.

    The collected corpus tracked commands to 0.22 mm per tick, so this gap is
    introduced at load time. If it ever collapses back to zero the head will
    train to a constant and look perfect while contributing nothing.
    """
    shard = _shard()
    generator = torch.Generator().manual_seed(5)
    batch = next(shard.sequence_batches(64, 24, 1.0, generator=generator))
    dead_reckoned = batch.previous_action[1:, :, -1, 0] * V_MAX * DT
    truth = batch.translation[1:, :, 0]
    residual = (dead_reckoned - truth).abs().mean()
    assert float(residual) > 1e-4      # metres per tick; ~0.2 mm would be zero


def test_noise_scale_zero_gives_exact_dead_reckoning():
    """The clean tier must have no slip, or 'clean' means nothing."""
    shard = _shard()
    generator = torch.Generator().manual_seed(6)
    batch = next(shard.sequence_batches(32, 16, 0.0, generator=generator))
    dead_reckoned = batch.previous_action[1:, :, -1, 0] * V_MAX * DT
    truth = batch.translation[1:, :, 0]
    assert float((dead_reckoned - truth).abs().max()) < 1e-4


def test_new_dynamics_corpus_does_not_receive_legacy_slip_twice():
    shard = _shard()
    shard.motion_model_version = 2
    batch = next(shard.sequence_batches(
        32, 16, 1.0, generator=torch.Generator().manual_seed(6)))
    dead_reckoned = batch.previous_action[1:, :, -1, 0] * V_MAX * DT
    truth = batch.translation[1:, :, 0]
    assert float((dead_reckoned - truth).abs().max()) < 1e-4


def test_episode_batches_start_only_at_real_resets():
    shard = _shard()
    batches = list(shard.episode_batches(
        3, 0.0, shuffle=False, generator=torch.Generator().manual_seed(9)))
    assert sum(batch.initial_belief.shape[0] for batch in batches) == EPISODES
    assert all(batch.lidar.shape[0] == LENGTH for batch in batches)
    assert all(bool(batch.valid_steps.all()) for batch in batches)


def test_a_belief_policy_consumes_the_batch_unmodified():
    """The data path and the model must agree without an adapter between them."""
    shard = _shard()
    generator = torch.Generator().manual_seed(7)
    batch = next(shard.sequence_batches(8, 16, 1.0, generator=generator))
    policy = BeliefNavPolicy()
    actions, beliefs = policy.rollout(batch.lidar, batch.gyro,
                                      batch.previous_action,
                                      batch.initial_belief)
    assert actions.shape == batch.action.shape
    assert beliefs.shape == batch.belief.shape
    assert torch.isfinite(actions).all() and torch.isfinite(beliefs).all()


def test_untrained_belief_tracks_truth_at_dead_reckoning_accuracy():
    """A sanity floor: at init the model is dead reckoning, so error is small.

    If this ever grows to the goal tolerance, the composition or the indexing has
    broken -- an earlier revision integrated the command about to be issued
    against the rotation just measured and sat at 8 cm here.
    """
    shard = _shard()
    generator = torch.Generator().manual_seed(8)
    batch = next(shard.sequence_batches(32, 32, 0.0, generator=generator))
    policy = BeliefNavPolicy()
    with torch.no_grad():
        _, beliefs = policy.rollout(batch.lidar, batch.gyro,
                                    batch.previous_action, batch.initial_belief)
    error = (beliefs[-1] - batch.belief[-1]).norm(dim=-1).mean()
    assert float(error) < 0.01
