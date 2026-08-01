"""Odometry drift and wheel slip: the corrections that made the old numbers honest.

The central assertion here is autocorrelation. The previous model drew a fresh
i.i.d. Gaussian per sample, which has none -- it averages out across the
four-frame history and a policy can low-pass it away for free. Real drift
integrates and never comes back, and that is the only kind that threatens
anything. A regression to white noise would not change a single shape, would not
fail a single existing test, and would quietly restore every optimistic number
this project used to report. So it is pinned here.
"""

from __future__ import annotations

import math

import torch

from sim2edge.noise import (ODOM_YAW_RATE_BIAS_RAD, OdometryDrift,
                            _segmented_cumsum, odometry_drift)
from sim2edge.spec import DT

EPISODE, COUNT = 190, 256


def _episodes(length: int = EPISODE, count: int = COUNT) -> torch.Tensor:
    return torch.arange(count, dtype=torch.int32).repeat_interleave(length)


def test_segmented_cumsum_restarts_at_every_episode():
    values = torch.ones(9, 1)
    episode = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.int32)
    result = _segmented_cumsum(values, episode).squeeze(1)
    assert result.tolist() == [1, 2, 3, 1, 2, 1, 2, 3, 4]


def test_drift_is_strongly_autocorrelated_within_an_episode():
    """The property white noise does not have, and the reason this exists."""
    generator = torch.Generator().manual_seed(0)
    episode = _episodes()
    drift = odometry_drift(episode, 1.0, generator)
    walk = drift[:, 0].reshape(COUNT, EPISODE)
    pairs = torch.stack([walk[:, 1:].flatten(), walk[:, :-1].flatten()])
    assert float(torch.corrcoef(pairs)[0, 1]) > 0.9


def test_drift_grows_along_an_episode():
    """A random walk's spread widens with sqrt(t); white noise's is flat."""
    generator = torch.Generator().manual_seed(1)
    drift = odometry_drift(_episodes(), 1.0, generator)
    walk = drift[:, :2].reshape(COUNT, EPISODE, 2)
    early = walk[:, 20].norm(dim=1).mean()
    late = walk[:, 180].norm(dim=1).mean()
    assert float(late) > 2.0 * float(early)


def test_drift_restarts_at_zero_each_episode():
    """Error must not be inherited across a reset: that is a different room."""
    generator = torch.Generator().manual_seed(2)
    drift = odometry_drift(_episodes(), 1.0, generator)
    first = drift.reshape(COUNT, EPISODE, 3)[:, 0]
    # The first sample carries exactly one increment, so it is small but nonzero.
    assert float(first.norm(dim=1).mean()) < 0.02


def test_yaw_bias_scales_with_rotation_not_with_time():
    """The scale-factor term must punish cornering, not merely elapsed steps.

    Compared at the episode's final sample rather than averaged over it: the
    bias term grows with accumulated rotation, so averaging over a window that
    starts at zero rotation halves the effect and blurs what is being asserted.
    Two identically-seeded runs differ only in whether the robot turned.
    """
    episode = _episodes()
    straight = torch.zeros(episode.shape[0])
    turning = torch.full((episode.shape[0],), 0.5 * DT)
    still = odometry_drift(episode, 1.0, torch.Generator().manual_seed(4),
                           yaw_increment=straight)
    turned = odometry_drift(episode, 1.0, torch.Generator().manual_seed(4),
                            yaw_increment=turning)
    final = lambda drift: drift[:, 2].reshape(COUNT, EPISODE)[:, -1].abs().mean()
    assert float(final(turned)) > 2.0 * float(final(still))


def test_live_drift_matches_the_offline_walk_in_scale():
    """Two implementations of one process; they must not disagree.

    The offline path cumsums a whole shard, the live path integrates a tick at a
    time because a running environment cannot see the future. If these drift
    apart, a policy is evaluated against a different error than it trained on.
    """
    generator = torch.Generator().manual_seed(5)
    live = OdometryDrift(4096, scale=1.0, generator=generator)
    increment = torch.zeros(4096)
    for _ in range(EPISODE):
        drift = live.step(increment)
    offline = odometry_drift(_episodes(EPISODE, 4096), 1.0,
                             torch.Generator().manual_seed(6))
    offline_final = offline.reshape(4096, EPISODE, 3)[:, -1]
    ratio = (drift[:, :2].norm(dim=1).mean()
             / offline_final[:, :2].norm(dim=1).mean())
    assert 0.8 < float(ratio) < 1.25


def test_live_drift_resets_per_environment():
    generator = torch.Generator().manual_seed(7)
    live = OdometryDrift(8, scale=1.0, generator=generator)
    for _ in range(50):
        live.step(torch.zeros(8))
    live.reset(torch.tensor([0, 1]))
    assert float(live.drift[:2].abs().max()) == 0.0
    assert float(live.drift[2:].abs().max()) > 0.0


def test_drift_scale_zero_produces_no_error():
    """Clean evaluation must be genuinely clean, for the noise=0 tier."""
    drift = odometry_drift(_episodes(), 0.0, torch.Generator().manual_seed(8))
    assert float(drift.abs().max()) == 0.0
