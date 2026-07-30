"""BC recorder policy without launching Isaac."""

import numpy as np
import torch

from scripts.collect_bc import Collector


def _record(c, steps, *, moving=True):
    for step in range(steps):
        ranges = torch.full((c.num_envs, 3), 1.0 + step)
        x = float(step) * (0.1 if moving else 0.0)
        pose = torch.tensor([[x, 0.0, 0.0]] * c.num_envs)
        goal = torch.tensor([[2.0, 0.0]] * c.num_envs)
        action = torch.tensor([[0.2, 0.0]] * c.num_envs)
        c.record(ranges, pose, goal, action, action)


def test_default_retains_short_collision_prefix(tmp_path):
    c = Collector(tmp_path, 100, 1, max_steps=10, n_beams=3, device="cpu")
    _record(c, 3)
    c.flush(torch.tensor([True]), torch.tensor([False]), torch.tensor([True]),
            torch.zeros(3), filter_failures=False)

    assert c.kept_truncated == 1
    assert c.dropped_short == 0
    assert c.steps_kept == 3
    assert c.pending[0]["outcome"].tolist() == [1, 1, 1]


def test_opt_in_filter_reports_discarded_samples(tmp_path):
    c = Collector(tmp_path, 100, 1, max_steps=10, n_beams=3, device="cpu")
    _record(c, 3)
    c.flush(torch.tensor([True]), torch.tensor([False]), torch.tensor([True]),
            torch.zeros(3), filter_failures=True)

    assert c.episodes_kept == 0
    assert c.dropped_short == 1
    assert c.steps_dropped == 3


def test_shards_are_capped_to_one_episode_of_overshoot(tmp_path):
    c = Collector(tmp_path, 4, 2, max_steps=10, n_beams=3, device="cpu")
    _record(c, 3)
    c.flush(torch.tensor([True, True]), torch.tensor([True, True]),
            torch.tensor([False, False]), torch.zeros(3),
            filter_failures=False)

    assert c.shards_written == 1
    with np.load(tmp_path / "bc_000.npz") as shard:
        assert len(shard["episode"]) == 6
        assert len(shard["episode"]) < c.shard_steps + c.max_steps
        assert shard["outcome"].tolist() == [0] * 6
