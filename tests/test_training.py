"""Corpus splitting, action loss and a complete tiny BC run."""

import math
from dataclasses import replace
import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sim2edge.dataset import save_shard
from sim2edge.spec import OMEGA_MAX, V_MAX
from sim2edge.training import (BCConfig, ShardCorpus, load_policy_checkpoint,
                               train_bc)
from sim2edge.training.bc import imitation_loss


def _write_shards(root, count=5, steps=64, beams=90):
    rng = np.random.default_rng(42)
    angles = np.linspace(0, 2 * math.pi, beams, endpoint=False)
    for shard_index in range(count):
        pose = np.zeros((steps, 3), np.float32)
        pose[:, 0] = np.linspace(0.0, 1.0, steps)
        goal = np.tile(np.array([2.0, 0.0], np.float32), (steps, 1))
        action = np.tile(np.array([0.3, 0.0], np.float32), (steps, 1))
        previous = np.vstack((np.zeros((1, 2), np.float32), action[:-1]))
        # Four complete episodes per shard. IDs need not be globally unique for
        # a shard-level split, though the real collector makes them so.
        episode = np.repeat(np.arange(4, dtype=np.int32), steps // 4)
        save_shard(
            root / f"bc_{shard_index:03d}.npz",
            ranges=rng.uniform(0.3, 5.0, (steps, beams)), angles=angles,
            pose=pose, goal=goal, action=action, prev_action=previous,
            episode=episode, outcome=np.zeros(steps, np.uint8),
        )


def test_normalized_loss_gives_equal_errors_equal_weight():
    target = torch.tensor([[V_MAX / 2, 0.0]])
    prediction = torch.tensor([[V_MAX * 0.6, OMEGA_MAX * 0.2]])
    _, v_loss, omega_loss = imitation_loss(prediction, target, omega_weight=1.0)
    assert v_loss == pytest.approx(omega_loss, rel=1e-6)


def test_corpus_split_is_deterministic_and_whole_shard(tmp_path):
    _write_shards(tmp_path)
    first = ShardCorpus(tmp_path, val_shards=2, seed=7)
    second = ShardCorpus(tmp_path, val_shards=2, seed=7)
    assert [x.path for x in first.validation] == [x.path for x in second.validation]
    assert first.digest == second.digest
    assert len(first.train) == 3 and len(first.validation) == 2
    assert {x.path for x in first.train}.isdisjoint(
        {x.path for x in first.validation})
    assert first.train_samples == 3 * 64


def test_prefetched_corpus_yields_each_shard_once(tmp_path):
    _write_shards(tmp_path, count=3)
    corpus = ShardCorpus(tmp_path, val_shards=1)
    loaded = list(corpus.loaded(corpus.train, "cpu", prefetch=True))
    assert [info.path for info, _ in loaded] == [info.path for info in corpus.train]
    assert all(len(shard) == 64 for _, shard in loaded)


def test_prefetch_decompresses_multiple_shards_concurrently(tmp_path,
                                                             monkeypatch):
    _write_shards(tmp_path, count=6)
    corpus = ShardCorpus(tmp_path, val_shards=1)
    lock = threading.Lock()
    active = peak = 0

    class FakeShard:
        def to(self, device):
            return self

    def fake_load(path, device="cpu"):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return FakeShard()

    monkeypatch.setattr("sim2edge.training.data.load_shard", fake_load)
    loaded = list(corpus.loaded(corpus.train, "cpu", workers=3,
                                buffer_size=4))
    assert peak >= 2
    assert [info.path for info, _ in loaded] == [x.path for x in corpus.train]


def test_tiny_training_run_writes_loadable_transparent_artifacts(tmp_path):
    data = tmp_path / "data"
    output = tmp_path / "run"
    data.mkdir()
    _write_shards(data, count=3)
    config = replace(
        BCConfig(), epochs=1, batch_size=32, learning_rate=1e-3,
        warmup_steps=0, val_shards=1, train_noise_scale=0.0,
        compile_model=False, amp="off", log_every=0,
        checkpoint_every_shards=1,
    )
    result = train_bc(data, output, config, device="cpu")

    assert result["epoch"] == 1
    assert not any(isinstance(value, np.generic) for value in result.values())
    assert (output / "config.json").is_file()
    assert (output / "dataset.json").is_file()
    assert (output / "metrics.jsonl").is_file()
    assert (output / "best.pt").is_file()
    assert (output / "final.pt").is_file()
    policy = load_policy_checkpoint(output / "final.pt")
    assert policy(torch.zeros(2, 185)).shape == (2, 2)

    # Reissuing resume after completion is idempotent, which is useful in job
    # scripts whose worker status can be lost independently of the checkpoint.
    legacy = torch.load(output / "latest.pt", weights_only=False)
    for key in ("prefetch_workers", "prefetch_shards"):
        legacy["training_config"].pop(key)
    legacy_path = output / "legacy.pt"
    torch.save(legacy, legacy_path)
    resumed = train_bc(data, output, config, device="cpu", resume=legacy_path)
    assert resumed["epoch"] == 1
