"""Streaming access to a corpus of compressed demonstration shards.

The collected corpus is 104 GB, while one decompressed shard is only a few
hundred MB. The useful abstraction is therefore a shuffled stream of shards,
not one giant ``Dataset``. One background thread decompresses the next shard
into host RAM while the GPU trains on the current one. This overlaps the only
meaningful CPU bottleneck without adding multiprocessing queues or duplicating
many gigabytes of data.

Validation holds out whole shards. The collector never splits an episode across
shards, so adjacent timesteps cannot leak across the split. It is still not a
layout-generalization test--the old shards do not store layout IDs--which is why
closed-loop evaluation uses an entirely new layout seed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterator, Sequence
import zipfile

import numpy as np

from ..dataset import Shard, load_shard
from ..spec import fingerprint

__all__ = ["ShardCorpus", "ShardInfo"]


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    samples: int
    bytes: int

    def record(self, root: Path) -> dict[str, int | str]:
        try:
            name = str(self.path.relative_to(root))
        except ValueError:
            name = str(self.path)
        return {"path": name, "samples": self.samples, "bytes": self.bytes}


def _npy_shape(archive: zipfile.ZipFile, member: str) -> tuple[int, ...]:
    """Read an NPY header without inflating the hundreds-of-MB array body."""
    with archive.open(member) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(stream)[0]
        if version in ((2, 0), (3, 0)):
            return np.lib.format.read_array_header_2_0(stream)[0]
        raise ValueError(f"unsupported NPY version {version} in {member}")


def inspect_shard(path: Path) -> ShardInfo:
    with zipfile.ZipFile(path) as archive:
        samples = int(_npy_shape(archive, "ranges.npy")[0])
        with archive.open("fingerprint.npy") as stream:
            got = str(np.lib.format.read_array(stream, allow_pickle=False))
    want = fingerprint()
    if got != want:
        raise ValueError(
            f"shard {path} was collected under spec {got}, this process is "
            f"{want}. Refusing to mix observation contracts."
        )
    return ShardInfo(path=path, samples=samples, bytes=path.stat().st_size)


class ShardCorpus:
    """A deterministic train/validation split over a directory of NPZ shards."""

    def __init__(self, root: str | Path, *, pattern: str = "bc_*.npz",
                 val_shards: int = 4, seed: int = 0,
                 max_train_shards: int | None = None):
        self.root = Path(root).resolve()
        paths = sorted(self.root.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no {pattern!r} shards under {self.root}")

        infos = [inspect_shard(path) for path in paths]
        ranked = sorted(
            infos,
            key=lambda item: hashlib.sha256(
                f"{seed}:{item.path.name}".encode()).digest(),
        )
        # A one-shard smoke corpus remains trainable and simply has no offline
        # validation. Real runs should retain at least one whole shard.
        count = min(max(0, val_shards), max(0, len(infos) - 1))
        validation_paths = {item.path for item in ranked[:count]}
        self.validation = [item for item in infos
                           if item.path in validation_paths]
        train = [item for item in infos if item.path not in validation_paths]

        if max_train_shards is not None:
            if max_train_shards < 1:
                raise ValueError("max_train_shards must be positive")
            # Select by the same stable ranking, then restore filename order.
            train_paths = {item.path for item in train}
            selected = [item for item in ranked if item.path in train_paths]
            keep = {item.path for item in selected[:max_train_shards]}
            train = [item for item in train if item.path in keep]
        self.train = train
        self.seed = seed

        payload = self.as_dict()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @property
    def train_samples(self) -> int:
        return sum(item.samples for item in self.train)

    @property
    def validation_samples(self) -> int:
        return sum(item.samples for item in self.validation)

    def train_order(self, epoch: int) -> list[ShardInfo]:
        order = list(self.train)
        random.Random(self.seed + epoch).shuffle(order)
        return order

    def as_dict(self) -> dict:
        return {
            "root": str(self.root),
            "spec_fingerprint": fingerprint(),
            "seed": self.seed,
            "train": [item.record(self.root) for item in self.train],
            "validation": [item.record(self.root) for item in self.validation],
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
        }

    def loaded(self, infos: Sequence[ShardInfo], device,
               *, prefetch: bool = True) -> Iterator[tuple[ShardInfo, Shard]]:
        """Yield shards on ``device`` while inflating one successor on CPU."""
        if not infos:
            return
        if not prefetch:
            for info in infos:
                yield info, load_shard(info.path, device=device)
            return

        with ThreadPoolExecutor(max_workers=1,
                                thread_name_prefix="shard-prefetch") as pool:
            iterator = iter(infos)
            current = next(iterator)
            future = pool.submit(load_shard, current.path, "cpu")
            while True:
                shard = future.result()
                try:
                    following = next(iterator)
                except StopIteration:
                    following = None
                if following is not None:
                    future = pool.submit(load_shard, following.path, "cpu")
                yield current, shard.to(device)
                if following is None:
                    break
                current = following
