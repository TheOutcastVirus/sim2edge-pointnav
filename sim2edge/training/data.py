"""Streaming access to a corpus of compressed demonstration shards.

The collected corpus is 104 GB, while one decompressed shard is only about
300 MB. The useful abstraction is therefore a shuffled stream of shards, not
one giant ``Dataset``. A small thread pool inflates several future shards into a
bounded RAM queue while the GPU trains on the current one. ZIP's zlib decoder
releases the GIL, so independent shards really do decompress concurrently; a
process pool would only add a 300 MB IPC copy per shard.

New corpora arrive with explicit ``train/`` and ``validation/`` directories;
the collector assigns whole layout IDs to one side and this loader verifies
that they do not overlap. Legacy flat corpora still hold out whole shards. Those
old shards do not store layout IDs, so only their closed-loop evaluation on a
new layout seed measures layout generalization.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
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
            f"{want}. Refusing to mix raw sensor contracts."
        )
    return ShardInfo(path=path, samples=samples, bytes=path.stat().st_size)


def _inspect_layout_split(paths: Sequence[Path], expected: str) -> set[tuple[int, int]]:
    """Return stable ``(layout_seed, layout_id)`` keys and verify provenance."""
    keys: set[tuple[int, int]] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            if "layout" not in shard.files or "layout_seed" not in shard.files:
                raise ValueError(f"explicit split shard {path} lacks layout metadata")
            if "split" in shard.files and str(shard["split"]) != expected:
                raise ValueError(
                    f"shard {path} says split={shard['split']}, expected {expected}")
            seed = int(shard["layout_seed"])
            keys.update((seed, int(value)) for value in np.unique(shard["layout"]))
    return keys


class ShardCorpus:
    """A deterministic train/validation split over a directory of NPZ shards."""

    def __init__(self, root: str | Path, *, pattern: str = "bc_*.npz",
                 val_shards: int = 4, seed: int = 0,
                 max_train_shards: int | None = None):
        self.root = Path(root).resolve()
        explicit_train = sorted((self.root / "train").glob(pattern))
        explicit_validation = sorted((self.root / "validation").glob(pattern))
        explicit_split = bool(explicit_train or explicit_validation)
        self.explicit_split = explicit_split
        if explicit_split:
            if not explicit_train or not explicit_validation:
                raise FileNotFoundError(
                    f"explicit corpus under {self.root} needs both train/ and "
                    f"validation/ shards")
            train_infos = [inspect_shard(path) for path in explicit_train]
            validation_infos = [inspect_shard(path)
                                for path in explicit_validation]
            train_layouts = _inspect_layout_split(explicit_train, "train")
            validation_layouts = _inspect_layout_split(
                explicit_validation, "validation")
            overlap = train_layouts & validation_layouts
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(
                    f"layout leakage across train/validation: {sample}")
            infos = train_infos + validation_infos
        else:
            paths = sorted(self.root.glob(pattern))
            if not paths:
                raise FileNotFoundError(
                    f"no {pattern!r} shards under {self.root}")
            infos = [inspect_shard(path) for path in paths]
        ranked = sorted(
            infos,
            key=lambda item: hashlib.sha256(
                f"{seed}:{item.path.name}".encode()).digest(),
        )
        # A one-shard smoke corpus remains trainable and simply has no offline
        # validation. Real runs should retain at least one whole shard.
        if explicit_split:
            self.validation = validation_infos
            train = train_infos
        else:
            count = min(max(0, val_shards), max(0, len(infos) - 1))
            validation_paths = {item.path for item in ranked[:count]}
            self.validation = [item for item in infos
                               if item.path in validation_paths]
            train = [item for item in infos
                     if item.path not in validation_paths]

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
        # Cumulative wall time spent waiting for the next decompressed shard.
        # The trainer snapshots this around each epoch so an input bottleneck is
        # visible in metrics rather than inferred from low GPU utilization.
        self.input_wait_seconds = 0.0

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
        value = {
            "root": str(self.root),
            "spec_fingerprint": fingerprint(),
            "seed": self.seed,
            "train": [item.record(self.root) for item in self.train],
            "validation": [item.record(self.root) for item in self.validation],
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
        }
        if self.explicit_split:
            value["split"] = "layout"
        return value

    def loaded(self, infos: Sequence[ShardInfo], device,
               *, prefetch: bool = True, workers: int = 4,
               buffer_size: int = 8) -> Iterator[tuple[ShardInfo, Shard]]:
        """Yield ordered shards while decompressing a bounded window in parallel.

        ``workers=4, buffer_size=8`` occupies roughly 2.4 GB when the queue is
        full--small on the 128 GB host, but deep enough to absorb variation in
        compression ratio. Results are consumed in input order, so changing
        worker timing cannot change training order or reproducibility.
        """
        if not infos:
            return
        if not prefetch:
            for info in infos:
                started = time.perf_counter()
                shard = load_shard(info.path, device=device)
                self.input_wait_seconds += time.perf_counter() - started
                yield info, shard
            return
        if workers < 1:
            raise ValueError("prefetch workers must be positive")
        if buffer_size < workers:
            raise ValueError("prefetch buffer_size must be at least workers")

        pool = ThreadPoolExecutor(max_workers=workers,
                                  thread_name_prefix="shard-prefetch")
        pending = deque()
        iterator = iter(infos)

        def fill() -> None:
            while len(pending) < buffer_size:
                try:
                    info = next(iterator)
                except StopIteration:
                    break
                pending.append((
                    info, pool.submit(load_shard, info.path, "cpu")))

        fill()
        try:
            while pending:
                info, future = pending.popleft()
                started = time.perf_counter()
                shard = future.result()
                self.input_wait_seconds += time.perf_counter() - started
                # Refill before the host-to-device copy so decompression also
                # overlaps that transfer and observation construction.
                fill()
                yield info, shard.to(device)
        finally:
            for _, future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
