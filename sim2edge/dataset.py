"""Demonstration storage. Raw in, observations out.

The shard holds **raw sensor readings and true poses**, never preprocessed
observations and never noisy ones. That single decision is what makes the
dataset survive design changes: retuning the noise model, changing the bin
count, or renormalizing the goal all become loader-side edits instead of a
recollection run. Only the raw quantities -- what the sensor measured, where
the robot was, what the oracle commanded -- are actually expensive to produce.

The observation is assembled per batch, on device, with fresh noise. Doing it
per sample in a ``Dataset.__getitem__`` would run 180-element torch ops one at a
time and spend more wall clock in kernel launches than in arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np
import torch

from . import noise as noise_mod
from .preprocess import bin_lidar, build_observation, goal_features
from .spec import OMEGA_MAX, V_MAX, fingerprint

__all__ = ["Shard", "save_shard", "load_shard"]


@dataclass
class Shard:
    """One collection run's raw data, resident on whichever device it was loaded to.

    ``action`` is the oracle's *commanded* (v, omega) -- see ``oracle.py`` for
    why it must not be the achieved velocity. ``prev_action`` is stored rather
    than derived, because deriving it means special-casing episode boundaries
    and getting that wrong silently leaks the previous episode's last command
    into the first observation of the next one.
    """

    ranges: torch.Tensor        # (T, B) float16 raw metres, inf where no return
    angles: torch.Tensor        # (B,)   float32 beam bearings, robot frame
    pose: torch.Tensor          # (T, 3) float32 true world x, y, yaw
    goal: torch.Tensor          # (T, 2) float32 world goal
    action: torch.Tensor        # (T, 2) float32 oracle command
    prev_action: torch.Tensor   # (T, 2) float32 command issued the tick before
    episode: torch.Tensor       # (T,)   int32   episode id, for splitting

    def __len__(self) -> int:
        return self.ranges.shape[0]

    def to(self, device) -> "Shard":
        for f in ("ranges", "angles", "pose", "goal", "action", "prev_action",
                  "episode"):
            setattr(self, f, getattr(self, f).to(device))
        return self

    def batches(self, batch_size: int, noise_scale: float = 1.0,
                shuffle: bool = True, generator: torch.Generator | None = None,
                drop_last: bool = False):
        """Yield ``(observation, action)`` with a fresh noise draw every pass.

        Because corruption happens here and not in the simulator, two epochs
        over the same shard show the policy two different corruptions of the
        same geometry. That is free regularization; a dataset with noise baked
        in gets exactly one sample per state, forever.
        """
        n = len(self)
        order = (torch.randperm(n, generator=generator, device=self.ranges.device)
                 if shuffle else torch.arange(n, device=self.ranges.device))

        for i in range(0, n, batch_size):
            sel = order[i:i + batch_size]
            if drop_last and sel.numel() < batch_size:
                break
            ranges = self.ranges[sel].float()

            r, a = noise_mod.corrupt_lidar(ranges, self.angles, noise_scale,
                                           generator)
            pose, goal = noise_mod.corrupt_goal(self.pose[sel], self.goal[sel],
                                                noise_scale, generator)

            obs = build_observation(
                bin_lidar(r, a),
                goal_features(pose, goal),
                _normalize_action(self.prev_action[sel]),
            )
            yield obs, self.action[sel]


def _normalize_action(a: torch.Tensor) -> torch.Tensor:
    """Physical (v, omega) -> [-1, 1], the scale the network sees.

    The *label* stays in SI units -- the exported graph squashes and rescales
    in-graph so the C++ node receives m/s directly -- but the last-action input
    is an activation and has to be conditioned like one.
    """
    return torch.stack([a[:, 0] / V_MAX, a[:, 1] / OMEGA_MAX], dim=1)


def save_shard(path, ranges, angles, pose, goal, action, prev_action, episode,
               **meta) -> None:
    """Write a shard, stamped with the spec fingerprint.

    Ranges go to float16: the A1's own error is 2 cm and float16 resolves better
    than 6 mm out at 12 m, so the storage format is two orders of magnitude
    finer than the sensor. It halves the shard, which is the only part of this
    that is large.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        fingerprint=np.array(fingerprint()),
        ranges=np.asarray(ranges, dtype=np.float16),
        angles=np.asarray(angles, dtype=np.float32),
        pose=np.asarray(pose, dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        prev_action=np.asarray(prev_action, dtype=np.float32),
        episode=np.asarray(episode, dtype=np.int32),
        **meta,
    )
    # The file remains an ordinary np.load-compatible NPZ.  Deflate level 1 is
    # 1.7x faster than NumPy's default level 6 on representative BC scans, for
    # only ~8% more disk, so synchronous shard writes stall simulation less.
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for key, value in arrays.items():
            with archive.open(f"{key}.npy", mode="w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(value),
                                          allow_pickle=False)


def load_shard(path, device="cpu") -> Shard:
    """Load and verify. A fingerprint mismatch raises rather than warns.

    A shard collected at 180 bins fed to a policy expecting 120 has compatible
    looking shapes right until it doesn't, and the resulting failure presents as
    a training problem. Refusing here is the cheapest place to catch it.
    """
    # ``np.load`` owns a ZipFile. Keeping it alive until garbage collection is
    # harmless for one shard and exhausts file descriptors in a 623-shard
    # training pass, so every array is materialized inside the context manager.
    with np.load(Path(path), allow_pickle=False) as z:
        got, want = str(z["fingerprint"]), fingerprint()
        if got != want:
            raise ValueError(
                f"shard {path} was collected under spec {got}, this process is "
                f"{want}. The observation contract changed; recollect or check "
                f"out the spec that produced it."
            )
        t = lambda k, d=torch.float32: torch.from_numpy(z[k]).to(
            device=device, dtype=d)
        return Shard(
            ranges=t("ranges", torch.float16), angles=t("angles"), pose=t("pose"),
            goal=t("goal"), action=t("action"), prev_action=t("prev_action"),
            episode=t("episode", torch.int32),
        )
