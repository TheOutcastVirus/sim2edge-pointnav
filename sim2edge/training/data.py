"""Demonstration storage and collection for behaviour cloning and DAgger.

**Storage format.** Observations are stored quantized: depth as ``uint8`` and
lidar as ``uint16`` over their normalized [0, 1] ranges; everything else as
float32. This is not a cosmetic saving. A single depth frame is 61 KB as float32,
so a 10k-episode dataset at 20 Hz is ~200 GB and simply cannot be collected.
As ``uint8`` it is 15 KB and the same dataset is ~50 GB.

The precision loss is 1/255 of the clipped depth range, i.e. 2.2 cm — smaller
than the depth sensor's own noise at any useful distance, and *smaller than what
INT8 quantization will do to this tensor at inference anyway*. Training on
exactly the precision the deployed model will see is a feature.

**Collection supports DAgger directly.** :func:`collect_episodes` takes the
policy that *drives* and always labels with the oracle, so passing the oracle
gives plain behaviour cloning and passing the current policy gives a DAgger
round, with no separate code path.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..envs.reference_env import EnvConfig, NavEnv
from ..oracle.expert import PurePursuitExpert
from ..spec import DEFAULT_SPEC, NavSpec

__all__ = ["Demonstrations", "collect_episodes", "DemoDataset", "save_shard",
           "load_shards"]


def _q8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def _dq8(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255.0


def _q16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 65535.0), 0, 65535).astype(np.uint16)


def _dq16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 65535.0


@dataclass
class Demonstrations:
    """A flat, quantized set of (observation, expert action) pairs.

    ``depth`` may be stored with a leading dimension of 0, which means "constant
    zeros" — the lidar-only configuration. That configuration still feeds a
    depth tensor to the network (the exported graph's signature must never
    change between configurations), but materializing 15 KB of zeros per sample
    would cost ~70 GB of RAM for a 10k-episode dataset to store no information
    at all. :meth:`dequantized` synthesizes the zeros on demand instead.
    """

    depth: np.ndarray        # (N, 1, H, W) uint8, or (0, 1, H, W) for "zeros"
    lidar: np.ndarray        # (N, bins)    uint16
    goal: np.ndarray         # (N, 3)       float32
    last_action: np.ndarray  # (N, 2)       float32
    action: np.ndarray       # (N, 2)       float32, expert command, SI units
    privileged: np.ndarray   # (N, P)       float32

    def __len__(self) -> int:
        return int(self.action.shape[0])

    @property
    def has_depth(self) -> bool:
        return self.depth.shape[0] > 0

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.depth, self.lidar, self.goal,
                                      self.last_action, self.action,
                                      self.privileged))

    @classmethod
    def empty(cls, spec: NavSpec = DEFAULT_SPEC) -> "Demonstrations":
        d = spec.obs.depth
        return cls(
            np.zeros((0, 1, d.height, d.width), np.uint8),
            np.zeros((0, spec.obs.lidar.bins), np.uint16),
            np.zeros((0, 3), np.float32),
            np.zeros((0, 2), np.float32),
            np.zeros((0, 2), np.float32),
            np.zeros((0, spec.obs.privileged.dim), np.float32),
        )

    @classmethod
    def concat(cls, parts: Sequence["Demonstrations"]) -> "Demonstrations":
        parts = [p for p in parts if len(p)]
        if not parts:
            raise ValueError("no non-empty demonstrations to concatenate")
        if len({p.has_depth for p in parts}) > 1:
            raise ValueError(
                "cannot concatenate depth-bearing and lidar-only demonstrations; "
                "they were collected under different env configurations")
        fields = ["lidar", "goal", "last_action", "action", "privileged"]
        merged = {f: np.concatenate([getattr(p, f) for p in parts], axis=0)
                  for f in fields}
        depth = (np.concatenate([p.depth for p in parts], axis=0)
                 if parts[0].has_depth else parts[0].depth)
        return cls(depth=depth, **merged)

    def dequantized(self, idx) -> Dict[str, np.ndarray]:
        """Decode a slice back to the float32 observation the policy consumes."""
        if self.has_depth:
            depth = _dq8(self.depth[idx])
        else:
            n = len(idx) if hasattr(idx, "__len__") else 1
            depth = np.zeros((n,) + self.depth.shape[1:], dtype=np.float32)
        return {
            "depth": depth,
            "lidar": _dq16(self.lidar[idx]),
            "goal": self.goal[idx],
            "last_action": self.last_action[idx],
            "privileged": self.privileged[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collection
# ─────────────────────────────────────────────────────────────────────────────


def collect_episodes(
    n_episodes: int,
    config: Optional[EnvConfig] = None,
    spec: NavSpec = DEFAULT_SPEC,
    seed_base: int = 0,
    driver=None,
    beta: float = 1.0,
    keep_failures: bool = False,
    progress_every: int = 0,
    env: Optional[NavEnv] = None,
) -> Tuple[Demonstrations, Dict[str, float]]:
    """Roll out episodes and record ``(observation, oracle action)`` pairs.

    ``driver`` is the policy that actually controls the robot. ``None`` means
    the oracle drives, which is plain behaviour cloning. Passing a trained
    policy makes this a DAgger round: the states visited are the *policy's*
    state distribution while the labels remain the oracle's actions, which is
    the entire point — a BC policy fails by drifting into states the expert
    never visited, and no amount of additional on-distribution expert data fixes
    that.

    ``beta`` mixes the two: with probability ``beta`` the oracle drives this
    step. ``beta=1`` is pure BC, ``beta=0`` is pure policy-driven DAgger.

    ``keep_failures`` keeps episodes that collided or timed out. Off by default
    for the initial BC set (imitating a trajectory that ends in a wall is
    actively harmful) but worth enabling for later DAgger rounds, where the
    interesting labels are precisely the recoveries near failure.
    """
    cfg = config or EnvConfig(spec=spec)
    if env is None:
        env = NavEnv(cfg, seed=seed_base)
    rng = np.random.default_rng(seed_base + 991)
    max_steps = cfg.max_episode_steps or spec.reward.max_episode_steps

    parts: List[Demonstrations] = []
    n_success = n_collision = n_timeout = n_nopath = 0
    total_steps = 0

    for i in range(n_episodes):
        obs = env.reset(seed=seed_base + i)
        expert = PurePursuitExpert(env.room, spec)
        if not expert.has_path:
            n_nopath += 1
            continue

        depth_l, lidar_l, goal_l, la_l, act_l, priv_l = [], [], [], [], [], []
        result = None

        for _ in range(max_steps):
            cmd = expert.act(*env.pose)
            expert_action = np.array([cmd.v, cmd.omega], dtype=np.float32)

            # Record the expert label for the state actually visited.
            if cfg.use_depth:
                depth_l.append(_q8(obs["depth"]))
            lidar_l.append(_q16(obs["lidar"]))
            goal_l.append(obs["goal"])
            la_l.append(obs["last_action"])
            act_l.append(expert_action)
            priv_l.append(obs["privileged"])

            if driver is None or rng.random() < beta:
                action = expert_action
            else:
                action = np.asarray(driver(obs), dtype=np.float32)

            result = env.step(action)
            obs = result.obs
            if result.terminated or result.truncated:
                break

        assert result is not None
        info = result.info
        ok = bool(info["success"])
        n_success += ok
        n_collision += bool(info["collision"])
        n_timeout += (not ok and not info["collision"])

        if ok or keep_failures:
            d = spec.obs.depth
            depth_arr = (np.stack(depth_l) if cfg.use_depth
                         else np.zeros((0, 1, d.height, d.width), np.uint8))
            parts.append(Demonstrations(
                depth_arr, np.stack(lidar_l), np.stack(goal_l),
                np.stack(la_l), np.stack(act_l), np.stack(priv_l),
            ))
            total_steps += len(act_l)

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [{i + 1}/{n_episodes}] kept {total_steps} steps, "
                  f"expert SR {n_success / (i + 1):.1%}")

    demos = Demonstrations.concat(parts) if parts else Demonstrations.empty(spec)
    n = max(1, n_episodes)
    stats = {
        "episodes": float(n_episodes),
        "steps": float(len(demos)),
        "success_rate": n_success / n,
        "collision_rate": n_collision / n,
        "timeout_rate": n_timeout / n,
        "no_path_rate": n_nopath / n,
        "megabytes": demos.nbytes / 1e6,
    }
    return demos, stats


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────


def save_shard(demos: Demonstrations, path: str | pathlib.Path) -> None:
    """Write one shard. Sharding keeps any single file loadable into RAM."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        depth=demos.depth, lidar=demos.lidar, goal=demos.goal,
        last_action=demos.last_action, action=demos.action,
        privileged=demos.privileged,
        spec_fingerprint=np.array(DEFAULT_SPEC.fingerprint()),
    )


def load_shards(paths: Sequence[str | pathlib.Path],
                check_fingerprint: bool = True) -> Demonstrations:
    """Load and concatenate shards, refusing data from a different contract.

    The fingerprint check is not paranoia: a dataset collected before a change
    to the depth clip range or the lidar bin count is silently wrong rather than
    obviously wrong, and it would train a policy that looks fine in sim and is
    subtly miscalibrated everywhere else.
    """
    parts = []
    want = DEFAULT_SPEC.fingerprint()
    for p in paths:
        with np.load(p) as z:
            got = str(z["spec_fingerprint"]) if "spec_fingerprint" in z else None
            if check_fingerprint and got != want:
                raise ValueError(
                    f"{p} was collected under spec fingerprint {got!r} but the "
                    f"current spec is {want!r}. Re-collect, or pass "
                    f"check_fingerprint=False if you are certain this is safe.")
            parts.append(Demonstrations(
                z["depth"], z["lidar"], z["goal"], z["last_action"],
                z["action"], z["privileged"]))
    return Demonstrations.concat(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Torch dataset
# ─────────────────────────────────────────────────────────────────────────────


class DemoDataset:
    """Minimal map-style dataset over :class:`Demonstrations`.

    Deliberately not a ``torch.utils.data.Dataset`` subclass at import time —
    this module is imported by tooling that has numpy but may not have torch.
    :meth:`torch_dataset` builds the torch-facing wrapper on demand.
    """

    def __init__(self, demos: Demonstrations, spec: NavSpec = DEFAULT_SPEC):
        self.demos = demos
        self.spec = spec

    def __len__(self) -> int:
        return len(self.demos)

    def batches(self, batch_size: int, shuffle: bool = True,
                rng: Optional[np.random.Generator] = None,
                drop_last: bool = False) -> Iterator[Dict[str, np.ndarray]]:
        """Yield dequantized numpy batches."""
        n = len(self.demos)
        idx = np.arange(n)
        if shuffle:
            (rng or np.random.default_rng()).shuffle(idx)
        stop = (n // batch_size) * batch_size if drop_last else n
        for start in range(0, stop, batch_size):
            sel = idx[start:start + batch_size]
            batch = self.demos.dequantized(sel)
            batch["action"] = self.demos.action[sel]
            yield batch

    def split(self, val_fraction: float = 0.05,
              seed: int = 0) -> Tuple["DemoDataset", "DemoDataset"]:
        """Random train/val split over samples."""
        n = len(self.demos)
        idx = np.random.default_rng(seed).permutation(n)
        cut = int(n * (1.0 - val_fraction))
        d = self.demos

        def pick(s):
            return Demonstrations(
                d.depth[s] if d.has_depth else d.depth,
                d.lidar[s], d.goal[s], d.last_action[s], d.action[s],
                d.privileged[s])

        return (DemoDataset(pick(idx[:cut]), self.spec),
                DemoDataset(pick(idx[cut:]), self.spec))
