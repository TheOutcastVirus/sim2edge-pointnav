"""Dump real observations for INT8 calibration of the deployed actor.

Quantization observers need the state distribution the policy actually meets,
not random tensors: the lidar block is a bounded, heavily structured signal and
its per-channel range decides the scale every downstream activation inherits.

The provenance of the samples is written alongside them because it is the first
thing to suspect when the quantization gate fails. In particular ``driver``
records whether these observations came from the oracle or from a policy — an
oracle corpus visits a different state distribution than the deployed actor and
is a known-weaker calibration source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sim2edge import spec
from sim2edge.dataset import load_shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="corpus directory containing train/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--shards", type=int, default=8,
                        help="shards to draw from; spreading samples across "
                             "shards covers more layouts than depth in one")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    split = args.data / "train"
    paths = sorted(split.glob("bc_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {split}")
    paths = paths[:args.shards]
    per_shard = max(1, args.samples // len(paths))

    generator = torch.Generator().manual_seed(args.seed)
    chunks: list[torch.Tensor] = []
    drivers: set[str] = set()
    contracts: set[str] = set()
    for path in paths:
        shard = load_shard(path, device="cpu")
        drivers.add(str(shard.driver))
        contracts.add(str(shard.sensor_contract_version))
        for observation, _action in shard.batches(
                per_shard, noise_scale=args.noise_scale, shuffle=True,
                generator=generator):
            chunks.append(observation.to(torch.float32))
            break  # one batch per shard keeps layout coverage wide and RAM flat

    observations = torch.cat(chunks)[:args.samples]
    if observations.shape[1] != spec.OBS_DIM:
        raise SystemExit(
            f"assembled width {observations.shape[1]} != spec OBS_DIM "
            f"{spec.OBS_DIM}; the corpus and the spec disagree")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": str(args.data.resolve()),
        "shards": [p.name for p in paths],
        "samples": int(observations.shape[0]),
        "observation_dim": int(observations.shape[1]),
        "noise_scale": args.noise_scale,
        "seed": args.seed,
        "driver": sorted(drivers),
        "sensor_contract_version": sorted(contracts),
        "spec_fingerprint": spec.fingerprint(),
        "observation_fingerprint": spec.observation_fingerprint(),
    }
    np.savez_compressed(args.out, observations=observations.numpy(),
                        meta=json.dumps(meta))
    print(f"[calib] {observations.shape[0]} observations "
          f"x {observations.shape[1]} -> {args.out}")
    print(f"[calib] driver={sorted(drivers)} noise_scale={args.noise_scale}")
    if drivers != {"policy"}:
        print("[calib] WARNING: not a policy-driven corpus; the deployed "
              "actor's own state distribution is the stronger calibration "
              "source if the quantization gate fails")


if __name__ == "__main__":
    main()
