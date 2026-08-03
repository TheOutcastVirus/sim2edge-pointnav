"""Compose an actor checkpoint with a separately validated belief estimator.

The two stages optimize different evidence: actor checkpoints are selected by
held-out navigation, while estimator checkpoints are selected by paired motion
error and lidar ablations.  This utility creates an inference checkpoint without
pretending that either source's optimizer state remains resumable.

    .venv/bin/python scripts/compose_belief_checkpoint.py \
      --actor runs/bc_swept_dagger_v4/best.pt \
      --estimator runs/belief_motion_v4/best.pt \
      --out runs/belief_dagger_v4/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2edge.spec import (MOTION_PRIOR_SOURCE, SENSOR_CONTRACT_VERSION,
                           fingerprint, observation_fingerprint)
from sim2edge.training import load_belief_checkpoint, load_policy_checkpoint
from sim2edge.training.bc import atomic_torch_save


def compose(actor_path: Path, estimator_path: Path, output: Path) -> dict:
    actor = load_policy_checkpoint(actor_path)
    policy = load_belief_checkpoint(estimator_path)
    if actor.config != policy.config.actor:
        raise ValueError("actor and belief estimator use different actor architectures")
    policy.actor.load_state_dict(actor.state_dict())
    payload = {
        "format_version": 2,
        "algorithm": "belief-composed",
        "spec_fingerprint": fingerprint(),
        "sensor_contract_version": SENSOR_CONTRACT_VERSION,
        "motion_prior_source": MOTION_PRIOR_SOURCE,
        "observation_fingerprint": observation_fingerprint(
            policy.config.actor.history_frames, belief=True),
        "model_config": policy.config.to_dict(),
        "model": policy.state_dict(),
        "actor_source": str(actor_path.resolve()),
        "estimator_source": str(estimator_path.resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--estimator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compose(args.actor, args.estimator, args.out)
    print(f"[compose] actor {args.actor}")
    print(f"[compose] estimator {args.estimator}")
    print(f"[compose] wrote {args.out}")


if __name__ == "__main__":
    main()
