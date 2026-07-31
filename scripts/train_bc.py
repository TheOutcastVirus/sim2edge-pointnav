"""Train the lidar navigation policy from collected BC shards.

Full 4090 run::

    .venv/bin/python scripts/train_bc.py data/overnight_v2 --out runs/bc_v1

Fast pipeline check::

    .venv/bin/python scripts/train_bc.py data/overnight_v2 --out runs/bc_smoke \
        --epochs 1 --max-train-shards 2 --val-shards 1 --no-compile
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from sim2edge.training import BCConfig, train_bc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/bc"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=BCConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=BCConfig.batch_size)
    parser.add_argument("--lr", type=float, default=BCConfig.learning_rate)
    parser.add_argument("--collision-tail-steps", type=int,
                        default=BCConfig.collision_tail_steps)
    parser.add_argument("--collision-tail-weight", type=float,
                        default=BCConfig.collision_tail_weight)
    parser.add_argument("--near-obstacle-m", type=float,
                        default=BCConfig.near_obstacle_m)
    parser.add_argument("--near-obstacle-weight", type=float,
                        default=BCConfig.near_obstacle_weight)
    parser.add_argument("--near-obstacle-overspeed-weight", type=float,
                        default=BCConfig.near_obstacle_overspeed_weight)
    parser.add_argument("--policy-data-weight", type=float,
                        default=BCConfig.policy_data_weight)
    parser.add_argument("--val-shards", type=int, default=BCConfig.val_shards)
    parser.add_argument("--max-train-shards", type=int)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=BCConfig.compile_model)
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction,
                        default=BCConfig.prefetch)
    parser.add_argument("--prefetch-workers", type=int,
                        default=BCConfig.prefetch_workers)
    parser.add_argument("--prefetch-shards", type=int,
                        default=BCConfig.prefetch_shards,
                        help="bounded number of decompressed shards held in RAM")
    parser.add_argument("--amp", choices=("bf16", "off"), default=BCConfig.amp)
    parser.add_argument("--seed", type=int, default=BCConfig.seed)
    args = parser.parse_args()

    config = replace(
        BCConfig(), epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, val_shards=args.val_shards,
        collision_tail_steps=args.collision_tail_steps,
        collision_tail_weight=args.collision_tail_weight,
        near_obstacle_m=args.near_obstacle_m,
        near_obstacle_weight=args.near_obstacle_weight,
        near_obstacle_overspeed_weight=args.near_obstacle_overspeed_weight,
        policy_data_weight=args.policy_data_weight,
        max_train_shards=args.max_train_shards,
        compile_model=args.compile, prefetch=args.prefetch, amp=args.amp,
        prefetch_workers=args.prefetch_workers,
        prefetch_shards=args.prefetch_shards, seed=args.seed,
    )
    train_bc(args.data, args.out, config, device=args.device,
             resume=args.resume)


if __name__ == "__main__":
    main()
