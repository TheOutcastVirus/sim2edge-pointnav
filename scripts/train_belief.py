"""Train the goal-belief policy on the shard corpus. No Isaac required.

    .venv/bin/python scripts/train_belief.py \
        --data data/overnight_v2 \
        --warm-start runs/ppo_v1/best.pt \
        --out runs/belief_v1

The actor arrives trained; the odometry head arrives at the dead-reckoning
baseline. What this run buys is the head, and the number to watch is
``belief_error_final_m`` -- mean metres of goal-belief error after a full
``seq_len`` window of the model integrating its own estimate. Dead reckoning
alone is the value at step zero of training; anything above that is the lidar
doing work.

Sanity floor: the goal tolerance is 0.25 m, and a belief error near that is a
policy that cannot reliably tell it has arrived.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2edge.models import BeliefNavPolicy, BeliefNavPolicyConfig
from sim2edge.training import ShardCorpus, BeliefConfig, belief_losses, warm_start
from sim2edge.training.bc import atomic_torch_save
from sim2edge.training.belief import (belief_checkpoint, lr_multiplier,
                                      _parameter_groups)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path,
                        help="NavPolicy checkpoint to initialize the actor from")
    parser.add_argument("--epochs", type=int, default=BeliefConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=BeliefConfig.batch_size)
    parser.add_argument("--seq-len", type=int, default=BeliefConfig.seq_len)
    parser.add_argument("--learning-rate", type=float,
                        default=BeliefConfig.learning_rate)
    parser.add_argument("--freeze-actor", action="store_true",
                        help="train only the odometry head")
    parser.add_argument("--max-train-shards", type=int)
    parser.add_argument("--val-shards", type=int, default=BeliefConfig.val_shards)
    parser.add_argument("--seed", type=int, default=BeliefConfig.seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--max-steps", type=int,
                        help="stop early; for smoke tests")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BeliefConfig(
        epochs=args.epochs, batch_size=args.batch_size, seq_len=args.seq_len,
        learning_rate=args.learning_rate, freeze_actor=args.freeze_actor,
        val_shards=args.val_shards, seed=args.seed,
        max_train_shards=args.max_train_shards)
    device = torch.device(args.device)
    sys.stdout.reconfigure(line_buffering=True)

    corpus = ShardCorpus(args.data, val_shards=config.val_shards,
                         seed=config.seed,
                         max_train_shards=config.max_train_shards)
    print(f"[belief] corpus {args.data}: {len(corpus.train)} train shards "
          f"({corpus.train_samples:,} samples), "
          f"{len(corpus.validation)} validation")

    if args.warm_start:
        policy = warm_start(args.warm_start, device=device)
        print(f"[belief] warm-started actor from {args.warm_start}")
    else:
        policy = BeliefNavPolicy(BeliefNavPolicyConfig()).to(device)
        print("[belief] cold actor -- expect this to need far more data")
    print(f"[belief] {policy.parameter_count:,} parameters "
          f"({policy.odometry_parameter_count:,} in the new odometry head)")

    optimizer = torch.optim.AdamW(_parameter_groups(policy, config),
                                  weight_decay=config.weight_decay)
    # Windows per shard is data-dependent; estimate for the schedule and let the
    # cosine clamp handle an over- or under-shoot rather than pretending to know.
    windows = max(1, corpus.train_samples // config.batch_size)
    total_steps = args.max_steps or (windows * config.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, total_steps, config))

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "metrics.jsonl"
    metrics = metrics_path.open("a")
    generator = torch.Generator(device=device.type).manual_seed(config.seed)

    step, best = 0, float("inf")
    started = time.perf_counter()
    stop = False
    for epoch in range(config.epochs):
        for info, shard in corpus.loaded(corpus.train_order(epoch), device,
                                         prefetch=config.prefetch,
                                         workers=config.prefetch_workers,
                                         buffer_size=config.prefetch_shards):
            for batch in shard.sequence_batches(
                    config.batch_size, config.seq_len,
                    noise_scale=config.train_noise_scale, generator=generator):
                policy.train()
                with torch.autocast(device_type=device.type,
                                    dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda"
                                             and config.amp == "bf16")):
                    loss, parts = belief_losses(policy, batch, config)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                               config.grad_clip)
                optimizer.step()
                scheduler.step()
                step += 1

                if step % config.log_every == 0 or step == 1:
                    record = {k: float(v) for k, v in parts.items()}
                    record.update(step=step, epoch=epoch,
                                  shard=info.path.name,
                                  lr=optimizer.param_groups[0]["lr"],
                                  elapsed_seconds=time.perf_counter() - started)
                    metrics.write(json.dumps(record, sort_keys=True) + "\n")
                    metrics.flush()
                    print(f"[belief] step {step:6d}  loss {record['loss']:.4f}  "
                          f"action {record['action_loss']:.4f}  "
                          f"belief_err {record['belief_error_m']*100:.2f} cm  "
                          f"final {record['belief_error_final_m']*100:.2f} cm  "
                          f"drift/m {record['drift_per_m']*100:.2f} cm/m")
                    if record["belief_error_final_m"] < best:
                        best = record["belief_error_final_m"]
                        atomic_torch_save(
                            belief_checkpoint(
                                policy, optimizer, scheduler, config, corpus,
                                epoch=epoch, global_step=step,
                                best_validation=best,
                                warm_start_path=str(args.warm_start)
                                if args.warm_start else None),
                            args.out / "best.pt")
                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    atomic_torch_save(
        belief_checkpoint(policy, optimizer, scheduler, config, corpus,
                          epoch=config.epochs - 1, global_step=step,
                          best_validation=best,
                          warm_start_path=str(args.warm_start)
                          if args.warm_start else None),
        args.out / "final.pt")
    metrics.close()
    print(f"[belief] wrote {args.out}/final.pt after {step} steps "
          f"({time.perf_counter() - started:.1f} s); "
          f"best final-step belief error {best*100:.2f} cm")


if __name__ == "__main__":
    main()
