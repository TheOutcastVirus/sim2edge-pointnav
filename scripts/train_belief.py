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
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2edge.models import (BeliefNavPolicy, BeliefNavPolicyConfig,
                             OdometryHead)
from sim2edge.training import (ShardCorpus, BeliefConfig, belief_losses,
                               load_belief_checkpoint, warm_start)
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
    parser.add_argument("--belief-start", type=Path,
                        help="belief checkpoint to initialize both actor and estimator")
    parser.add_argument("--resume", type=Path,
                        help="resume model/optimizer from a periodic belief checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=200,
                        help="write OUT/latest.pt every N optimizer steps (0 disables)")
    parser.add_argument("--epochs", type=int, default=BeliefConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=BeliefConfig.batch_size)
    parser.add_argument("--seq-len", type=int, default=BeliefConfig.seq_len)
    parser.add_argument("--learning-rate", type=float,
                        default=BeliefConfig.learning_rate)
    parser.add_argument("--train-actor", action="store_true",
                        help="also apply offline action loss (motion stage freezes it)")
    parser.add_argument("--freeze-estimator", action="store_true",
                        help="actor-only belief-distribution adaptation; requires "
                             "--belief-start and --train-actor")
    parser.add_argument("--freeze-actor", action="store_true",
                        help="compatibility spelling; this is already the default")
    parser.add_argument(
        "--persistent-calibration", action="store_true",
        help="carry a bounded forward wheel-scale state across each episode")
    parser.add_argument(
        "--motion-state-size", type=int, default=0,
        help="learn an explicit recurrent motion state of this width (0 disables)")
    parser.add_argument(
        "--derotate-scan-history", action="store_true",
        help="gyro-align the scan history into the newest frame before encoding")
    parser.add_argument(
        "--scaled-scan-difference", action="store_true",
        help="mask and rescale the cross-frame range difference channel")
    parser.add_argument(
        "--scan-vote-bins", type=int, default=0,
        help="enable the consensus scan vote on a BINS x BINS candidate grid "
             "(0 disables); requires --reset-odometry, since it changes the "
             "head's input width")
    parser.add_argument(
        "--scan-vote-sigma", type=float,
        help="consensus kernel width in metres (default 0.01)")
    parser.add_argument(
        "--confidence-gate-bias", type=float,
        help="initial bias of the learned confidence gate (default -2.0); "
             "requires --reset-odometry, since it is an initialization")
    parser.add_argument(
        "--reset-odometry", action="store_true",
        help="reinitialize the odometry head, keeping the warm-started actor; "
             "use when a change to the scan encoding invalidates its weights")
    parser.add_argument("--action-coef", type=float,
                        help="action imitation coefficient (default: 1 when "
                             "--train-actor, otherwise 0)")
    parser.add_argument("--translation-coef", type=float,
                        default=BeliefConfig.translation_coef,
                        help="weight on per-tick translation error. This term "
                             "rewards low variance, where command dead "
                             "reckoning is unbeatable; the belief error that "
                             "actually matters is an integral, where bias is")
    parser.add_argument("--belief-coef", type=float,
                        default=BeliefConfig.belief_coef,
                        help="weight on accumulated belief error -- the "
                             "integral the deployed system is judged on")
    parser.add_argument("--mismatched-lidar-coef", type=float,
                        default=BeliefConfig.mismatched_lidar_coef,
                        help="force mismatched scans back to command dead reckoning")
    parser.add_argument("--confidence-coef", type=float,
                        default=BeliefConfig.confidence_coef,
                        help="rank the confidence gate on matched vs mismatched "
                             "scans; without it the gate gets no usable gradient")
    parser.add_argument("--windowed", action="store_true",
                        help="legacy fixed windows; complete episodes are the default")
    parser.add_argument("--max-train-shards", type=int)
    parser.add_argument("--val-shards", type=int, default=BeliefConfig.val_shards)
    parser.add_argument("--seed", type=int, default=BeliefConfig.seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--max-steps", type=int,
                        help="stop early; for smoke tests")
    return parser


@torch.no_grad()
def _validate(policy, corpus, config, device, generator):
    """Score held-out episodes; checkpoint selection never sees training batches."""
    policy.eval()
    totals: dict[str, float] = {}
    batches = 0
    for _, shard in corpus.loaded(corpus.validation, device, prefetch=False):
        source = (shard.episode_batches(config.batch_size,
                                        config.validation_noise_scale,
                                        shuffle=False, generator=generator)
                  if config.full_episodes else
                  shard.sequence_batches(config.batch_size, config.seq_len,
                                         config.validation_noise_scale,
                                         shuffle=False, generator=generator))
        for batch in source:
            _, parts = belief_losses(policy, batch, config)
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            batches += 1
            if batches >= config.validation_batches:
                break
        if batches >= config.validation_batches:
            break
    if not batches:
        return None
    return {key: value / batches for key, value in totals.items()}


def main() -> None:
    args = build_parser().parse_args()
    starts = [args.resume, args.warm_start, args.belief_start]
    if sum(value is not None for value in starts) > 1:
        raise ValueError("--resume, --warm-start, and --belief-start are "
                         "mutually exclusive")
    if args.freeze_estimator and not (args.belief_start and args.train_actor):
        raise ValueError("--freeze-estimator requires --belief-start and "
                         "--train-actor")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every cannot be negative")
    if args.motion_state_size < 0:
        raise ValueError("--motion-state-size cannot be negative")
    if args.persistent_calibration and args.motion_state_size:
        raise ValueError(
            "--persistent-calibration and --motion-state-size are alternatives")
    config = BeliefConfig(
        epochs=args.epochs, batch_size=args.batch_size, seq_len=args.seq_len,
        learning_rate=args.learning_rate, freeze_actor=not args.train_actor,
        freeze_estimator=args.freeze_estimator,
        action_coef=((1.0 if args.action_coef is None else args.action_coef)
                     if args.train_actor else 0.0),
        mismatched_lidar_coef=args.mismatched_lidar_coef,
        confidence_coef=args.confidence_coef,
        full_episodes=not args.windowed,
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

    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("algorithm") != "belief":
            raise ValueError(f"{args.resume} is not a belief checkpoint")
        if resume.get("dataset_digest") != corpus.digest:
            raise ValueError("resume checkpoint was trained on a different corpus")
        if resume.get("training_config") != asdict(config):
            raise ValueError("resume checkpoint training config differs from this run")
        policy = BeliefNavPolicy(BeliefNavPolicyConfig.from_dict(
            resume["model_config"])).to(device)
        policy.load_state_dict(resume["model"])
        print(f"[belief] resumed model from {args.resume}")
    elif args.belief_start:
        policy = load_belief_checkpoint(args.belief_start, device=device)
        print(f"[belief] initialized actor and estimator from {args.belief_start}")
    elif args.warm_start:
        policy = warm_start(args.warm_start, device=device)
        print(f"[belief] warm-started actor from {args.warm_start}")
    else:
        policy = BeliefNavPolicy(BeliefNavPolicyConfig()).to(device)
        print("[belief] cold actor -- expect this to need far more data")
    if args.derotate_scan_history and not policy.config.derotate_scan_history:
        if resume is not None:
            raise ValueError("cannot change scan derotation while resuming")
        policy.config = replace(policy.config, derotate_scan_history=True)
        policy.odometry.config = policy.config
        print("[belief] gyro-aligning the scan history before encoding")
    if args.scaled_scan_difference and not policy.config.scaled_scan_difference:
        if resume is not None:
            raise ValueError("cannot change difference scaling while resuming")
        policy.config = replace(policy.config, scaled_scan_difference=True)
        policy.odometry.config = policy.config
        print("[belief] masking and rescaling the scan difference channel")
    if args.scan_vote_bins and not policy.config.scan_vote_bins:
        if resume is not None:
            raise ValueError("cannot enable the scan vote while resuming")
        if not args.reset_odometry:
            raise ValueError("--scan-vote-bins widens the head's trunk input; "
                             "pass --reset-odometry")
        policy.config = replace(policy.config,
                                scan_vote_bins=args.scan_vote_bins)
        policy.odometry.config = policy.config
        print(f"[belief] consensus scan vote on a {args.scan_vote_bins}x"
              f"{args.scan_vote_bins} candidate grid")
    if args.scan_vote_sigma is not None:
        if not args.scan_vote_bins:
            raise ValueError("--scan-vote-sigma requires --scan-vote-bins")
        policy.config = replace(policy.config,
                                scan_vote_sigma_m=args.scan_vote_sigma)
        policy.odometry.config = policy.config
        print(f"[belief] vote kernel width {args.scan_vote_sigma * 100:.1f} cm")
    if args.confidence_gate_bias is not None:
        if not args.reset_odometry:
            raise ValueError("--confidence-gate-bias only takes effect on a "
                             "fresh head; pass --reset-odometry")
        policy.config = replace(
            policy.config, confidence_gate_bias=args.confidence_gate_bias)
        policy.odometry.config = policy.config
        print(f"[belief] confidence gate starts at bias "
              f"{args.confidence_gate_bias:+.2f}")
    if args.reset_odometry:
        if resume is not None:
            raise ValueError("cannot reset the odometry head while resuming")
        policy.odometry = OdometryHead(policy.config).to(device)
        print("[belief] reinitialized the odometry head")
    if args.persistent_calibration and not policy.config.persistent_calibration:
        policy.config = replace(policy.config, persistent_calibration=True)
        policy.odometry.config = policy.config
        print("[belief] enabled persistent forward calibration state")
    if args.motion_state_size:
        if resume is not None and policy.config.motion_state_size != args.motion_state_size:
            raise ValueError("cannot change motion-state size while resuming")
        if policy.config.motion_state_size != args.motion_state_size:
            upgraded_config = replace(
                policy.config, persistent_calibration=False,
                motion_state_size=args.motion_state_size)
            upgraded = BeliefNavPolicy(upgraded_config).to(device)
            incompatible = upgraded.load_state_dict(
                policy.state_dict(), strict=False)
            allowed = ("odometry.motion_memory.",
                       "odometry.motion_scale_head.")
            bad_missing = [key for key in incompatible.missing_keys
                           if not key.startswith(allowed)]
            if bad_missing or incompatible.unexpected_keys:
                raise ValueError(
                    "motion-state upgrade mismatch: "
                    f"missing={bad_missing}, "
                    f"unexpected={incompatible.unexpected_keys}")
            policy = upgraded
            print(f"[belief] added {args.motion_state_size}-float learned "
                  "motion state")
    print(f"[belief] {policy.parameter_count:,} parameters "
          f"({policy.odometry_parameter_count:,} in the new odometry head)")

    optimizer = torch.optim.AdamW(_parameter_groups(policy, config),
                                  weight_decay=config.weight_decay)
    # Windows per shard is data-dependent; estimate for the schedule and let the
    # cosine clamp handle an over- or under-shoot rather than pretending to know.
    windows = max(1, corpus.train_samples // (
        config.batch_size * (config.seq_len if config.full_episodes else 1)))
    total_steps = args.max_steps or (windows * config.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, total_steps, config))
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    (args.out / "dataset.json").write_text(
        json.dumps(corpus.as_dict(), indent=2, sort_keys=True) + "\n")
    metrics_path = args.out / "metrics.jsonl"
    metrics = metrics_path.open("a")
    generator = torch.Generator(device=device.type).manual_seed(config.seed)
    if resume is not None and "generator_state" in resume:
        generator.set_state(resume["generator_state"])

    step = int(resume.get("global_step", 0)) if resume is not None else 0
    best = float(resume.get("best_validation", float("inf"))) \
        if resume is not None else float("inf")
    start_epoch = int(resume.get("epoch", 0)) if resume is not None else 0
    resume_shard = int(resume.get("next_shard_cursor", 0)) \
        if resume is not None else 0
    warm_start_path = (resume.get("warm_start") if resume is not None else
                       str(args.belief_start) if args.belief_start else
                       str(args.warm_start) if args.warm_start else None)

    def save_checkpoint(path, epoch, next_shard_cursor):
        value = belief_checkpoint(
            policy, optimizer, scheduler, config, corpus, epoch=epoch,
            global_step=step, best_validation=best,
            warm_start_path=warm_start_path)
        value["next_shard_cursor"] = next_shard_cursor
        value["generator_state"] = generator.get_state()
        atomic_torch_save(value, path)

    started = time.perf_counter()
    stop = False
    for epoch in range(start_epoch, config.epochs):
        ordered_shards = corpus.train_order(epoch)
        shard_offset = resume_shard if epoch == start_epoch else 0
        for shard_cursor, (info, shard) in enumerate(corpus.loaded(
                                         ordered_shards[shard_offset:], device,
                                         prefetch=config.prefetch,
                                         workers=config.prefetch_workers,
                                         buffer_size=config.prefetch_shards),
                                         start=shard_offset):
            source = (shard.episode_batches(
                        config.batch_size, noise_scale=config.train_noise_scale,
                        generator=generator)
                      if config.full_episodes else
                      shard.sequence_batches(
                        config.batch_size, config.seq_len,
                        noise_scale=config.train_noise_scale,
                        generator=generator))
            for batch in source:
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
                if args.checkpoint_every and step % args.checkpoint_every == 0:
                    # Resume at the next shard. This deliberately sacrifices at
                    # most the unconsumed tail of one shard rather than trying
                    # to serialize a shuffled Python generator mid-iteration.
                    save_checkpoint(args.out / "latest.pt", epoch,
                                    shard_cursor + 1)
                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break
            if stop:
                break
        validation = _validate(policy, corpus, config, device, generator)
        if validation is not None:
            validation_record = {
                f"validation_{key}": value for key, value in validation.items()}
            validation_record.update(step=step, epoch=epoch,
                                     elapsed_seconds=time.perf_counter() - started)
            metrics.write(json.dumps(validation_record, sort_keys=True) + "\n")
            metrics.flush()
            score = (validation["action_loss"] if config.freeze_estimator else
                     validation["belief_error_final_m"])
            if config.freeze_estimator:
                print(f"[belief] validation action loss {score:.5f}  "
                      f"belief final "
                      f"{validation['belief_error_final_m']*100:.2f} cm  "
                      f"p95 "
                      f"{validation['belief_error_final_p95_m']*100:.2f} cm  "
                      f"drift/m {validation['drift_per_m']*100:.2f} cm/m")
            else:
                print(f"[belief] validation final {score*100:.2f} cm  "
                      f"p95 "
                      f"{validation['belief_error_final_p95_m']*100:.2f} cm  "
                      f"drift/m {validation['drift_per_m']*100:.2f} cm/m")
            if score < best:
                best = score
                save_checkpoint(args.out / "best.pt", epoch + 1, 0)
        if stop:
            break

    save_checkpoint(args.out / "final.pt", config.epochs, 0)
    metrics.close()
    if best == float("inf"):
        suffix = "no validation shards; no best.pt selected"
    elif config.freeze_estimator:
        suffix = f"best validation action loss {best:.5f}"
    else:
        suffix = f"best validation final error {best*100:.2f} cm"
    print(f"[belief] wrote {args.out}/final.pt after {step} steps "
          f"({time.perf_counter() - started:.1f} s); {suffix}")


if __name__ == "__main__":
    main()
