"""Paired held-out ablations for the lidar/IMU goal-belief estimator.

This is the inexpensive causal gate to run before closed-loop Isaac evaluation.
Every mode sees the same validation episodes and the same sampled sensor noise:

``belief``          the learned estimator with the matching lidar history
``dead-reckoning``  the learned correction disabled; commands plus IMU only
``lidar-shuffled``  a different episode's coherent lidar history
``lidar-zeroed``    no valid lidar returns (the deployable fallback)

The useful comparison is belief against dead reckoning.  Shuffled and zeroed
lidar should fall back toward dead reckoning; if belief is unchanged by either
ablation, the checkpoint has not demonstrated causal use of lidar.

    .venv/bin/python scripts/eval_belief_offline.py \
        runs/belief_motion_v4/best.pt --data data/overnight_v3 \
        --out runs/belief_motion_v4/offline_ablation.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2edge.spec import LIDAR_MAX_M, fingerprint
from sim2edge.training import ShardCorpus, load_belief_checkpoint


MODES = ("belief", "dead-reckoning", "lidar-shuffled", "lidar-zeroed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, nargs="+",
                        help="one or more belief checkpoints")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-shards", type=int, default=4)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=32,
                        help="0 evaluates every held-out episode")
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    return parser


@contextlib.contextmanager
def _dead_reckoning(policy):
    """Disable only the learned correction, restoring parameters afterwards."""
    head = policy.odometry.head
    weight = head.weight.detach().clone()
    bias = head.bias.detach().clone()
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-100.0)
    try:
        yield
    finally:
        with torch.no_grad():
            head.weight.copy_(weight)
            head.bias.copy_(bias)


def _ablate(batch, mode: str):
    raw, valid = batch.raw_lidar, batch.raw_valid
    if mode == "lidar-shuffled":
        if raw.shape[1] > 1:
            raw, valid = raw.roll(1, dims=1), valid.roll(1, dims=1)
        else:
            # A one-episode tail still needs a mismatched control. Preserve its
            # internal scan windows but offset them from the robot trajectory.
            shift = max(1, raw.shape[0] // 2)
            raw, valid = raw.roll(shift, dims=0), valid.roll(shift, dims=0)
    elif mode == "lidar-zeroed":
        raw = torch.full_like(raw, LIDAR_MAX_M)
        valid = torch.zeros_like(valid)
    return raw, valid


@torch.inference_mode()
def _score(policy, batch, mode: str) -> dict[str, object]:
    raw, valid = _ablate(batch, mode)
    context = _dead_reckoning(policy) if mode == "dead-reckoning" else \
        contextlib.nullcontext()
    with context:
        _, predicted = policy.rollout(
            batch.lidar, batch.gyro, batch.previous_action,
            batch.initial_belief, raw, valid)
        confidence = policy.motion_estimate(
            batch.lidar[1:].reshape(-1, *batch.lidar.shape[2:]),
            batch.gyro[1:].reshape(-1, batch.gyro.shape[-1]),
            batch.previous_action[1:].reshape(
                -1, *batch.previous_action.shape[2:]),
            raw[1:].reshape(-1, *raw.shape[2:]),
            valid[1:].reshape(-1, *valid.shape[2:]),
        )[2]

    distance = (predicted - batch.belief).float().norm(dim=-1)
    lengths = batch.valid_steps.long().sum(dim=0).clamp_min(1)
    last = lengths - 1
    final = distance.gather(0, last[None]).squeeze(0)
    travelled = (batch.translation.float().norm(dim=-1)
                 * batch.valid_steps).sum(dim=0)
    motion_valid = batch.valid_steps[1:].reshape(-1)
    return {
        "error_sum": float(distance[batch.valid_steps].sum()),
        "step_count": int(batch.valid_steps.sum()),
        "final": final.cpu().tolist(),
        "travelled": travelled.cpu().tolist(),
        "confidence_sum": float(confidence[motion_valid].sum()),
        "motion_count": int(motion_valid.sum()),
    }


def _merge(total: dict[str, object], part: dict[str, object]) -> None:
    for key in ("error_sum", "step_count", "confidence_sum", "motion_count"):
        total[key] += part[key]
    total["final"].extend(part["final"])
    total["travelled"].extend(part["travelled"])


def _finish(total: dict[str, object]) -> dict[str, float | int]:
    final = torch.tensor(total["final"], dtype=torch.float64)
    travelled = torch.tensor(total["travelled"], dtype=torch.float64)
    return {
        "episodes": final.numel(),
        "steps": int(total["step_count"]),
        "belief_error_m": total["error_sum"] / max(1, total["step_count"]),
        "belief_error_final_m": float(final.mean()),
        "belief_error_final_p95_m": float(final.quantile(0.95)),
        "drift_per_m": float(final.sum() / travelled.sum().clamp_min(1.0e-9)),
        "scan_confidence": (total["confidence_sum"]
                            / max(1, total["motion_count"])),
    }


def evaluate(checkpoint: Path, corpus: ShardCorpus, args,
             device: torch.device) -> dict:
    policy = load_belief_checkpoint(checkpoint, device=device).eval()
    totals = {mode: {
        "error_sum": 0.0, "step_count": 0, "final": [], "travelled": [],
        "confidence_sum": 0.0, "motion_count": 0,
    } for mode in MODES}
    generator = torch.Generator(device=device.type).manual_seed(args.seed)
    batches = 0
    for _, shard in corpus.loaded(corpus.validation, device, prefetch=False):
        for batch in shard.episode_batches(
                args.batch_size, args.noise_scale, shuffle=False,
                generator=generator):
            for mode in MODES:
                _merge(totals[mode], _score(policy, batch, mode))
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break
        if args.max_batches and batches >= args.max_batches:
            break
    if not batches:
        raise RuntimeError("validation split produced no episode batches")

    modes = {mode: _finish(total) for mode, total in totals.items()}
    learned = modes["belief"]["belief_error_final_m"]
    baseline = modes["dead-reckoning"]["belief_error_final_m"]
    shuffled = modes["lidar-shuffled"]["belief_error_final_m"]
    zeroed = modes["lidar-zeroed"]["belief_error_final_m"]
    causal = {
        "beats_dead_reckoning": learned < baseline,
        "improvement_over_dead_reckoning_m": baseline - learned,
        "shuffled_lidar_degradation_m": shuffled - learned,
        "zeroed_lidar_degradation_m": zeroed - learned,
        "lidar_causal": (learned < baseline and shuffled > learned
                         and zeroed > learned),
    }
    return {
        "checkpoint": str(checkpoint.resolve()),
        "batches": batches,
        "modes": modes,
        "causal": causal,
    }


def _print(result: dict) -> None:
    print(f"[offline] {result['checkpoint']}")
    for mode, metrics in result["modes"].items():
        print(f"[offline] {mode:>15s}  final "
              f"{metrics['belief_error_final_m'] * 100:.2f} cm  p95 "
              f"{metrics['belief_error_final_p95_m'] * 100:.2f} cm  drift/m "
              f"{metrics['drift_per_m'] * 100:.2f} cm/m  conf "
              f"{metrics['scan_confidence']:.3f}")
    causal = result["causal"]
    print(f"[offline] lidar_causal={causal['lidar_causal']}  "
          f"vs DR {causal['improvement_over_dead_reckoning_m'] * 100:+.2f} cm  "
          f"shuffle {causal['shuffled_lidar_degradation_m'] * 100:+.2f} cm  "
          f"zero {causal['zeroed_lidar_degradation_m'] * 100:+.2f} cm")


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.val_shards < 1 or args.max_batches < 0:
        raise ValueError("batch-size and val-shards must be positive; "
                         "max-batches cannot be negative")
    device = torch.device(args.device)
    corpus = ShardCorpus(args.data, val_shards=args.val_shards, seed=args.seed)
    results = [evaluate(path, corpus, args, device)
               for path in args.checkpoint]
    for result in results:
        _print(result)
    payload = {
        "data": str(args.data.resolve()),
        "dataset_digest": corpus.digest,
        "spec_fingerprint": fingerprint(),
        "noise_scale": args.noise_scale,
        "seed": args.seed,
        "paired_modes": True,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n")
    print(f"[offline] wrote {args.out}")


if __name__ == "__main__":
    main()
