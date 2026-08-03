"""Export the deployable navigation actor and verify graph parity/latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from sim2edge.spec import (DT, MOTION_PRIOR_SOURCE, SENSOR_CONTRACT_VERSION,
                           observation_fingerprint)
from sim2edge.training import load_policy_checkpoint


def _operators(exported) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in exported.graph.nodes:
        if node.op in ("call_function", "call_method"):
            name = str(node.target).replace("torch.ops.", "").replace("aten.", "")
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _latency(module, observation: torch.Tensor, repeats: int) -> dict[str, float]:
    torch.set_num_threads(1)
    with torch.inference_mode():
        for _ in range(10):
            module(observation)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            module(observation)
            samples.append(1_000.0 * (time.perf_counter() - started))
    samples.sort()
    return {
        "median_ms": samples[len(samples) // 2],
        "p95_ms": samples[min(len(samples) - 1, int(0.95 * len(samples)))],
        "max_ms": samples[-1],
        "control_period_ms": DT * 1_000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--latency-repeats", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--motion-prior-source", default=MOTION_PRIOR_SOURCE)
    args = parser.parse_args()

    actor = load_policy_checkpoint(args.checkpoint, device="cpu").eval()
    generator = torch.Generator().manual_seed(20260803)
    observation = torch.randn(1, actor.config.observation_dim,
                              generator=generator)
    # Exercise realistic bounded lidar/action/goal regions rather than tracing
    # only zeros, which would hide preprocessing or scaling mistakes.
    lidar_width = actor.config.history_frames * actor.config.lidar_bins
    observation[:, :lidar_width] = torch.rand(
        1, lidar_width, generator=generator)
    observation[:, lidar_width:lidar_width + 2] *= 3.0
    observation[:, lidar_width + 2] = torch.rand(1, generator=generator)
    observation[:, lidar_width + 3:] = torch.rand(
        1, actor.config.history_frames * 2, generator=generator) * 2.0 - 1.0

    exported = torch.export.export(actor, (observation,), strict=False)
    with torch.inference_mode():
        expected = actor(observation)
        actual = exported.module()(observation)
    difference = float((expected - actual).abs().max())
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "sensor_contract_version": SENSOR_CONTRACT_VERSION,
        "motion_prior_source": args.motion_prior_source,
        "observation_fingerprint": observation_fingerprint(
            actor.config.history_frames),
        "parameter_count": actor.parameter_count,
        "operators": _operators(exported),
        "parity": {"max_abs_difference": difference,
                   "tolerance": args.tolerance},
        "latency": _latency(exported.module(), observation,
                            args.latency_repeats),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, args.out / "actor.pt2")
    (args.out / "export_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[export] parity max |difference| {difference:.3e}")
    print(f"[export] host latency median {report['latency']['median_ms']:.3f} ms, "
          f"p95 {report['latency']['p95_ms']:.3f} ms")
    print(f"[export] wrote {args.out / 'actor.pt2'}")
    if difference > args.tolerance:
        raise SystemExit(
            f"export parity failed: {difference:.3e} > {args.tolerance:.3e}")


if __name__ == "__main__":
    main()
