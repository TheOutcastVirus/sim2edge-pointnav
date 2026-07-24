#!/usr/bin/env python3
"""Measure ``.pte`` inference latency against the control-loop budget.

Run this on **Day 1** with an untrained model, before any training exists. The
number it produces decides whether the NPU path is on the critical path at all:
if CPU inference already fits inside the 50 ms budget, a QNN problem stops being
schedule-threatening and becomes an optimization.

Note the reference repos documented ~5 Hz for ResNet18 on this board. That figure
was pipeline-limited, not NPU-limited -- the NPU was considerably faster. This
policy is roughly a tenth of ResNet18's size, so treat any result near 5 Hz as a
measurement bug in the harness rather than a property of the hardware.

    python3 tools/bench_latency.py --pte models/nav_policy_qnn.pte --iters 500
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim2edge.spec import DEFAULT_SPEC  # noqa: E402


def bench(run, inputs, iters: int, warmup: int) -> dict:
    for _ in range(warmup):
        run(inputs)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run(inputs)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "mean": statistics.fmean(samples),
        "median": samples[len(samples) // 2],
        "p95": samples[int(len(samples) * 0.95)],
        "p99": samples[int(len(samples) * 0.99)],
        "max": samples[-1],
        "min": samples[0],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pte", default=None, help="Lowered model to benchmark.")
    p.add_argument("--weights", default=None,
                   help="Also benchmark this FP32 checkpoint in eager torch.")
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    args = p.parse_args()

    spec = DEFAULT_SPEC
    budget_ms = 1000.0 / spec.action.control_rate_hz
    inputs = tuple(torch.zeros((1,) + tuple(s), dtype=torch.float32)
                   for s in spec.actor_input_shapes)
    print(f"Control budget: {budget_ms:.1f} ms at "
          f"{spec.action.control_rate_hz:.0f} Hz\n")

    results = {}

    if args.weights is not None or args.pte is None:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from export_common import load_actor

        actor = load_actor(args.weights, spec, strict_fingerprint=False)
        with torch.no_grad():
            results["torch-eager-fp32"] = bench(
                lambda x: actor(*x), inputs, args.iters, args.warmup)

    if args.pte:
        from executorch.runtime import Runtime

        method = Runtime.get().load_program(args.pte).load_method("forward")
        results[pathlib.Path(args.pte).name] = bench(
            lambda x: method.execute(x), inputs, args.iters, args.warmup)

    print(f"{'backend':<32}{'mean':>8}{'p95':>8}{'p99':>8}{'max':>8}   verdict")
    print("-" * 78)
    for name, r in results.items():
        # p99 is the number that matters: a control loop is judged by its worst
        # ticks, not its average one.
        ok = r["p99"] < budget_ms
        headroom = budget_ms / r["p99"] if r["p99"] > 0 else float("inf")
        verdict = (f"OK ({headroom:.1f}x headroom)" if ok
                   else "OVER BUDGET at p99")
        print(f"{name:<32}{r['mean']:>7.2f}{r['p95']:>8.2f}{r['p99']:>8.2f}"
              f"{r['max']:>8.2f}   {verdict}")

    print("\nIf the p99 exceeds budget, in order: confirm the graph is fully "
          "delegated (a partial partition falls back to CPU per-op and is the "
          "usual cause), then check qnn-platform-validator, then reduce the "
          "depth input resolution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
