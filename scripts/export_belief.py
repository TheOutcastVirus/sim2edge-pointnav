"""Export the deployable belief step, and prove the exported graph is the model.

This is the front half of Phase 5. It runs on the training host and needs
neither the board nor ExecuTorch: ``torch.export`` is the exact front end
``to_edge_transform_and_lower`` consumes, so everything that can go wrong in the
graph itself goes wrong here, where there is a debugger.

What this checks, in the order the failures actually happen
==========================================================
1. **The signature exports at all.** The deployed step takes explicit state --
   belief, covariance, arrival count, optional calibration -- in and out. A
   recurrent model with carried state is the case export helpers handle worst,
   and the previous build's log records the four-input signature as its one
   genuine unknown on day one.
2. **Every operator is named.** An op outside the delegated set does not fail;
   it silently falls back to CPU and costs a partition boundary. The report
   below lists the operator set so it can be diffed against the set the
   reference models proved delegated, rather than discovered as a latency
   regression on the robot.
3. **The graph reproduces eager numerics.** Not approximately -- this compares
   the exported program's output against the module it was traced from, on
   random and adversarial state, and fails on any drift beyond float noise.

What this deliberately does *not* do
====================================
It does not quantize, and it does not lower to QNN. Both need the board (the
QNN AOT pybind is patched on for aarch64 there) and both are gated on a
measured sim success-rate drop rather than on an action error, which needs the
simulator. Those stay in ``eval_belief.py``'s territory.

Where the scan vote goes
========================
:class:`~sim2edge.models.belief.ScanVote` is deliberately kept *out* of the
exported graph, and ``--include-vote`` exists only to demonstrate what including
it would cost. Two reasons, both structural:

* it holds no parameters. It is fixed geometry -- the same status the deskew and
  the angular binning already have, and those live in the C++ node's
  preprocessing translation unit, not in the network;
* it is an argmax over a consensus surface measuring *centimetres*. INT8 is
  about 1 mm across the candidate grid at the extents used, and quantizing a
  surface whose whole content is the sharpness of one peak is the kind of thing
  that degrades without erroring.

Excluding it also keeps the network inside the operator set the reference models
proved out: including it adds ``gather``, ``cumsum``, ``flip``, ``remainder``,
``floor`` and a ``set_grad_enabled`` higher-order wrapper, none of which appear
in the actor-only graph.

    .venv/bin/python scripts/export_belief.py runs/belief_gate_v21/final.pt \
        --out runs/belief_gate_v21/export
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from sim2edge.models.belief import BeliefNavPolicy
from sim2edge.spec import (DT, LIDAR_MAX_M, MOTION_PRIOR_SOURCE,
                           SENSOR_CONTRACT_VERSION)
from sim2edge.training.belief import load_belief_checkpoint


class DeployedStep(nn.Module):
    """The exact call the C++ node makes, as a traceable module.

    The wrapper exists because ``torch.export`` traces a module's ``forward``,
    and the deployed entry point is :meth:`BeliefNavPolicy.step_state`. Keeping
    it a thin forwarder rather than a reimplementation is the same rule
    ``rollout`` follows against ``step``: a second copy of the inference path is
    the most expensive bug class this project has had.
    """

    def __init__(self, policy: BeliefNavPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, *state_and_sensors):
        return self.policy.step_state(*state_and_sensors)


class NetworkOnlyStep(nn.Module):
    """The step with the parameter-free vote lifted out into its inputs.

    This is the graph that ships. The vote runs ahead of it, in float, on the
    CPU, and hands in two small tensors. Deleting the vote module instead would
    not work and is worth saying out loud: the trunk's first layer is sized to
    include the vote's features, so a checkpoint trained with it produces a
    shape error, not a graceful fallback.
    """

    def __init__(self, policy: BeliefNavPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, surfaces, scalars, sharpness, *state_and_sensors):
        return self.policy.step_state(
            *state_and_sensors, vote=(surfaces, scalars, sharpness))


def operator_report(exported) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in exported.graph.nodes:
        if node.op in ("call_function", "call_method"):
            name = str(node.target).replace("torch.ops.", "").replace("aten.", "")
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def flatten(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    out: list[torch.Tensor] = []
    for item in value:
        out.extend(flatten(item))
    return out


def parity(module: nn.Module, exported, inputs) -> dict[str, float]:
    """Largest absolute disagreement between the graph and the module."""
    with torch.no_grad():
        expected = flatten(module(*inputs))
        actual = flatten(exported.module()(*inputs))
    if len(expected) != len(actual):
        raise RuntimeError(
            f"export changed the output arity: {len(expected)} -> {len(actual)}")
    worst = 0.0
    for want, got in zip(expected, actual):
        if want.dtype.is_floating_point:
            worst = max(worst, float((want - got).abs().max()))
        elif not torch.equal(want, got):
            raise RuntimeError("an integer output changed under export")
    return {"max_abs_difference": worst}


def adversarial_inputs(policy: BeliefNavPolicy, batch: int, generator):
    """State the robot really reaches, not just zeros.

    Zeros pass almost anything. These carry a non-trivial belief, an inflated
    covariance, a partially elapsed arrival count, and a scan with a realistic
    fraction of missing returns -- the case where the validity mask decides the
    answer.
    """
    config = policy.config
    frames = config.actor.history_frames
    goal = torch.randn(batch, 2, generator=generator) * 2.0
    belief, covariance, count = policy.initial_state(goal)
    covariance = covariance + torch.eye(2)[None] * 0.01
    count = torch.full_like(count, config.arrival_confirm_ticks - 1)
    raw = torch.rand(batch, config.motion_history_frames,
                     config.raw_lidar_bins, generator=generator) * 4.0 + 0.3
    valid = torch.rand(batch, config.motion_history_frames,
                       config.raw_lidar_bins, generator=generator) > 0.25
    raw = torch.where(valid, raw, torch.full_like(raw, LIDAR_MAX_M))
    inputs = [
        belief, covariance, count,
        torch.rand(batch, frames, config.actor.lidar_bins, generator=generator),
        raw, valid,
        torch.randn(batch, config.motion_history_frames,
                    generator=generator) * 0.05,
        torch.rand(batch, config.motion_history_frames, 2,
                   generator=generator) * 2.0 - 1.0,
    ]
    if config.persistent_calibration or config.motion_state_size > 0:
        inputs.append(policy.initial_calibration(goal))
    return tuple(inputs)


def measure_latency(module: nn.Module, inputs, repeats: int) -> dict[str, float]:
    """Host CPU latency. A proxy for the budget, not a board measurement.

    The control period is 50 ms and the board's HTP is faster than this host's
    CPU for the convolutional bulk, so a host figure comfortably inside the
    period is evidence the budget is not obviously blown -- and nothing more.
    Anything near 50 ms here is worth resolving before the board sees it.
    """
    with torch.no_grad():
        for _ in range(5):
            module(*inputs)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            module(*inputs)
            samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return {
        "median_ms": samples[len(samples) // 2],
        "p95_ms": samples[int(len(samples) * 0.95)],
        "max_ms": samples[-1],
        "control_period_ms": DT * 1e3,
        "median_fraction_of_period": samples[len(samples) // 2] / (DT * 1e3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--include-vote", action="store_true",
                        help="trace the scan vote into the graph as well; for "
                             "measuring what that costs, not for shipping")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    policy = load_belief_checkpoint(args.checkpoint, device="cpu").eval()
    generator = torch.Generator().manual_seed(args.seed)
    inputs = adversarial_inputs(policy, args.batch_size, generator)

    if args.include_vote or policy.config.scan_vote_bins == 0:
        module, traced_inputs = DeployedStep(policy), inputs
    else:
        # The vote's inputs are the raw scan history, the gyro and the previous
        # commands -- exactly the tensors already in `inputs`, at the positions
        # step_state takes them.
        lidar, raw, valid, gyro, previous = inputs[3:8]
        with torch.no_grad():
            vote = policy.odometry.compute_vote(lidar, gyro, previous, raw, valid)
        module, traced_inputs = NetworkOnlyStep(policy), (*vote, *inputs)
    module.eval()

    started = time.perf_counter()
    exported = torch.export.export(module, traced_inputs, strict=False)
    export_seconds = time.perf_counter() - started

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "sensor_contract_version": SENSOR_CONTRACT_VERSION,
        "motion_prior_source": MOTION_PRIOR_SOURCE,
        "includes_scan_vote": bool(args.include_vote),
        "scan_vote_bins": policy.config.scan_vote_bins,
        "batch_size": args.batch_size,
        "export_seconds": export_seconds,
        "parameter_count": policy.parameter_count,
        "operators": operator_report(exported),
        "parity": parity(module, exported, traced_inputs),
        "latency": measure_latency(module, traced_inputs, args.latency_repeats),
    }
    report["operator_count"] = len(report["operators"])

    args.out.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, args.out / "belief_step.pt2")
    (args.out / "export_report.json").write_text(json.dumps(report, indent=2))

    worst = report["parity"]["max_abs_difference"]
    latency = report["latency"]
    print(f"[export] {args.checkpoint}")
    print(f"[export] {report['operator_count']} distinct operators, "
          f"{report['parameter_count']:,} parameters")
    print(f"[export] parity max |difference| {worst:.3e} "
          f"(tolerance {args.tolerance:.1e})")
    print(f"[export] host latency median {latency['median_ms']:.2f} ms, p95 "
          f"{latency['p95_ms']:.2f} ms, of a {latency['control_period_ms']:.0f} "
          f"ms control period")
    print(f"[export] wrote {args.out / 'belief_step.pt2'}")
    if worst > args.tolerance:
        raise SystemExit(
            f"export does not reproduce the module: {worst:.3e} > "
            f"{args.tolerance:.1e}")


if __name__ == "__main__":
    main()
