#!/usr/bin/env python3
"""Day-1 smoke test: does the four-input export path work on this board?

Run this **before any training exists**. It answers the one genuinely open
question in the deployment plan: the reference projects
(`ventuno-collision-avoidance`, `ventuno-object-tracking`) both exported
single-tensor models, and this policy takes four input tensors. Whether
`build_executorch_binary` accepts a 4-tuple example input is unverified, and
finding out on Day 9 instead of Day 1 is what turns it into a schedule problem.

The model is untrained on purpose. The subject of the test is the toolchain, not
the policy.

Stages, each reported independently so a failure localizes immediately:

  1. torch.export with the 4-input signature
  2. operator check against the set proven delegated on HTP V75
  3. XNNPACK lowering (the Gate-A fallback, and a control for stage 4)
  4. QNN/HTP lowering
  5. latency against the 20 Hz budget

Exit code is the number of failed stages, so it is usable in a bring-up script.

    python3 tools/smoke_export.py --out-dir models/smoke
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from export_common import example_inputs  # noqa: E402
from sim2edge.models.nav_policy import build_policy  # noqa: E402
from sim2edge.spec import DEFAULT_SPEC  # noqa: E402

# Operators the reference models proved fully delegated on HTP V75 at 8a8w.
PROVEN_OPS = {
    "aten::conv1d", "aten::conv2d", "aten::linear", "aten::batch_norm",
    "aten::relu", "aten::relu_", "aten::tanh", "aten::cat", "aten::flatten",
    "aten::slice", "aten::add", "aten::mul", "aten::unsqueeze",
}


class Stage:
    def __init__(self):
        self.failures = 0

    def run(self, name, fn):
        print(f"\n── {name} " + "─" * max(0, 60 - len(name)))
        try:
            fn()
            print(f"   PASS: {name}")
            return True
        except Exception as exc:  # noqa: BLE001 - report and continue
            self.failures += 1
            print(f"   FAIL: {name}\n         {type(exc).__name__}: {exc}")
            return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="models/smoke")
    p.add_argument("--soc-model", default="QCS8300")
    p.add_argument("--skip-qnn", action="store_true")
    args = p.parse_args()

    spec = DEFAULT_SPEC
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"spec fingerprint   {spec.fingerprint()}")
    print(f"actor input shapes {[tuple(s) for s in spec.actor_input_shapes]}")
    print(f"QNN_SDK_ROOT       {os.environ.get('QNN_SDK_ROOT', '(unset)')}")

    actor = build_policy(spec).eval().export_module()
    example = example_inputs(spec)
    state = {}
    stages = Stage()

    def s1_export():
        program = torch.export.export(actor, example)
        with torch.no_grad():
            ref = actor(*example)
        got = program.module()(*example)
        assert torch.allclose(ref, got, atol=1e-6), "exported graph diverges"
        state["program"] = program
        print(f"   4-input signature accepted; output {tuple(got.shape)} "
              f"= (v={got[0, 0]:.4f}, omega={got[0, 1]:+.4f})")

    def s2_ops():
        used = set()
        for node in state["program"].graph.nodes:
            if node.op == "call_function" and hasattr(node.target, "_schema"):
                used.add(node.target._schema.name)
        unproven = used - PROVEN_OPS
        print(f"   {len(used)} distinct operators, all previously delegated"
              if not unproven else f"   unproven: {sorted(unproven)}")
        assert not unproven, (
            f"operators not proven on HTP V75: {sorted(unproven)}. An "
            f"unsupported op silently splits the graph and falls back to CPU "
            f"per-op, which costs far more than it appears to.")

    def s3_xnnpack():
        result = subprocess.run(
            [sys.executable, str(REPO / "tools/export_cpu.py"),
             "--output", str(out / "smoke_xnnpack.pte")],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr[-1500:]
        size = (out / "smoke_xnnpack.pte").stat().st_size / 1e6
        print(f"   smoke_xnnpack.pte  {size:.2f} MB")

    def s4_qnn():
        result = subprocess.run(
            [sys.executable, str(REPO / "tools/export_qnn.py"),
             "--output", str(out / "smoke_qnn.pte"),
             "--soc-model", args.soc_model],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr[-1500:] +
                "\n\nIf the helper flow rejects the 4-tuple example input, "
                "retry with --flow explicit rather than reshaping the model: "
                "a model contorted to fit a helper no longer matches the C++ "
                "node.")
        size = (out / "smoke_qnn.pte").stat().st_size / 1e6
        print(f"   smoke_qnn.pte  {size:.2f} MB")

    def s5_latency():
        budget = 1000.0 / spec.action.control_rate_hz
        with torch.no_grad():
            for _ in range(20):
                actor(*example)
            t0 = time.perf_counter()
            for _ in range(100):
                actor(*example)
        ms = (time.perf_counter() - t0) * 10.0
        print(f"   torch eager fp32: {ms:.2f} ms/inference "
              f"(budget {budget:.0f} ms)")
        print("   Run tools/bench_latency.py --pte <file> for the real number; "
              "eager torch is only a sanity floor.")

    stages.run("1. torch.export with 4 inputs", s1_export)
    stages.run("2. operator set vs HTP-proven ops", s2_ops)
    stages.run("3. XNNPACK lowering (Gate-A fallback)", s3_xnnpack)
    if not args.skip_qnn:
        stages.run("4. QNN/HTP lowering", s4_qnn)
    stages.run("5. latency sanity", s5_latency)

    print("\n" + "=" * 62)
    if stages.failures == 0:
        print("Gate A smoke test PASSED. The multi-input export path works.")
    else:
        print(f"{stages.failures} stage(s) failed.\n"
              "Gate-A fallback: ship the XNNPACK .pte and debug QNN in "
              "parallel. This policy is ~2.7M parameters, so CPU inference has "
              "a real chance of holding 20 Hz -- do not block training on the "
              "NPU.")
    return stages.failures


if __name__ == "__main__":
    sys.exit(main())
