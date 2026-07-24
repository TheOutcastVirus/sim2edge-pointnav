#!/usr/bin/env python3
"""Lower the navigation actor to an XNNPACK (CPU) ``.pte``.

This is a **live fallback**, not a curiosity. The policy is ~2.7 M parameters —
far smaller than the ResNet18 the reference project ran — so CPU inference has a
real chance of holding 20 Hz. That makes it the Gate-A fallback: if QNN lowering
fights back on Day 1, ship this and debug the NPU in parallel rather than
blocking training on the board bring-up.

Note the board build in the reference project used ``-DBUILD_XNNPACK_BACKEND=OFF``,
which makes ``backend:=cpu`` a dead option. Build ExecuTorch with
``EXECUTORCH_BUILD_XNNPACK=ON`` for this project so the fallback actually runs.

Also supports ``--quantize`` for INT8 XNNPACK, which is useful for reproducing a
quantization-related bug off the NPU, where the debugging tools are better.

Usage:
    python3 tools/export_cpu.py --weights runs/bc/bc_final.pt \\
        --output models/nav_policy_xnnpack.pte
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from export_common import (  # noqa: E402
    example_inputs,
    load_actor,
    load_calibration,
    report_parity,
    synthetic_calibration,
    write_sidecar,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sim2edge.spec import DEFAULT_SPEC  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=None)
    p.add_argument("--output", default="models/nav_policy_xnnpack.pte")
    p.add_argument("--quantize", action="store_true",
                   help="INT8 (8a8w) XNNPACK instead of fp32.")
    p.add_argument("--calibration", default=None)
    p.add_argument("--calibration-samples", type=int, default=512)
    p.add_argument("--no-strict-fingerprint", action="store_true")
    args = p.parse_args()

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    spec = DEFAULT_SPEC
    actor = load_actor(args.weights, spec,
                       strict_fingerprint=not args.no_strict_fingerprint)
    example = example_inputs(spec)
    print(f"Actor inputs: {[tuple(t.shape) for t in example]}")

    print("Exporting to ExecuTorch IR ...")
    ep = torch.export.export(actor, example)

    if args.quantize:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e,
            prepare_pt2e,
        )

        calibration = (load_calibration(args.calibration, spec,
                                        args.calibration_samples)
                       if args.calibration else synthetic_calibration(spec, 64))
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True))
        prepared = prepare_pt2e(ep.module(), quantizer)
        print(f"Calibrating on {len(calibration)} samples ...")
        with torch.no_grad():
            for sample in calibration:
                prepared(*sample)
        converted = convert_pt2e(prepared)

        with torch.no_grad():
            ref = torch.cat([actor(*s) for s in calibration[:64]], dim=0)
            got = torch.cat([converted(*s) for s in calibration[:64]], dim=0)
        report_parity(ref, got, spec, "INT8 XNNPACK vs fp32 (pre-lowering)")

        ep = torch.export.export(converted, example)

    print("Lowering with the XNNPACK partitioner ...")
    # _check_ir_validity=False mirrors the reference export: the quantized graph
    # carries ops the strict IR validator rejects, and the partitioner handles
    # them correctly regardless.
    program = to_edge_transform_and_lower(
        ep,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(program.buffer)
    print(f"Saved {out} ({out.stat().st_size / 1024:.1f} KB)")

    write_sidecar(out, spec, backend="xnnpack",
                  quant="8a8w" if args.quantize else "fp32",
                  extra={"calibration_source": args.calibration or
                         ("synthetic" if args.quantize else "n/a")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
