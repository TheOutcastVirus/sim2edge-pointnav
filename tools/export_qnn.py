#!/usr/bin/env python3
"""Lower the navigation actor to an ExecuTorch ``.pte`` for the Qualcomm HTP NPU.

Runs **on the Ventuno Q**, inside the pinned ExecuTorch venv, with the QAIRT/QNN
environment sourced (``scripts/install_ventuno_deps.sh`` sets this up and exports
``QNN_SDK_ROOT``). Nothing is cross-compiled; everything builds natively on the
board, as in both reference projects.

``--soc-model QCS8300`` is not a typo. ExecuTorch has no QCS8275 entry, so the
board bring-up patches a fake ``QCS8300`` chipset enum that maps to HTP V75.
That patch lives in ``/opt/executorch`` and will break on an ExecuTorch upgrade —
pin the working commit and do not upgrade mid-project.

Two lowering flows are provided:

* ``--flow helper`` (default) mirrors the reference scripts' one-call
  ``build_executorch_binary``.
* ``--flow explicit`` does ``torch.export`` -> ``to_edge_transform_and_lower``
  with the QNN partitioner by hand.

The explicit flow exists because the reference exports were all single-tensor
while this policy takes four, and the helper's handling of a multi-input example
tuple was unverified. If the helper fights the 4-input signature, switch flows
rather than reshaping the model — a model contorted to fit a helper function is a
model that no longer matches the C++ node.

Usage (on the board):

    python3 tools/export_qnn.py \\
        --weights runs/bc/bc_final.pt \\
        --output models/nav_policy_qnn.pte \\
        --calibration calib/nav_calib.npz
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from export_common import (  # noqa: E402
    example_inputs,
    load_actor,
    load_calibration,
    synthetic_calibration,
    write_sidecar,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sim2edge.spec import DEFAULT_SPEC  # noqa: E402


def _lower_helper(actor, calibration, args, spec):
    """One-call lowering, mirroring ``export_resnet18_qnn.py``."""
    from executorch.backends.qualcomm.export_utils import (
        QnnConfig,
        build_executorch_binary,
        make_quantizer,
    )
    from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchBackendType,
    )

    qnn_config = QnnConfig(
        soc_model=args.soc_model,
        build_folder=args.build_folder,
        backend="htp",
        target="aarch64-oe-linux",
        compile_only=False,
        enable_x86_64=False,
    )
    quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_8a8w,
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=args.soc_model,
    )

    # build_executorch_binary appends .pte to file_name.
    file_stem = str(pathlib.Path(args.output).with_suffix(""))
    build_executorch_binary(
        model=actor,
        qnn_config=qnn_config,
        file_name=file_stem,
        dataset=calibration,
        custom_quantizer=quantizer,
    )


def _lower_explicit(actor, calibration, args, spec):
    """Manual export -> quantize -> partition -> lower.

    More code, but every stage is visible, which is what you want when the
    helper misbehaves on a multi-input signature.
    """
    from executorch.backends.qualcomm.partition.qnn_partitioner import (
        QnnPartitioner,
    )
    from executorch.backends.qualcomm.quantizer.quantizer import (
        QnnQuantizer,
        QuantDtype,
    )
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_soc_to_chipset_map,
    )
    from executorch.exir import to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import (
        convert_pt2e,
        prepare_pt2e,
    )

    example = example_inputs(spec)

    print("[1/5] torch.export")
    exported = torch.export.export(actor, example, strict=True)

    print("[2/5] prepare_pt2e (8a8w)")
    quantizer = QnnQuantizer()
    quantizer.set_default_quant_config(QuantDtype.use_8a8w)
    prepared = prepare_pt2e(exported.module(), quantizer)

    print(f"[3/5] calibrating on {len(calibration)} samples")
    with torch.no_grad():
        for sample in calibration:
            prepared(*sample)

    print("[4/5] convert_pt2e")
    converted = convert_pt2e(prepared)

    print("[5/5] partition + lower to HTP")
    compiler_spec = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[args.soc_model],
        backend_options=generate_htp_compiler_spec(use_fp16=False),
    )
    lowered = to_edge_transform_and_lower(
        torch.export.export(converted, example, strict=True),
        partitioner=[QnnPartitioner(compiler_spec)],
    ).to_executorch()

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        lowered.write_to_file(fh)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=None,
                   help="Trained checkpoint. Omit for an untrained smoke test.")
    p.add_argument("--output", default="models/nav_policy_qnn.pte")
    p.add_argument("--calibration", default=None,
                   help="npz from tools/calib_dataset.py. Strongly recommended.")
    p.add_argument("--calibration-samples", type=int, default=512)
    p.add_argument("--soc-model", default="QCS8300",
                   help="Patched-in stand-in for QCS8275 / HTP V75.")
    p.add_argument("--build-folder", default=os.path.join(
        os.environ.get("EXECUTORCH_ROOT", "/opt/executorch"), "build-x86"))
    p.add_argument("--flow", choices=("helper", "explicit"), default="helper")
    p.add_argument("--no-strict-fingerprint", action="store_true")
    p.add_argument("--skip-env-check", action="store_true")
    args = p.parse_args()

    spec = DEFAULT_SPEC
    if "QNN_SDK_ROOT" not in os.environ and not args.skip_env_check:
        raise SystemExit(
            "QNN_SDK_ROOT is not set. Source the board environment first "
            "(see docs/deployment.md), or pass --skip-env-check to build the "
            "graph without the QNN SDK for inspection only.")

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    actor = load_actor(args.weights, spec,
                       strict_fingerprint=not args.no_strict_fingerprint)
    n_params = sum(p_.numel() for p_ in actor.parameters())
    print(f"Actor: {n_params / 1e6:.2f}M parameters, "
          f"inputs {[tuple(s) for s in spec.actor_input_shapes]}")

    calibration = (load_calibration(args.calibration, spec,
                                    args.calibration_samples)
                   if args.calibration
                   else synthetic_calibration(spec, n=64))

    print(f"Lowering via '{args.flow}' flow to {out} ...")
    if args.flow == "helper":
        _lower_helper(actor, calibration, args, spec)
    else:
        _lower_explicit(actor, calibration, args, spec)

    if not out.exists():
        raise SystemExit(f"ERROR: lowering did not produce {out}")
    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {out} ({size_mb:.2f} MB)")

    write_sidecar(out, spec, backend="qnn-htp", quant="8a8w",
                  extra={"soc_model": args.soc_model, "flow": args.flow,
                         "calibration_samples": len(calibration),
                         "calibration_source": args.calibration or "synthetic",
                         "parameters": n_params})

    print("\nNext: verify on the NPU and gate on parity before any robot run:")
    print(f"  python3 tools/parity_check.py --pte {out} --weights {args.weights}")
    print(f"  python3 tools/bench_latency.py --pte {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
