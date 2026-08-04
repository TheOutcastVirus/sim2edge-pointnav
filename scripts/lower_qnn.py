"""Lower the deployable actor to a QNN .pte. Runs on the board, not the host.

The AOT pybind only exists on the board (see docs/deployment notes: the
aarch64 pybind patch), so this is deliberately a board-side script.

Two things it refuses to let you get wrong:

* **Delegation is reported, never assumed.** A graph that silently falls back
  to CPU for one operator still runs and still produces plausible numbers; it
  just misses the latency budget for a reason no benchmark explains. The
  report counts delegated calls against leftover aten operators.
* **Quantization error is measured against the float module** on the same
  calibration observations, so a broken observer shows up here rather than as
  an unexplained sim-to-real gap.

``--soc-model QCS8300`` is not a typo and not a different chip: ExecuTorch has
no QCS8275 entry, and the patched QCS8300 enum maps to HTP V75, which is what
``qnn-platform-validator --backend dsp --coreVersion`` reports on this board.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from executorch.backends.qualcomm.quantizer.quantizer import (QnnQuantizer,
                                                              QuantDtype)
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec, generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn)
from torchao.quantization.pt2e.observer import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from sim2edge import spec
from sim2edge.models.nav_policy import rewrite_circular_convolutions
from sim2edge.training import load_policy_checkpoint

QUANT_DTYPES = {
    "use_8a8w": QuantDtype.use_8a8w,
    "use_16a8w": QuantDtype.use_16a8w,
    "use_16a16w": QuantDtype.use_16a16w,
}


def _delegation(edge_program) -> dict:
    """Count delegate calls against operators the partitioner declined."""
    delegated, fallback = 0, {}
    for node in edge_program.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if "executorch_call_delegate" in target:
            delegated += 1
        elif target.startswith(("aten.", "torch.ops.aten")) or "aten" in target:
            name = target.replace("torch.ops.", "")
            fallback[name] = fallback.get(name, 0) + 1
    return {"delegate_calls": delegated,
            "fallback_operators": dict(sorted(fallback.items())),
            "fully_delegated": delegated > 0 and not fallback}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--soc-model", default="QCS8300")
    parser.add_argument("--quant-dtype", default="use_8a8w",
                        choices=sorted(QUANT_DTYPES))
    parser.add_argument("--fp16", action="store_true",
                        help="skip quantization and lower in float16")
    parser.add_argument("--per-channel", action="store_true",
                        help="per-channel conv/linear weight observers; the "
                             "second escalation when the parity gate fails")
    parser.add_argument("--calibration-batch", type=int, default=32)
    parser.add_argument("--moving-average-observer", action="store_true",
                        help="use the quantizer's default activation observer "
                             "instead of true min/max. It averages per-batch "
                             "extremes, which narrows the pre-tanh range and "
                             "clips saturated turn commands.")
    parser.add_argument("--keep-circular-padding", action="store_true",
                        help="do not rewrite circular Conv1d padding; the "
                             "QNN partitioner has no visitor for the ops it "
                             "lowers to and will raise KeyError")
    args = parser.parse_args()

    original = load_policy_checkpoint(args.checkpoint, device="cpu").eval()
    if args.keep_circular_padding:
        actor = original
        rewrite_difference = None
    else:
        actor = rewrite_circular_convolutions(original).eval()
    payload = np.load(args.calibration, allow_pickle=True)
    observations = torch.from_numpy(payload["observations"]).float()
    calibration_meta = json.loads(str(payload["meta"])) if "meta" in payload \
        else {}
    if observations.shape[1] != actor.config.observation_dim:
        raise SystemExit(
            f"calibration width {observations.shape[1]} != actor "
            f"observation_dim {actor.config.observation_dim}")

    example = (observations[:1].clone(),)
    with torch.inference_mode():
        reference = original(observations)
        if not args.keep_circular_padding:
            # The rewrite claims to be an identity. Check it on the real
            # calibration observations rather than trusting the argument.
            rewrite_difference = float(
                (actor(observations) - reference).abs().max())
            if rewrite_difference != 0.0:
                raise SystemExit(
                    "circular padding rewrite changed the model: max |diff| "
                    f"{rewrite_difference:.3e}, expected exactly 0")

    module = actor
    if not args.fp16:
        quantizer = QnnQuantizer()
        # The default activation observer tracks a *moving average* of each
        # batch's min and max, so a range only the rare batch reaches gets
        # averaged away. The commands that live at that range are the hard
        # avoidance turns: measured on this actor, the pre-tanh logit clipped
        # at about +/-0.72 and every sample with |omega| near the 1.9 rad/s
        # limit collapsed onto the same clamped output, for a worst-case error
        # of 37% of the limit that did not improve with more bits. True min/max
        # is not the usual advice, but here the tail *is* the operating
        # envelope and tanh bounds the output anyway.
        observer = None if args.moving_average_observer else MinMaxObserver
        quantizer.set_default_quant_config(
            QUANT_DTYPES[args.quant_dtype],
            is_conv_per_channel=args.per_channel,
            is_linear_per_channel=args.per_channel,
            **({} if observer is None else {"act_observer": observer}))
        captured = torch.export.export(actor, example, strict=False).module()
        prepared = prepare_pt2e(captured, quantizer)
        with torch.no_grad():
            for start in range(0, observations.shape[0],
                               args.calibration_batch):
                prepared(observations[start:start + args.calibration_batch])
        module = convert_pt2e(prepared)
        with torch.no_grad():
            quantized_out = module(observations)
        error = (quantized_out - reference).abs()
        # Reported per action axis and relative to that axis's own envelope:
        # v and omega differ by roughly an order of magnitude in SI units, so a
        # pooled figure hides which command is actually being degraded.
        limits = (actor.config.v_max, actor.config.omega_max)
        quant_error = {
            "max_abs": float(error.max()),
            "mean_abs": float(error.mean()),
            "reference_abs_mean": float(reference.abs().mean()),
            "per_axis": {
                name: {
                    "limit": limit,
                    "mean_abs": float(error[:, index].mean()),
                    "max_abs": float(error[:, index].max()),
                    "mean_fraction_of_limit":
                        float(error[:, index].mean()) / limit,
                    "max_fraction_of_limit":
                        float(error[:, index].max()) / limit,
                }
                for index, (name, limit) in enumerate(zip(("v", "omega"),
                                                          limits))
            },
        }
    else:
        quant_error = None

    backend_options = generate_htp_compiler_spec(use_fp16=args.fp16)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.soc_model),
        backend_options=backend_options)

    started = time.perf_counter()
    edge = to_edge_transform_and_lower_to_qnn(module, example, compiler_specs)
    delegation = _delegation(edge.exported_program())
    executorch_program = edge.to_executorch()
    lowering_seconds = time.perf_counter() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as handle:
        executorch_program.write_to_file(handle)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "pte": str(args.out.resolve()),
        "pte_bytes": args.out.stat().st_size,
        "soc_model": args.soc_model,
        "precision": "fp16" if args.fp16 else args.quant_dtype,
        "per_channel": args.per_channel,
        "activation_observer": ("moving_average_min_max"
                                if args.moving_average_observer
                                else "min_max"),
        "delegation": delegation,
        "circular_padding_rewritten": not args.keep_circular_padding,
        "circular_rewrite_max_abs_difference": rewrite_difference,
        "quantization_error": quant_error,
        "lowering_seconds": lowering_seconds,
        "sensor_contract_version": spec.SENSOR_CONTRACT_VERSION,
        "observation_fingerprint": spec.observation_fingerprint(),
        "calibration": calibration_meta,
    }
    report_path = args.out.with_suffix(".lowering.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"[qnn] wrote {args.out} ({report['pte_bytes'] / 1024:.0f} KiB) "
          f"in {lowering_seconds:.1f}s")
    print(f"[qnn] delegate calls {delegation['delegate_calls']}, "
          f"fully_delegated={delegation['fully_delegated']}")
    if delegation["fallback_operators"]:
        print(f"[qnn] CPU fallback: {delegation['fallback_operators']}")
    if quant_error is not None:
        print(f"[qnn] quantization |error| mean {quant_error['mean_abs']:.5f} "
              f"max {quant_error['max_abs']:.5f} against action scale "
              f"{quant_error['reference_abs_mean']:.5f}")
    print(f"[qnn] report {report_path}")


if __name__ == "__main__":
    main()
