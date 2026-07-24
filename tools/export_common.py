"""Shared plumbing for the ExecuTorch export scripts.

Runs on the Ventuno Q inside the pinned ExecuTorch venv, so it must stay light:
torch and numpy only, no torchvision, no simulator, no scipy.

The single most important thing in this file is :func:`write_sidecar`. Every
``.pte`` is written with a JSON sidecar recording the spec fingerprint it was
quantized under, and the ROS node refuses to load a model whose fingerprint does
not match the one compiled into it. Both reference projects lost significant time
to calibration/runtime preprocessing mismatches, whose defining property is that
they produce plausible-looking garbage instead of an error. This turns that
silent failure into a loud one.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim2edge.models.nav_policy import ActorMean, build_policy  # noqa: E402
from sim2edge.spec import DEFAULT_SPEC, NavSpec  # noqa: E402

__all__ = ["load_actor", "example_inputs", "write_sidecar", "read_sidecar",
           "load_calibration", "synthetic_calibration", "report_parity"]


def load_actor(
    weights: Optional[str],
    spec: NavSpec = DEFAULT_SPEC,
    strict_fingerprint: bool = True,
) -> ActorMean:
    """Build the actor-mean module and load trained weights into it.

    ``weights=None`` yields an untrained network, which is exactly what the
    Day-1 smoke test wants: the point of that test is to prove the *four-input
    export path* works on this board, and it must not be blocked on having a
    trained policy.
    """
    policy = build_policy(spec)
    if weights:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        policy.load_state_dict(state)

        got = ckpt.get("spec_fingerprint") if isinstance(ckpt, dict) else None
        want = spec.fingerprint()
        if got and got != want:
            msg = (f"checkpoint was trained under spec fingerprint {got!r} but "
                   f"this environment's spec is {want!r}")
            if strict_fingerprint:
                raise SystemExit(
                    f"ERROR: {msg}.\nThe preprocessing contract changed since "
                    f"training. Re-train, or re-generate the spec, or pass "
                    f"--no-strict-fingerprint if you are certain.")
            print(f"WARNING: {msg}", file=sys.stderr)
    policy.eval()
    return policy.export_module()


def example_inputs(spec: NavSpec = DEFAULT_SPEC,
                   batch: int = 1) -> Tuple[torch.Tensor, ...]:
    """One example input tuple, in ``ObservationSpec.actor_inputs`` order."""
    return tuple(torch.zeros((batch,) + shape, dtype=torch.float32)
                 for shape in spec.actor_input_shapes)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data
# ─────────────────────────────────────────────────────────────────────────────


def load_calibration(
    path: str,
    spec: NavSpec = DEFAULT_SPEC,
    limit: int = 512,
) -> List[Tuple[torch.Tensor, ...]]:
    """Load a calibration set produced by ``tools/calib_dataset.py``.

    The file must have been written by that tool, which guarantees the samples
    went through the *identical* preprocessing the C++ node performs. Loading
    raw observations from anywhere else defeats the purpose.
    """
    import numpy as np  # noqa: PLC0415

    with np.load(path) as z:
        got = str(z["spec_fingerprint"]) if "spec_fingerprint" in z else None
        want = spec.fingerprint()
        if got != want:
            raise SystemExit(
                f"ERROR: calibration set {path} was built under spec "
                f"fingerprint {got!r}, current spec is {want!r}. "
                f"Rebuild it with tools/calib_dataset.py.")
        n = min(limit, int(z["lidar"].shape[0]))
        out = []
        for i in range(n):
            out.append(tuple(
                torch.from_numpy(z[name][i]).unsqueeze(0).float()
                for name in spec.obs.actor_inputs))
    print(f"Calibration: {n} real observation tuples from {path}")
    return out


def synthetic_calibration(
    spec: NavSpec = DEFAULT_SPEC,
    n: int = 64,
    seed: int = 0,
) -> List[Tuple[torch.Tensor, ...]]:
    """Fallback calibration from plausible-but-fake observations.

    Quantizes noticeably worse than real data and is only appropriate for the
    Day-1 smoke test. Real calibration must come from sim rollouts of the final
    policy plus the recorded rosbags, so the observed ranges match deployment.
    """
    print("WARNING: using SYNTHETIC calibration data. Activation ranges will "
          "not match deployment and the quantized policy may behave poorly. "
          "Build a real set with tools/calib_dataset.py.", file=sys.stderr)
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        sample = []
        for name, shape in zip(spec.obs.actor_inputs, spec.actor_input_shapes):
            if name in ("depth", "lidar"):
                # Sensor channels are normalized to [0, 1] and skewed toward
                # "far"; uniform noise would give the quantizer a range that
                # never occurs in practice.
                t = torch.clamp(
                    torch.rand(shape, generator=g) * 0.6 + 0.4, 0.0, 1.0)
            elif name == "goal":
                d = torch.rand(1, generator=g)
                a = (torch.rand(1, generator=g) * 2 - 1) * 3.14159
                t = torch.cat([d, torch.sin(a), torch.cos(a)])
            else:
                t = torch.rand(shape, generator=g) * 2 - 1
            sample.append(t.unsqueeze(0).float())
        out.append(tuple(sample))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sidecar metadata
# ─────────────────────────────────────────────────────────────────────────────


def write_sidecar(pte_path: str | pathlib.Path, spec: NavSpec, backend: str,
                  quant: str, extra: Optional[Dict] = None) -> pathlib.Path:
    """Write ``<model>.json`` next to the ``.pte``."""
    pte_path = pathlib.Path(pte_path)
    meta = {
        "model": pte_path.name,
        "backend": backend,
        "quantization": quant,
        "spec_version": spec.version,
        "spec_fingerprint": spec.fingerprint(),
        "actor_inputs": list(spec.obs.actor_inputs),
        "input_shapes": [list(s) for s in spec.actor_input_shapes],
        "output": ["v_mps", "omega_radps"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta.update(extra or {})
    side = pte_path.with_suffix(".json")
    side.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Sidecar: {side}  (fingerprint {meta['spec_fingerprint']})")
    return side


def read_sidecar(pte_path: str | pathlib.Path) -> Dict:
    side = pathlib.Path(pte_path).with_suffix(".json")
    if not side.exists():
        raise FileNotFoundError(f"no sidecar next to {pte_path}")
    return json.loads(side.read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Parity reporting
# ─────────────────────────────────────────────────────────────────────────────


def report_parity(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    spec: NavSpec = DEFAULT_SPEC,
    label: str = "quantized vs fp32",
) -> Dict[str, float]:
    """Compare two action batches and report error *relative to action range*.

    Absolute MSE on ``(v, omega)`` is close to meaningless because the two
    channels have ranges differing by 10x. What matters operationally is the
    error as a fraction of each channel's usable range.
    """
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()
    a = spec.action
    ranges = torch.tensor([a.v_max - a.v_min, 2.0 * a.omega_max])

    err = (cand - ref).abs()
    rel = err / ranges
    stats = {
        "mse": float(((cand - ref) ** 2).mean()),
        "v_mae": float(err[:, 0].mean()),
        "omega_mae": float(err[:, 1].mean()),
        "v_rel_mae": float(rel[:, 0].mean()),
        "omega_rel_mae": float(rel[:, 1].mean()),
        "v_rel_max": float(rel[:, 0].max()),
        "omega_rel_max": float(rel[:, 1].max()),
    }
    print(f"\n── Parity: {label} ({len(ref)} samples) ──")
    print(f"  v      MAE {stats['v_mae']:.5f} m/s   "
          f"({stats['v_rel_mae']:.2%} of range, max {stats['v_rel_max']:.2%})")
    print(f"  omega  MAE {stats['omega_mae']:.5f} rad/s "
          f"({stats['omega_rel_mae']:.2%} of range, max {stats['omega_rel_max']:.2%})")
    return stats
