#!/usr/bin/env python3
"""The parity gate. Nothing goes on the robot until this passes.

Two independent checks, both of which the implementation plan makes mandatory
before the first robot trial (§5, §6.2):

**1. Preprocessing parity (Python vs C++).** The two implementations of the
observation pipeline are compared on identical raw sensor data. This is the
mitigation for the failure mode that cost both reference projects the most time:
a calibration/runtime preprocessing mismatch produces silent garbage, never an
error. The check is:

    python3 tools/parity_check.py --emit-inputs /tmp/parity_in.bin
    ros2 run nav_policy_node preprocess_parity /tmp/parity_in.bin /tmp/parity_cpp.bin
    python3 tools/parity_check.py --emit-inputs /tmp/parity_in.bin \\
        --cpp-outputs /tmp/parity_cpp.bin

**2. Quantization parity (FP32 vs INT8 .pte).** The lowered artifact is replayed
on the same observations as the FP32 checkpoint, and — more importantly — is run
through the *full sim benchmark*, because an action MSE that looks small can
still cost success rate at exactly the moments that matter. The gate is a sim SR
drop under 2 percentage points, not a weight-error norm.

    python3 tools/parity_check.py --pte models/nav_policy_qnn.pte \\
        --weights runs/bc/bc_final.pt --sim-episodes 200
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim2edge.preprocess import Preprocessor  # noqa: E402
from sim2edge.spec import DEFAULT_SPEC, NavSpec  # noqa: E402

IN_MAGIC = b"S2EP"
OUT_MAGIC = b"S2EO"
FORMAT_VERSION = 1

#: Tolerance for preprocessing parity. Both sides do the same arithmetic in
#: float32, so agreement should be near-exact; anything above this indicates a
#: real algorithmic difference (a different pooling rule, a different invalid
#: convention, a transposed image), not floating-point drift.
PREPROCESS_TOL = 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Raw-sample interchange format (mirrored in preprocess_parity.cpp)
# ─────────────────────────────────────────────────────────────────────────────


def write_inputs(path: str, samples: List[Dict], spec: NavSpec) -> None:
    with open(path, "wb") as fh:
        fh.write(IN_MAGIC)
        fh.write(struct.pack("<ii", FORMAT_VERSION, len(samples)))
        for s in samples:
            depth = np.ascontiguousarray(s["depth_m"], dtype=np.float32)
            ranges = np.ascontiguousarray(s["ranges"], dtype=np.float32)
            fh.write(struct.pack("<ii", depth.shape[0], depth.shape[1]))
            fh.write(depth.tobytes())
            fh.write(struct.pack("<i", ranges.shape[0]))
            fh.write(struct.pack("<ffff", s["angle_min"], s["angle_increment"],
                                 s["range_min"], s["range_max"]))
            fh.write(ranges.tobytes())
            fh.write(struct.pack("<ffff", s["goal_x"], s["goal_y"],
                                 s["last_v"], s["last_omega"]))


def read_outputs(path: str, spec: NavSpec) -> List[Dict[str, np.ndarray]]:
    d, l = spec.obs.depth, spec.obs.lidar
    sizes = [("depth", d.numel), ("lidar", l.bins), ("goal", 3),
             ("last_action", 2)]
    out = []
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != OUT_MAGIC:
            raise SystemExit(f"ERROR: {path} is not a parity output file "
                             f"(magic {magic!r})")
        version, n = struct.unpack("<ii", fh.read(8))
        if version != FORMAT_VERSION:
            raise SystemExit(f"ERROR: {path} format version {version}, "
                             f"expected {FORMAT_VERSION}")
        for _ in range(n):
            sample = {}
            for name, count in sizes:
                buf = fh.read(4 * count)
                if len(buf) != 4 * count:
                    raise SystemExit(f"ERROR: {path} truncated in '{name}'")
                arr = np.frombuffer(buf, dtype="<f4").copy()
                sample[name] = (arr.reshape(1, d.height, d.width)
                                if name == "depth" else arr)
            out.append(sample)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sample generation
# ─────────────────────────────────────────────────────────────────────────────


def make_samples(n: int, spec: NavSpec, seed: int = 3) -> List[Dict]:
    """Raw sensor samples spanning the awkward cases.

    Deliberately includes the corner cases that break naive implementations:
    all-invalid scans, holes, values outside the clip range, negative and NaN
    depth, wrapped bearings. A parity test on clean data proves almost nothing —
    the two implementations diverge precisely on how they treat garbage.
    """
    from sim2edge.envs.reference_env import EnvConfig, NavEnv

    rng = np.random.default_rng(seed)
    env = NavEnv(EnvConfig(spec=spec, use_depth=True), seed=seed)
    samples: List[Dict] = []

    env.reset(seed=seed)
    while len(samples) < n:
        raw = env.raw_sensors()
        ranges, angle_min, angle_inc = raw["scan"]
        depth = raw["depth"].copy()
        ranges = ranges.copy()

        k = len(samples)
        if k % 7 == 1:  # stereo holes and speckle
            depth[rng.random(depth.shape) < 0.25] = 0.0
        elif k % 7 == 2:  # NaN / negative depth from a bad driver frame
            depth[rng.random(depth.shape) < 0.05] = np.nan
            depth[rng.random(depth.shape) < 0.05] = -1.0
        elif k % 7 == 3:  # beyond the far clip
            depth = depth * 3.0
        elif k % 7 == 4:  # scan fully invalid
            ranges[:] = np.inf
        elif k % 7 == 5:  # returns below the near clip
            ranges = np.minimum(ranges, 0.05)
        elif k % 7 == 6:  # scan reported over [0, 2pi) instead of [-pi, pi)
            angle_min = 0.0

        samples.append({
            "depth_m": depth,
            "ranges": ranges,
            "angle_min": float(angle_min),
            "angle_increment": float(angle_inc),
            "range_min": 0.15,
            "range_max": 12.0,
            "goal_x": float(rng.uniform(-9, 9)),
            "goal_y": float(rng.uniform(-9, 9)),
            "last_v": float(rng.uniform(0, spec.action.v_max)),
            "last_omega": float(rng.uniform(-spec.action.omega_max,
                                            spec.action.omega_max)),
        })

        res = env.step((rng.uniform(0, spec.action.v_max),
                        rng.uniform(-1, 1)))
        if res.terminated or res.truncated:
            env.reset()
    return samples


def python_reference(samples: List[Dict], spec: NavSpec) -> List[Dict]:
    pre = Preprocessor(spec)
    return [pre.build(
        depth_m=s["depth_m"],
        scan_ranges=s["ranges"],
        scan_angle_min=s["angle_min"],
        scan_angle_increment=s["angle_increment"],
        goal_xy_robot=(s["goal_x"], s["goal_y"]),
        last_action=(s["last_v"], s["last_omega"]),
        scan_range_min=s["range_min"],
        scan_range_max=s["range_max"],
    ) for s in samples]


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────


def check_preprocessing(samples, cpp_path: str, spec: NavSpec) -> bool:
    py = python_reference(samples, spec)
    cpp = read_outputs(cpp_path, spec)
    if len(cpp) != len(py):
        print(f"FAIL: C++ produced {len(cpp)} samples, expected {len(py)}")
        return False

    print(f"\n── Preprocessing parity: Python vs C++ ({len(py)} samples) ──")
    ok = True
    for name in spec.obs.actor_inputs:
        worst = 0.0
        worst_i = -1
        for i, (a, b) in enumerate(zip(py, cpp)):
            diff = float(np.max(np.abs(a[name].ravel() - b[name].ravel())))
            if diff > worst:
                worst, worst_i = diff, i
        status = "OK  " if worst <= PREPROCESS_TOL else "FAIL"
        if worst > PREPROCESS_TOL:
            ok = False
        print(f"  {status} {name:12s} max |diff| = {worst:.3e}"
              + (f"  (sample {worst_i})" if worst > PREPROCESS_TOL else ""))

    if not ok:
        print("\nPreprocessing implementations DISAGREE. Do not run on the "
              "robot: quantization calibration and runtime would see different "
              "inputs, which produces confident wrong actions rather than an "
              "error. Compare sim2edge/preprocess.py against "
              "ros_ws/src/nav_policy_node/src/preprocess.cpp.")
    return ok


def check_quantization(args, spec: NavSpec) -> bool:
    import torch

    from sim2edge.envs.reference_env import EnvConfig
    from sim2edge.eval.harness import evaluate
    from sim2edge.eval.torch_policy import ExecuTorchPolicy, TorchPolicy

    fp32 = TorchPolicy.from_checkpoint(args.weights, spec)
    quant = ExecuTorchPolicy(args.pte, spec)

    # ── Action-level comparison on a fixed observation batch ─────────────────
    if args.calibration:
        with np.load(args.calibration) as z:
            obs_batch = [{n: z[n][i] for n in spec.obs.actor_inputs}
                         for i in range(min(args.action_samples,
                                            z["lidar"].shape[0]))]
    else:
        samples = make_samples(min(args.action_samples, 256), spec)
        obs_batch = python_reference(samples, spec)

    ref = torch.tensor([fp32(o) for o in obs_batch])
    got = torch.tensor([quant(o) for o in obs_batch])
    from export_common import report_parity

    stats = report_parity(ref, got, spec, "INT8 .pte vs FP32 checkpoint")
    # Reported for the writeup, but deliberately NOT the gate. A small action
    # error can still cost success at exactly the moments that matter, which is
    # why the real criterion below is the sim benchmark.
    if stats["omega_rel_max"] > 0.25:
        print("  NOTE: worst-case angular error exceeds 25% of range. Even if "
              "the success-rate gate passes, inspect the calibration set: this "
              "usually means it under-covers the turning regime.")

    # ── The gate that actually matters: success rate in the sim benchmark ────
    if args.sim_episodes > 0:
        cfg = EnvConfig(spec=spec, use_depth=not args.lidar_only)
        print(f"\n── Sim benchmark, {args.sim_episodes} held-out episodes ──")
        m_fp32 = evaluate(fp32, args.sim_episodes, cfg, spec=spec)
        print(f"  FP32 : {m_fp32.summary()}")
        m_int8 = evaluate(quant, args.sim_episodes, cfg, spec=spec)
        print(f"  INT8 : {m_int8.summary()}")
        drop = m_fp32.success_rate - m_int8.success_rate
        print(f"\n  SR drop: {drop:+.1%} (gate: < {args.max_sr_drop:.1%})")
        if drop > args.max_sr_drop:
            print("\nFAIL: quantization costs too much success rate.\n"
                  "Escalation order (cheapest first):\n"
                  "  1. fix the calibration data — is it from the FINAL policy, "
                  "and does it include real rosbag observations?\n"
                  "  2. per-channel weight observers\n"
                  "  3. QAT")
            return False
    print("\nPASS: quantization parity within gate.")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emit-inputs", default=None,
                   help="Write raw sensor samples for the C++ tool to consume.")
    p.add_argument("--cpp-outputs", default=None,
                   help="C++ preprocessing output to compare against.")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--pte", default=None, help="Lowered model for quant parity.")
    p.add_argument("--weights", default=None, help="FP32 reference checkpoint.")
    p.add_argument("--calibration", default=None)
    p.add_argument("--action-samples", type=int, default=1000)
    p.add_argument("--sim-episodes", type=int, default=200)
    p.add_argument("--max-sr-drop", type=float, default=0.02)
    p.add_argument("--lidar-only", action="store_true")
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()

    spec = DEFAULT_SPEC
    ok = True

    if args.emit_inputs:
        samples = make_samples(args.samples, spec, args.seed)
        write_inputs(args.emit_inputs, samples, spec)
        print(f"Wrote {len(samples)} raw samples to {args.emit_inputs}")
        print(f"Spec fingerprint: {spec.fingerprint()}")
        if args.cpp_outputs:
            ok &= check_preprocessing(samples, args.cpp_outputs, spec)
        else:
            print("\nNow run the C++ side:\n"
                  f"  ros2 run nav_policy_node preprocess_parity "
                  f"{args.emit_inputs} /tmp/parity_cpp.bin\n"
                  "then re-run this command with --cpp-outputs /tmp/parity_cpp.bin")

    if args.pte:
        if not args.weights:
            raise SystemExit("--pte requires --weights for the FP32 reference")
        ok &= check_quantization(args, spec)

    if not args.emit_inputs and not args.pte:
        p.print_help()
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
