#!/usr/bin/env python3
"""Build the INT8 quantization calibration set.

Calibration decides the activation ranges the quantizer bakes into the ``.pte``.
Get it wrong and the model does not error — it produces confident, plausible,
wrong actions. Two rules follow, and this tool exists to enforce them
structurally rather than by discipline:

1. **Calibration data must pass through the identical preprocessing the C++ node
   performs.** This tool never constructs an observation by hand; it calls
   ``sim2edge.preprocess``, the same code the environment and the parity test
   use.
2. **Calibration data must cover the deployment distribution.** Sim rollouts of
   the *final* policy (not the oracle — the deployed policy visits its own
   states) mixed with real recorded rosbags. Sim-only calibration systematically
   misses the real sensor's noise floor and hole statistics.

The output carries the spec fingerprint, and the export scripts refuse a
calibration set whose fingerprint does not match.

Usage:
    # sim rollouts of a trained policy
    python3 tools/calib_dataset.py --weights runs/bc/bc_final.pt \\
        --episodes 40 --output calib/nav_calib.npz

    # mix in real data recorded on the robot
    python3 tools/calib_dataset.py --weights runs/bc/bc_final.pt \\
        --episodes 30 --bag-npz calib/real_obs.npz --output calib/nav_calib.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim2edge.envs.reference_env import EnvConfig, NavEnv  # noqa: E402
from sim2edge.eval.harness import oracle_policy  # noqa: E402
from sim2edge.spec import DEFAULT_SPEC  # noqa: E402


def collect_sim(args, spec) -> dict:
    """Roll out a policy and keep its observations."""
    cfg = EnvConfig(spec=spec, use_depth=not args.lidar_only)
    env = NavEnv(cfg, seed=args.seed)

    if args.weights:
        from sim2edge.eval.torch_policy import TorchPolicy

        policy = TorchPolicy.from_checkpoint(args.weights, spec)
        print(f"Rolling out {args.weights}")
    else:
        # The oracle is a fallback only. Its state distribution is not the
        # deployed policy's, so the activation ranges it produces are the wrong
        # ones — acceptable for a first smoke export, not for a robot run.
        policy = oracle_policy(env, spec)
        print("WARNING: no --weights; calibrating on ORACLE rollouts. The "
              "deployed policy visits different states, so re-build this set "
              "from the final policy before any robot trial.", file=sys.stderr)

    buf = {name: [] for name in spec.obs.actor_inputs}
    max_steps = spec.reward.max_episode_steps
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        for t in range(max_steps):
            # Subsample within an episode: consecutive frames at 20 Hz are
            # near-duplicates and would let a few long episodes dominate the
            # observed activation ranges.
            if t % args.stride == 0:
                for name in spec.obs.actor_inputs:
                    buf[name].append(obs[name])
            res = env.step(policy(obs))
            obs = res.obs
            if res.terminated or res.truncated:
                break
        if (ep + 1) % 10 == 0:
            print(f"  episode {ep + 1}/{args.episodes}: "
                  f"{len(buf['lidar'])} samples")
    return {k: np.stack(v).astype(np.float32) for k, v in buf.items()}


def load_bag(path, spec) -> dict:
    """Load real observations previously extracted from a rosbag.

    Produce this file with ``tools/bag_to_obs.py``, which replays recorded
    ``/scan``, depth and odometry messages through the same preprocessing.
    """
    with np.load(path) as z:
        got = str(z["spec_fingerprint"]) if "spec_fingerprint" in z else None
        if got != spec.fingerprint():
            raise SystemExit(
                f"ERROR: {path} was built under spec fingerprint {got!r}, "
                f"current spec is {spec.fingerprint()!r}. Re-run bag_to_obs.py.")
        return {name: z[name].astype(np.float32)
                for name in spec.obs.actor_inputs}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=None)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--stride", type=int, default=5,
                   help="Keep every Nth control step within an episode.")
    p.add_argument("--bag-npz", default=None,
                   help="Real observations from tools/bag_to_obs.py.")
    p.add_argument("--real-fraction", type=float, default=0.35,
                   help="Target share of real samples in the final mix.")
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--lidar-only", action="store_true")
    p.add_argument("--output", default="calib/nav_calib.npz")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    spec = DEFAULT_SPEC
    sim = collect_sim(args, spec)
    n_sim = len(sim["lidar"])
    print(f"Sim samples: {n_sim}")

    parts = [sim]
    if args.bag_npz:
        real = load_bag(args.bag_npz, spec)
        n_real = len(real["lidar"])
        print(f"Real samples: {n_real}")
        # Trim sim so the real data is not swamped. Real data is what pins the
        # noise floor and the hole statistics, which sim only approximates.
        want_real = min(n_real, int(args.max_samples * args.real_fraction))
        want_sim = min(n_sim, args.max_samples - want_real)
        rng = np.random.default_rng(args.seed)
        sim = {k: v[rng.choice(n_sim, want_sim, replace=False)]
               for k, v in sim.items()}
        real = {k: v[rng.choice(n_real, want_real, replace=False)]
                for k, v in real.items()}
        parts = [sim, real]
        print(f"Mixed: {want_sim} sim + {want_real} real")

    merged = {name: np.concatenate([p_[name] for p_ in parts], axis=0)
              for name in spec.obs.actor_inputs}
    n = len(merged["lidar"])
    if n > args.max_samples:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(n, args.max_samples, replace=False)
        merged = {k: v[sel] for k, v in merged.items()}
        n = args.max_samples

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, spec_fingerprint=np.array(spec.fingerprint()),
                        **merged)

    print(f"\nWrote {out}: {n} observation tuples "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    print(f"Spec fingerprint: {spec.fingerprint()}")
    print("\nPer-channel observed ranges (what the quantizer will see):")
    for name in spec.obs.actor_inputs:
        a = merged[name]
        print(f"  {name:12s} shape {str(a.shape[1:]):16s} "
              f"min {a.min():+.4f}  max {a.max():+.4f}  "
              f"mean {a.mean():+.4f}  std {a.std():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
