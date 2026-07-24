#!/usr/bin/env python3
"""Training entry point: BC (+DAgger), then optional PPO fine-tuning.

The two stages are deliberately separable. The BC checkpoint must stay
deployable at all times -- it is the Gate-B fallback, and with DAgger it is
usually demo-grade on its own. PPO is an upgrade that is allowed to fail.

    # Stage 1 only: the guaranteed-demo path
    python3 scripts/train.py bc --episodes 10000 --lidar-only

    # Stage 2: fine-tune a BC checkpoint
    python3 scripts/train.py ppo --init runs/bc/bc_final.pt --iterations 3000

    # Both
    python3 scripts/train.py all --episodes 10000

This drives the portable reference environment. Isaac Lab training uses
``isaaclab_env/train_rsl_rl.py``; both consume the same spec, so checkpoints are
interchangeable.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim2edge.envs.reference_env import EnvConfig  # noqa: E402
from sim2edge.eval.harness import evaluate, evaluate_noise_tiers  # noqa: E402
from sim2edge.eval.torch_policy import TorchPolicy  # noqa: E402
from sim2edge.models.nav_policy import build_policy  # noqa: E402
from sim2edge.spec import DEFAULT_SPEC  # noqa: E402
from sim2edge.training.bc import BCConfig, run_bc_pipeline  # noqa: E402
from sim2edge.training.ppo import PPOConfig, PPOTrainer  # noqa: E402


def _env_config(args, spec):
    return EnvConfig(spec=spec, use_depth=not args.lidar_only)


def cmd_bc(args, spec):
    cfg = BCConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        dagger_rounds=[args.dagger_episodes] * args.dagger_rounds,
        dagger_beta=[max(0.1, 0.5 / (i + 1)) for i in range(args.dagger_rounds)],
    )
    report = run_bc_pipeline(
        n_episodes=args.episodes,
        config=cfg,
        env_config=_env_config(args, spec),
        spec=spec,
        out_dir=args.out,
        eval_episodes=args.eval_episodes,
    )
    (pathlib.Path(args.out) / "bc_report.json").write_text(
        json.dumps(report, indent=2, default=float))
    return report


def cmd_ppo(args, spec):
    policy = build_policy(spec)
    baseline_sr = None
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        got = ckpt.get("spec_fingerprint")
        if got and got != spec.fingerprint():
            raise SystemExit(
                f"ERROR: {args.init} was trained under spec fingerprint {got!r}, "
                f"current spec is {spec.fingerprint()!r}.")
        policy.load_state_dict(ckpt["model"])
        baseline_sr = ckpt.get("metrics", {}).get("success_rate")
        print(f"Initialized from {args.init}"
              + (f" (BC baseline SR {baseline_sr:.1%})" if baseline_sr else ""))

    trainer = PPOTrainer(
        policy,
        PPOConfig(
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            iterations=args.iterations,
            critic_warmup_iters=args.critic_warmup,
            anchor_beta=args.anchor_beta,
            device=args.device,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
        ),
        env_config=_env_config(args, spec),
        spec=spec,
        # Anchor to the BC policy as loaded, not to the live one: anchoring to a
        # moving copy of the policy is a no-op regularizer.
        bc_reference=build_policy(spec) if not args.init else None,
    )
    trainer.train(out_dir=args.out, bc_baseline_sr=baseline_sr)
    return {"iterations": args.iterations}


def cmd_eval(args, spec):
    policy = TorchPolicy.from_checkpoint(args.init, spec, args.device)
    tiers = evaluate_noise_tiers(policy, args.eval_episodes,
                                 base_config=_env_config(args, spec), spec=spec)
    print("\n── Noise-tier robustness ──")
    for name, metrics in tiers.items():
        print(f"  {name:12s} {metrics.summary()}")

    safety_cfg = _env_config(args, spec)
    safety_cfg = EnvConfig(spec=spec, use_depth=safety_cfg.use_depth,
                           apply_safety=True)
    with_safety = evaluate(policy, args.eval_episodes, safety_cfg, spec=spec)
    print(f"\n  with safety layer: {with_safety.summary()}")
    print("  (interventions are a first-class metric: a run that succeeds only "
          "because the watchdog kept saving it is a finding, not a success)")

    out = {name: m.to_dict() for name, m in tiers.items()}
    out["with_safety"] = with_safety.to_dict()
    if args.out:
        pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(args.out) / "eval.json").write_text(
            json.dumps(out, indent=2))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("bc", "ppo", "all", "eval"))
    p.add_argument("--episodes", type=int, default=10_000,
                   help="Oracle episodes for the initial BC set.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dagger-rounds", type=int, default=2)
    p.add_argument("--dagger-episodes", type=int, default=2000)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--critic-warmup", type=int, default=500)
    p.add_argument("--anchor-beta", type=float, default=1.0)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--lidar-only", action="store_true",
                   help="Skip depth. Trains far faster; this is the Gate-B "
                        "guaranteed demo configuration.")
    p.add_argument("--init", default=None, help="Checkpoint to start from.")
    p.add_argument("--out", default="runs/train")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    spec = DEFAULT_SPEC
    print(f"spec fingerprint {spec.fingerprint()}  device {args.device}  "
          f"{'lidar-only' if args.lidar_only else 'depth+lidar'}")

    if args.stage in ("bc", "all"):
        cmd_bc(args, spec)
        if args.stage == "all":
            args.init = str(pathlib.Path(args.out) / "bc_final.pt")
    if args.stage in ("ppo", "all"):
        cmd_ppo(args, spec)
    if args.stage == "eval":
        if not args.init:
            raise SystemExit("eval requires --init <checkpoint>")
        cmd_eval(args, spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
