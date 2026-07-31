"""PPO fine-tuning of a behaviour-cloned navigation policy, in Isaac Lab.

    ~/isaac-venv/bin/python scripts/train_ppo.py \
        --init runs/bc_stack_dagger_v2_e4_final/best.pt \
        --num_envs 2048 --iterations 300 --out runs/ppo_v1

This file owns the rollout and nothing else: the algorithm lives in
``sim2edge/training/ppo.py``, where it is tested without a simulator, and the
reward lives in ``isaaclab_env/nav_task.py``, where the geodesic field it reads
already exists. What is here is the loop between them.

Two properties of that loop are worth stating because they are easy to get
silently wrong:

**The observation is corrupted here, unlike during collection.** On-policy data
is consumed once, so there is no stored shard whose noise model must stay
retunable; `--noise-scale` therefore drives ``NavEnvCfg.observation_noise_scale``
and the policy trains on the input distribution BC trained on and evaluation
measures.

**Truncation is treated as termination in the advantage estimate.** Bootstrapping
a timed-out episode correctly needs the observation from *before* the automatic
reset, which the Direct workflow does not hand back. Held-out evaluation puts
timeouts under 1% of episodes against a 40% collision rate, so the bias is far
below what this run is trying to move. Watch ``timeout_rate`` in the log: if it
climbs, that reasoning has expired.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time


#: Algorithm knobs whose default is whatever ``PPOConfig`` says. They are left
#: unset here rather than duplicated, so ``--help`` costs no torch import --
#: which matters because torch must not be imported before Isaac's app launches.
_PPO_OVERRIDES = ("iterations", "num_steps", "critic_warmup_iters",
                  "anchor_coef", "actor_lr", "critic_lr", "clip_range",
                  "entropy_coef", "init_std", "seed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", type=Path,
                        help="BC checkpoint to initialize and anchor the actor")
    parser.add_argument("--out", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--resume", type=Path,
                        help="PPO checkpoint to continue, from this run")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--num-layouts", type=int, default=4096)
    parser.add_argument("--layout-seed", type=int, default=24680,
                        help="training layouts; keep disjoint from evaluation")
    parser.add_argument("--starts-per-layout", type=int, default=32)
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="live sensor/odometry corruption, 1.0 = the "
                             "ranges BC trained and evaluation reports at")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num-steps", type=int, dest="num_steps")
    parser.add_argument("--critic-warmup", type=int, dest="critic_warmup_iters",
                        help="iterations of value-only training")
    parser.add_argument("--anchor-coef", type=float, dest="anchor_coef",
                        help="0 disables the BC anchor entirely")
    parser.add_argument("--actor-lr", type=float, dest="actor_lr")
    parser.add_argument("--critic-lr", type=float, dest="critic_lr")
    parser.add_argument("--clip-range", type=float, dest="clip_range")
    parser.add_argument("--entropy-coef", type=float, dest="entropy_coef")
    parser.add_argument("--init-std", type=float, dest="init_std")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.init is not None and not args.init.is_file():
        raise FileNotFoundError(args.init)

    # Redirected to a file, stdout is block-buffered: a run whose shutdown
    # wedges would never show its progress at all.
    sys.stdout.reconfigure(line_buffering=True)

    # The app must launch before anything under isaaclab.sim is imported.
    from isaaclab.app import AppLauncher
    AppLauncher(headless=not args.gui)

    import torch

    from isaaclab_env.nav_task import NavEnvCfg, Sim2EdgeNavEnv
    from sim2edge.models import NavActorCritic, NavPolicy
    from sim2edge.spec import OBS_DIM
    from sim2edge.training import load_policy_checkpoint
    from sim2edge.training.ppo import (EpisodeStats, PPOConfig, PPOTrainer,
                                       RolloutBuffer, load_ppo_checkpoint,
                                       save_ppo_checkpoint)

    overrides = {name: getattr(args, name) for name in _PPO_OVERRIDES
                 if getattr(args, name, None) is not None}
    config = replace(PPOConfig(), **overrides)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume is None and (output / "latest.pt").exists():
        raise FileExistsError(
            f"run directory {output} already holds latest.pt; pass --resume or "
            f"choose a new output directory")

    env_cfg = NavEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.num_layouts = args.num_layouts
    env_cfg.layout_seed = args.layout_seed
    env_cfg.starts_per_layout = args.starts_per_layout
    env_cfg.seed = args.seed
    # PPO consumes the full task API: observations, reward and dones.
    env_cfg.collection_mode = False
    env_cfg.observation_noise_scale = args.noise_scale

    print(f"[PPO] building {args.num_envs} environments over "
          f"{args.num_layouts} layouts (seed {args.layout_seed}) ...")
    env = Sim2EdgeNavEnv(env_cfg)
    device = torch.device(env.device)
    torch.manual_seed(args.seed)

    # ── policy, anchor, trainer ──────────────────────────────────────────────
    if args.init is not None:
        actor = load_policy_checkpoint(args.init, device=device)
        print(f"[PPO] initialized the actor from {args.init}")
    else:
        actor = NavPolicy().to(device)
        print("[PPO] no --init: training the actor from scratch")
    if actor.config.observation_dim != OBS_DIM:
        raise ValueError(
            f"checkpoint expects a {actor.config.observation_dim}-element "
            f"observation, this task emits {OBS_DIM}")
    actor.train()

    policy = NavActorCritic(actor=actor, init_std=config.init_std).to(device)
    # Frozen at the weights BC ended on. A copy of the live actor would track
    # every update and regularize nothing. Snapshotted here, *before* any
    # --resume overwrites the actor, so a resumed run anchors to the same BC
    # policy the original run did rather than to wherever it drifted to.
    anchor = None
    if args.init is not None and config.anchor_coef > 0.0:
        anchor = copy.deepcopy(policy.actor).to(device)
    trainer = PPOTrainer(policy, config, anchor, device=device)

    start_iteration = 0
    best_success = 0.0
    if args.resume is not None:
        saved = load_ppo_checkpoint(args.resume, policy, trainer, device=device)
        start_iteration = trainer.iteration
        best_success = float(saved.get("metrics", {}).get("success_rate", 0.0))
        print(f"[PPO] resumed at iteration {start_iteration}")

    (output / "config.json").write_text(json.dumps(
        {"ppo": asdict(config), "num_envs": args.num_envs,
         "num_layouts": args.num_layouts, "layout_seed": args.layout_seed,
         "noise_scale": args.noise_scale,
         "init": str(args.init) if args.init else None},
        indent=2, sort_keys=True) + "\n")
    metrics_path = output / "metrics.jsonl"

    transitions = args.num_envs * config.num_steps
    print(f"[PPO] actor {policy.actor.parameter_count:,} parameters, "
          f"critic {sum(p.numel() for p in policy.critic.parameters()):,}")
    print(f"[PPO] {transitions:,} transitions/iteration, "
          f"{config.critic_warmup_iters} warmup iterations, "
          f"anchor {'on' if anchor is not None else 'off'}")

    # ── the loop ─────────────────────────────────────────────────────────────
    buffer = RolloutBuffer(config.num_steps, args.num_envs, OBS_DIM, device)
    stats = EpisodeStats(args.num_envs, device)
    rollout_generator = torch.Generator(device=device.type)
    rollout_generator.manual_seed(args.seed + 101)

    observation = env.reset(seed=args.seed)[0]["policy"]
    started = time.perf_counter()
    record: dict[str, float] = {}
    for iteration in range(start_iteration, config.iterations):
        iteration_started = time.perf_counter()
        buffer.reset()
        stats.reset()

        for _ in range(config.num_steps):
            with torch.no_grad():
                raw, action, log_prob, value = policy.act(
                    observation, rollout_generator)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            buffer.add(observation, raw, log_prob, value, reward, done)
            stats.update(reward, terminated, truncated,
                         env._reached, env._collided)
            # For a finished environment this is already the observation of a
            # fresh episode; GAE's not_done term is what keeps value from
            # crossing that seam.
            observation = next_observation["policy"]

        with torch.no_grad():
            last_value = policy.critic(observation)
        buffer.compute_advantages(last_value, config.gamma, config.gae_lambda)

        record = {**trainer.update(buffer), **stats.summary()}
        elapsed = time.perf_counter() - iteration_started
        record["steps_per_second"] = transitions / elapsed
        record["elapsed_seconds"] = time.perf_counter() - started
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

        if config.log_every and (iteration + 1) % config.log_every == 0:
            print(f"[PPO] iter {record['iteration']:>4}/{config.iterations}  "
                  f"SR {record['success_rate']:.1%}  "
                  f"coll {record['collision_rate']:.1%}  "
                  f"return {record['episode_return']:+.2f}  "
                  f"ev {record['explained_variance']:.2f}  "
                  f"kl {record['approx_kl']:.4f}  "
                  f"clip {record['clip_fraction']:.2f}  "
                  f"std {record['std_v']:.3f}/{record['std_omega']:.3f}  "
                  f"beta {record['anchor_weight']:.2f}  "
                  f"{record['steps_per_second']:,.0f} steps/s"
                  + ("  [warmup]" if record["warming_up"] else ""))

        # Selection is on closed-loop success, never on a loss. The offline
        # metric is what the BC phase already showed can improve while
        # navigation does not.
        if (not record["warming_up"] and record["episodes"] > 0
                and record["success_rate"] > best_success):
            best_success = record["success_rate"]
            save_ppo_checkpoint(output / "best.pt", policy, trainer, record)
        if (iteration + 1) % config.checkpoint_every == 0:
            save_ppo_checkpoint(output / "latest.pt", policy, trainer, record)

    if record:
        save_ppo_checkpoint(output / "latest.pt", policy, trainer, record)
        save_ppo_checkpoint(output / "final.pt", policy, trainer, record)
    print(f"[PPO] complete in {(time.perf_counter() - started) / 3600:.2f} h  "
          f"best rollout success {best_success:.1%}")

    # Kit shutdown has repeatedly wedged after all durable work completed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
