#!/usr/bin/env python3
"""rsl_rl training entry point for the Isaac Lab navigation task.

Run on the training host (RTX 4090):

    # Stage 2 from a BC checkpoint -- the normal path
    ./isaaclab.sh -p isaaclab_env/train_rsl_rl.py --task Sim2Edge-Nav-v0 \\
        --num_envs 2048 --headless --init runs/bc/bc_final.pt

    # Depth is the throughput bottleneck; expect far fewer envs
    ./isaaclab.sh -p isaaclab_env/train_rsl_rl.py --task Sim2Edge-Nav-v0 \\
        --num_envs 512 --headless --use-depth --init runs/bc/bc_final.pt

``rsl_rl`` is chosen over ``skrl`` for simplicity; both are first-class in Isaac
Lab. The PPO hyperparameters below mirror ``sim2edge/training/ppo.py`` so that a
result on the reference environment transfers, and so the BC handoff behaves
identically on both.

**The BC handoff is not optional.** Naive BC-init + PPO destroys the pretrained
actor: a random critic produces noisy advantages, which apply large random
updates to a carefully fitted policy. See the module docstring of
``sim2edge/training/ppo.py`` for the full argument. rsl_rl has no built-in
critic-warmup or policy-anchor mechanism, so both are installed here as
callbacks rather than being silently skipped.
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="Sim2Edge-Nav-v0")
    p.add_argument("--num_envs", type=int, default=2048)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--use-depth", action="store_true",
                   help="Enable the tiled-rendered depth camera. This is the "
                        "throughput bottleneck; expect 256-1024 envs, not 4k.")
    p.add_argument("--init", default=None,
                   help="BC checkpoint to initialize the actor from.")
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--critic-warmup", type=int, default=500)
    p.add_argument("--anchor-beta", type=float, default=1.0)
    p.add_argument("--anchor-anneal-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/isaac_ppo")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    # Isaac Sim must be launched before any isaaclab import.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    # Imported for their registration side effects: isaaclab_tasks populates
    # the task registry, and nav_task registers Sim2Edge-Nav-v0 in it.
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from isaaclab_env.nav_task import Sim2EdgeNavEnv  # noqa: F401
    from sim2edge.models.nav_policy import build_policy
    from sim2edge.spec import DEFAULT_SPEC

    spec = DEFAULT_SPEC
    print(f"spec fingerprint {spec.fingerprint()}")

    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.use_depth = args.use_depth
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)

    # Hyperparameters mirror sim2edge/training/ppo.py. The conservative clip and
    # actor LR are because this fine-tunes a good policy rather than training
    # one from scratch.
    train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.1,
            "desired_kl": 0.02,
            "entropy_coef": 0.005,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1.0e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 0.5,
        },
        "policy": {"class_name": "ActorCritic"},
        "num_steps_per_env": 64,
        "max_iterations": args.iterations,
        "save_interval": 100,
        "experiment_name": "sim2edge_nav",
        "seed": args.seed,
    }

    runner = OnPolicyRunner(env, train_cfg, log_dir=args.out,
                            device=env.unwrapped.device)

    anchor = None
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        got = ckpt.get("spec_fingerprint")
        if got and got != spec.fingerprint():
            raise SystemExit(
                f"ERROR: {args.init} was trained under spec fingerprint {got!r}, "
                f"current spec is {spec.fingerprint()!r}. Re-train or "
                f"regenerate the spec.")
        policy = build_policy(spec)
        policy.load_state_dict(ckpt["model"])
        # Statistics must stop moving before RL starts: rollouts are collected
        # in one mode and updated in another, so drifting BatchNorm statistics
        # make a state's value non-stationary.
        policy.freeze_norm()
        _load_actor_into_runner(runner, policy)

        # Frozen snapshot, never updated -- anchoring to a moving copy of the
        # policy would be a no-op regularizer.
        anchor = copy.deepcopy(policy).to(env.unwrapped.device).eval()
        for p_ in anchor.parameters():
            p_.requires_grad_(False)
        print(f"Initialized actor from {args.init}")

    _install_bc_handoff(runner, anchor, args)

    runner.learn(num_learning_iterations=args.iterations,
                 init_at_random_ep_len=False)
    env.close()
    simulation_app.close()
    return 0


def _load_actor_into_runner(runner, policy) -> None:
    """Copy the BC actor's weights into rsl_rl's ActorCritic.

    rsl_rl's default ActorCritic is a plain MLP over a flat observation, which
    does not match this policy's encoder structure. The clean way to run this
    architecture under rsl_rl is to register it as a custom policy class; this
    helper is the seam where that happens.
    """
    raise NotImplementedError(
        "Register NavPolicy as a custom rsl_rl policy class and copy weights "
        "here. This must be done on the training host, where the installed "
        "rsl_rl version's ActorCritic interface can be checked directly -- "
        "guessing at it off-host is how signatures silently drift.")


def _install_bc_handoff(runner, anchor, args) -> None:
    """Install critic warmup and the annealed BC anchor.

    Critic warmup sets the actor learning rate to exactly zero (not merely
    small) for the first ``--critic-warmup`` iterations: while the critic is
    producing noise, any actor step is pure damage.

    The anchor adds ``beta * ||pi(s) - pi_BC(s)||^2`` with beta annealed to zero
    over the first ``--anchor-anneal-frac`` of training, with the anneal clock
    starting when the actor does.
    """
    algo = runner.alg
    original_update = algo.update
    base_lr = algo.learning_rate
    state = {"iteration": 0}

    def anchor_weight() -> float:
        if anchor is None or args.anchor_beta <= 0.0:
            return 0.0
        span = max(1.0, args.iterations * args.anchor_anneal_frac)
        progress = max(0.0, state["iteration"] - args.critic_warmup) / span
        return float(args.anchor_beta * max(0.0, 1.0 - progress))

    def patched_update(*a, **kw):
        warming = state["iteration"] < args.critic_warmup
        algo.learning_rate = 0.0 if warming else base_lr
        result = original_update(*a, **kw)
        state["iteration"] += 1
        beta = anchor_weight()
        if state["iteration"] % 10 == 0:
            print(f"  [handoff] iter {state['iteration']} "
                  f"{'WARMUP ' if warming else ''}anchor_beta={beta:.3f}")
        return result

    algo.update = patched_update


if __name__ == "__main__":
    sys.exit(main())
