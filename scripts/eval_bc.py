"""Closed-loop Isaac evaluation of a behaviour-cloning checkpoint.

Every environment contributes exactly one episode. Re-seeding the initial reset
therefore gives each noise tier the same layouts; policy divergence cannot
silently change the evaluation set. The environment uses collection mode and
this script assembles observations from raw sensors through the same functions
used by the dataset loader.

    ~/isaac-venv/bin/python scripts/eval_bc.py runs/bc_v1/best.pt \
        --episodes 512 --out runs/bc_v1/eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=512,
                        help="also the number of parallel environments")
    parser.add_argument("--num-layouts", type=int, default=512)
    parser.add_argument("--layout-seed", type=int, default=10_000_000,
                        help="must differ from the collection layout seed")
    parser.add_argument("--noise-scales", type=float, nargs="+",
                        default=(0.0, 1.0, 2.0))
    parser.add_argument("--out", type=Path, default=Path("runs/bc_eval.json"))
    parser.add_argument("--gui", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")

    # Isaac's native modules become importable only after AppLauncher starts.
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=not args.gui).app

    import torch

    from isaaclab_env.nav_task import NavEnvCfg, Sim2EdgeNavEnv, normalize_action
    from sim2edge import noise as noise_mod
    from sim2edge.preprocess import bin_lidar, build_observation, goal_features
    from sim2edge.spec import (DT, GOAL_TOLERANCE_M, OMEGA_MAX, V_MAX,
                               fingerprint)
    from sim2edge.training import load_policy_checkpoint

    sys.stdout.reconfigure(line_buffering=True)
    config = NavEnvCfg()
    config.scene.num_envs = args.episodes
    config.num_layouts = args.num_layouts
    config.layout_seed = args.layout_seed
    config.seed = args.layout_seed
    config.collection_mode = True

    print(f"[eval] building {args.episodes} environments with held-out layout "
          f"seed {args.layout_seed} ...")
    env = Sim2EdgeNavEnv(config)
    device = torch.device(env.device)
    policy = load_policy_checkpoint(args.checkpoint, device=device)
    print(f"[eval] policy ready: {policy.parameter_count:,} parameters on {device}")

    reports = []
    reference_layouts = None
    for scale in args.noise_scales:
        report, layouts = _evaluate_tier(env, policy, float(scale),
                                         args.layout_seed, torch,
                                         noise_mod, bin_lidar,
                                         build_observation, goal_features,
                                         normalize_action, DT, V_MAX, OMEGA_MAX,
                                         GOAL_TOLERANCE_M)
        if (reference_layouts is not None
                and not torch.equal(layouts, reference_layouts)):
            raise RuntimeError("re-seeded evaluation produced different layouts")
        if reference_layouts is None:
            reference_layouts = layouts
        reports.append(report)
        print(f"[eval] noise {scale:g}x  SR {report['success_rate']:.1%}  "
              f"SPL {report['spl']:.3f}  collision "
              f"{report['collision_rate']:.1%}  timeout "
              f"{report['timeout_rate']:.1%}  jerk {report['mean_jerk']:.3f}")

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "spec_fingerprint": fingerprint(),
        "episodes": args.episodes,
        "num_layouts": args.num_layouts,
        "layout_seed": args.layout_seed,
        "paired_layouts_across_tiers": True,
        "tiers": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[eval] wrote {args.out}")

    # Kit shutdown has repeatedly wedged after all durable work completed. This
    # is a standalone worker and the OS can release its native resources safely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _evaluate_tier(env, policy, scale, seed, torch, noise_mod,
                   bin_lidar, build_observation, goal_features,
                   normalize_action, dt, v_max, omega_max, goal_tolerance):
    """Run one initial episode per environment and aggregate on the GPU."""
    torch.manual_seed(seed)
    env.reset(seed=seed)
    layout_ids = env._layout.detach().cpu().clone()
    n = env.num_envs
    device = torch.device(env.device)
    noise_generator = torch.Generator(device=device.type).manual_seed(seed + 17)

    active = torch.ones(n, dtype=torch.bool, device=device)
    steps = torch.zeros(n, dtype=torch.long, device=device)
    distance = torch.zeros(n, device=device)
    jerk_sum = torch.zeros(n, device=device)
    previous = torch.zeros(n, 2, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    collision = torch.zeros(n, dtype=torch.bool, device=device)
    timeout = torch.zeros(n, dtype=torch.bool, device=device)

    # Each env gets one initial layout. Auto-resets after completion are ignored,
    # which makes this exact layout vector reproducible at every noise tier.
    # The environment records the geodesic for the start sampled on this reset.
    # SPL targets the success region rather than the goal cell at its centre.
    optimal = (env._optimal_path_m.clone() - goal_tolerance).clamp_min(0.0)
    inference_seconds = 0.0
    while bool(active.any()):
        pose = env._robot_pose()
        ranges, angles = env.raw_scan()
        goal = env._goal_w - env.scene.env_origins[:, :2]
        noisy_ranges, noisy_angles = noise_mod.corrupt_lidar(
            ranges, angles, scale, noise_generator)
        noisy_pose, noisy_goal = noise_mod.corrupt_goal(
            pose, goal, scale, noise_generator)
        last_action = torch.stack((env._action[:, 0] / v_max,
                                   env._action[:, 1] / omega_max), dim=1)
        observation = build_observation(
            bin_lidar(noisy_ranges, noisy_angles),
            goal_features(noisy_pose, noisy_goal), last_action)

        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            command = policy(observation)
        # Synchronize only for honest inference timing; Isaac dominates this
        # evaluator, so the small measurement cost does not affect conclusions.
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started

        command = torch.where(active[:, None], command, torch.zeros_like(command))
        normalized = torch.stack((command[:, 0] / v_max,
                                  command[:, 1] / omega_max), dim=1)
        previous_normalized = torch.stack((previous[:, 0] / v_max,
                                           previous[:, 1] / omega_max), dim=1)
        jerk_sum[active] += torch.linalg.vector_norm(
            normalized[active] - previous_normalized[active], dim=1) / dt
        distance[active] += command[active, 0].abs() * dt
        steps[active] += 1
        previous[active] = command[active]

        _, _, terminated, truncated, _ = env.step(normalize_action(command))
        newly_done = active & (terminated | truncated)
        if bool(newly_done.any()):
            success[newly_done] = env._reached[newly_done]
            collision[newly_done] = env._collided[newly_done]
            timeout[newly_done] = truncated[newly_done] & ~terminated[newly_done]
            active[newly_done] = False

    successful_path_ratio = torch.where(
        success,
        optimal / torch.maximum(optimal, distance.clamp_min(1.0e-6)),
        torch.zeros_like(optimal),
    )
    successful_time = steps[success].float() * dt
    result = {
        "noise_scale": scale,
        "success_rate": float(success.float().mean()),
        "spl": float(successful_path_ratio.mean()),
        "collision_rate": float(collision.float().mean()),
        "timeout_rate": float(timeout.float().mean()),
        "mean_time_to_goal_s": (float(successful_time.mean())
                                 if bool(success.any()) else math.nan),
        "mean_jerk": float((jerk_sum / steps.clamp_min(1)).mean()),
        "mean_inference_ms": 1000.0 * inference_seconds / int(steps.max()),
    }
    return result, layout_ids


if __name__ == "__main__":
    main()
