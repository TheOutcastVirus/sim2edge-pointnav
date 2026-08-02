"""PPO-adapt a swept-scan actor to the model's odometry-free goal belief.

PPO is not allowed to reshape localization through policy gradients. During
critic warmup the motion estimator is frozen; afterwards each actor update is
followed by a small supervised live-distribution motion update using simulator
pose strictly as a label. Run ``train_belief.py`` first, then use this stage to
close the loop between accumulated belief error and collision avoidance.

    ~/isaac-venv/bin/python scripts/train_belief_ppo.py \
        --init runs/belief_motion_v2/best.pt --out runs/belief_ppo_v2
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--num-layouts", type=int, default=4096)
    parser.add_argument("--layout-seed", type=int, default=24680)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--critic-warmup", type=int)
    parser.add_argument("--actor-lr", type=float)
    parser.add_argument("--estimator-lr", type=float, default=3.0e-5)
    parser.add_argument("--motion-batch-size", type=int, default=256,
                        help="live estimator examples retained every 8 rollout steps")
    parser.add_argument("--timeout-penalty", type=float,
                        help="terminal cost for running out the episode clock")
    parser.add_argument("--freeze-estimator", action="store_true",
                        help="skip alternating live scan-motion supervision")
    parser.add_argument("--freeze-actor", action="store_true",
                        help="adapt only the estimator on live policy trajectories")
    parser.add_argument("--reset-motion-optimizer", action="store_true",
                        help="do not restore the estimator optimizer from --resume")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    return parser


def _body_goal(pose, goal, torch):
    delta = goal - pose[:, :2]
    yaw = pose[:, 2]
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([cos * delta[:, 0] + sin * delta[:, 1],
                        -sin * delta[:, 0] + cos * delta[:, 1]], dim=1)


def main() -> None:
    args = build_parser().parse_args()
    if args.freeze_actor and args.freeze_estimator:
        raise ValueError("--freeze-actor and --freeze-estimator are mutually exclusive")
    if args.motion_batch_size < 1 or args.motion_batch_size > args.num_envs:
        raise ValueError("--motion-batch-size must lie in [1, --num-envs]")
    sys.stdout.reconfigure(line_buffering=True)
    if not args.init.is_file():
        raise FileNotFoundError(args.init)

    from isaaclab.app import AppLauncher
    AppLauncher(headless=not args.gui)

    import torch

    from isaaclab_env.nav_task import NavEnvCfg, Sim2EdgeNavEnv
    from sim2edge.models import NavActorCritic, compose_belief
    from sim2edge.spec import (DT, GOAL_TOLERANCE_M, V_MAX, fingerprint,
                               observation_fingerprint)
    from sim2edge.training import load_belief_checkpoint
    from sim2edge.training.bc import atomic_torch_save
    from sim2edge.training.ppo import (EpisodeStats, PPOConfig, PPOTrainer,
                                       RolloutBuffer)

    overrides = {}
    if args.iterations is not None:
        overrides["iterations"] = args.iterations
    if args.num_steps is not None:
        overrides["num_steps"] = args.num_steps
    if args.critic_warmup is not None:
        overrides["critic_warmup_iters"] = args.critic_warmup
    if args.actor_lr is not None:
        overrides["actor_lr"] = args.actor_lr
    config = replace(PPOConfig(), seed=args.seed, **overrides)

    env_cfg = NavEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.num_layouts = args.num_layouts
    env_cfg.layout_seed = args.layout_seed
    env_cfg.seed = args.seed
    env_cfg.observation_noise_scale = args.noise_scale
    if args.timeout_penalty is not None:
        if args.timeout_penalty < 0.0:
            raise ValueError("--timeout-penalty cannot be negative")
        env_cfg.timeout_penalty = args.timeout_penalty
    env = Sim2EdgeNavEnv(env_cfg)
    device = torch.device(env.device)

    belief_policy = load_belief_checkpoint(args.init, device=device)
    belief_policy.odometry.eval()
    actor_critic = NavActorCritic(
        actor=belief_policy.actor, init_std=config.init_std).to(device)
    anchor = copy.deepcopy(belief_policy.actor).eval().requires_grad_(False)
    trainer = PPOTrainer(actor_critic, config, anchor, device=device)
    motion_optimizer = torch.optim.AdamW(
        belief_policy.odometry.parameters(), lr=args.estimator_lr,
        weight_decay=1.0e-5)

    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    if args.resume is None and (output / "latest.pt").exists():
        raise FileExistsError(f"{output} already contains a run")

    start_iteration = 0
    best_score = -float("inf")
    selection_ema = None
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        belief_policy.load_state_dict(saved["model"])
        actor_critic.critic.load_state_dict(saved["critic"])
        actor_critic.log_std.data.copy_(saved["log_std"].to(device))
        trainer.optimizer.load_state_dict(saved["optimizer"])
        if "motion_optimizer" in saved and not args.reset_motion_optimizer:
            motion_optimizer.load_state_dict(saved["motion_optimizer"])
        trainer.iteration = int(saved["iteration"])
        start_iteration = trainer.iteration
        best_score = float(saved.get("metrics", {}).get("selection_score", best_score))
        selection_ema = saved.get("metrics", {}).get("selection_score")

    if args.freeze_actor:
        # Keep the held-out-selected actor exactly fixed while adapting the
        # estimator to that actor's trajectory distribution.  Optimizer groups
        # may still contain these parameters after resume, but parameters with
        # requires_grad=False never receive an Adam update.
        belief_policy.actor.requires_grad_(False)
        best_score = -float("inf")
        selection_ema = None

    run_config = {
        "init": str(args.init), "ppo": asdict(config),
        "num_envs": args.num_envs, "num_layouts": args.num_layouts,
        "layout_seed": args.layout_seed, "noise_scale": args.noise_scale,
        "estimator_lr": args.estimator_lr,
        "motion_batch_size": args.motion_batch_size,
        "estimator_frozen": args.freeze_estimator,
        "actor_frozen": args.freeze_actor,
        "motion_optimizer_reset": args.reset_motion_optimizer,
        "timeout_penalty": env_cfg.timeout_penalty,
    }
    (output / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n")

    def checkpoint(metrics):
        return {
            "format_version": 2,
            "algorithm": "belief-ppo",
            "spec_fingerprint": fingerprint(),
            "observation_fingerprint": observation_fingerprint(
                belief_policy.config.actor.history_frames, belief=True),
            "model_config": belief_policy.config.to_dict(),
            "training_config": run_config,
            "model": belief_policy.state_dict(),
            "critic": actor_critic.critic.state_dict(),
            "log_std": actor_critic.log_std.detach().cpu(),
            "optimizer": trainer.optimizer.state_dict(),
            "motion_optimizer": motion_optimizer.state_dict(),
            "iteration": trainer.iteration,
            "metrics": metrics,
        }

    env.reset(seed=args.seed)
    pose = env._robot_pose()
    goal = env._goal_w - env.scene.env_origins[:, :2]
    belief, covariance, arrival_count = belief_policy.initial_state(
        _body_goal(pose, goal, torch))
    calibration = belief_policy.initial_calibration(belief)
    actor_observation = belief_policy.observation(
        env._observation_history.lidar, belief,
        env._observation_history.action)

    buffer = RolloutBuffer(config.num_steps, args.num_envs,
                           belief_policy.config.actor.observation_dim, device)
    stats = EpisodeStats(args.num_envs, device)
    generator = torch.Generator(device=device.type).manual_seed(args.seed + 101)
    started = time.perf_counter()
    record = {}

    for iteration in range(start_iteration, config.iterations):
        tick = time.perf_counter()
        buffer.reset()
        stats.reset()
        declared_success = torch.zeros((), device=device)
        false_arrival = torch.zeros((), device=device)
        confidence_total = torch.zeros((), device=device)
        belief_error_total = torch.zeros((), device=device)
        motion_batches = []

        for rollout_step in range(config.num_steps):
            current_observation = actor_observation
            pose_before = env._robot_pose()
            with torch.no_grad():
                raw, normalized_action, log_prob, value = actor_critic.act(
                    current_observation, generator)
            _, reward, terminated, truncated, _ = env.step(normalized_action)
            done = terminated | truncated

            pose = env._robot_pose()
            goal = env._goal_w - env.scene.env_origins[:, :2]
            truth = _body_goal(pose, goal, torch)
            raw_lidar, raw_valid = env.motion_scan()
            with torch.no_grad():
                if (belief_policy.config.persistent_calibration
                        or belief_policy.config.motion_state_size > 0):
                    (translation, variance, confidence,
                     advanced_calibration) = (
                        belief_policy.calibrated_motion_estimate(
                            calibration, env._observation_history.lidar,
                            env.gyro_increment(), env.motion_action(),
                            raw_lidar, raw_valid))
                    advanced = compose_belief(
                        belief, translation, env.gyro_increment()[:, -1])
                else:
                    translation, variance, confidence = (
                        belief_policy.motion_estimate(
                            env._observation_history.lidar,
                            env.gyro_increment(), env.motion_action(),
                            raw_lidar, raw_valid))
                    advanced = belief_policy.update_belief(
                        belief, env._observation_history.lidar,
                        env.gyro_increment(), env.motion_action(),
                        raw_lidar, raw_valid)
                    advanced_calibration = calibration
                advanced_covariance = belief_policy._propagate_covariance(
                    covariance, variance, env.gyro_increment()[:, -1])
                largest = belief_policy._largest_covariance_eigenvalue(
                    advanced_covariance)
                bound = (advanced.norm(dim=1)
                         + belief_policy.config.arrival_sigma
                         * largest.clamp_min(0.0).sqrt())
                candidate = bound <= belief_policy.config.arrival_radius_m
                next_count = torch.where(candidate, arrival_count + 1,
                                         torch.zeros_like(arrival_count))
                new_arrival = ((next_count
                                >= belief_policy.config.arrival_confirm_ticks)
                               & (arrival_count
                                  < belief_policy.config.arrival_confirm_ticks))
                true_distance = truth.norm(dim=1)
                belief_error_total += torch.linalg.vector_norm(
                    advanced - truth, dim=1).sum()
                declared_success += (new_arrival
                                     & (true_distance <= GOAL_TOLERANCE_M)).sum()
                new_false = new_arrival & (true_distance > GOAL_TOLERANCE_M)
                false_arrival += new_false.sum()
                confidence_total += confidence.mean()

                fresh_belief, fresh_covariance, fresh_count = (
                    belief_policy.initial_state(truth))
                belief = torch.where(done[:, None], fresh_belief, advanced)
                covariance = torch.where(
                    done[:, None, None], fresh_covariance, advanced_covariance)
                arrival_count = torch.where(done, fresh_count, next_count)
                fresh_calibration = belief_policy.initial_calibration(truth)
                calibration = torch.where(
                    done, fresh_calibration, advanced_calibration)
                actor_observation = belief_policy.observation(
                    env._observation_history.lidar, belief,
                    env._observation_history.action)

                # A small, bounded live-distribution batch for the alternating
                # estimator update. Pose is a label only and is never included
                # in the saved model's runtime inputs.
                if rollout_step % 8 == 0:
                    eligible = torch.nonzero(
                        ~done, as_tuple=False).squeeze(1)[:args.motion_batch_size]
                    if eligible.numel():
                        world = pose[eligible, :2] - pose_before[eligible, :2]
                        yaw0 = pose_before[eligible, 2]
                        cos0, sin0 = torch.cos(yaw0), torch.sin(yaw0)
                        target = torch.stack([
                            cos0 * world[:, 0] + sin0 * world[:, 1],
                            -sin0 * world[:, 0] + cos0 * world[:, 1],
                        ], dim=1)
                        motion_batches.append((
                            env._observation_history.lidar[eligible].detach().clone(),
                            env.gyro_increment()[eligible].detach().clone(),
                            env.motion_action()[eligible].detach().clone(),
                            raw_lidar[eligible].detach().clone(),
                            raw_valid[eligible].detach().clone(),
                            target.detach().clone(),
                        ))

            # A premature stop is terminally bad on hardware even though the
            # simulator would otherwise let the actor continue and hide it.
            adjusted_reward = reward - 10.0 * new_false.float()
            buffer.add(current_observation, raw, log_prob, value,
                       adjusted_reward, done)
            stats.update(adjusted_reward, terminated, truncated,
                         env._reached, env._collided)

        with torch.no_grad():
            last_value = actor_critic.critic(actor_observation)
        buffer.compute_advantages(last_value, config.gamma, config.gae_lambda)
        record = {**trainer.update(buffer), **stats.summary()}
        motion_loss_value = 0.0
        mismatched_lidar_loss_value = 0.0
        if (motion_batches and not args.freeze_estimator
                and trainer.iteration > config.critic_warmup_iters):
            fields = [torch.cat(items, dim=0) for items in zip(*motion_batches)]
            lidar_batch, gyro_batch, action_batch, raw_batch, valid_batch, target = fields
            prediction, variance, _ = belief_policy.motion_estimate(
                lidar_batch, gyro_batch, action_batch, raw_batch, valid_batch)
            error = prediction - target
            translation_loss = torch.nn.functional.smooth_l1_loss(
                prediction / (V_MAX * DT), target / (V_MAX * DT), beta=0.1)
            calibration_loss = torch.nn.functional.smooth_l1_loss(
                variance.log(), (error.detach().square() + 1.0e-7).log(), beta=0.5)
            # Preserve the scan-causality constraint used by the validated
            # offline trainer.  Without this counterfactual, a live update can
            # minimize one-step error by learning average command slip and
            # quietly stop using lidar at all.
            counter_index = torch.arange(
                0, raw_batch.shape[0], 4, device=device)
            counter_translation, _, _ = belief_policy.motion_estimate(
                lidar_batch[counter_index], gyro_batch[counter_index],
                action_batch[counter_index],
                raw_batch.roll(1, dims=0)[counter_index],
                valid_batch.roll(1, dims=0)[counter_index])
            counter_v = action_batch[counter_index, -1, 0] * V_MAX
            counter_prior = torch.stack(
                [counter_v * DT, torch.zeros_like(counter_v)], dim=1)
            mismatched_lidar_loss = torch.nn.functional.smooth_l1_loss(
                counter_translation / (V_MAX * DT),
                counter_prior / (V_MAX * DT), beta=0.1)
            motion_loss = (translation_loss + mismatched_lidar_loss
                           + 0.01 * calibration_loss)
            motion_optimizer.zero_grad(set_to_none=True)
            motion_loss.backward()
            torch.nn.utils.clip_grad_norm_(belief_policy.odometry.parameters(), 1.0)
            motion_optimizer.step()
            motion_loss_value = float(motion_loss.detach())
            mismatched_lidar_loss_value = float(
                mismatched_lidar_loss.detach())
        arrivals = float((declared_success + false_arrival).cpu())
        completed = max(1.0, record["episodes"])
        record.update(
            declared_arrivals=arrivals,
            declared_arrival_precision=(float(declared_success.cpu())
                                        / max(1.0, arrivals)),
            declared_arrival_success_rate=(float(declared_success.cpu())
                                           / completed),
            false_arrival_rate=float(false_arrival.cpu()) / completed,
            scan_confidence=(float(confidence_total.cpu()) / config.num_steps),
            belief_error_mean_m=(float(belief_error_total.cpu())
                                 / (args.num_envs * config.num_steps)),
            motion_aux_loss=motion_loss_value,
            mismatched_lidar_loss=mismatched_lidar_loss_value,
            elapsed_seconds=time.perf_counter() - started,
            steps_per_second=(args.num_envs * config.num_steps
                              / (time.perf_counter() - tick)),
        )
        raw_selection = (-record["belief_error_mean_m"] if args.freeze_actor
                         else (record["success_rate"]
                               - record["false_arrival_rate"]))
        # One 32-step rollout contains only tens of completed episodes at the
        # memory-safe environment counts. Selecting on that single fraction
        # promoted a 71.9% statistical spike whose deterministic held-out score
        # was worse than the final actor. Smooth selection across iterations;
        # the raw value remains logged for diagnosis.
        selection_ema = (raw_selection if selection_ema is None else
                         0.9 * selection_ema + 0.1 * raw_selection)
        record["selection_score_raw"] = raw_selection
        record["selection_score"] = selection_ema
        if args.freeze_actor:
            record["actor_lr"] = 0.0
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[belief-ppo] {record['iteration']:4d}/{config.iterations}  "
              f"SR {record['success_rate']:.1%}  coll {record['collision_rate']:.1%}  "
              f"false-arrival {record['false_arrival_rate']:.1%}  "
              f"conf {record['scan_confidence']:.2f}  "
              f"{record['steps_per_second']:,.0f} steps/s")

        if (not record["warming_up"]
                and record["selection_score"] > best_score):
            best_score = record["selection_score"]
            atomic_torch_save(checkpoint(record), output / "best.pt")
        if (iteration + 1) % config.checkpoint_every == 0:
            atomic_torch_save(checkpoint(record), output / "latest.pt")

    if record:
        atomic_torch_save(checkpoint(record), output / "latest.pt")
        atomic_torch_save(checkpoint(record), output / "final.pt")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
