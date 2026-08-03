"""Closed-loop actor evaluation with the CPU exact-bearing localizer."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("checkpoint", type=Path)
    value.add_argument("--episodes", type=int, default=32)
    value.add_argument("--num-layouts", type=int, default=512)
    value.add_argument("--layout-seed", type=int, default=20_000_000)
    value.add_argument("--noise-scales", type=float, nargs="+", default=[0, 1, 2])
    value.add_argument("--correction-gain", type=float, default=0.30)
    value.add_argument("--yaw-correction-gain", type=float, default=0.50)
    value.add_argument("--arrival-radius-m", type=float, default=0.15)
    value.add_argument("--max-submap-scans", type=int, default=128)
    value.add_argument("--bearing-bias-scale", type=float, default=1.0,
                       help="diagnostic multiplier on the per-episode mount error")
    value.add_argument("--max-correction-m", type=float, default=0.04)
    value.add_argument("--max-correction-yaw-deg", type=float, default=2.0)
    value.add_argument("--motion-scale-update-rate", type=float, default=0.10)
    value.add_argument("--max-yaw-departure-deg", type=float, default=math.inf)
    value.add_argument("--point-stride", type=int, default=2)
    value.add_argument("--prior-source", choices=("command-gyro", "create3-ekf"),
                       default="create3-ekf")
    value.add_argument("--encoder-noise-scale", type=float, default=1.0,
                       help="diagnostic multiplier on Create3 increment noise")
    value.add_argument("--gyro-noise-scale", type=float, default=1.0,
                       help="diagnostic multiplier on gyro noise")
    value.add_argument("--out", type=Path, required=True)
    return value


def _body_offset(delta, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack((c * delta[:, 0] + s * delta[:, 1],
                            -s * delta[:, 0] + c * delta[:, 1]))


def main() -> None:
    args = parser().parse_args()
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=True).app

    import torch
    from isaaclab_env.nav_task import NavEnvCfg, Sim2EdgeNavEnv, normalize_action
    from sim2edge.localization import LocalPoseTracker
    from sim2edge.noise import ODOM_XY_WALK_M
    from sim2edge.preprocess import build_observation, relative_goal_features
    from sim2edge.sensors import PolarSweep
    from sim2edge.spec import DT, GOAL_TOLERANCE_M, SENSOR_CONTRACT_VERSION
    from sim2edge.training import load_policy_checkpoint

    cfg = NavEnvCfg()
    cfg.scene.num_envs = args.episodes
    cfg.num_layouts = args.num_layouts
    cfg.layout_seed = args.layout_seed
    cfg.seed = args.layout_seed
    cfg.collection_mode = False
    cfg.terminate_on_goal = False
    env = Sim2EdgeNavEnv(cfg)
    device = torch.device(env.device)
    actor = load_policy_checkpoint(args.checkpoint, device=device).eval()
    tiers = []

    for scale in args.noise_scales:
        env.cfg.observation_noise_scale = float(scale)
        env._gyro.scale = float(scale) * args.gyro_noise_scale
        env._odometry_drift.scale = float(scale)
        env.reset(seed=args.layout_seed)
        env._lidar_bearing_bias *= args.bearing_bias_scale
        simulated_bearing_bias = env._lidar_bearing_bias.detach().cpu().numpy().copy()
        simulated_wheel_calibration = env._wheel_calibration.detach().cpu().numpy().copy()
        n = env.num_envs
        trackers = [LocalPoseTracker(
            max_submap_scans=args.max_submap_scans,
            correction_gain=args.correction_gain,
            yaw_correction_gain=args.yaw_correction_gain,
            max_correction_m=args.max_correction_m,
            max_correction_yaw_rad=math.radians(
                args.max_correction_yaw_deg),
            motion_scale_update_rate=args.motion_scale_update_rate,
            max_yaw_departure_rad=math.radians(
                args.max_yaw_departure_deg),
            point_stride=args.point_stride,
            motion_prior_source=args.prior_source)
                    for _ in range(n)]
        initial_pose = env._robot_pose().detach().cpu().numpy().copy()
        previous_truth_pose = initial_pose.copy()
        encoder_rng = np.random.default_rng(args.layout_seed + 71)
        goal_world = (env._goal_w - env.scene.env_origins[:, :2]).detach().cpu().numpy()
        initial_goal = _body_offset(goal_world - initial_pose[:, :2],
                                    initial_pose[:, 2])
        active = np.ones(n, dtype=bool)
        arrived = np.zeros(n, dtype=bool)
        false_arrival = np.zeros(n, dtype=bool)
        collision = np.zeros(n, dtype=bool)
        timeout = np.zeros(n, dtype=bool)
        ever_reached = np.zeros(n, dtype=bool)
        arrival_count = np.zeros(n, dtype=np.int64)
        travelled = np.zeros(n)
        final_error = np.zeros(n)
        final_dead_error = np.zeros(n)
        final_yaw_error = np.zeros(n)
        dead_pose = np.zeros((n, 3))
        final_truth_distance = np.zeros(n)
        final_belief_distance = np.zeros(n)
        final_sigma = np.zeros(n)
        last_pose = np.zeros((n, 3))
        observable = accepted = 0
        first = True
        inference_s = 0.0

        while active.any():
            canonical = env.canonical_sweep().to("cpu")
            gyro = env.gyro_increment()[:, -1].detach().cpu().numpy()
            speed = env._action[:, 0].detach().cpu().numpy()
            current_truth_pose = env._robot_pose().detach().cpu().numpy()
            goal_body = np.zeros((n, 2), dtype=np.float32)
            for i in np.flatnonzero(active):
                if not first:
                    heading = dead_pose[i, 2] + 0.5 * float(gyro[i])
                    dead_pose[i, :2] += float(speed[i]) * DT * np.array(
                        [math.cos(heading), math.sin(heading)])
                    dead_pose[i, 2] = math.atan2(
                        math.sin(dead_pose[i, 2] + float(gyro[i])),
                        math.cos(dead_pose[i, 2] + float(gyro[i])))
                one = PolarSweep(canonical.ranges[i], canonical.bearings[i],
                                 canonical.ages[i], canonical.valid[i])
                prior_delta = None
                if args.prior_source == "create3-ekf" and not first:
                    world_delta = (current_truth_pose[i, :2]
                                   - previous_truth_pose[i, :2])
                    yaw0 = previous_truth_pose[i, 2]
                    c0, s0 = math.cos(yaw0), math.sin(yaw0)
                    encoder_noise = encoder_rng.normal(
                        0.0, ODOM_XY_WALK_M * float(scale)
                        * args.encoder_noise_scale, size=2)
                    prior_delta = np.array([
                        c0 * world_delta[0] + s0 * world_delta[1]
                            + encoder_noise[0],
                        -s0 * world_delta[0] + c0 * world_delta[1]
                            + encoder_noise[1],
                        float(gyro[i])])
                estimate = trackers[i].update(
                    one, 0.0 if first else float(gyro[i]),
                    0.0 if first else float(speed[i]),
                    prior_delta_body=prior_delta)
                estimate_pose = estimate.relative_pose
                estimate_covariance = estimate.covariance
                last_pose[i] = estimate_pose
                observable += int(estimate.observable)
                accepted += 1
                travelled[i] += abs(float(speed[i])) * DT if not first else 0.0
                delta = initial_goal[i] - estimate_pose[:2]
                yaw = estimate_pose[2]
                c, s = math.cos(yaw), math.sin(yaw)
                goal_body[i] = (c * delta[0] + s * delta[1],
                                -s * delta[0] + c * delta[1])
                largest = float(np.linalg.eigvalsh(
                    estimate_covariance[:2, :2])[-1])
                # Preserve a calibrated systematic scale term separately from
                # the scan matcher's random covariance.
                sigma = math.sqrt(max(0.0, largest)
                                  + (0.01 * travelled[i]) ** 2)
                final_belief_distance[i] = np.linalg.norm(goal_body[i])
                final_sigma[i] = sigma
                candidate = (final_belief_distance[i] + 1.96 * sigma
                             <= args.arrival_radius_m)
                arrival_count[i] = arrival_count[i] + 1 if candidate else 0
                arrived[i] = arrival_count[i] >= 3

            goal_tensor = torch.from_numpy(goal_body).to(device)
            observation = build_observation(
                env._observation_history.lidar,
                relative_goal_features(goal_tensor),
                env._observation_history.action)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                command = actor(observation)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_s += time.perf_counter() - started

            active_t = torch.from_numpy(active).to(device)
            arrived_t = torch.from_numpy(arrived & active).to(device)
            command = torch.where((active_t & ~arrived_t)[:, None], command,
                                  torch.zeros_like(command))
            current_pose = env._robot_pose().detach().cpu().numpy()
            truth_distance = np.linalg.norm(goal_world - current_pose[:, :2], axis=1)
            false_arrival |= arrived & active & (
                truth_distance > GOAL_TOLERANCE_M)
            env.set_external_done(arrived_t)
            _, _, terminated, truncated, _ = env.step(normalize_action(command))
            ever_reached |= active & env._reached.detach().cpu().numpy()
            done = active & (terminated | truncated).detach().cpu().numpy()
            if done.any():
                hit = env._collided.detach().cpu().numpy()
                collision[done] = hit[done]
                timeout[done] = truncated.detach().cpu().numpy()[done]
                now = env._pose.detach().cpu().numpy()
                final_truth_distance[done] = np.linalg.norm(
                    goal_world[done] - now[done, :2], axis=1)
                truth_delta = _body_offset(
                    now[:, :2] - initial_pose[:, :2], initial_pose[:, 2])
                for i in np.flatnonzero(done):
                    final_error[i] = np.linalg.norm(
                        last_pose[i, :2] - truth_delta[i])
                    final_dead_error[i] = np.linalg.norm(
                        dead_pose[i, :2] - truth_delta[i])
                    truth_yaw = math.atan2(
                        math.sin(now[i, 2] - initial_pose[i, 2]),
                        math.cos(now[i, 2] - initial_pose[i, 2]))
                    final_yaw_error[i] = abs(math.atan2(
                        math.sin(last_pose[i, 2] - truth_yaw),
                        math.cos(last_pose[i, 2] - truth_yaw)))
                active[done] = False
            first = False
            previous_truth_pose = current_truth_pose

        correct = arrived & ~false_arrival
        tier = {
            "noise_scale": float(scale),
            "correct_declared_arrival_rate": float(correct.mean()),
            "false_arrival_rate": float(false_arrival.mean()),
            "collision_rate": float(collision.mean()),
            "timeout_rate": float(timeout.mean()),
            "truth_reached_rate": float(ever_reached.mean()),
            "mean_final_belief_error_m": float(final_error.mean()),
            "p95_final_belief_error_m": float(np.quantile(final_error, 0.95)),
            "mean_dead_reckoning_error_m": float(final_dead_error.mean()),
            "improvement_over_dead_reckoning": float(
                1.0 - final_error.mean() / max(final_dead_error.mean(), 1.0e-12)),
            "mean_final_yaw_error_deg": float(np.degrees(final_yaw_error).mean()),
            "mean_final_truth_distance_m": float(final_truth_distance.mean()),
            "mean_final_belief_distance_m": float(final_belief_distance.mean()),
            "mean_final_sigma_m": float(final_sigma.mean()),
            "timeout_mean_truth_distance_m": (
                float(final_truth_distance[timeout].mean()) if timeout.any()
                else math.nan),
            "timeout_mean_belief_distance_m": (
                float(final_belief_distance[timeout].mean()) if timeout.any()
                else math.nan),
            "timeout_mean_sigma_m": (
                float(final_sigma[timeout].mean()) if timeout.any() else math.nan),
            "observable_update_rate": observable / max(accepted, 1),
            "mean_actor_inference_ms": 1000.0 * inference_s / max(accepted / n, 1),
            "episode_final_error_m": final_error.tolist(),
            "episode_dead_reckoning_error_m": final_dead_error.tolist(),
            "episode_final_yaw_error_deg": np.degrees(final_yaw_error).tolist(),
            "episode_final_truth_distance_m": final_truth_distance.tolist(),
            "episode_final_belief_distance_m": final_belief_distance.tolist(),
            "episode_arrived": arrived.tolist(),
            "episode_false_arrival": false_arrival.tolist(),
            "episode_collision": collision.tolist(),
            "diagnostic_sim_bearing_bias_rad": simulated_bearing_bias.tolist(),
            "diagnostic_sim_wheel_calibration": simulated_wheel_calibration.tolist(),
            "episode_final_motion_scale": [tracker.motion_scale
                                             for tracker in trackers],
        }
        tiers.append(tier)
        print(f"[localized] {scale:g}x arrivals {tier['correct_declared_arrival_rate']:.1%} "
              f"false {tier['false_arrival_rate']:.1%} collision "
              f"{tier['collision_rate']:.1%} error "
              f"{tier['mean_final_belief_error_m'] * 100:.2f} cm", flush=True)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "sensor_contract_version": SENSOR_CONTRACT_VERSION,
        "motion_prior_source": args.prior_source,
        "correction_gain": args.correction_gain,
        "yaw_correction_gain": args.yaw_correction_gain,
        "arrival_radius_m": args.arrival_radius_m,
        "max_submap_scans": args.max_submap_scans,
        "bearing_bias_scale": args.bearing_bias_scale,
        "max_correction_m": args.max_correction_m,
        "max_correction_yaw_deg": args.max_correction_yaw_deg,
        "motion_scale_update_rate": args.motion_scale_update_rate,
        "max_yaw_departure_deg": args.max_yaw_departure_deg,
        "point_stride": args.point_stride,
        "encoder_noise_scale": args.encoder_noise_scale,
        "gyro_noise_scale": args.gyro_noise_scale,
        "episodes": args.episodes,
        "layout_seed": args.layout_seed,
        "tiers": tiers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[localized] wrote {args.out}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
