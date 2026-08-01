"""Closed-loop Isaac evaluation of the goal-belief policy.

Every number this project has reported about the belief design so far came from
teacher-forced 32-step windows: the belief was reset to truth at the start of
each window and the actor's mistakes never fed back into its own inputs. Both of
those are the opposite of deployment. This script closes the loop -- the belief
is initialized once at episode start and then integrates its own estimate for the
full ~220 steps, and the actor drives on that estimate.

Four modes, run against identical layouts, and the differences between them are
the measurement:

``reference``       a ``NavPolicy`` checkpoint driving on odometry-derived goal
                    features. Reproduces the pre-belief number.
``pose``            the belief checkpoint's *actor*, same odometry-derived goal.
                    Isolates what joint training did to the actor, separately
                    from anything the belief does.
``dead-reckoning``  belief carried, odometry head zeroed. This is option A from
                    the 2026-07-31 log -- commanded dead reckoning -- and it is
                    the baseline the head has to beat to justify existing.
``belief``          the deliverable: no pose input anywhere.

``reference`` -> ``pose`` isolates actor regression, ``pose`` -> ``belief``
isolates the cost of localizing, and ``dead-reckoning`` -> ``belief`` is the
odometry head's actual contribution in closed loop.

Two success criteria are reported and they answer different questions.
``success_rate`` is the environment's, measured on the true pose, and is what
compares against the PPO evaluation. ``belief_arrival`` is measured on what the
robot *believed*: on hardware nothing announces arrival, so a policy that reaches
the goal without knowing it has not solved the deployed task. A large gap between
the two means the belief is drifting even where navigation still succeeds.

    ~/isaac-venv/bin/python scripts/eval_belief.py runs/belief_v1/final.pt \
        --reference runs/ppo_v1/best.pt --allow-legacy-reference \
        --episodes 512 --out runs/belief_v1/eval.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time

MODES = ("reference", "pose", "dead-reckoning", "belief")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path,
                        help="belief checkpoint (algorithm='belief')")
    parser.add_argument("--reference", type=Path,
                        help="NavPolicy checkpoint for the 'reference' mode")
    parser.add_argument("--allow-legacy-reference", action="store_true",
                        help="admit a pre-spinning-sensor reference checkpoint; "
                             "measuring what the corrected sensor model costs it "
                             "is the only honest use of this")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--episodes", type=int, default=512,
                        help="also the number of parallel environments")
    parser.add_argument("--num-layouts", type=int, default=512)
    parser.add_argument("--layout-seed", type=int, default=10_000_000,
                        help="must differ from the collection layout seed")
    parser.add_argument("--noise-scales", type=float, nargs="+",
                        default=(0.0, 1.0, 2.0))
    parser.add_argument("--no-wheel-slip", action="store_true",
                        help="disable simulator wheel slip, making dead "
                             "reckoning exact; a debugging control, not a result")
    parser.add_argument("--out", type=Path,
                        default=Path("runs/belief_eval.json"))
    parser.add_argument("--gui", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if "reference" in args.modes and args.reference is None:
        raise ValueError("mode 'reference' needs --reference CHECKPOINT")

    # Isaac's native modules become importable only after AppLauncher starts.
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=not args.gui).app

    import torch

    from isaaclab_env.nav_task import NavEnvCfg, Sim2EdgeNavEnv, normalize_action
    from sim2edge import noise as noise_mod
    from sim2edge.spec import fingerprint, observation_fingerprint
    from sim2edge.training import load_belief_checkpoint, load_policy_checkpoint

    sys.stdout.reconfigure(line_buffering=True)
    config = NavEnvCfg()
    config.scene.num_envs = args.episodes
    config.num_layouts = args.num_layouts
    config.layout_seed = args.layout_seed
    config.seed = args.layout_seed
    config.collection_mode = True
    config.wheel_slip = not args.no_wheel_slip

    print(f"[eval] building {args.episodes} environments with held-out layout "
          f"seed {args.layout_seed} (wheel_slip={config.wheel_slip}) ...")
    env = Sim2EdgeNavEnv(config)
    device = torch.device(env.device)

    policy = load_belief_checkpoint(args.checkpoint, device=device)
    print(f"[eval] belief policy: {policy.parameter_count:,} parameters "
          f"({policy.odometry_parameter_count:,} in the odometry head)")
    reference = None
    if args.reference is not None:
        reference = load_policy_checkpoint(
            args.reference, device=device,
            allow_legacy_sensor_model=args.allow_legacy_reference)
        print(f"[eval] reference policy: {reference.parameter_count:,} "
              f"parameters from {args.reference}")

    reports = []
    reference_layouts = None
    for mode in args.modes:
        for scale in args.noise_scales:
            report, layouts = _evaluate_tier(
                env, policy, reference, mode, float(scale), args.layout_seed,
                torch, noise_mod)
            if (reference_layouts is not None
                    and not torch.equal(layouts, reference_layouts)):
                raise RuntimeError("re-seeded evaluation produced different layouts")
            if reference_layouts is None:
                reference_layouts = layouts
            reports.append(report)
            _print_tier(report)

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "reference": (str(args.reference.resolve())
                      if args.reference is not None else None),
        "spec_fingerprint": fingerprint(),
        "observation_fingerprint": observation_fingerprint(
            policy.config.actor.history_frames, belief=True),
        "episodes": args.episodes,
        "num_layouts": args.num_layouts,
        "layout_seed": args.layout_seed,
        "wheel_slip": config.wheel_slip,
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


def _print_tier(report: dict) -> None:
    line = (f"[eval] {report['mode']:>14s} {report['noise_scale']:g}x  "
            f"SR {report['success_rate']:.1%}  "
            f"SPL {report['spl']:.3f}  "
            f"coll {report['collision_rate']:.1%}  "
            f"timeout {report['timeout_rate']:.1%}")
    if report["belief_error_final_m"] == report["belief_error_final_m"]:
        line += (f"  belief_final {report['belief_error_final_m'] * 100:.1f} cm"
                 f"  drift/m {report['drift_per_m'] * 100:.2f} cm/m"
                 f"  arrived {report['belief_arrival_rate']:.1%}"
                 f"  arrival_err {report['belief_arrival_error_m'] * 100:.1f} cm")
    print(line)


@contextlib.contextmanager
def _dead_reckoning(policy, torch, enabled: bool):
    """Zero the odometry head, making ``translation`` pure dead reckoning.

    The residual parameterization means option A is the zero-weights special
    case of this model (``models/belief.py``), so the baseline needs no separate
    implementation -- and, more importantly, no separate code path that could
    drift from the one being measured. Weights are restored on exit.
    """
    if not enabled:
        yield
        return
    head = policy.odometry.head
    saved = (head.weight.detach().clone(), head.bias.detach().clone())
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            head.weight.copy_(saved[0])
            head.bias.copy_(saved[1])


def _body_goal(pose, goal, torch):
    """(N, 2) true goal offset in the robot frame -- the belief's ground truth."""
    d = goal - pose[:, :2]
    yaw = pose[:, 2]
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([cos * d[:, 0] + sin * d[:, 1],
                        -sin * d[:, 0] + cos * d[:, 1]], dim=1)


def _evaluate_tier(env, policy, reference, mode, scale, seed, torch, noise_mod):
    """Run one initial episode per environment and aggregate on the GPU."""
    from sim2edge.preprocess import (ObservationHistory, bin_lidar,
                                     goal_features, relative_goal_features,
                                     wrap_angle)
    from sim2edge.sensors import GyroModel, advance_phase, deskew, sweep_scan
    from sim2edge.spec import (DESKEW_WITH_GYRO, DT, GOAL_TOLERANCE_M, OMEGA_MAX,
                               SPIN_TICKS, V_MAX)
    from isaaclab_env.nav_task import normalize_action

    uses_belief = mode in ("belief", "dead-reckoning")
    actor = reference if mode == "reference" else policy.actor
    frames = policy.config.actor.history_frames
    bins = policy.config.actor.lidar_bins

    torch.manual_seed(seed)
    env.reset(seed=seed)
    layout_ids = env._layout.detach().cpu().clone()
    n = env.num_envs
    device = torch.device(env.device)
    # Every tier draws from a generator seeded identically, so a difference
    # between two tiers is the policy and not the sensor draw.
    generator = torch.Generator(device=device.type).manual_seed(seed + 17)
    kw = {"generator": generator, "device": device}

    active = torch.ones(n, dtype=torch.bool, device=device)
    steps = torch.zeros(n, dtype=torch.long, device=device)
    distance = torch.zeros(n, device=device)
    jerk_sum = torch.zeros(n, device=device)
    previous = torch.zeros(n, 2, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    collision = torch.zeros(n, dtype=torch.bool, device=device)
    timeout = torch.zeros(n, dtype=torch.bool, device=device)

    history = ObservationHistory(n, frames=frames, lidar_bins=bins, device=device)
    gyro_model = GyroModel(n, scale=scale, device=device, generator=generator)
    odometry = noise_mod.OdometryDrift(n, scale=scale, device=device,
                                       generator=generator)
    # One mount error per episode rather than per frame: the bias is a property
    # of how the unit is bolted on, and redrawing it every tick would let the
    # policy average away a systematic error it cannot average away on hardware.
    bearing_bias = torch.randn((n, 1), **kw) * (
        noise_mod.BEARING_BIAS_RAD * scale)
    # The sweep phase is a property of the motor and keeps advancing regardless
    # of what the robot does, so it is drawn once and stepped, never redrawn.
    phase = torch.rand((n,), **kw)
    gyro_history = torch.zeros(n, frames, device=device)
    scans = None
    belief = torch.zeros(n, 2, device=device)

    belief_error_sum = torch.zeros(n, device=device)
    belief_error_final = torch.zeros(n, device=device)
    belief_error_max = torch.zeros(n, device=device)
    arrived = torch.zeros(n, dtype=torch.bool, device=device)
    arrival_error = torch.zeros(n, device=device)
    previous_yaw = torch.zeros(n, device=device)

    optimal = (env._optimal_path_m.clone() - GOAL_TOLERANCE_M).clamp_min(0.0)
    inference_seconds = 0.0
    first = True

    with _dead_reckoning(policy, torch, mode == "dead-reckoning"):
        while bool(active.any()):
            pose = env._robot_pose()
            ranges, angles = env.raw_scan()
            goal = env._goal_w - env.scene.env_origins[:, :2]
            noisy_ranges, noisy_angles = noise_mod.corrupt_lidar(
                ranges, angles, scale, generator, bearing_bias=bearing_bias)

            # ── gyro ────────────────────────────────────────────────────────
            # The increment is backward-looking: the rotation measured over the
            # tick that just ended, against the command executed over that same
            # tick. Mixing those two ticks cost 8 cm of belief error once
            # already (2026-07-31 log, 6.1) and is invisible in the shapes.
            yaw = pose[:, 2]
            true_rate = wrap_angle(yaw - previous_yaw) / DT
            measured_rate = gyro_model.step(true_rate)
            if first:
                # No preceding tick exists; synthesize_gyro zeroes the first
                # sample of each episode for the same reason.
                measured_rate = torch.zeros_like(measured_rate)
            previous_yaw = yaw
            gyro_history = torch.cat(
                [gyro_history[:, 1:], (measured_rate * DT).unsqueeze(1)], dim=1)

            # ── spinning lidar ──────────────────────────────────────────────
            if first:
                scans = noisy_ranges.unsqueeze(1).expand(
                    -1, SPIN_TICKS, -1).clone()
            else:
                scans = torch.cat([scans[:, 1:], noisy_ranges.unsqueeze(1)],
                                  dim=1)
            swept, ages = sweep_scan(scans, phase)
            beam_angles = noisy_angles
            if DESKEW_WITH_GYRO:
                # Deskew translation uses the command executed over the tick
                # just ended -- the newest one the robot actually has. The
                # offline loader reads the command *at* the frame instead;
                # online that value has not been decided yet, and the resulting
                # one-tick offset is far below the range noise this corrects.
                swept, beam_angles = deskew(swept, beam_angles, ages,
                                            measured_rate, env._action[:, 0])
            binned = bin_lidar(swept, beam_angles)
            phase = advance_phase(phase)

            last_action = torch.stack((env._action[:, 0] / V_MAX,
                                       env._action[:, 1] / OMEGA_MAX), dim=1)
            history.push(binned, last_action)

            # ── belief and action ───────────────────────────────────────────
            truth = _body_goal(pose, goal, torch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                if uses_belief:
                    if first:
                        # belief0 is correct *at* step zero by definition, so
                        # integrating here would double-count the tick before
                        # the episode opened. ``rollout`` skips it identically.
                        belief = policy.initial_belief(truth)
                        command = actor(policy.observation(
                            history.lidar, belief, history.action))
                    else:
                        command, belief = policy.step(
                            belief, history.lidar, gyro_history, history.action)
                else:
                    drift = odometry.step(env._action[:, 1] * DT)
                    drifted = pose + drift
                    noisy_pose, noisy_goal = noise_mod.corrupt_goal(
                        drifted, goal, scale, generator)
                    command = actor(history.observation(
                        goal_features(noisy_pose, noisy_goal)))
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started

            if uses_belief:
                error = torch.linalg.vector_norm(belief - truth, dim=1)
                belief_error_sum[active] += error[active]
                belief_error_final[active] = error[active]
                belief_error_max[active] = torch.maximum(
                    belief_error_max[active], error[active])
                # The robot's own arrival decision, which on hardware is the
                # only one available. Recorded at the first tick it fires, with
                # the true distance at that instant -- that gap is the number a
                # deployment cares about.
                believes = (torch.linalg.vector_norm(belief, dim=1)
                            < GOAL_TOLERANCE_M)
                newly = active & believes & ~arrived
                arrived[newly] = True
                arrival_error[newly] = torch.linalg.vector_norm(
                    truth[newly], dim=1)

            command = torch.where(active[:, None], command,
                                  torch.zeros_like(command))
            normalized = torch.stack((command[:, 0] / V_MAX,
                                      command[:, 1] / OMEGA_MAX), dim=1)
            previous_normalized = torch.stack((previous[:, 0] / V_MAX,
                                               previous[:, 1] / OMEGA_MAX), dim=1)
            jerk_sum[active] += torch.linalg.vector_norm(
                normalized[active] - previous_normalized[active], dim=1) / DT
            distance[active] += command[active, 0].abs() * DT
            steps[active] += 1
            previous[active] = command[active]

            _, _, terminated, truncated, _ = env.step(normalize_action(command))
            newly_done = active & (terminated | truncated)
            if bool(newly_done.any()):
                success[newly_done] = env._reached[newly_done]
                collision[newly_done] = env._collided[newly_done]
                timeout[newly_done] = (truncated[newly_done]
                                       & ~terminated[newly_done])
                active[newly_done] = False
            first = False

    successful_path_ratio = torch.where(
        success,
        optimal / torch.maximum(optimal, distance.clamp_min(1.0e-6)),
        torch.zeros_like(optimal),
    )
    successful_time = steps[success].float() * DT
    nan = float("nan")
    result = {
        "mode": mode,
        "noise_scale": scale,
        "success_rate": float(success.float().mean()),
        "spl": float(successful_path_ratio.mean()),
        "collision_rate": float(collision.float().mean()),
        "timeout_rate": float(timeout.float().mean()),
        "mean_time_to_goal_s": (float(successful_time.mean())
                                if bool(success.any()) else nan),
        "mean_jerk": float((jerk_sum / steps.clamp_min(1)).mean()),
        "mean_inference_ms": 1000.0 * inference_seconds / int(steps.max()),
        "mean_steps": float(steps.float().mean()),
        "mean_path_m": float(distance.mean()),
        # Belief diagnostics are meaningless when the goal came from a pose.
        "belief_error_m": (float((belief_error_sum / steps.clamp_min(1)).mean())
                           if uses_belief else nan),
        "belief_error_final_m": (float(belief_error_final.mean())
                                 if uses_belief else nan),
        "belief_error_final_p95_m": (
            float(belief_error_final.quantile(0.95)) if uses_belief else nan),
        "belief_error_max_m": (float(belief_error_max.mean())
                               if uses_belief else nan),
        "drift_per_m": (float((belief_error_final
                               / distance.clamp_min(1.0e-3)).mean())
                        if uses_belief else nan),
        # Fraction of episodes in which the robot ever concluded it had
        # arrived, and how far it truly was when it first concluded that.
        "belief_arrival_rate": (float(arrived.float().mean())
                                if uses_belief else nan),
        "belief_arrival_error_m": (
            float(arrival_error[arrived].mean())
            if uses_belief and bool(arrived.any()) else nan),
    }
    return result, layout_ids


if __name__ == "__main__":
    main()
