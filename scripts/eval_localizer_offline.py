"""Evidence-gated offline evaluation for the CPU scan-to-submap localizer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from sim2edge import noise as noise_mod
from sim2edge.localization import LocalPoseTracker
from sim2edge.preprocess import wrap_angle
from sim2edge.sensors import (PolarSweep, advance_phase, deskew_history,
                              sweep_scan, synthesize_gyro)
from sim2edge.spec import (DT, LIDAR_MAX_M, LIDAR_MIN_M, SENSOR_CONTRACT_VERSION,
                           SPIN_TICKS)
from sim2edge.training.data import ShardCorpus


def _truth_relative(pose: torch.Tensor) -> np.ndarray:
    delta = pose[-1, :2] - pose[0, :2]
    yaw = float(pose[0, 2])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * float(delta[0]) + s * float(delta[1]),
                     -s * float(delta[0]) + c * float(delta[1]),
                     float(wrap_angle(pose[-1, 2] - pose[0, 2]))])


def _sweep(shard, indices: torch.Tensor, phase: torch.Tensor, noise_scale: float,
           generator: torch.Generator,
           gyro_increment: torch.Tensor) -> PolarSweep:
    ranges = shard.ranges[indices].float()
    beams = ranges.shape[-1]
    bias = torch.randn(1, generator=generator) * (
        noise_mod.BEARING_BIAS_RAD * noise_scale)
    noisy, angles = noise_mod.corrupt_lidar(
        ranges, shard.angles, noise_scale, generator,
        bearing_bias=bias.reshape(1, 1).expand(SPIN_TICKS, 1))
    swept, ages = sweep_scan(noisy.unsqueeze(0), phase, subtick=True)

    speed = shard.prev_action[indices, 0]
    frame_angles = angles[-1].unsqueeze(0)
    swept, frame_angles = deskew_history(
        swept, frame_angles, ages, gyro_increment.unsqueeze(0), speed.unsqueeze(0),
        subtick=True)
    valid = (torch.isfinite(swept) & (swept >= LIDAR_MIN_M)
             & (swept <= LIDAR_MAX_M))
    return PolarSweep(swept[0], frame_angles[0], ages[0], valid[0])


def evaluate(data: Path, noise_scale: float, max_episodes: int,
             *, shuffled: bool = False, correction_gain: float = 0.10,
             yaw_correction_gain: float = 0.05,
             max_submap_scans: int = 16) -> dict:
    corpus = ShardCorpus(data)
    generator = torch.Generator().manual_seed(1729 + int(noise_scale * 100))
    final, dead_final, drift, dead_drift = [], [], [], []
    observable, updates = 0, 0
    episodes = 0
    for _, shard in corpus.loaded(corpus.validation, "cpu", prefetch=False):
        starts = torch.nonzero(torch.cat((
            torch.tensor([True]), shard.episode[1:] != shard.episode[:-1])),
            as_tuple=False).squeeze(1)
        ends = torch.cat((starts[1:] - 1, starts.new_tensor([len(shard) - 1])))
        pairs = list(zip(starts.tolist(), ends.tolist()))
        for episode_index, (start, end) in enumerate(pairs):
            if end - start + 1 < 2:
                continue
            tracker = LocalPoseTracker(
                max_submap_scans=max_submap_scans,
                correction_gain=correction_gain,
                yaw_correction_gain=yaw_correction_gain)
            dead = np.zeros(3)
            phase = torch.rand(1, generator=generator)
            episode_slice = slice(start, end + 1)
            episode_ids = torch.zeros(end - start + 1, dtype=torch.long)
            measured_gyro = synthesize_gyro(
                shard.pose[episode_slice, 2].float(), episode_ids,
                noise_scale, generator)
            for t in range(start, end + 1):
                scan_t, scan_start = t, start
                if shuffled:
                    other_start, other_end = pairs[(episode_index + 1) % len(pairs)]
                    scan_t = min(other_start + (t - start), other_end)
                    scan_start = other_start
                indices = torch.arange(
                    scan_t - SPIN_TICKS + 1, scan_t + 1).clamp_min(scan_start)
                gyro_indices = (indices - scan_start).clamp(
                    0, measured_gyro.numel() - 1)
                sweep = _sweep(shard, indices, phase, noise_scale, generator,
                               measured_gyro[gyro_indices])
                if t == start:
                    gyro = speed = 0.0
                else:
                    gyro = float(measured_gyro[t - start])
                    speed = float(shard.prev_action[t, 0])
                    heading = dead[2] + 0.5 * gyro
                    dead[:2] += speed * DT * np.array(
                        [math.cos(heading), math.sin(heading)])
                    dead[2] = math.atan2(math.sin(dead[2] + gyro),
                                         math.cos(dead[2] + gyro))
                estimate = tracker.update(sweep, gyro, speed)
                observable += int(estimate.observable)
                updates += 1
                phase = advance_phase(phase)
            truth = _truth_relative(shard.pose[start:end + 1])
            err = float(np.linalg.norm(estimate.relative_pose[:2] - truth[:2]))
            dead_err = float(np.linalg.norm(dead[:2] - truth[:2]))
            distance = float(torch.linalg.vector_norm(
                shard.pose[start + 1:end + 1, :2]
                - shard.pose[start:end, :2], dim=1).sum())
            final.append(err)
            dead_final.append(dead_err)
            drift.append(err / max(distance, 1.0e-6))
            dead_drift.append(dead_err / max(distance, 1.0e-6))
            episodes += 1
            if max_episodes and episodes >= max_episodes:
                break
        if max_episodes and episodes >= max_episodes:
            break
    values = np.asarray(final)
    dead_values = np.asarray(dead_final)
    return {
        "episodes": episodes,
        "mean_final_error_m": float(values.mean()),
        "p95_final_error_m": float(np.quantile(values, 0.95)),
        "mean_drift_m_per_m": float(np.mean(drift)),
        "dead_reckoning_mean_final_error_m": float(dead_values.mean()),
        "dead_reckoning_mean_drift_m_per_m": float(np.mean(dead_drift)),
        "improvement_over_dead_reckoning": float(
            1.0 - values.mean() / max(dead_values.mean(), 1.0e-12)),
        "observable_update_rate": observable / max(updates, 1),
        "lidar_shuffled": shuffled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--noise-scales", type=float, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-episodes", type=int, default=64)
    parser.add_argument("--correction-gain", type=float, default=0.10)
    parser.add_argument("--yaw-correction-gain", type=float, default=0.05)
    parser.add_argument("--max-submap-scans", type=int, default=16)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    tiers = {str(scale): evaluate(
        args.data, scale, args.max_episodes,
        correction_gain=args.correction_gain,
        yaw_correction_gain=args.yaw_correction_gain,
        max_submap_scans=args.max_submap_scans)
             for scale in args.noise_scales}
    shuffled = evaluate(
        args.data, 1.0, args.max_episodes, shuffled=True,
        correction_gain=args.correction_gain,
        yaw_correction_gain=args.yaw_correction_gain,
        max_submap_scans=args.max_submap_scans)
    baseline = tiers.get("1.0", tiers.get("1", None))
    report = {
        "sensor_contract_version": SENSOR_CONTRACT_VERSION,
        "motion_prior_source": "command-gyro",
        "correction_gain": args.correction_gain,
        "yaw_correction_gain": args.yaw_correction_gain,
        "max_submap_scans": args.max_submap_scans,
        "tiers": tiers,
        "lidar_shuffled_1x": shuffled,
        "shuffled_lidar_degradation_m": (
            shuffled["mean_final_error_m"] - baseline["mean_final_error_m"]
            if baseline is not None else None),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
