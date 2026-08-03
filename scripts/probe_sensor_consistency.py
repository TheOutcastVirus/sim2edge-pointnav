"""Stage sensor errors only after applying the true SE(2) transform.

Arms isolate untouched instantaneous scans, exact-bearing reconstructed sweeps,
uniform angular rasterization, and sensor noise.  The matcher searches only for
the residual left after truth; reporting a full pose estimate here would mix
scene motion with reconstruction error and recreate the invalid v1 diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import torch

from eval_localizer_offline import _sweep
from sim2edge.preprocess import rasterize_lidar, wrap_angle
from sim2edge.sensors import PolarSweep
from sim2edge.spec import (DT, LIDAR_MAX_M, LIDAR_MOUNT_XYZ, LIDAR_SPIN_HZ,
                           SENSOR_CONTRACT_VERSION, SPIN_TICKS)
from sim2edge.training.data import ShardCorpus


def _points(sweep: PolarSweep) -> np.ndarray:
    ranges = sweep.ranges.float().numpy()
    angles = sweep.bearings.float().numpy()
    valid = sweep.valid.numpy() & np.isfinite(ranges) & (ranges < LIDAR_MAX_M)
    ranges, angles = ranges[valid][::2], angles[valid][::2]
    return np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))


def _place(points: np.ndarray, pose) -> np.ndarray:
    x, y, yaw = map(float, pose)
    c, s = math.cos(yaw), math.sin(yaw)
    return points @ np.array([[c, s], [-s, c]]) + (x, y)


def _score(tree: cKDTree, points: np.ndarray) -> float:
    distance = tree.query(points, k=1)[0]
    return float(np.exp(-0.5 * (distance / 0.03) ** 2).mean())


def _measure(old: PolarSweep, new: PolarSweep, truth) -> dict | None:
    old_points, new_points = _points(old), _points(new)
    if min(len(old_points), len(new_points)) < 60:
        return None
    tree = cKDTree(old_points)
    aligned = _place(new_points, truth)
    zero_score = _score(tree, aligned)
    static_score = _score(tree, new_points)

    def objective(residual):
        distance = tree.query(_place(aligned, residual), k=1)[0]
        return float(np.minimum(distance, 0.20).mean())

    fit = minimize(objective, np.zeros(3), method="Powell",
                   bounds=((-0.08, 0.08), (-0.08, 0.08),
                           (-math.radians(3), math.radians(3))),
                   options={"maxiter": 40, "xtol": 2e-4, "ftol": 1e-5})
    return {"residual": fit.x, "score_at_zero": zero_score,
            "static_score": static_score}


def _instant(shard, index: int) -> PolarSweep:
    ranges = shard.ranges[index].float()
    valid = torch.isfinite(ranges) & (ranges < LIDAR_MAX_M)
    return PolarSweep(ranges, shard.angles, torch.zeros_like(ranges), valid)


def _raster(sweep: PolarSweep) -> PolarSweep:
    ranges, valid = rasterize_lidar(
        sweep.ranges.unsqueeze(0), sweep.bearings.unsqueeze(0),
        sweep.ranges.numel())
    bins = ranges.shape[1]
    bearings = -math.pi + (torch.arange(bins) + 0.5) * (2 * math.pi / bins)
    return PolarSweep(ranges[0], bearings, torch.zeros_like(bearings), valid[0])


def _truth(old_pose: torch.Tensor, new_pose: torch.Tensor) -> np.ndarray:
    def lidar_xy(pose):
        c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
        return pose[:2].double().numpy() + np.array([
            c * LIDAR_MOUNT_XYZ[0] - s * LIDAR_MOUNT_XYZ[1],
            s * LIDAR_MOUNT_XYZ[0] + c * LIDAR_MOUNT_XYZ[1]])
    delta = lidar_xy(new_pose) - lidar_xy(old_pose)
    yaw = float(old_pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * delta[0] + s * delta[1],
                     -s * delta[0] + c * delta[1],
                     float(wrap_angle(new_pose[2] - old_pose[2]))])


def _summary(rows: list[dict]) -> dict:
    residual = np.stack([row["residual"] for row in rows])
    zero = np.array([row["score_at_zero"] for row in rows])
    static = np.array([row["static_score"] for row in rows])
    norm = np.linalg.norm(residual[:, :2], axis=1)
    return {
        "samples": len(rows),
        "score_at_zero_mean": float(zero.mean()),
        "truth_vs_static_score_margin_mean": float((zero - static).mean()),
        "residual_translation_bias_cm": (residual[:, :2].mean(0) * 100).tolist(),
        "residual_translation_median_cm": float(np.median(norm) * 100),
        "residual_translation_p95_cm": float(np.quantile(norm, 0.95) * 100),
        "residual_yaw_bias_deg": float(np.degrees(residual[:, 2].mean())),
        "residual_yaw_scatter_deg": float(np.degrees(residual[:, 2].std())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--stride", type=int, default=53)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = {name: [] for name in
            ("instantaneous_exact", "reconstructed_exact",
             "angular_rasterization", "sensor_noise")}
    corpus = ShardCorpus(args.data)
    generator = torch.Generator().manual_seed(991)
    for _, shard in corpus.loaded(corpus.validation, "cpu", prefetch=False):
        for t in range(args.span, len(shard), args.stride):
            old_t = t - args.span
            if shard.episode[t] != shard.episode[old_t]:
                continue
            truth = _truth(shard.pose[old_t], shard.pose[t])
            instant_old, instant_new = _instant(shard, old_t), _instant(shard, t)
            measured = _measure(instant_old, instant_new, truth)
            if measured is not None:
                rows["instantaneous_exact"].append(measured)

            old_index = torch.arange(old_t - SPIN_TICKS + 1,
                                     old_t + 1).clamp_min(0)
            new_index = torch.arange(t - SPIN_TICKS + 1, t + 1).clamp_min(0)
            # Clamp within this episode, never into the preceding trajectory.
            episode_start = old_t
            while episode_start > 0 and shard.episode[episode_start - 1] == shard.episode[t]:
                episode_start -= 1
            old_index.clamp_min_(episode_start)
            new_index.clamp_min_(episode_start)
            phase_old = torch.tensor([(old_t * LIDAR_SPIN_HZ * DT) % 1.0])
            phase_new = torch.tensor([(t * LIDAR_SPIN_HZ * DT) % 1.0])
            exact_old = _sweep(shard, old_index, phase_old, 0.0, generator)
            exact_new = _sweep(shard, new_index, phase_new, 0.0, generator)
            noisy_old = _sweep(shard, old_index, phase_old, 1.0, generator)
            noisy_new = _sweep(shard, new_index, phase_new, 1.0, generator)
            for name, pair in (
                ("reconstructed_exact", (exact_old, exact_new)),
                ("angular_rasterization", (_raster(exact_old), _raster(exact_new))),
                ("sensor_noise", (noisy_old, noisy_new)),
            ):
                measured = _measure(*pair, truth)
                if measured is not None:
                    rows[name].append(measured)
            if len(rows["instantaneous_exact"]) >= args.max_pairs:
                break
        if len(rows["instantaneous_exact"]) >= args.max_pairs:
            break

    report = {
        "sensor_contract_version": SENSOR_CONTRACT_VERSION,
        "truth_applied_before_residual_search": True,
        "arms": {name: _summary(value) for name, value in rows.items()},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
