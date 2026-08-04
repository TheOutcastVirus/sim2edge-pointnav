"""Benchmark the CPU scan-to-submap localizer at production settings.

Only the actor has ever been timed. The actor is 290k parameters and lowers to
the NPU; the localizer is dense NumPy nearest-neighbour work that stays on the
CPU, so on a Cortex-A55/A78C it -- not the network -- is what decides whether
the 20 Hz control loop closes.

Cost is reported against submap occupancy because it is not constant: the
submap fills to ``--max-submap-scans`` over the first seconds of an episode and
the correspondence search grows with it, so an average over a short run
understates the steady state that matters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from sim2edge.localization import LocalPoseTracker
from sim2edge.sensors import PolarSweep
from sim2edge.spec import DT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=Path, required=True,
                        help="npz with ranges (T, B) and angles (B,)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-submap-scans", type=int, default=128)
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--correction-gain", type=float, default=0.3)
    parser.add_argument("--yaw-correction-gain", type=float, default=0.5)
    parser.add_argument("--max-correction-m", type=float, default=0.04)
    parser.add_argument("--max-correction-yaw-deg", type=float, default=2.0)
    parser.add_argument("--motion-scale-update-rate", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=1,
                        help="torch/BLAS threads; the deployed node owns one "
                             "core for this, so 1 is the honest default")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    payload = np.load(args.scans)
    ranges = torch.from_numpy(payload["ranges"]).float()
    angles = torch.from_numpy(payload["angles"]).float()
    speeds = payload["speeds"] if "speeds" in payload else None
    gyro = payload["gyro"] if "gyro" in payload else None

    tracker = LocalPoseTracker(
        max_submap_scans=args.max_submap_scans,
        point_stride=args.point_stride,
        correction_gain=args.correction_gain,
        yaw_correction_gain=args.yaw_correction_gain,
        max_correction_m=args.max_correction_m,
        max_correction_yaw_rad=np.radians(args.max_correction_yaw_deg),
        motion_scale_update_rate=args.motion_scale_update_rate)

    # Episodes are walked separately with a reset between them: a submap
    # stitched across a teleport is not geometry the tracker ever sees, and it
    # would make the correspondence search look artificially cheap.
    episode = (payload["episode"] if "episode" in payload
               else np.zeros(ranges.shape[0], dtype=np.int32))

    ages = torch.zeros(ranges.shape[1])
    samples: list[float] = []
    occupancy: list[int] = []
    valid_points: list[int] = []
    previous_episode = None
    for index in range(ranges.shape[0]):
        if episode[index] != previous_episode:
            tracker.reset()
            previous_episode = episode[index]
        row = ranges[index]
        valid = torch.isfinite(row) & (row > 0)
        sweep = PolarSweep(ranges=row, bearings=angles, ages=ages, valid=valid)
        speed = float(speeds[index]) if speeds is not None else 0.25
        gyro_delta = float(gyro[index]) if gyro is not None else 0.0
        started = time.perf_counter()
        tracker.update(sweep, gyro_delta, speed, dt=DT)
        samples.append(1_000.0 * (time.perf_counter() - started))
        occupancy.append(len(tracker._scans))
        valid_points.append(int(valid.sum()))

    array = np.array(samples)
    full = array[np.array(occupancy) >= args.max_submap_scans]
    steady = full if full.size else array[len(array) // 2:]

    report = {
        "scans": str(args.scans),
        "ticks": int(array.size),
        "control_period_ms": DT * 1_000.0,
        "max_submap_scans": args.max_submap_scans,
        "point_stride": args.point_stride,
        "threads": args.threads,
        "mean_valid_returns": float(np.mean(valid_points)),
        "submap_filled_at_tick": (int(np.argmax(np.array(occupancy)
                                                >= args.max_submap_scans))
                                  if full.size else None),
        "all_ticks": {
            "median_ms": float(np.median(array)),
            "p95_ms": float(np.percentile(array, 95)),
            "max_ms": float(array.max()),
        },
        "steady_state": {
            "ticks": int(steady.size),
            "median_ms": float(np.median(steady)),
            "p95_ms": float(np.percentile(steady, 95)),
            "max_ms": float(steady.max()),
            "fraction_of_control_period":
                float(np.median(steady)) / (DT * 1_000.0),
        },
    }
    print(f"[localizer] {report['ticks']} ticks, "
          f"{report['mean_valid_returns']:.0f} valid returns/scan, "
          f"submap {args.max_submap_scans} scans")
    steady_report = report["steady_state"]
    print(f"[localizer] steady state median {steady_report['median_ms']:.2f} ms "
          f"p95 {steady_report['p95_ms']:.2f} ms max "
          f"{steady_report['max_ms']:.2f} ms "
          f"against a {DT * 1000:.0f} ms control period "
          f"({100 * steady_report['fraction_of_control_period']:.1f}%)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[localizer] wrote {args.out}")


if __name__ == "__main__":
    main()
