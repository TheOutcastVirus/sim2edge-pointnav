#!/usr/bin/env python3
"""Replay a rosbag through the reference preprocessing to produce observations.

Feeds two things the project cannot get from simulation alone:

* **Calibration data for INT8 quantization.** Sim-only calibration
  systematically misses the real sensor's noise floor and its stereo-hole
  statistics, so the activation ranges baked into the ``.pte`` are subtly wrong
  in exactly the regime the robot operates in.
* **Measured domain-randomization ranges.** ``--stats`` reports what the real
  sensors actually do, which is how the ranges in ``RandomizationSpec`` get set
  from evidence rather than from guesses. See ``docs/domain_randomization.md``.

Run on the board (or any machine with ROS 2 and the bag):

    ros2 bag record /scan /oak/stereo/image_raw /odom -o bags/room1
    python3 tools/bag_to_obs.py --bag bags/room1 --output calib/real_obs.npz --stats

Messages are paired by nearest timestamp with the same staleness limits the ROS
node enforces, so an observation is only emitted when the node would have been
willing to act on one.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim2edge.preprocess import (  # noqa: E402
    preprocess_depth,
    preprocess_goal,
    preprocess_lidar,
    preprocess_last_action,
)
from sim2edge.spec import DEFAULT_SPEC, NavSpec  # noqa: E402


def _read_bag(path: str, topics: Dict[str, str]):
    """Yield ``(topic, message, timestamp_ns)`` from a rosbag2 directory."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ROS 2 Python packages are required to read a bag. Run this on the "
            f"board or a ROS machine. Original error: {exc}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = set(topics.values())

    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic not in wanted:
            continue
        msg_type = get_message(type_map[topic])
        yield topic, deserialize_message(data, msg_type), stamp


def _depth_to_metres(msg, spec: NavSpec) -> np.ndarray:
    """16UC1 millimetres -> float32 metres, honouring row padding."""
    dtype = ">u2" if msg.is_bigendian else "<u2"
    raw = np.frombuffer(msg.data, dtype=dtype)
    row_words = msg.step // 2
    grid = raw.reshape(msg.height, row_words)[:, :msg.width]
    return grid.astype(np.float32) * spec.obs.depth.raw_scale_to_m


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", required=True, help="rosbag2 directory")
    p.add_argument("--scan-topic", default="/scan")
    p.add_argument("--depth-topic", default="/oak/stereo/image_raw")
    p.add_argument("--odom-topic", default="/odom")
    p.add_argument("--output", default="calib/real_obs.npz")
    p.add_argument("--stride", type=int, default=3,
                   help="Keep every Nth paired sample; consecutive 20 Hz frames "
                        "are near-duplicates.")
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--stats", action="store_true",
                   help="Report measured sensor statistics for setting the "
                        "domain-randomization ranges.")
    args = p.parse_args()

    spec = DEFAULT_SPEC
    topics = {"scan": args.scan_topic, "depth": args.depth_topic,
              "odom": args.odom_topic}

    latest: Dict[str, Tuple[object, int]] = {}
    out: Dict[str, List[np.ndarray]] = {n: [] for n in spec.obs.actor_inputs}
    raw_stats = {"depth_invalid_frac": [], "lidar_invalid_frac": [],
                 "depth_max_m": [], "lidar_max_m": []}
    kept = 0
    seen = 0

    lidar_age = spec.safety.lidar_max_age_s
    depth_age = spec.safety.depth_max_age_s

    for topic, msg, stamp in _read_bag(args.bag, topics):
        key = next(k for k, v in topics.items() if v == topic)
        latest[key] = (msg, stamp)
        # Emit on each new scan, since the lidar is the tightest staleness bound.
        if key != "scan" or len(latest) < 3:
            continue

        seen += 1
        if seen % args.stride:
            continue
        if kept >= args.max_samples:
            break

        # Enforce the same staleness limits the ROS node applies, so the
        # calibration set contains only observations the node would have acted on.
        if abs(stamp - latest["depth"][1]) * 1e-9 > depth_age:
            continue
        if abs(stamp - latest["odom"][1]) * 1e-9 > lidar_age:
            continue

        scan = latest["scan"][0]
        depth_m = _depth_to_metres(latest["depth"][0], spec)
        odom = latest["odom"][0]

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        out["depth"].append(preprocess_depth(depth_m, spec))
        out["lidar"].append(preprocess_lidar(
            ranges, scan.angle_min, scan.angle_increment, spec,
            scan.range_min, scan.range_max))
        # No real goal in a recording: sample plausible relative goals so the
        # goal channel's activation range is covered. The sensor channels are
        # what the real data is actually needed for.
        rng = np.random.default_rng(kept)
        bearing = rng.uniform(-math.pi, math.pi)
        dist = rng.uniform(0.0, spec.obs.goal.max_distance_m)
        out["goal"].append(preprocess_goal(dist * math.cos(bearing),
                                           dist * math.sin(bearing), spec))
        out["last_action"].append(preprocess_last_action(
            float(odom.twist.twist.linear.x),
            float(odom.twist.twist.angular.z), spec))

        if args.stats:
            raw_stats["depth_invalid_frac"].append(
                float(np.mean(~np.isfinite(depth_m) | (depth_m <= 0))))
            raw_stats["lidar_invalid_frac"].append(
                float(np.mean(~np.isfinite(ranges))))
            valid_d = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
            raw_stats["depth_max_m"].append(
                float(valid_d.max()) if valid_d.size else 0.0)
            valid_r = ranges[np.isfinite(ranges)]
            raw_stats["lidar_max_m"].append(
                float(valid_r.max()) if valid_r.size else 0.0)
        kept += 1

    if kept == 0:
        raise SystemExit(
            "No samples survived pairing. Check the topic names against "
            "`ros2 bag info` -- and remember the Create3 namespace differs "
            "between setups.")

    merged = {k: np.stack(v).astype(np.float32) for k, v in out.items()}
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, spec_fingerprint=np.array(spec.fingerprint()),
                        **merged)
    print(f"Wrote {path}: {kept} observations "
          f"({path.stat().st_size / 1e6:.1f} MB)")
    print(f"Spec fingerprint: {spec.fingerprint()}")

    if args.stats:
        print("\n── Measured sensor statistics ──")
        print("Use these to set the ranges in RandomizationSpec. Bracket what "
              "you actually observe; do not guess.\n")
        for name, values in raw_stats.items():
            a = np.asarray(values)
            if a.size:
                print(f"  {name:22s} mean {a.mean():7.4f}  p95 "
                      f"{np.percentile(a, 95):7.4f}  max {a.max():7.4f}")
        print("\n  depth_invalid_frac -> RandomizationSpec.depth_hole_fraction")
        print("  lidar_invalid_frac -> RandomizationSpec.lidar_dropout")
        print("\n  Latency is NOT measurable from this bag alone. Measure "
              "sensor->cmd_vel latency directly and set action_delay_steps to "
              "bracket it; it is the single most important randomization for "
              "sim-to-real on a real ROS stack.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
