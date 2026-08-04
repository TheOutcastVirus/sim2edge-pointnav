#!/usr/bin/env python3
"""Record raw /scan frames to an .npz for offline analysis.

Deliberately dumb: it stores the ranges verbatim and decides nothing. The
dropout question -- what fraction of beams return nothing, and whether those
beams are independent or arrive in contiguous arcs -- is answered afterwards,
so the lidar only has to be connected for the capture.
"""
import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="", help="what this scene is")
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("scan_capture")
    frames, stamps, meta = [], [], {}

    def on_scan(m):
        if not meta:
            meta.update(
                angle_min=float(m.angle_min), angle_max=float(m.angle_max),
                angle_increment=float(m.angle_increment),
                time_increment=float(m.time_increment),
                scan_time=float(m.scan_time),
                range_min=float(m.range_min), range_max=float(m.range_max),
                frame_id=m.header.frame_id)
        frames.append(np.asarray(m.ranges, dtype=np.float32))
        stamps.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        n = len(frames)
        if n % 50 == 0:
            print(f"  {n}/{args.scans}", file=sys.stderr, flush=True)

    node.create_subscription(LaserScan, "scan", on_scan, qos_profile_sensor_data)

    deadline = time.time() + args.scans / 5.0 + 30.0   # generous wall-clock cap
    while len(frames) < args.scans and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

    if not frames:
        print("FAIL: no scans received")
        return 1

    widths = {f.shape[0] for f in frames}
    if len(widths) > 1:
        print(f"WARNING: beam count varied {sorted(widths)}; truncating")
        w = min(widths)
        frames = [f[:w] for f in frames]

    ranges = np.stack(frames)
    np.savez_compressed(args.out, ranges=ranges,
                        stamps=np.asarray(stamps, dtype=np.float64),
                        label=args.label, **meta)
    finite = np.isfinite(ranges)
    print(f"\nsaved {args.out}")
    print(f"  {ranges.shape[0]} scans x {ranges.shape[1]} beams")
    print(f"  label: {args.label!r}")
    print(f"  finite fraction {finite.mean():.4f}  "
          f"-> invalid {1.0 - finite.mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
