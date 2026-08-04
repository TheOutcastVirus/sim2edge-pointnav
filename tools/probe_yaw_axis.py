#!/usr/bin/env python3
"""Which gyro channel is yaw, and with what sign.

The stack consumes ``angular_velocity.z`` as the robot's yaw rate. The
accelerometer at rest puts gravity on the OAK-D's +y, which says the vertical
axis is y instead -- and if that is right, deskew is being handed a pitch or
roll channel. A wrong axis here does not fail; it inverts or nulls the deskew
rotation and presents as a localization failure.

No timing coordination with the operator: the recording runs long, motion is
found by segmenting on gyro magnitude, and each burst is integrated separately.
Turning left and then back right therefore produces two bursts with opposite
signs on the same axis, which is what distinguishes a real yaw channel from
drift.
"""
import argparse
import math
import statistics
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

AXES = ("x", "y", "z")
STACK_AXIS = "z"          # what navigator_node integrates today
MIN_BURST_S = 0.30        # shorter than this is a bump, not a turn
MERGE_GAP_S = 0.40        # two bursts closer than this are one turn
MIN_TURN_DEG = 20.0       # a burst worth drawing a conclusion from


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("yaw_axis_probe")
    samples = []
    node.create_subscription(
        Imu, "imu",
        lambda m: samples.append((
            m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
            m.linear_acceleration.x, m.linear_acceleration.y,
            m.linear_acceleration.z)),
        qos_profile_sensor_data)

    end = time.time() + args.seconds
    last = 0.0
    print(f"recording {args.seconds:.0f} s -- turn the robot whenever you are "
          f"ready", file=sys.stderr)
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - last > 10.0:
            last = time.time()
            print(f"  {end - time.time():.0f} s left, {len(samples)} samples",
                  file=sys.stderr)
    node.destroy_node()
    rclpy.shutdown()

    if len(samples) < 200:
        print(f"FAIL: only {len(samples)} samples")
        return 1

    t = [s[0] for s in samples]
    gyro = {a: [s[1 + i] for s in samples] for i, a in enumerate(AXES)}
    accel = {a: [s[4 + i] for s in samples] for i, a in enumerate(AXES)}

    # Most of the recording is the robot sitting still, so the median is a
    # robust first guess at each axis's bias.
    bias = {a: statistics.median(gyro[a]) for a in AXES}
    norm = [math.sqrt(sum((gyro[a][i] - bias[a]) ** 2 for a in AXES))
            for i in range(len(samples))]
    quiet = statistics.median(norm)
    mad = statistics.median([abs(v - quiet) for v in norm])
    threshold = max(0.08, quiet + 8.0 * mad)   # rad/s

    still = [i for i, v in enumerate(norm) if v < threshold]
    if len(still) > 50:
        bias = {a: statistics.fmean([gyro[a][i] for i in still]) for a in AXES}

    # Contiguous runs above threshold, merged across short gaps.
    moving = [v >= threshold for v in norm]
    bursts, start = [], None
    for i, m in enumerate(moving):
        if m and start is None:
            start = i
        elif not m and start is not None:
            bursts.append((start, i))
            start = None
    if start is not None:
        bursts.append((start, len(moving) - 1))

    merged = []
    for b in bursts:
        if merged and t[b[0]] - t[merged[-1][1]] < MERGE_GAP_S:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    bursts = [b for b in merged if t[b[1]] - t[b[0]] >= MIN_BURST_S]

    print(f"\n=== gyro yaw-axis probe, {len(samples)} samples over "
          f"{t[-1]-t[0]:.1f} s ===\n")
    print(f"  motion threshold          {threshold:.4f} rad/s")
    print(f"  still samples             {len(still)} "
          f"({100.0*len(still)/len(samples):.0f}%)")
    print("  bias, from still          " +
          "  ".join(f"{a}={bias[a]:+.5f}" for a in AXES) + " rad/s")

    g = {a: statistics.fmean([accel[a][i] for i in still or range(len(samples))])
         for a in AXES}
    up = max(AXES, key=lambda a: abs(g[a]))
    print(f"  gravity at rest           " +
          "  ".join(f"{a}={g[a]:+.3f}" for a in AXES) +
          f" m/s^2  -> vertical axis is '{up}' ({g[up]:+.2f})")

    if not bursts:
        print("\n  NO MOTION DETECTED. The robot never turned far enough to "
              "clear the\n  noise floor -- re-run and rotate it a full quarter "
              "turn.")
        return 1

    print(f"\n  {len(bursts)} motion burst(s), integrated per axis "
          f"(bias removed):\n")
    print(f"  {'#':>2} {'start':>7} {'dur':>6}   " +
          "".join(f"{'d'+a+' (deg)':>13}" for a in AXES))
    totals = {a: [] for a in AXES}
    for n, (i0, i1) in enumerate(bursts, 1):
        d = {a: 0.0 for a in AXES}
        for i in range(i0, i1):
            dt = t[i + 1] - t[i]
            if not (0.0 < dt < 0.1):
                continue
            for a in AXES:
                d[a] += (gyro[a][i] - bias[a]) * dt
        for a in AXES:
            totals[a].append(math.degrees(d[a]))
        print(f"  {n:>2} {t[i0]-t[0]:>6.1f}s {t[i1]-t[i0]:>5.2f}s   " +
              "".join(f"{math.degrees(d[a]):>+13.2f}" for a in AXES))

    # The yaw axis is the one that actually moved.
    peak = {a: max(abs(v) for v in totals[a]) for a in AXES}
    winner = max(AXES, key=lambda a: peak[a])
    runner = sorted(AXES, key=lambda a: peak[a])[-2]

    print(f"\n  peak |rotation| per axis  " +
          "  ".join(f"{a}={peak[a]:7.2f}deg" for a in AXES))

    print("\nVERDICT\n")
    if peak[winner] < MIN_TURN_DEG:
        print(f"  INCONCLUSIVE: the largest rotation seen was "
              f"{peak[winner]:.1f} deg on '{winner}'.\n"
              f"  Re-run and turn the robot a full quarter turn in place.")
        return 1

    ratio = peak[winner] / max(peak[runner], 1e-6)
    print(f"  The turning axis is '{winner}' "
          f"({peak[winner]:.1f} deg peak, {ratio:.1f}x the next axis "
          f"'{runner}' at {peak[runner]:.1f} deg).")

    if winner == up:
        print(f"  It agrees with the accelerometer, which puts gravity on "
              f"'{up}'. Consistent.")
    else:
        print(f"  ** It does NOT agree with the accelerometer, which puts "
              f"gravity on '{up}'.\n     Something is off -- was the robot "
              f"tilted or lifted rather than turned in place?")

    # Only bursts that actually turned the robot say anything about sign. A
    # nudge while letting go of it clears the motion threshold easily and would
    # otherwise be read as "the first turn", inverting the conclusion.
    real = [v for v in totals[winner] if abs(v) >= MIN_TURN_DEG]
    skipped = len(totals[winner]) - len(real)
    if skipped:
        print(f"  ({skipped} burst(s) under {MIN_TURN_DEG:.0f} deg ignored as "
              f"handling, not turns.)")
    if not real:
        print(f"  INCONCLUSIVE: no burst on '{winner}' reached "
              f"{MIN_TURN_DEG:.0f} deg.")
        return 1

    first = real[0]
    sense = "POSITIVE" if first > 0 else "NEGATIVE"
    print(f"\n  First real turn on '{winner}' is {first:+.1f} deg -> {sense}.")
    print(f"  If that turn was counter-clockwise seen from above (front "
          f"sweeping LEFT),\n  then '{winner}' "
          f"{'already follows' if first > 0 else 'is INVERTED against'}"
          f" the REP-103 convention the stack assumes\n  (CCW = positive), so "
          f"gyro_yaw_sign is {'+1' if first > 0 else '-1'}.")

    # Consecutive same-sign bursts are one turn the operator paused mid-way, so
    # sum them before looking for the return leg.
    legs, run = [], real[0]
    for v in real[1:]:
        if v * run > 0:
            run += v
        else:
            legs.append(run)
            run = v
    legs.append(run)

    if len(legs) >= 2:
        out, back = legs[0], legs[1]
        close = abs(out + back)
        print(f"\n  Out {out:+.1f} deg, back {back:+.1f} deg -- closes to "
              f"{close:.2f} deg over {abs(out):.0f}.")
        if close < 0.1 * abs(out):
            print("  The return leg cancels the outbound one, so this is a "
                  "real rotation\n  signal and not drift.")
        else:
            print("  ** The legs do not cancel. Either the robot did not come "
                  "back to where it\n     started, or something is drifting.")
    else:
        print("\n  Only one turn direction seen. Turn it back the other way to "
              "confirm the\n  signal cancels rather than drifts.")

    print(f"\n  The stack currently reads angular_velocity.{STACK_AXIS}.")
    if winner == STACK_AXIS:
        print(f"  That is the axis that moved. No remapping needed; only the "
              f"sign is left to\n  confirm against the direction you turned.")
    else:
        print(f"  ** That is NOT the axis that moved. '{STACK_AXIS}' saw only "
              f"{peak[STACK_AXIS]:.1f} deg while\n     '{winner}' saw "
              f"{peak[winner]:.1f} deg. The gyro must be remapped, or the "
              f"OAK-D\n     re-mounted, before anything drives.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
