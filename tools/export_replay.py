#!/usr/bin/env python3
"""Record held-out episodes as a replay the board can run without Python.

The board has ROS 2, ExecuTorch and a QNN runtime. It does not have torch, the
corpus, this repository's Python environment, or a network connection. So the
parity gate cannot be "run both implementations and compare"; it has to be
"record what Python did, ship it, and let the C++ answer for itself".

That is what this writes. Every tick carries:

  * the **inputs** a control tick receives -- one lidar revolution *before*
    deskew, with per-beam nominal bearings and acquisition ages, plus the gyro
    increment and the base velocity over the tick;
  * the **reference outputs** Python produced from them -- deskewed and binned
    lidar, the tracker's pose, the believed goal offset, the full 731-number
    observation, and the actor's command.

Deskew is deliberately on the C++ side of the boundary rather than baked into
the recorded input. It is the correction whose absence is an action-correlated
sim-to-real gap, and a parity test that assumed it correct would be checking the
easy half.

One asymmetry between this and the robot is worth stating plainly, because it
is the difference between a replay and a trial. In simulation the sweep must be
*reconstructed*: Isaac produces an instantaneous 720-beam snapshot per tick, and
``sensors.sweep_scan`` stitches one revolution out of several of them. A real A1
produces the revolution directly. So the recorded pre-deskew sweep is a
simulated stand-in for what a driver hands over -- correct in structure, per-beam
age and bearing, and untested against the physical unit's actual beam count and
spin rate. Nothing here is a sim-to-real result.

Usage::

    .venv/bin/python tools/export_replay.py \\
        --data data/body_frame_v2_prod/validation \\
        --episodes 12 --out ros2_ws/replays/validation_12.s2e
"""

from __future__ import annotations

import argparse
import math
import pathlib
import struct
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sim2edge import noise as noise_mod                                # noqa: E402
from sim2edge.dataset import load_shard                                # noqa: E402
from sim2edge.localization import LocalPoseTracker                     # noqa: E402
from sim2edge.preprocess import (bin_lidar, build_observation,         # noqa: E402
                                 relative_goal_features)
from sim2edge.sensors import (PolarSweep, deskew_history, sweep_scan,  # noqa: E402
                              synthesize_gyro)
from sim2edge.spec import (DT, HISTORY_FRAMES, LIDAR_BINS, LIDAR_MAX_M,  # noqa: E402
                           LIDAR_MIN_M, LIDAR_SPIN_HZ, OBS_DIM,
                           OMEGA_MAX, SENSOR_CONTRACT_VERSION, SPIN_TICKS,
                           V_MAX, observation_fingerprint)

MAGIC = b"S2EREPL\x00"
VERSION = 1

#: Must match ros2_ws/src/sim2edge_core/src/replay.cpp::kScalarCount.
SCALARS = 13

#: The dataset loader reconstructs sweeps with sub-tick interpolation, and the
#: deployed checkpoint was trained on that. Keep the two together: interpolating
#: the range without interpolating the motion correction applied to it is
#: strictly worse than rounding both.
SUBTICK = True


#: Header flag: the reference action came from an actor checkpoint rather than
#: the corpus's stored oracle command. It changes what an action comparison
#: MEANS -- against the actor it measures quantization error, against the oracle
#: it measures how far the policy is from its teacher -- so the reader is told
#: which, instead of the number being reported under one label for both.
FLAG_REFERENCE_ACTION_FROM_ACTOR = 1 << 0


def _header(episodes: int, ticks: int, beams: int, flags: int) -> bytes:
    def fixed(text: str, width: int) -> bytes:
        raw = text.encode()
        if len(raw) >= width:
            raise ValueError(f"{text!r} does not fit in {width} bytes")
        return raw.ljust(width, b"\x00")

    return struct.pack(
        "<8sII24s24sIIII", MAGIC, VERSION, flags,
        fixed(observation_fingerprint(), 24),
        fixed(SENSOR_CONTRACT_VERSION, 24),
        episodes, ticks, beams, 0)


def _episode_indices(shard, count: int, min_length: int) -> list[np.ndarray]:
    """Whole episodes, long enough to reach steady state, longest first.

    Longest first because the submap fills over the first seconds and the cost
    that matters is the one paid afterwards; a replay of short episodes would
    measure the transient.
    """
    episode = shard.episode.cpu().numpy()
    out: list[np.ndarray] = []
    for value in np.unique(episode):
        index = np.flatnonzero(episode == value)
        if index.size >= min_length:
            out.append(index)
    out.sort(key=len, reverse=True)
    return out[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=pathlib.Path, required=True,
                        help="a split directory holding bc_*.npz shards")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--min-length", type=int, default=80,
                        help="skip episodes shorter than this many ticks")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--checkpoint", type=pathlib.Path,
                        help="actor checkpoint; without it the reference "
                             "action is the stored oracle command and the "
                             "board's action parity is skipped")
    parser.add_argument("--max-submap-scans", type=int, default=128)
    parser.add_argument("--point-stride", type=int, default=2)
    args = parser.parse_args()

    shards = sorted(args.data.glob("*.npz"))
    if not shards:
        print(f"no shards under {args.data}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    actor = None
    if args.checkpoint is not None:
        from sim2edge.training import load_policy_checkpoint
        actor = load_policy_checkpoint(args.checkpoint, device="cpu").eval()

    # One shard is plenty: a shard holds many episodes and the gate is about
    # agreement per tick, not corpus coverage.
    shard = load_shard(shards[0], device="cpu")
    print(f"shard {shards[0].name}: {len(shard)} samples, "
          f"driver={shard.driver}, contract={shard.sensor_contract_version}")

    episodes = _episode_indices(shard, args.episodes, args.min_length)
    if not episodes:
        print(f"no episode reaches {args.min_length} ticks", file=sys.stderr)
        return 1

    beams = int(shard.ranges.shape[1])
    gyro_all = synthesize_gyro(shard.pose[:, 2], shard.episode,
                               scale=args.noise_scale, generator=generator)

    episode_records: list[bytes] = []
    tick_blobs: list[bytes] = []
    first_tick = 0

    for order, index in enumerate(episodes):
        steps = index.size
        # A tick needs one full revolution of scan history behind it, so the
        # first SPIN_TICKS-1 samples of an episode cannot be served.
        usable = steps - (SPIN_TICKS - 1)
        if usable <= 0:
            continue

        raw = shard.ranges[index].float()                      # (T, B)
        # One mount/bearing error per episode, not per frame: the sensor is
        # bolted on once.
        bias = torch.randn((1, 1), generator=generator) * (
            noise_mod.BEARING_BIAS_RAD * args.noise_scale)
        noisy, angles = noise_mod.corrupt_lidar(
            raw, shard.angles, args.noise_scale, generator,
            bearing_bias=bias.expand(steps, 1))

        gyro = gyro_all[index]                                 # (T,)
        speed = shard.prev_action[index, 0]                    # (T,)

        # The motion history the *robot* has, not the one the corpus offers.
        #
        # The first tick a replay can serve is SPIN_TICKS-1 into the episode,
        # because a sweep needs a revolution of scan history behind it. The
        # corpus happily supplies the gyro and speed from before that point,
        # but a robot powering up does not have them -- its deskew history is
        # empty and fills over the first revolution, and the first tick has no
        # preceding interval at all.
        #
        # Recording the corpus's version would make the reference unreachable
        # by any correct implementation and turn a warm-up into a permanent
        # parity failure. Zeroing it here records the cold start the C++
        # actually performs, which is also what the hardware does.
        start_pad = SPIN_TICKS - 1
        gyro = gyro.clone()
        speed = speed.clone()
        gyro[:start_pad + 1] = 0.0
        speed[:start_pad + 1] = 0.0
        pose_true = shard.pose[index].numpy().astype(np.float64)
        goal_world = shard.goal[index].numpy().astype(np.float64)

        # Phase is a property of the motor and free-running, so it advances
        # across the episode rather than being redrawn per tick.
        phase0 = float(torch.rand((1,), generator=generator).item())

        tracker = LocalPoseTracker(max_submap_scans=args.max_submap_scans,
                                   point_stride=args.point_stride)

        # The goal, in the body frame at the first served tick. This is the one
        # and only place ground truth enters, exactly as in eval_localized.py.
        start = SPIN_TICKS - 1
        yaw0 = float(pose_true[start, 2])
        delta0 = goal_world[start] - pose_true[start, :2]
        c0, s0 = math.cos(yaw0), math.sin(yaw0)
        goal_body0 = np.array([c0 * delta0[0] + s0 * delta0[1],
                               -s0 * delta0[0] + c0 * delta0[1]])

        lidar_history = torch.zeros(1, HISTORY_FRAMES, LIDAR_BINS)
        action_history = torch.zeros(1, HISTORY_FRAMES, 2)
        history_ready = False
        emitted = 0

        for t in range(start, steps):
            window = noisy[t - SPIN_TICKS + 1:t + 1].unsqueeze(0)   # (1,K,B)
            phase = torch.tensor([(phase0 + LIDAR_SPIN_HZ * DT * t) % 1.0])
            swept, ages = sweep_scan(window, phase, subtick=SUBTICK)

            nominal = angles[t].unsqueeze(0)                        # (1, B)
            gyro_window = gyro[t - SPIN_TICKS + 1:t + 1].unsqueeze(0)
            speed_window = speed[t - SPIN_TICKS + 1:t + 1].unsqueeze(0)

            deskewed, bearings = deskew_history(
                swept, nominal, ages, gyro_window, speed_window, subtick=SUBTICK)

            bins = bin_lidar(deskewed, bearings)                    # (1, 180)

            valid = (torch.isfinite(deskewed) & (deskewed >= LIDAR_MIN_M)
                     & (deskewed <= LIDAR_MAX_M))
            sweep = PolarSweep(ranges=deskewed[0], bearings=bearings[0],
                               ages=ages[0], valid=valid[0])

            tick_gyro = 0.0 if t == start else float(gyro[t])
            tick_speed = 0.0 if t == start else float(speed[t])
            estimate = tracker.update(sweep, tick_gyro, tick_speed, DT)

            # The Create 3 odometry arm, recorded alongside so the same replay
            # can exercise either motion prior without a second export. It is
            # the true body-frame increment plus encoder noise -- which sees
            # through the wheel slip a real encoder would miss, so it is the
            # optimistic arm and is labelled as such in the runbook.
            if t == start:
                prior_delta = np.zeros(3)
            else:
                world_delta = pose_true[t, :2] - pose_true[t - 1, :2]
                yaw_prev = pose_true[t - 1, 2]
                cp, sp = math.cos(yaw_prev), math.sin(yaw_prev)
                encoder = rng.normal(
                    0.0, noise_mod.ODOM_XY_WALK_M * args.noise_scale, size=2)
                prior_delta = np.array([
                    cp * world_delta[0] + sp * world_delta[1] + encoder[0],
                    -sp * world_delta[0] + cp * world_delta[1] + encoder[1],
                    float(gyro[t])])

            # Believed goal offset, in the body frame, from the tracker's pose.
            pose_est = estimate.relative_pose
            delta = goal_body0 - pose_est[:2]
            ce, se = math.cos(pose_est[2]), math.sin(pose_est[2])
            goal_body = np.array([ce * delta[0] + se * delta[1],
                                  -se * delta[0] + ce * delta[1]])

            previous = shard.prev_action[index[t]]
            normalized = torch.tensor([[float(previous[0]) / V_MAX,
                                        float(previous[1]) / OMEGA_MAX]])
            if not history_ready:
                lidar_history[:] = bins.unsqueeze(1)
                action_history[:] = normalized.unsqueeze(1)
                history_ready = True
            else:
                lidar_history[:, :-1] = lidar_history[:, 1:].clone()
                action_history[:, :-1] = action_history[:, 1:].clone()
                lidar_history[:, -1] = bins
                action_history[:, -1] = normalized

            goal_features = relative_goal_features(
                torch.tensor([[goal_body[0], goal_body[1]]],
                             dtype=torch.float32))
            observation = build_observation(lidar_history, goal_features,
                                            action_history)

            if actor is not None:
                with torch.inference_mode():
                    command = actor(observation)[0].numpy().astype(np.float32)
            else:
                command = shard.action[index[t]].numpy().astype(np.float32)

            tick_blobs.append(_pack_tick(
                deskew_input_ranges=swept[0].numpy().astype(np.float32),
                nominal_bearings=nominal[0].numpy().astype(np.float32),
                ages=ages[0].numpy().astype(np.float32),
                gyro_delta=tick_gyro, speed=tick_speed,
                prior_delta=prior_delta,
                prev_v=float(previous[0]), prev_omega=float(previous[1]),
                ref_pose=pose_est, ref_goal_body=goal_body,
                ref_deskewed_ranges=deskewed[0].numpy().astype(np.float32),
                ref_deskewed_bearings=bearings[0].numpy().astype(np.float32),
                ref_bins=bins[0].numpy().astype(np.float32),
                ref_observation=observation[0].numpy().astype(np.float32),
                ref_action=command,
                observable=bool(estimate.observable)))
            emitted += 1

        episode_records.append(struct.pack(
            "<IIdd", first_tick, emitted, float(goal_body0[0]),
            float(goal_body0[1])))
        first_tick += emitted
        print(f"  episode {order}: {emitted} ticks, "
              f"goal ({goal_body0[0]:+.2f}, {goal_body0[1]:+.2f}) m")

    if not tick_blobs:
        print("nothing to write", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        flags = FLAG_REFERENCE_ACTION_FROM_ACTOR if actor is not None else 0
        handle.write(_header(len(episode_records), len(tick_blobs), beams,
                             flags))
        for record in episode_records:
            handle.write(record)
        for blob in tick_blobs:
            handle.write(blob)

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"\nwrote {args.out} -- {len(tick_blobs)} ticks in "
          f"{len(episode_records)} episodes, {beams} beams, {size_mb:.1f} MiB")
    print(f"  observation fingerprint {observation_fingerprint()}")
    print(f"  reference action from   "
          f"{'actor ' + str(args.checkpoint) if actor else 'stored oracle command'}")
    return 0


def _pack_tick(*, deskew_input_ranges, nominal_bearings, ages, gyro_delta,
               speed, prior_delta, prev_v, prev_omega, ref_pose,
               ref_goal_body, ref_deskewed_ranges, ref_deskewed_bearings,
               ref_bins, ref_observation, ref_action, observable) -> bytes:
    """Serialize one tick. Layout is mirrored in replay.cpp; keep them together.

    Doubles lead and the record is padded to eight bytes so every tick starts
    aligned whatever the beam count is -- see the note in replay.cpp.
    """
    scalars = np.array([
        gyro_delta, speed, DT,
        prior_delta[0], prior_delta[1], prior_delta[2],
        prev_v, prev_omega,
        ref_pose[0], ref_pose[1], ref_pose[2],
        ref_goal_body[0], ref_goal_body[1],
    ], dtype=np.float64)
    assert scalars.size == SCALARS, scalars.size
    assert ref_bins.size == LIDAR_BINS
    assert ref_observation.size == OBS_DIM
    assert ref_action.size == 2

    floats = np.concatenate([
        np.asarray(deskew_input_ranges, dtype=np.float32),
        np.asarray(nominal_bearings, dtype=np.float32),
        np.asarray(ages, dtype=np.float32),
        np.asarray(ref_deskewed_ranges, dtype=np.float32),
        np.asarray(ref_deskewed_bearings, dtype=np.float32),
        ref_bins, ref_observation,
        np.asarray(ref_action, dtype=np.float32),
    ])

    body = scalars.tobytes() + floats.tobytes() + bytes([1 if observable else 0])
    padding = (-len(body)) % 8
    return body + b"\x00" * padding


if __name__ == "__main__":
    raise SystemExit(main())
