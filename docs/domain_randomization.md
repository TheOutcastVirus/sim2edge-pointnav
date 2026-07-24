# Domain randomization: measure, don't guess

Every range in `RandomizationSpec` is a hypothesis about the real robot. The
ranges shipped in `sim2edge/spec.py` are *starting estimates*, and they must be
replaced with measurements from the actual hardware in the first two days —
before a large training run bakes them in.

## Why the ranges matter more than the mechanism

Randomizing over the wrong range is worse than not randomizing:

- Too narrow, and the policy overfits to a sensor that does not exist.
- Too wide, and the policy learns to be so conservative it never moves. A policy
  trained against 30% lidar dropout learns that lidar is untrustworthy, and then
  ignores it on a robot whose lidar is fine.

This is why the evaluation harness reports three noise tiers (clean /
deployment / 2× deployment). A policy that scores well at 1.0× but collapses at
2.0× has fit the ranges rather than become robust — and since the ranges are
estimates from a handful of rosbags, that distinction matters.

## How to measure

```bash
# On the robot: drive it around the demo space, including near obstacles,
# in the lighting the demo will actually run in.
ros2 bag record /scan /oak/stereo/image_raw /odom -o bags/room1

# Off the bag: report what the sensors actually do
python3 tools/bag_to_obs.py --bag bags/room1 --output calib/real_obs.npz --stats
```

The `--stats` output maps directly onto spec fields:

| Measurement | Sets |
|---|---|
| `depth_invalid_frac` | `depth_hole_fraction` |
| `lidar_invalid_frac` | `lidar_dropout` |
| observed max range | sanity-check `LidarSpec.max_m`, `DepthSpec.max_m` |

Set each range to **bracket** the observed value: low end near zero, high end
somewhat above the p95. The policy should see the real robot as an ordinary
draw from the distribution, not as an extreme one.

## What the bag cannot tell you

**Latency.** `action_delay_steps` is the single most important randomization for
sim-to-real on a real ROS stack, and it is not recoverable from a sensor
recording. Measure it directly:

1. Timestamp a `/scan` message.
2. Timestamp the `/cmd_vel` the node emits in response.
3. The difference, in control periods, is the delay to bracket.

Sensor→`cmd_vel` latency on a real robot is never below one control period
(50 ms at 20 Hz). The default range of 1–3 steps assumes 50–150 ms. If the
measurement lands outside that, change the spec rather than hoping.

**Dynamics.** `velocity_tracking_gain`, `motor_tau_s` and `wheel_slip` need a
step-response test: command a step in `v`, log `/odom` velocity, and fit a
first-order lag. The Create3's published limits are a starting point, not a
substitute.

## Noise is applied to raw sensor output

Both in `sim2edge/randomization.py` and in the Isaac Lab task, corruption is
applied **before** preprocessing, never after. This is not a stylistic choice:
min-pooling interacts strongly with per-beam noise and dropout — it biases
pooled values downward — and that interaction is exactly what the real robot
will do too. Injecting noise into an already-pooled observation would be a
different, and wrong, distribution.

The depth corruption model deliberately reproduces three real OAK-D failure
modes rather than adding Gaussian noise:

- **Range-dependent noise**, growing as distance squared (disparity quantization).
- **Holes at depth discontinuities** — block matching fails where the two views
  disagree, which is why holes are drawn correlated with the local depth
  gradient rather than uniformly.
- **Speckle**, isolated invalid pixels.

## Planned ablations

For the writeup (Days 10–12), one training run each:

| Run | Change | Expected finding |
|---|---|---|
| no-latency-DR | `action_delay_steps = (0, 0)` | Largest sim-to-real drop. This is the prediction the ablation is testing. |
| no-noise-DR | sensor noise ranges → 0 | Moderate drop, concentrated in cluttered layouts. |
| full DR | baseline | — |

Evaluate all three on the *same* held-out layouts and on the robot benchmark.
The interesting number is not sim success rate — all three will look fine in
their own training distribution — but the sim-to-real gap each one leaves.
