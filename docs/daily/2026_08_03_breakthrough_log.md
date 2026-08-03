# 2026-08-03 — Body-frame v2 breakthrough and usable actor

## Outcome

The lidar contract was the binding defect. It is repaired, a fresh production
corpus and actor were built, and the selected full stack now clears the
navigation gates at every simulated noise tier. The usable configuration is:

- sensor contract: `body-frame-v2`;
- actor: `runs/body_v2_actor_prod/best.pt`;
- motion prior: Create3/EKF incremental body-frame odometry;
- correction: exact-bearing CPU scan-to-submap tracker;
- arrival: three consecutive uncertainty-aware confirmations within 15 cm.

The run artifacts are intentionally ignored by Git and remain local. The code,
tests, plan, and this evidence record are versioned.

## What changed

The Isaac raycaster had rotated ray origins with the robot while leaving ray
directions fixed in the world. Beam metadata was also off by pi: the configured
pattern begins at `-pi`, not zero. `body-frame-v2` now uses base-aligned rays and
bearings beginning at `-pi + LIDAR_MOUNT_YAW`. Contract versions and observation
fingerprints prevent old shards and checkpoints from being consumed silently.

The repaired asymmetric-room analytic bring-up produced correlation 1.000 at
0, 90, and 180 degrees, with median range disagreement of 0.256, 0.297, and
0.285 cm. The corrected staged probe over 32 pairs measured:

| Probe arm | Median disagreement | p95 disagreement |
|---|---:|---:|
| instantaneous exact-bearing | 0.185 cm | 0.468 cm |
| reconstructed exact-bearing | 0.178 cm | 0.480 cm |
| rasterized | 0.220 cm | 0.512 cm |
| 1x sensor noise | 0.431 cm | 0.985 cm |

This invalidates the earlier world-fixed-v1 localization conclusions and
closes the false 1.095 cm reconstruction-floor diagnosis in the companion log.

## Data and actor

`data/body_frame_v2_prod` contains 65,576 episodes and 16,096,381 steps from
16,384 parallel environments over 8,192 layouts. The layout-separated split
has 77 training shards (15,256,970 samples) and five validation shards (839,411
samples). The 290,610-parameter circular-convolution actor reached a best clean
validation loss of 0.00334.

Pose-fed evaluation on 512 held-out layouts establishes the actor ceiling:

| Noise | Success | Collision | SPL |
|---:|---:|---:|---:|
| 0x | 96.29% | 3.32% | 0.935 |
| 1x | 96.48% | 3.32% | 0.937 |
| 2x | 97.07% | 2.93% | 0.941 |

## End-to-end result

The lidar/command-only prior did not meet the localization gate, so the plan's
evidence-gated fallback was exercised. The selected evaluator synthesizes the
same incremental body-frame quantity expected from Create3 odometry, using
achieved simulated motion plus encoder noise and the measured gyro increment;
ground truth is not passed to the actor or localizer as a pose estimate.

`runs/body_v2_actor_prod/eval_localized_create3_prod_64.json` evaluated 64
held-out layouts at each tier:

| Noise | Correct arrival | False arrival | Collision | Timeout | Mean / p95 belief error | Improvement over command DR |
|---:|---:|---:|---:|---:|---:|---:|
| 0x | 98.44% | 0% | 0% | 1.56% | 0.91 / 2.32 cm | 91.4% |
| 1x | 96.88% | 0% | 0% | 3.12% | 4.65 / 9.00 cm | 65.2% |
| 2x | 95.31% | 0% | 0% | 4.69% | 7.03 / 17.90 cm | 38.7% |

All three tiers clear the navigation gates: at least 85% correct arrivals, at
most 5% false arrivals, at most 10% collisions, and improvement over command
dead reckoning. The strict mean/p95 localization-error gate (5/12.5 cm) passes
at 0x and 1x but not at 2x. Given that normal indoor operation is expected to
be nearer the 1x tier and that the user accepted the present result, this is the
current usable candidate—not a claim of hardware validation.

## Export

`runs/body_v2_actor_prod/export/actor.pt2` is stamped with `body-frame-v2` and
`create3-ekf`. Export parity is exact (`max_abs_difference = 0`). On the host,
actor latency is 0.191 ms median and 0.225 ms p95 against the 50 ms control
period. This measures only the exported actor, not the CPU localizer or target
board.

## Standing issues

1. The 2x tier misses the strict localization-error target: 7.03 cm mean and
   17.90 cm p95 versus 5 cm and 12.5 cm.
2. The full-stack sample is 64 layouts per tier, smaller than the 512-layout
   pose-fed actor evaluation. A larger end-to-end run is desirable before a
   release claim.
3. No robot trial has run. The actual Create3 `/odom` increment, covariance,
   timestamps, lidar mounting error, wheel slip, occlusions, and environmental
   dynamics still need measurement against an external pose reference.
4. The production ROS 2/C++ path that feeds Create3 odometry into the tracker is
   not built. The evaluator's sensor synthesis must be replaced by real
   timestamped messages without changing the body-frame interface.
5. `LIDAR_SPIN_HZ = 5.5` remains unverified against the physical A1 setup.
6. The 128-scan CPU localizer's target-compute latency and memory are not yet
   benchmarked. Only actor latency has been measured.
7. INT8 calibration, QNN lowering, preprocessing parity, and board latency are
   still outstanding. The checked-in export is a Torch exported program, not a
   deployable Qualcomm artifact.
8. Isaac wheel-visual behavior remains cosmetic and unresolved; it did not
   affect the planar navigation measurements.

## Decision

Stop exploratory training here. Preserve `body-frame-v2` plus the
Create3/EKF-assisted localizer as the selected candidate, and make the next
work hardware integration and validation rather than another simulator tuning
cycle.
