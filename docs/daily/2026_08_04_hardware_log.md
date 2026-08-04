# 2026-08-04 (third) — Real sensors, and three bugs that could only appear here

The ROS 2 log from earlier today ended with a list whose items 2 and 3 were
"attach the A1" and "attach the OAK-D". Both are attached. The stack now runs
end to end on real sensor data, and it did not do so on the first attempt:
three defects were sitting in code that passed every replay test, because the
replay could not express the conditions that trigger them. One of the three
would not have shown up as a failure at all — it would have looked like the
localizer being bad.

## Outcome

The acceptance run — shipped defaults, physical lidar and gyro, navigator
pinned to `cpu2,3`, actor INT8 on the Hexagon:

| | value |
|---|---:|
| ticks | **2,513** |
| ticks over the 50 ms period | **0** |
| ticks held on stale sensors | **0** |
| tick median | **8.8 ms** |
| tick p95 | 34.0 ms |
| tick max | 41.1 ms |
| actor | 0.26–0.59 ms |
| localizer observable | **100%** |

That is with `max_submap_scans` reduced from 128 to 64, which is the one
production default this day changed and the reason is below.

## The A1 is not the sensor `spec.py` describes

| | `spec.py` / sim | measured | |
|---|---:|---:|---|
| spin rate | 5.5 Hz | **7.32 Hz** | +33% |
| beams per revolution | 720 | **1080** | +50% |
| **valid** returns per revolution | 720 (all) | **855** (79%) | +19% |
| angle increment | — | 0.3336° | |
| `time_increment` | — | 121 µs, nonzero | |
| `range_min` | 0.15 m | 0.05 m (driver's) | |

Three of these are worth separating, because they do not have the same
consequence.

**The spin rate does not affect the deployed stack at all.** `beamGeometry`
reads `time_increment` from the message and only falls back to
`spec::kSpinPeriodS` when the driver reports zero. This driver reports 121 µs,
so the fallback is never reached and the 5.5 Hz constant is inert at runtime.
It is *not* inert in training: `SPIN_TICKS` and the sweep reconstruction are
derived from it, so the policy was trained against a sweep smeared 33% more
than the hardware smears it. That is the benign direction — less skew than
trained for — but it is a real sim-to-real gap and it belongs to whoever
decides whether to re-derive the corpus. **`LIDAR_SPIN_HZ` has deliberately
not been changed**: it feeds `sensor_model_fingerprint()`, and editing it
silently reinterprets every stored dataset.

**The driver's own report is wrong and should not be believed.** `sllidar` logs
`scan mode: Sensitivity, sample rate: 8 Khz, ..., scan frequency:10.0 Hz`. The
motor does not turn at 10 Hz. 8000 samples/s over 1080 bins is 7.4 rev/s, which
is what arrives, and the measured 7.32 Hz agrees. The 10.0 is the mode's
nominal, not a measurement.

**The valid-return count moves in the opposite direction from what was
assumed.** The morning log reasoned that "a real RPLIDAR A1 delivers roughly
half [720] per revolution, so the board figure is probably pessimistic". It
delivers 855 valid — *more* than the sim, not half. The localizer estimate was
optimistic, not pessimistic. The log flagged this as a reason to re-measure
rather than assume margin, and re-measuring is what reversed the sign.

## Bug 1: the IMU published timestamps in the wrong epoch

The navigator held on `imu stale` forever and served **zero** ticks. The age it
reported was `1785869327` seconds — 56 years.

`oak_imu_node` called `getTimestampDevice()`, which counts from the *device's*
boot, and wrote it into `header.stamp` as though it were a ROS epoch time. The
stamps came out around 241 while every consumer compared them against 1.79e9.

Nothing failed loudly. The node logged a clean startup, `/imu` published at a
perfect 200.4 Hz, the intervals between samples were correct to 4 µs, and
`ros2 topic echo` showed plausible gyro values. Every property anyone would
think to check was right except the epoch.

The fix keeps what the device clock is *for* — intervals free of USB scheduling
jitter, which is the whole reason the node publishes at the sensor's rate — by
using `getTimestamp()` (the device capture time already translated into the
host's steady clock by depthai's own sync) and rebasing it into the ROS epoch
with an offset sampled once per poll. After the fix, stamp-versus-now lag is
6.5–20.8 ms and the sample interval is still 4.9908 ms with σ = 4 µs.

This could not have been caught by `sim2edge_parity` or `replay_test.launch.py`.
`replay_node` synthesizes its own stamps from `now()`, so the replay path never
executes the line that was wrong.

## Bug 2: `scan_max_age_s` was unreachable by construction

With the IMU fixed, the loop still held on `scan stale` about one tick in six,
with ages of 0.257–0.268 s against a 0.25 s threshold.

Two compounding errors, both in the same comment:

1. It assumed a 182 ms revolution. The real one is 137 ms.
2. It assumed a scan's age at arrival is ~0. **`sllidar` stamps the scan at the
   start of the sweep and publishes it at the end**, so a scan is already one
   full revolution old the instant it arrives. Measured stamp-to-arrival lag:
   137 ms median, exactly one revolution.

Scan age therefore sweeps 137 → 274 ms between messages and spends the top
~24 ms of every cycle above 250 ms. The threshold was not tight; it was below
the floor.

0.35 s clears the 274 ms normal worst case and still holds on a genuinely
dropped message (3 revolutions = 411 ms), which is the wanted behaviour — a
hold publishes zero.

The `trailing` computation in `navigator_node` — `scan_age - scan_time` — is
correct under this stamping and needed no change. The bug was in the
freshness threshold only.

## The localizer costs 2.6× what the replay said, and the reason is density

With both bugs fixed and production `max_submap_scans: 128`, the loop ran but
missed its period on **245 of 1107 ticks (22%)**: median 18.1 ms, p95 68 ms,
max 79 ms. The replay reported 0 of 2,114 at the same configuration.

| | replay | real lidar | ratio |
|---|---:|---:|---:|
| submap points | 40,407 | 54,603 | 1.35× |
| localizer median | 6.90 ms | 18.11 ms | 2.62× |
| **ns per submap point** | 170.8 | 331.6 | **1.94×** |

The point count explains 1.35× of it. The rest is that each point costs nearly
twice as much, which means the uniform bucket grid has stopped pruning.

Sweeping the submap on real data shows why:

| submap | median | p95 | max | submap points | ns/point |
|---:|---:|---:|---:|---:|---:|
| 128 | 18.11 ms | 65.4 ms | 73.7 ms | 54,603 | 331.6 |
| **64** | **8.54 ms** | 34.3 ms | 38.2 ms | 27,303 | 312.8 |
| 32 | 5.15 ms | 15.1 ms | 17.6 ms | 13,656 | 377.1 |

Cost per point is flat across submap size, so total cost is *linear* in submap
points with no pruning benefit whatsoever — every correspondence query is
effectively scanning the whole cloud. The replay's own sweep was sublinear
(128→64 cost 0.66×, not 0.50×), which is what a working spatial index looks
like.

**The mechanism is that the submap has no keyframe gate.**
`LocalPoseTracker::addScan` pushes a scan on every served tick unconditionally
and drops the oldest past `max_submap_scans`. A stationary or slow robot
therefore packs 128 near-identical scans into the same 0.20 m grid cells — the
cell is sized at the ICP rejection radius, so a 3×3 cell scan that is meant to
be an exact search over a small region becomes a near-brute-force search over
128× redundancy describing what 427 points already describe. This bench robot
is stationary and the scene is close (median valid return 0.56 m), which is the
worst case; driving spreads the scans out and should cost less.

The corpus could not show this. It was oracle-driven and always moving, so
consecutive scans were always displaced.

### What changed, and why 64 rather than 32

`max_submap_scans: 64`. The accuracy case is not an assumption —
`eval_localized_submap64_2x_32.json` and `eval_localized_submap128_2x_32.json`
already exist and report the **same success rate (0.84375)**, the same
false-arrival rate (0.0) and the same collision rate (0.15625), with 64 costing
2.6 mm of mean final belief error and 1.1 cm at p95. 32 is faster still and has
no evaluation at this configuration, which is the same reason `point_stride 4`
remains unadopted.

## What is now known about the gyro, and what is not

The OAK-D is on the bus and publishing. At rest:

| | value |
|---|---:|
| rate | 200.4 Hz, intervals σ = 4 µs |
| `angular_velocity.z` bias | −0.00285 rad/s (−0.163 °/s) |
| `angular_velocity.z` noise (1σ) | 0.00351 rad/s (0.201 °/s) |
| bias over one 137 ms revolution | −0.022° |
| bias over 60 s | −9.8° |

The bias is negligible per revolution, which is what deskew needs, and large
over a minute, which is what the scan-match correction exists to absorb.

## Bug 3: the stack was reading the wrong gyro axis

The accelerometer at rest put gravity on **+y** (+9.568 m/s²), which says the
vertical axis is y — while the stack consumed `angular_velocity.z`. Turning the
robot a quarter turn left and back settles it:

| burst | dx | dy | dz | |
|---|---:|---:|---:|---|
| 1 | −1.18° | **+34.93°** | +1.25° | left, first part |
| 2 | −2.13° | **+65.62°** | +3.15° | left, second part |
| 3 | +2.94° | **−100.90°** | −3.32° | back to the right |

**y carries 100.9° against z's 3.3° — a factor of 30.** The outbound legs sum
to +100.55° and the return is −100.90°, closing to 0.35° over 100°, so this is
a real rotation signal and not drift. It agrees with the accelerometer.

So `angular_velocity.z` was never the yaw rate on this mount. Deskew was being
handed a pitch/roll channel that reads ~3% of the true rotation — it would not
have failed, it would have quietly under-corrected the sweep by 97% and
presented as a localization failure.

Counter-clockwise came out **positive**, so the sign already matches the
REP-103 convention the stack assumes; only the channel was wrong.

The fix is `gyro_yaw_axis: "y"` and `gyro_yaw_sign: 1.0` in `navigator.yaml`,
read by `NavigatorNode::yawRateOf`. It is a parameter rather than a constant
because it is a *mount* fact: re-mount the OAK-D and it must be re-measured.
`tools/probe_yaw_axis.py` is the measurement, and it needs no timing
coordination — it segments on gyro magnitude and integrates each burst, so
"turn it left, then back right, whenever you like" is the whole protocol.

The verification the operator should insist on is the out-and-back cancelling.
A single turn tells you which channel moved; only the return leg distinguishes
rotation from drift.

## Also done

`ubuntu` was not in `dialout`, so `/dev/ttyUSB0` (`crw-rw---- root:dialout`)
could not be opened by any node. Fixed with `usermod -aG dialout`.

## What is not done

1. **The quantization gate**, which has been top of this list since this
   morning and remains untouched. Nothing here changes that.
3. **No Create 3.** The acceptance run used `require_odom:=false`, so the motion
   prior ran without the base's velocity term and the robot never moved. Every
   number here is latency and numerics; none of it is navigation.
4. **The missing keyframe gate.** Reducing the submap treats the symptom. The
   cause is that a slow robot fills its submap with redundant scans, and a gate
   on distance or rotation travelled since the last inserted scan would fix the
   cost and the degeneracy at once — but it changes what the localizer matches
   against and would invalidate the existing accuracy evaluations.
5. **`LIDAR_SPIN_HZ` is still 5.5** against a measured 7.32, deliberately. It is
   inert at runtime and load-bearing in training.

## Order of work

1. The quantization gate on the host. It has been the top of this list all day
   and nothing measured here displaces it.
2. The keyframe gate on the submap, with a fresh accuracy evaluation — this is
   the real fix for the localizer cost, and `max_submap_scans: 64` is a
   workaround for it.
3. Connect the Create 3 and confirm `/odom` and `/cmd_vel`.
4. Then a stationary trial with the wheels off the ground. The gyro axis is
   settled, so deskew can now be checked against a rotation the robot performs
   itself rather than one applied by hand: command a known yaw, and the
   integrated `gyro_yaw_axis` should track it.

## A note on what the replay harness can and cannot prove

Three bugs, all in code that passed `sim2edge_parity` and
`replay_test.launch.py`, and each invisible for a different reason:

| bug | why the replay could not see it |
|---|---|
| IMU epoch | `replay_node` synthesizes stamps from `now()`, so the wrong line never ran |
| `scan_max_age_s` | the replay publishes on its own schedule, with no sweep-start stamping |
| gyro yaw axis | the corpus stores a scalar yaw rate; there are no axes to get wrong |

The pattern is that a replay proves the arithmetic and cannot prove the
*contract with the hardware* — every one of these was a disagreement about what
a field means, not about what to compute with it. Worth remembering the next
time a parity pass reads like coverage.
