# 2026-08-04 (third) — Real sensors, and five bugs that could only appear here

The ROS 2 log from earlier today ended with a list whose items 2 and 3 were
"attach the A1" and "attach the OAK-D." Both are attached. The stack now
runs end to end on real sensor data, and it did not do so on the first
attempt: five defects were sitting in code that passed every replay test,
because the replay could not express the conditions that trigger them. Two
of the five would not have shown up as failures at all — they'd have looked
like the localizer being bad.

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

That's with `max_submap_scans` reduced from 128 to 64, the one production
default this day changed — reason below.

## The A1 is not the sensor `spec.py` describes

| | `spec.py` / sim | measured | |
|---|---:|---:|---|
| spin rate | 5.5 Hz | **7.32 Hz** | +33% |
| beams per revolution | 720 | **1080** | +50% |
| **valid** returns per revolution | 720 (all) | **855** (79%) | +19% |
| angle increment | — | 0.3336° | |
| `time_increment` | — | 121 µs, nonzero | |
| `range_min` | 0.15 m | 0.05 m (driver's) | |

Three of these matter differently. **The spin rate doesn't affect the
deployed stack at all** — `beamGeometry` reads `time_increment` from the
message and only falls back to `spec::kSpinPeriodS` when the driver reports
zero; this driver reports 121 µs, so the 5.5 Hz constant is inert at
runtime. It is *not* inert in training: `SPIN_TICKS` and the sweep
reconstruction derive from it, so the policy trained against a sweep
smeared 33% more than hardware smears it — the benign direction (less skew
than trained for), but a real sim-to-real gap belonging to whoever decides
to re-derive the corpus. **`LIDAR_SPIN_HZ` has deliberately not been
changed** — it feeds `sensor_model_fingerprint()`, and editing it silently
reinterprets every stored dataset.

**The driver's own report is wrong and shouldn't be believed.** `sllidar`
logs `scan mode: Sensitivity, sample rate: 8 Khz, ..., scan frequency:10.0
Hz`, but the motor doesn't turn at 10 Hz — 8000 samples/s over 1080 bins is
7.4 rev/s, matching the measured 7.32 Hz. The 10.0 is the mode's nominal,
not a measurement.

**The valid-return count moves opposite the assumption.** The morning log
reasoned a real A1 delivers roughly half [720] per revolution, "so the
board figure is probably pessimistic." It delivers 855 valid — *more* than
sim, not half. The localizer estimate was optimistic, not pessimistic;
re-measuring is what reversed the sign.

## Bug 1: the IMU published timestamps in the wrong epoch

The navigator held on `imu stale` forever, serving **zero** ticks. Reported
age: `1785869327` seconds — 56 years.

`oak_imu_node` called `getTimestampDevice()`, which counts from the
*device's* boot, and wrote it into `header.stamp` as a ROS epoch time.
Stamps came out around 241 while every consumer compared against 1.79e9.

Nothing failed loudly: clean startup, `/imu` at a perfect 200.4 Hz,
intervals correct to 4 µs, `ros2 topic echo` showing plausible gyro values.
Every property anyone would think to check was right except the epoch.

Fix keeps what the device clock is *for* — intervals free of USB scheduling
jitter — using `getTimestamp()` (device capture time, already translated
into the host's steady clock by depthai's own sync) and rebasing into the
ROS epoch with an offset sampled once per poll. After the fix, stamp-vs-now
lag is 6.5–20.8 ms and the sample interval is still 4.9908 ms, σ = 4 µs.

Could not have been caught by `sim2edge_parity` or `replay_test.launch.py`
— `replay_node` synthesizes its own stamps from `now()`, so the replay path
never executes the wrong line.

## Bug 2: `scan_max_age_s` was unreachable by construction

With the IMU fixed, the loop still held on `scan stale` about one tick in
six, ages 0.257–0.268 s against a 0.25 s threshold.

Two compounding errors in the same comment: it assumed a 182 ms revolution
(the real one is 137 ms), and it assumed a scan's age at arrival is ~0 —
but **`sllidar` stamps the scan at the start of the sweep and publishes it
at the end**, so a scan is already one full revolution old on arrival
(measured stamp-to-arrival lag: 137 ms median, exactly one revolution).
Scan age therefore sweeps 137→274 ms between messages, spending the top
~24 ms of every cycle above 250 ms — the threshold wasn't tight, it was
below the floor.

0.35 s clears the 274 ms normal worst case and still holds on a genuinely
dropped message (3 revolutions = 411 ms). The `trailing` computation in
`navigator_node` (`scan_age - scan_time`) was correct under this stamping
and needed no change — the bug was only in the freshness threshold.

## The localizer costs 2.6× what the replay said, and the reason is density

With both bugs fixed and production `max_submap_scans: 128`, the loop
missed its period on **245 of 1107 ticks (22%)**: median 18.1 ms, p95 68 ms,
max 79 ms. The replay reported 0 of 2,114 at the same configuration.

| | replay | real lidar | ratio |
|---|---:|---:|---:|
| submap points | 40,407 | 54,603 | 1.35× |
| localizer median | 6.90 ms | 18.11 ms | 2.62× |
| **ns per submap point** | 170.8 | 331.6 | **1.94×** |

Point count explains 1.35× of it; the rest is that each point costs nearly
twice as much — the uniform bucket grid has stopped pruning. Sweeping the
submap on real data shows why:

| submap | median | p95 | max | submap points | ns/point |
|---:|---:|---:|---:|---:|---:|
| 128 | 18.11 ms | 65.4 ms | 73.7 ms | 54,603 | 331.6 |
| **64** | **8.54 ms** | 34.3 ms | 38.2 ms | 27,303 | 312.8 |
| 32 | 5.15 ms | 15.1 ms | 17.6 ms | 13,656 | 377.1 |

Cost per point is flat across submap size, so total cost is *linear* in
submap points with no pruning benefit — every correspondence query
effectively scans the whole cloud. The replay's own sweep was sublinear
(128→64 cost 0.66×, not 0.50×), what a working spatial index looks like.

**The mechanism: the submap has no keyframe gate.** `LocalPoseTracker::
addScan` pushes a scan on every served tick unconditionally, dropping the
oldest past `max_submap_scans`. A stationary or slow robot packs 128
near-identical scans into the same 0.20 m grid cells — the cell is sized at
the ICP rejection radius, so a search meant to be exact over a small region
becomes near-brute-force over 128× redundancy describing what 427 points
already describe. This bench robot is stationary with a close scene (median
valid return 0.56 m) — the worst case; driving should cost less. The corpus
couldn't show this: it was oracle-driven and always moving, so consecutive
scans were always displaced.

### What changed, and why 64 rather than 32

`max_submap_scans: 64`. Not an assumption — `eval_localized_submap64_2x_32.json`
and `eval_localized_submap128_2x_32.json` already report the **same success
rate (0.84375)**, false-arrival rate (0.0), and collision rate (0.15625),
with 64 costing 2.6 mm mean / 1.1 cm p95 final belief error. 32 is faster
still but has no evaluation at this configuration, same reason
`point_stride 4` remains unadopted.

## What is now known about the gyro, and what is not

The OAK-D is on the bus and publishing. At rest:

| | value |
|---|---:|
| rate | 200.4 Hz, intervals σ = 4 µs |
| `angular_velocity.z` bias | −0.00285 rad/s (−0.163 °/s) |
| `angular_velocity.z` noise (1σ) | 0.00351 rad/s (0.201 °/s) |
| bias over one 137 ms revolution | −0.022° |
| bias over 60 s | −9.8° |

Negligible per revolution (what deskew needs), large over a minute (what
the scan-match correction exists to absorb).

## Bug 3: the stack was reading the wrong gyro axis

The accelerometer at rest put gravity on **+y** (+9.568 m/s²) — the
vertical axis is y — while the stack consumed `angular_velocity.z`. Turning
the robot a quarter turn left and back settles it:

| burst | dx | dy | dz | |
|---|---:|---:|---:|---|
| 1 | −1.18° | **+34.93°** | +1.25° | left, first part |
| 2 | −2.13° | **+65.62°** | +3.15° | left, second part |
| 3 | +2.94° | **−100.90°** | −3.32° | back to the right |

**y carries 100.9° against z's 3.3° — a factor of 30.** Outbound legs sum
to +100.55°, return is −100.90°, closing to 0.35° over 100° — a real
rotation signal, not drift, and it agrees with the accelerometer.

So `angular_velocity.z` was never the yaw rate on this mount — deskew was
being handed a pitch/roll channel reading ~3% of true rotation. It wouldn't
have failed; it would have quietly under-corrected the sweep by 97% and
presented as a localization failure. Counter-clockwise came out **positive**,
matching the REP-103 convention the stack assumes — only the channel was
wrong.

Fix: `gyro_yaw_axis: "y"` and `gyro_yaw_sign: 1.0` in `navigator.yaml`, read
by `NavigatorNode::yawRateOf`. It's a parameter rather than a constant
because it's a *mount* fact — re-mount the OAK-D and re-measure.
`tools/probe_yaw_axis.py` needs no timing coordination: it segments on
gyro magnitude and integrates each burst, so "turn it left, then back
right, whenever you like" is the whole protocol. Insist on the out-and-back
cancelling — one turn shows which channel moved, only the return leg
separates rotation from drift.

## The Create 3, connected late in the day — two more contract bugs

**`/imu` had two publishers.** The Create 3 runs Iron at the root namespace
and publishes its own `/imu` from `robot_state`; `oak_imu` published there
too, so the navigator's single subscription interleaved two sensors with
different mountings, axis conventions, and clocks. It doesn't fail — it
feeds deskew a blend, applying `gyro_yaw_axis: "y"` (measured for the
OAK-D) to Create 3 messages, where it reads a pitch/roll channel. The
OAK-D now publishes `/oak/imu`.

**The Create 3's clock is 3.73 years behind the board** — no RTC, no sync.
Board 2026-08-04 20:12 UTC against the base's 2022-11-11 06:19 UTC, so every
stamp it writes looks 1.2e8 s old and `/odom` could never pass its
freshness check; with `require_odom: true` the loop held forever.
`odom_stamp_` is now arrival time — costs nothing real, since the base's
clock never enters the arithmetic (the tick reads `twist.linear.x`, a
scalar, and the optional pose prior differences consecutive poses; neither
is a function of time). Weaker than the scan's and IMU's checks, which
still use device stamps — revert if the base is ever synced.

With both fixed, the full stack ran against the real base: 703 ticks, 0
over the period, `odom` age 0.043 s, `/imu` down to one publisher and zero
subscribers. One `scan stale` hold at 0.373 s in ~700 ticks — a genuinely
dropped scan, the 0.35 s threshold doing its job rather than a regression.

### The gadget link, which cost more time than the bugs

`docs/create3_connection.md` in the sibling `ventuno-object-tracking`
project is the authority; reading it earlier would have saved an hour. Two
failure modes that chain: (1) base's multicast arrives while `ping` fails
and `ip neigh` reads `FAILED` — a **wedged gadget TX endpoint**, the link
is one-way, and nothing at the DDS layer fixes it (diagnosing it as a
multicast-delivery problem is the trap fallen into here, since inbound
multicast was the half that worked); (2) re-binding the gadget bounces the
base's USB and silences its ROS app — still pingable, web UI answering, no
SPDP — fixed with `curl -X POST http://192.168.186.2/api/restart-app`,
never the Reboot button, which re-wedges the gadget.

## Also done

`ubuntu` was not in `dialout`, so `/dev/ttyUSB0` (`crw-rw---- root:dialout`)
couldn't be opened by any node. Fixed with `usermod -aG dialout`.

## What is not done

1. **The quantization gate** — top of this list since this morning, still
   untouched.
2. **No Create 3 in the acceptance run.** It used `require_odom:=false`, so
   the motion prior ran without the base's velocity term and the robot
   never moved — every number above is latency and numerics, none of it
   navigation.
3. **The missing keyframe gate.** Reducing the submap treats the symptom.
   A slow robot fills its submap with redundant scans; a gate on
   distance/rotation travelled since the last inserted scan would fix cost
   and degeneracy at once, but changes what the localizer matches against
   and would invalidate existing accuracy evaluations.
4. **`LIDAR_SPIN_HZ` is still 5.5** against a measured 7.32, deliberately —
   inert at runtime, load-bearing in training.

## Order of work

1. The quantization gate on the host — top of the list all day, nothing
   measured here displaces it.
2. The keyframe gate on the submap, with a fresh accuracy evaluation — the
   real fix for localizer cost; `max_submap_scans: 64` is a workaround.
3. Connect the Create 3 and confirm `/odom` and `/cmd_vel`.
4. A stationary trial with the wheels off the ground. The gyro axis is
   settled, so deskew can be checked against a rotation the robot performs
   itself: command a known yaw, integrated `gyro_yaw_axis` should track it.

## A note on what the replay harness can and cannot prove

Three bugs, all in code that passed `sim2edge_parity` and
`replay_test.launch.py`, each invisible for a different reason:

| bug | why the replay could not see it |
|---|---|
| IMU epoch | `replay_node` synthesizes stamps from `now()`, so the wrong line never ran |
| `scan_max_age_s` | the replay publishes on its own schedule, with no sweep-start stamping |
| gyro yaw axis | the corpus stores a scalar yaw rate; there are no axes to get wrong |
| `/imu` collision | there is no second publisher in a replay, only the one the harness starts |
| Create 3 clock | the replay stamps every topic from one machine's clock |

A replay proves the arithmetic and cannot prove the *contract with the
hardware* — every one of these was a disagreement about what a field means,
not about what to compute with it. Worth remembering next time a parity
pass reads like coverage.
