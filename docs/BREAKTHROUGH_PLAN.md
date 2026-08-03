# Breakthrough Plan: Repair the Lidar Contract and Rebuild Localization

## Summary

The ceiling is caused by the simulated sensor contract, not proven lidar
observability:

- Isaac Lab's local `ray_alignment="yaw"` path rotates the ray origins but
  leaves ray directions world-fixed.
- The repo labels beam 0 as `0 + mount_yaw`, while the configured Isaac pattern
  begins at `-pi`; bearings are mislabeled by pi.
- Held-out corpus scans match world-fixed analytic raycasts at 0.98-0.99
  correlation and about 1.2 cm median error, while the intended body-frame
  raycasts miss by 0.5-1.05 m.
- `probe_sensor_consistency.py` estimates the complete transform and compares it
  with truth; it does not first apply truth and measure reconstruction residual.
  Untouched instantaneous scans perform even worse under that matcher,
  invalidating the claimed 1.095 cm reconstruction floor.

Repair the sensor frame first, invalidate the affected localization evidence,
then replace per-tick learned slip correction with accumulated scan-to-submap
localization.

## Completion status — 2026-08-03

The plan produced a usable candidate. The sensor contract is repaired, a fresh
body-frame-v2 corpus and actor were built, and the evidence-gated Create3/EKF
prior was selected after the lidar/command-only path failed its localization
gate. On 64 held-out end-to-end layouts per tier, correct declared arrival is
98.44% / 96.88% / 95.31% at 0x / 1x / 2x noise, with no false arrivals or
collisions. Mean/p95 belief error is 0.91/2.32 cm, 4.65/9.00 cm, and
7.03/17.90 cm respectively.

Thus all navigation gates pass; the strict 5/12.5 cm localization-error gate
passes at 0x and 1x and remains open at 2x. Export parity is exact and host
actor latency is 0.191 ms median. Hardware odometry ingestion, target-board
deployment, full-stack latency, and physical validation remain standing work.
See [the final daily log](daily/2026_08_03_breakthrough_log.md) for the complete
evidence and issue list.

## Implementation Changes

### 1. Establish sensor contract v2

- Configure the raycaster with `ray_alignment="base"` so directions follow
  robot yaw; the planar pose controller already keeps the base level.
- Define body-frame beam bearings as
  `-pi + beam_index * resolution + LIDAR_MOUNT_YAW`.
- Add a sensor-contract version to corpus fingerprints and checkpoints. Unknown
  or mismatched versions must fail loudly.
- Add an asymmetric-room bring-up check at several robot yaws. Require
  analytic/raw range correlation >=0.98 and median range disagreement <=2 cm.
- Verify the A1's configured spin rate before freezing the new fingerprint.

### 2. Retire invalid data and diagnostics

- Mark existing corpora as world-fixed-v1. They may remain actor baselines but
  must not train or evaluate localization.
- Permit pose-based circular resampling of v1 scans only for bootstrapping the
  180-bin actor, where 0.5-degree interpolation is below actor resolution.
  Never use converted v1 scans for ego-motion evaluation.
- Recollect a body-frame-v2 localization corpus with live slip and the
  corrected raycaster, preserving the layout-level train/validation split.
- Rewrite the consistency probe as a staged instrument:
  1. untouched instantaneous exact-bearing scans;
  2. reconstructed exact-bearing sweeps;
  3. angular rasterization;
  4. sensor noise.
- In every arm, apply the true SE(2) transform first and search only for the
  remaining residual. Report score at zero, residual bias/scatter, and
  truth-versus-static score margins.
- Do not launch further estimator training until the instantaneous arm passes
  its analytic control.

### 3. Rebuild the navigation stack around canonical scans

- Introduce a canonical `PolarSweep` interface carrying metric ranges, exact
  body-frame bearings, ages, and validity. The localization path must not re-bin
  to 720 uniform angles.
- Retrain the actor on canonical 180-bin scans. Change its 360-degree
  convolutions to circular padding; keep the existing actor only as a legacy
  benchmark.
- Replace the learned per-tick translation residual with a parameter-free local
  pose tracker outside the quantized graph:
  - integrate gyro and commanded motion as the prior;
  - accumulate exact-bearing points into a short-lived local submap;
  - robustly correlate the newest sweep against that submap;
  - update accumulated pose and a slowly varying forward-motion scale estimate;
  - expose covariance/observability and reject ambiguous corrections.
- Continue carrying the two-float relative-goal belief outside the graph. Derive
  it from the local pose tracker, then feed it to the actor.
- Preserve the calibrated systematic drift term in arrival uncertainty. Arrival
  remains confirmed only when the upper confidence bound is inside the radius.
- Keep scan matching and map state on CPU; only learned actor components cross
  the NPU boundary.

### 4. Add an evidence-gated odometry fallback

- First evaluate the repaired lidar/IMU-only tracker.
- Require, on corrected held-out data:
  - mean and p95 final belief error <=5 cm and <=12.5 cm;
  - drift <=2 cm/m and at least 20% below command dead reckoning;
  - shuffled lidar worsens final error by at least 1 cm;
  - no estimator output when observability is rejected.
- If any offline requirement fails, extend the motion-prior interface to accept
  Create3 wheel/EKF delta and covariance. Use it only as the prior; retain lidar
  submap corrections, uncertainty, and the same actor interface.
- Record the selected prior source in checkpoints and evaluation reports so
  lidar-only and odometry-assisted results cannot be conflated.

## Public Interfaces and Compatibility

- Add `sensor_contract_version` and `motion_prior_source` to shards,
  checkpoints, exports, and reports.
- Add `PolarSweep(ranges, bearings, ages, valid)` as the canonical preprocessing
  output.
- Add a stateful localizer interface returning relative pose, covariance,
  motion scale, and observability.
- Old checkpoints default explicitly to the legacy world-fixed contract and
  cannot run silently against v2 observations.
- Existing localization checkpoints are diagnostic artifacts only; actor
  weights may be reused only through an explicit legacy adapter or warm start.

## Test and Acceptance Plan

- Unit-test Isaac pattern directions against stored bearing metadata, including
  the `-pi` origin and mount rotation.
- Integration-test an asymmetric room at 0, 90, and 180 degrees yaw; scans must
  rotate in the body frame while analytic geometry remains aligned.
- Test sweep deskew with exact poses, exact irregular bearings, turns,
  translation, depth edges, reset history, and missing returns.
- Require the rewritten consistency probe to distinguish raw-sensor, sweep,
  rasterization, and noise errors on synthetic rooms with known injected
  defects.
- Test localizer rejection in open space, corridors, symmetric scenes, sparse
  returns, and abrupt occlusion changes.
- Run paired pose-fed actor evaluation first: >=85% success and <=10% collision.
- Run the full existing belief gate at 0x/1x/2x noise: >=85% correct declared
  arrivals, <=5% false arrivals, <=10% collisions, required belief-error bounds,
  and improvement over dead reckoning at every tier.
- Only after passing these gates, repeat export parity, INT8 calibration, CPU
  preprocessing parity, and the 50 ms deployment latency budget.

## Assumptions

- The lidar/IMU-only interface remains the first target; Create3 odometry is an
  evidence-gated fallback.
- No more compute is spent tuning the current estimator or interpreting its
  reconstruction-floor results.
- Corrected localization data is recollected rather than pretending world-fixed
  v1 scans are sufficiently precise for motion estimation.
