"""Constants that more than one module has to agree on. Nothing else.

The rule for this file: a value lives here **iff at least two modules read it**.
Room sampling parameters live in ``rooms.py``, noise ranges in ``noise.py``,
because each has exactly one consumer. Resist growing this file; a constant with
one reader does not need a contract.

**Stdlib only.** It is imported from the Isaac Lab CUDA stack, from a plain
numpy process, and (later) transcribed into a C++ header for the board. A numpy
import here would break one of those.

Why it exists at all: ``INVALID_FILL`` must be identical in the simulator that
produces absent returns and in the loader that fills dropped beams. Those are
different files. When they disagreed in the previous build, BC validation loss
fell fourfold at a 0% success rate -- the failure is silent, which is precisely
why the value is centralized.
"""

from __future__ import annotations

import hashlib
import math

__all__ = ["LIDAR_BINS", "LIDAR_MIN_M", "LIDAR_MAX_M", "INVALID_FILL",
           "GOAL_MAX_M", "GOAL_TOLERANCE_M", "V_MAX", "OMEGA_MAX", "ACCEL_MAX", "ALPHA_MAX", "DT",
           "ROBOT_RADIUS_M", "BODY_RADIUS_M", "LIDAR_MOUNT_XYZ", "LIDAR_MOUNT_YAW",
           "LIDAR_OCCLUSION_ARCS_RAD",
           "BIN_WIDTH_RAD", "HISTORY_FRAMES", "OBS_DIM", "fingerprint",
           "observation_fingerprint",
           "LIDAR_SPIN_HZ", "SPIN_TICKS", "DESKEW_WITH_GYRO", "GYRO_CHANNELS",
           "MOTION_HISTORY_FRAMES",
           "SENSOR_CONTRACT_VERSION", "MOTION_PRIOR_SOURCE",
           "sensor_model_fingerprint", "legacy_observation_fingerprint"]

# ── Lidar ────────────────────────────────────────────────────────────────────
# 180 bins over the full 360 degrees. Coarse on purpose: the policy needs
# geometry, not surface detail, and every bin is an INT8 activation on the NPU.
# Full circle rather than a forward arc because backing out of a dead end
# requires seeing behind -- forward-only makes recovery unlearnable, not just
# harder.
LIDAR_BINS = 180
LIDAR_MIN_M = 0.15          # RPLIDAR A1 usable range; outside it is "invalid"
LIDAR_MAX_M = 12.0

# What an absent return normalizes to. 1.0 == "nothing out to max range".
# NOT 0.0 -- zero denormalizes to LIDAR_MIN_M, i.e. an obstacle 15 cm dead ahead
# filling every bin, which trains a policy that refuses to move.
INVALID_FILL = 1.0

BIN_WIDTH_RAD = 2.0 * math.pi / LIDAR_BINS

# ── Goal ─────────────────────────────────────────────────────────────────────
# Goal is (distance / GOAL_MAX_M, sin bearing, cos bearing) in the robot frame.
# sin/cos rather than a raw angle: an angle is discontinuous at +/-pi, exactly
# where the robot needs to turn around, and quantizing a wrapping input to INT8
# fails silently.
GOAL_MAX_M = 8.0
# Success is reaching this radius around the goal. It changes episode outcome,
# not observation construction, so it is deliberately outside the fingerprint.
GOAL_TOLERANCE_M = 0.25

# ── Action: physical (v, omega), the units the Twist carries ─────────────────
# Create3's real limits. An oracle allowed to exceed them produces a policy that
# faithfully imitates commands the base cannot track, and the resulting error
# looks like perception while actually being dynamics.
V_MAX = 0.46                # m/s
OMEGA_MAX = 1.90            # rad/s
ACCEL_MAX = 0.90            # m/s^2
ALPHA_MAX = 3.00            # rad/s^2
DT = 0.05                   # 20 Hz control period

# ── Robot ────────────────────────────────────────────────────────────────────
# Create3 body radius is 0.164 m (irobot_create_description: body_radius =
# 16.4 cm). Planning inflates by a little more than that so the bumper is not
# the thing that discovers the wall.
ROBOT_RADIUS_M = 0.18

# The *actual* body radius, for deciding whether a pose is in collision.
#
# These are different questions and conflating them was a real bug: episodes
# were labelled collided whenever clearance fell below the 0.18 planning margin,
# even though nothing was touching, and the flag is sticky for a whole episode.
# It reported false collisions on 16 mm of deliberate safety margin.
#
# Not in the fingerprint: this changes which episodes are *selected*, not how an
# observation is *constructed* -- the same reason noise ranges are excluded.
BODY_RADIUS_M = 0.164

# RPLIDAR mount on the TB4 Lite, in base_link. From
# turtlebot4_description/urdf/lite/turtlebot4.urdf.xacro, with the create3
# base_link_z_offset of 6.42 cm folded in.
LIDAR_MOUNT_XYZ = (0.0039, 0.0, 0.1395)

# The sensor is bolted on rotated a quarter turn. Beam bearings therefore arrive
# 90 degrees off the robot's forward axis, and every consumer must rotate them
# into base_link before binning -- the goal bearing is in the robot frame, so a
# scan left in the lidar frame would put obstacles at right angles to where the
# policy believes they are. This is in the fingerprint precisely because it
# changes how the observation is constructed.
LIDAR_MOUNT_YAW = math.pi / 2

# Bearings the robot's own camera mount permanently blocks, in base_link, as
# (start, end) pairs swept counter-clockwise. An arc may run past +pi; the
# consumer wraps.
#
# MEASURED, 2026-08-04, over 500 revolutions: 17.93% of beams returned nothing,
# which reads as a 6x miss against DROPOUT_PROB = 0.03 until it is decomposed.
#
#   persistent -- the same bearings blocked in every single scan   14.51%
#   everything else                                                 3.43%
#
# The random component was right all along. The 14.51% is three contiguous arcs
# totalling 49.3 degrees, and the give-away that they are structure rather than
# scene is that P(invalid | neighbour invalid) is 0.91 against a 0.18 base rate:
# real dropout does not do that. Three arcs, three standoffs.
#
# 97% of it lies behind the robot -- the forward +/-35 degree sector the safety
# node watches loses 3 bins of 210 -- which is the benign place for it, but the
# simulator had no occlusion at all, so the policy has been trained believing it
# has 360 degrees of coverage against the robot's ~308.
#
# This is a MOUNT property, like LIDAR_MOUNT_YAW above. Move the camera and it
# is wrong, and wrong here is silent: it presents as a localizer that
# mysteriously has fewer points on one side. Re-derive with
# tools/capture_scans.py and tools/analyze_dropout.py rather than editing by
# hand.
#
# It lives in sensor_model_fingerprint() and NOT fingerprint(), because it is
# applied when an observation is built and not when a scan is collected -- the
# shards still hold clean instantaneous ranges. Putting it in fingerprint()
# would invalidate 51 GB of corpus for a mask the collector never saw.
LIDAR_OCCLUSION_ARCS_RAD = (
    (-2.96253, -2.72378),   # -169.7 to -156.1 deg, 42 of 1080 bins
    (-2.38604, -2.16476),   # -136.7 to -124.0 deg, 39 bins
    (+2.84898, +3.23331),   # +163.2 to +185.3 deg, 67 bins -- crosses +pi
)

# Raw range/angle semantics.  This is deliberately human-readable in every
# artifact as well as folded into the hash: a hash mismatch says that something
# changed, while this field says whether a scan is safe to use for localization.
SENSOR_CONTRACT_VERSION = "body-frame-v2"

# Default prior used by the repaired localization stack.  An odometry-assisted
# fallback must override and stamp this explicitly so its evidence cannot be
# reported as lidar/IMU-only.
MOTION_PRIOR_SOURCE = "command-gyro"

# Four frames are the smallest planned temporal representation: at 20 Hz they
# expose 150 ms of motion. The original build chose this to avoid recurrent
# state in the deployed graph; the belief policy reintroduces recurrence
# deliberately and in a bounded, semantically-pinned form -- see
# ``sim2edge.models.belief``. Four frames survives because it is also the
# window the odometry head integrates translation over.
HISTORY_FRAMES = 4

# Flat observation layout, oldest to newest within each history block:
# [lidar(H, B) | current_goal(3) | previous_action(H, 2)].
OBS_DIM = HISTORY_FRAMES * LIDAR_BINS + 3 + HISTORY_FRAMES * 2

# ── Spinning lidar ───────────────────────────────────────────────────────────
# The A1 is a *mechanically spinning* 2-D lidar, and the first build modelled it
# as an instantaneous 720-beam snapshot arriving every control tick. Neither
# half of that is true, and both halves matter:
#
#   1. One revolution takes 1/LIDAR_SPIN_HZ. At the measured 7.32 Hz that is
#      137 ms, which is 2.7 control periods -- so a "scan" is stitched from
#      beams acquired at three different robot poses. The result is motion skew, and the
#      skew is a function of the command just issued. It is worst exactly when
#      the robot turns hard to avoid something.
#   2. A complete fresh revolution is therefore *not* available every tick.
#      Beam age varies continuously with bearing, from zero at the beam the
#      sweep just passed to a full revolution at the beam in front of it.
#
# MEASURED on the physical unit, 2026-08-04: 7.32 Hz over 80 revolutions, with
# the driver's own time_increment implying 7.63. It is NOT the 5.5 Hz nominal
# this constant carried until then, and it is not the "scan frequency: 10.0 Hz"
# the sllidar driver prints for Sensitivity mode either -- that figure is the
# mode's nominal and the motor does not turn at it. 8000 samples/s over the
# 1080 bins the driver emits is 7.4 rev/s, which is what arrives.
#
# Re-measure with tools/probe_scan.py if the scan mode or the motor PWM changes.
LIDAR_SPIN_HZ = 7.32

# How many control ticks of raw scan history one revolution spans, plus the
# partial tick at each end. Anything reconstructing a swept scan needs this many
# stored frames to have a measurement for every bearing.
SPIN_TICKS = int(math.ceil(1.0 / (LIDAR_SPIN_HZ * DT))) + 1

# Whether the gyro is used to rotationally deskew the swept scan before binning.
# This is fixed, non-learned sensor preprocessing in the same category as
# ``bin_lidar`` -- it does not make the policy less end-to-end, it just declines
# to spend network capacity relearning a rotation matrix that is exactly known.
DESKEW_WITH_GYRO = True

# The belief policy's **actor observation is byte-identical to OBS_DIM**, and
# that is a deliberate constraint rather than a coincidence.
#
# The obvious design feeds the gyro to the actor alongside the lidar. Doing so
# widens the trunk's input and orphans every trunk weight in
# ``runs/ppo_v1/best.pt`` -- 300 PPO iterations and the whole BC corpus behind
# them. Routing the gyro to the odometry head *only* costs the actor nothing it
# cannot already infer (it reads commanded omega through the previous-action
# block) and buys a complete warm start: the actor transfers whole, and the
# single randomly-initialized module in the new model is the odometry head.
#
# What changes for the actor is the *provenance* of its three goal features --
# a belief the model maintains, rather than a subtraction against ground-truth
# pose -- not their layout. Same three numbers, same meaning, same encoder.
GYRO_CHANNELS = HISTORY_FRAMES

# The actor stays on four frames for checkpoint compatibility. Ego-motion is a
# much smaller signal and needs a longer baseline: one second makes centimetres
# of accumulated slip visible above per-beam lidar noise without introducing
# opaque recurrent state.
MOTION_HISTORY_FRAMES = 20


def fingerprint() -> str:
    """16 hex chars over the raw sensor contract; stamped into datasets.

    A dataset collected at 180 bins and loaded by a policy expecting 120 has
    compatible-looking shapes right up until it doesn't. Consumers compare this
    on load and refuse rather than silently misinterpreting.
    """
    # Listed explicitly rather than scraped from globals(): the set of values
    # under contract is a decision, not whatever happens to be uppercase.
    # Derived values (BIN_WIDTH_RAD, OBS_DIM) are omitted -- they are functions
    # of the others and would only add redundant entropy.
    values = [LIDAR_BINS, LIDAR_MIN_M, LIDAR_MAX_M, INVALID_FILL, GOAL_MAX_M,
              V_MAX, OMEGA_MAX, ACCEL_MAX, ALPHA_MAX, DT, ROBOT_RADIUS_M,
              LIDAR_MOUNT_XYZ, LIDAR_MOUNT_YAW, SENSOR_CONTRACT_VERSION]
    canon = ";".join(repr(v) for v in values)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def sensor_model_fingerprint() -> str:
    """The parts of the sensor model applied at *load* time, not collection time.

    Spin rate and gyro deskew change how an observation is *built* from a stored
    scan, but not what the collector had to write down: shards still hold clean
    instantaneous ranges and true poses, and the sweep is reconstructed from
    them. That is the same durability argument that keeps noise ranges out of
    :func:`fingerprint`, and it is why 104 GB of ``overnight_v2`` survived this
    change intact.
    """
    values = [LIDAR_SPIN_HZ, SPIN_TICKS, DESKEW_WITH_GYRO,
              "subtick-canonical-v3", MOTION_HISTORY_FRAMES,
              LIDAR_OCCLUSION_ARCS_RAD]
    return ";".join(repr(v) for v in values)


def legacy_observation_fingerprint(history_frames: int = HISTORY_FRAMES) -> str:
    """The observation contract as it stood before the spinning-sensor model.

    Kept so that checkpoints trained against instantaneous scans and white-noise
    odometry can be *recognized* rather than merely rejected. Those checkpoints
    are not interchangeable with current ones and must never load by default --
    but running one against the corrected sensor model is the measurement that
    says how much the correction was worth, and refusing to load it at all would
    make that number unobtainable.
    """
    canon = f"{fingerprint()};history-v1;{history_frames}"
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def observation_fingerprint(history_frames: int = HISTORY_FRAMES,
                            belief: bool = False) -> str:
    """Fingerprint the loader/model layout without invalidating raw shards.

    History is assembled from sequential samples at load time, so changing its
    length requires a new model checkpoint but not an expensive recollection.
    The same now goes for the spinning-sensor reconstruction and the gyro
    channel: both are load-time, both change what the network must expect, and
    neither touches a stored byte.

    ``belief`` selects the goal-belief observation layout, which carries the
    extra gyro channels. A checkpoint trained against a ground-truth-pose goal
    and one trained against a believed goal are not interchangeable even at
    identical shapes -- the second maintains state the first does not have -- so
    they must not share a fingerprint.
    """
    # v2 adds a 720-bin metric scan/mask stream and uncertainty-carrying runtime
    # state. Old belief checkpoints must not silently load into that contract.
    layout = "belief-v2" if belief else "history-v1"
    canon = (f"{fingerprint()};{layout};{history_frames};"
             f"{sensor_model_fingerprint()}")
    return hashlib.sha256(canon.encode()).hexdigest()[:16]
