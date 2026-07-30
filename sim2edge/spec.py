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
           "GOAL_MAX_M", "V_MAX", "OMEGA_MAX", "ACCEL_MAX", "ALPHA_MAX", "DT",
           "ROBOT_RADIUS_M", "BODY_RADIUS_M", "LIDAR_MOUNT_XYZ", "LIDAR_MOUNT_YAW",
           "BIN_WIDTH_RAD", "OBS_DIM", "fingerprint"]

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

# Flat observation layout: [lidar | goal(3) | last_action(2)].
OBS_DIM = LIDAR_BINS + 3 + 2


def fingerprint() -> str:
    """16 hex chars over the values above; stamped into datasets and checkpoints.

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
              LIDAR_MOUNT_XYZ, LIDAR_MOUNT_YAW]
    canon = ";".join(repr(v) for v in values)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]
