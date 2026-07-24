"""Classical safety layer — the Python twin of ``nav_safety``'s C++ logic.

This runs *outside* the network, on raw lidar ranges, and it is the last thing
between the policy and the motors. It is classical on purpose: a learned policy
that has been quantized to INT8 and is running on an NPU is not something to
trust with the only stopping authority.

Two implementations exist for the same reason preprocessing has two: the sim
must be able to model the intervention behaviour the robot will apply, so that
intervention counts measured in sim mean the same thing as intervention counts
measured on hardware. ``tests/test_safety_parity.py`` checks them against each
other.

Note the asymmetry: the bubble zeroes *linear* velocity but never blocks
rotation. A robot that has stopped 25 cm from a wall must still be able to turn
away from it, and a safety layer that freezes both axes turns a recoverable
situation into a deadlock.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .spec import DEFAULT_SPEC, NavSpec

__all__ = ["SafetyLayer", "forward_min_range"]


def forward_min_range(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    sector_rad: float,
    range_min: float = 0.05,
) -> float:
    """Minimum valid range within +/-``sector_rad`` of straight ahead.

    Returns ``inf`` when the forward sector contains no valid return at all —
    the caller must treat that as sensor loss rather than as clear space.
    """
    r = np.asarray(ranges, dtype=np.float32)
    if r.size == 0:
        return float("inf")
    angles = angle_min + angle_increment * np.arange(r.size, dtype=np.float32)
    wrapped = (angles + np.pi) % (2.0 * np.pi) - np.pi
    sector = np.abs(wrapped) <= sector_rad
    valid = sector & np.isfinite(r) & (r >= range_min)
    if not valid.any():
        return float("inf")
    return float(r[valid].min())


class SafetyLayer:
    """Bubble + clamps + rate limiting, with intervention accounting."""

    def __init__(self, spec: NavSpec = DEFAULT_SPEC):
        self.spec = spec
        self._prev_v = 0.0
        self._prev_w = 0.0
        self.intervention_count = 0

    def reset(self) -> None:
        self._prev_v = 0.0
        self._prev_w = 0.0
        self.intervention_count = 0

    def filter(
        self,
        v: float,
        omega: float,
        ranges: Optional[np.ndarray] = None,
        angle_min: float = -math.pi,
        angle_increment: float = 2.0 * math.pi / 360.0,
        sensor_ok: bool = True,
    ) -> Tuple[float, float, bool]:
        """Return ``(v_safe, omega_safe, intervened)``.

        ``intervened`` is True whenever the layer changed the command by a
        meaningful amount. It is a first-class evaluation metric: a run that
        succeeds only because the watchdog kept rescuing it is a finding, not a
        success, and it must be counted rather than hidden.
        """
        s, a = self.spec.safety, self.spec.action
        v_in, w_in = v, omega

        # Hard clamps to the base's limits come first, so everything downstream
        # reasons about achievable commands.
        v = float(np.clip(v, a.v_min, a.v_max))
        omega = float(np.clip(omega, -a.omega_max, a.omega_max))

        if not sensor_ok:
            # Sensor loss or a stale policy output: stop translating. Rotation
            # is also zeroed here because without sensors there is no basis for
            # choosing a direction to turn.
            v, omega = 0.0, 0.0
        elif ranges is not None:
            d = forward_min_range(ranges, angle_min, angle_increment,
                                  s.forward_sector_rad)
            if not math.isfinite(d):
                v = 0.0  # no valid forward returns: do not drive blind
            elif d < s.stop_distance_m:
                v = 0.0  # rotation deliberately still allowed
            elif d < s.slow_distance_m:
                span = max(1e-6, s.slow_distance_m - s.stop_distance_m)
                v *= (d - s.stop_distance_m) / span

        # Rate limiting last, so the bubble's decision is what gets ramped
        # rather than being ramped away.
        dt = a.dt
        v = float(np.clip(v, self._prev_v - a.accel_max * dt,
                          self._prev_v + a.accel_max * dt))
        omega = float(np.clip(omega, self._prev_w - a.alpha_max * dt,
                              self._prev_w + a.alpha_max * dt))
        self._prev_v, self._prev_w = v, omega

        intervened = (abs(v - v_in) > 1e-3) or (abs(omega - w_in) > 1e-3)
        self.intervention_count += int(intervened)
        return v, omega, intervened
