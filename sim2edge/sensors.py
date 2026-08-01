"""The spinning lidar and the rate gyro, modelled as the hardware behaves.

Both live here for the same reason ``preprocess.py`` exists: this code runs in
two places that must agree exactly -- on a GPU batch during training, and in C++
on the board -- and the agreement is the whole point. Everything is explicit
index arithmetic on batched tensors, with no operator whose semantics differ
between torch and the C++ standard library.

**Why this file had to exist.** The first build modelled the A1 as an
instantaneous 720-beam snapshot delivered every control tick. A real A1 is a
motor spinning at about 5.5 Hz, so:

* one revolution takes 182 ms, which is 3.6 control periods at 20 Hz;
* the beams in one "scan" were therefore acquired at three or four different
  robot poses, and the scan is smeared -- *motion skew*;
* the smear is a function of the command just issued, so it is largest exactly
  when the robot turns hard to avoid something;
* and a complete fresh revolution is simply not available every tick. Beam age
  varies continuously with bearing.

A policy trained on instantaneous scans meets skewed ones on hardware, and the
mismatch is correlated with the action. That is a systematic sim2real gap, not
noise, and it cannot be averaged away. It matters most for the goal-belief
policy, whose translation head infers ego-motion from exactly the frame-to-frame
change that skew corrupts.

**The split this file maintains.** :func:`sweep_scan` produces *what the driver
hands you*: ranges tagged with nominal body bearings and a per-beam age, with no
knowledge of where the robot actually was. :func:`deskew` is what you *do about
it*, using the gyro and the commanded velocity -- the same two things the robot
has. Keeping those apart is what stops the simulator from quietly solving a
problem the hardware still has.

Deskew is fixed, non-learned preprocessing in the same category as
:func:`~sim2edge.preprocess.bin_lidar`. It does not make the policy less
end-to-end; it declines to spend network capacity relearning a rotation matrix
that is exactly known.
"""

from __future__ import annotations

import math

import torch

from .noise import (GYRO_ARW_RAD_PER_SQRT_S, GYRO_BIAS_RAD_S,
                    GYRO_BIAS_WALK_RAD_S, GYRO_SCALE_ERROR)
from .preprocess import wrap_angle
from .spec import DT, LIDAR_SPIN_HZ, SPIN_TICKS

__all__ = ["SPIN_PERIOD_S", "beam_ages", "advance_phase", "sweep_scan",
           "deskew", "GyroModel", "synthesize_gyro"]

#: Seconds per revolution. The quantity every function here is really about.
SPIN_PERIOD_S = 1.0 / LIDAR_SPIN_HZ


def beam_ages(phase: torch.Tensor, beams: int) -> torch.Tensor:
    """``(N,)`` sweep phase -> ``(N, beams)`` seconds since each beam was swept.

    ``phase`` is the fraction of a revolution completed at the current control
    tick, in ``[0, 1)``. Beam ``b`` sits at sweep position ``b / beams``; the
    sweep last passed it ``((phase - b/beams) mod 1)`` revolutions ago.

    So the freshest beam is the one the sweep just crossed and the stalest is
    the one immediately ahead of it -- a full revolution old. That gradient
    around the circle is the thing the old instantaneous model was missing.
    """
    position = torch.arange(beams, device=phase.device,
                            dtype=phase.dtype) / beams
    revolutions = (phase[:, None] - position[None, :]) % 1.0
    return revolutions * SPIN_PERIOD_S


def advance_phase(phase: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """Move the sweep on by one control period.

    At 5.5 Hz and 20 Hz control this is 0.275 of a revolution per tick, which is
    why no two consecutive observations share a freshest bearing.
    """
    return (phase + LIDAR_SPIN_HZ * dt) % 1.0


def sweep_scan(ranges_history: torch.Tensor,
               phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stitch one revolution out of stored instantaneous scans.

    ``ranges_history`` is ``(N, K, B)``, oldest to newest, where ``K`` is at
    least :data:`~sim2edge.spec.SPIN_TICKS` -- enough control ticks to cover a
    full revolution. Returns ``(ranges, ages)``, both ``(N, B)``.

    Each beam takes its range from whichever stored tick is nearest its own
    acquisition time. That is not an approximation of the measurement: the raw
    return really was produced at that pose, and presenting it at the nominal
    body bearing is precisely the error the hardware makes. The skew is
    *implicit* in the result rather than synthesized on top of it, which is why
    there is no reprojection step here and no appeal to ground-truth pose.

    Quantizing acquisition time to the nearest stored tick costs at most half a
    control period of age, or 25 ms -- against beam ages that span 182 ms. The
    alternative is re-raycasting per beam sector, which costs a full extra
    sensor update per sub-tick for an error already well under the range noise.
    """
    n, k, beams = ranges_history.shape
    if k < SPIN_TICKS:
        raise ValueError(
            f"sweeping a {SPIN_PERIOD_S * 1e3:.0f} ms revolution needs "
            f"{SPIN_TICKS} ticks of scan history, got {k}")

    ages = beam_ages(phase, beams)                       # (N, B) seconds
    # Ticks back from the newest frame, rounded to the nearest stored scan.
    back = torch.round(ages / DT).long().clamp_(0, k - 1)
    frame = (k - 1) - back                               # (N, B) into history
    # Advanced indexing rather than gather: each (env, beam) pair selects its
    # own frame, so the index is genuinely two-dimensional and gather would need
    # a full expand over the beam axis to say the same thing.
    env = torch.arange(n, device=ranges_history.device)[:, None]
    beam = torch.arange(beams, device=ranges_history.device)[None, :]
    return ranges_history[env, frame, beam], ages


def deskew(ranges: torch.Tensor, angles: torch.Tensor, ages: torch.Tensor,
           yaw_rate: torch.Tensor,
           forward_speed: torch.Tensor | None = None
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Undo motion skew using the gyro, and optionally the commanded speed.

    ``angles`` is ``(B,)`` or ``(N, B)`` nominal body bearings, ``ages`` is
    ``(N, B)`` seconds, ``yaw_rate`` is ``(N,)`` rad/s **as the gyro measured
    it**, and ``forward_speed`` is ``(N,)`` m/s as commanded. Returns corrected
    ``(ranges, angles)``, ready for :func:`~sim2edge.preprocess.bin_lidar`.

    A feature seen at body bearing ``phi`` when the robot had yaw ``theta_acq``
    sits at ``phi - (theta_now - theta_acq)`` now. With a constant-rate
    assumption over the beam's age that correction is ``-yaw_rate * age``, which
    is exact to the extent the rate was constant over 182 ms and the gyro was
    right about it. Both assumptions fail gracefully: the residual is small and
    unbiased, where the uncorrected error is large and action-correlated.

    Translation is corrected only if ``forward_speed`` is given, and only from
    the *commanded* speed, because that is all the robot knows without the wheel
    odometry this design is removing. It is the weaker of the two corrections --
    at 0.46 m/s and a 182 ms age the displacement is 8 cm, against a 2 cm range
    noise -- but it is worth making because it is free and it removes a bias
    rather than a variance.
    """
    if angles.dim() == 1:
        angles = angles.unsqueeze(0).expand_as(ages)
    corrected = angles - yaw_rate[:, None] * ages

    if forward_speed is None:
        return ranges, wrap_angle(corrected)

    # Re-express each return as a point in the *current* body frame. The origin
    # moved forward by speed * age along the body x axis of the acquiring frame,
    # which after the rotation correction above is the current x axis.
    finite = torch.isfinite(ranges)
    safe = torch.where(finite, ranges, torch.zeros_like(ranges))
    x = safe * torch.cos(corrected) - forward_speed[:, None] * ages
    y = safe * torch.sin(corrected)
    moved = torch.sqrt(x * x + y * y)
    return (torch.where(finite, moved, ranges),
            torch.where(finite, torch.atan2(y, x), wrap_angle(corrected)))


class GyroModel:
    """Per-environment rate-gyro error state, stepped one control tick at a time.

    Three terms, because they dominate at different timescales and only the
    first of them is the one people remember:

    * white rate noise (angle random walk) -- negligible over an episode, and
      modelled anyway so that its negligibility is demonstrated;
    * a residual bias, constant within an episode after power-on calibration,
      which is the dominant term over ten seconds;
    * a scale-factor error, also per-episode, whose contribution is proportional
      to how much the robot turned rather than to how long it ran.

    The bias itself wanders (rate random walk), which is why it is state here
    rather than a constant drawn at reset.
    """

    def __init__(self, batch_size: int, *, scale: float = 1.0, device=None,
                 generator: torch.Generator | None = None):
        self.scale = float(scale)
        self.generator = generator
        self.device = device
        self.bias = torch.zeros(batch_size, device=device)
        self.scale_error = torch.zeros(batch_size, device=device)
        self.reset(torch.arange(batch_size, device=device))

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()
        kw = {"generator": self.generator, "device": self.device}
        self.bias[env_ids] = torch.randn((count,), **kw) * (
            GYRO_BIAS_RAD_S * self.scale)
        self.scale_error[env_ids] = torch.randn((count,), **kw) * (
            GYRO_SCALE_ERROR * self.scale)

    def step(self, true_yaw_rate: torch.Tensor,
             dt: float = DT) -> torch.Tensor:
        """``(N,)`` true rad/s -> ``(N,)`` measured rad/s, advancing bias state."""
        n = true_yaw_rate.shape[0]
        kw = {"generator": self.generator, "device": self.device}
        self.bias += torch.randn((n,), **kw) * (
            GYRO_BIAS_WALK_RAD_S * self.scale * math.sqrt(dt))
        white = torch.randn((n,), **kw) * (
            GYRO_ARW_RAD_PER_SQRT_S * self.scale / math.sqrt(dt))
        return true_yaw_rate * (1.0 + self.scale_error) + self.bias + white


def synthesize_gyro(yaw: torch.Tensor, episode: torch.Tensor,
                    scale: float = 1.0,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """``(T,)`` true yaw + episode ids -> ``(T,)`` measured yaw *increment*.

    The offline counterpart of :class:`GyroModel`, for the stored corpus. Shards
    hold true poses at every control tick, so the gyro never had to be collected
    -- it is derivable, and derivable at load time means the noise model stays a
    training knob instead of becoming a dataset property. This is the same
    argument that keeps lidar corruption out of the simulator.

    The returned quantity is the yaw change *over the preceding tick*, which is
    what the belief integrator composes and what the policy reads. The first
    sample of each episode gets a zero increment: there is no preceding tick,
    and fabricating one leaks the previous episode's heading.
    """
    from .noise import _episode_starts, _segmented_cumsum

    steps = yaw.shape[0]
    device, dtype = yaw.device, torch.float32
    kw = {"generator": generator, "device": device, "dtype": dtype}

    first = _episode_starts(episode)
    true_increment = torch.zeros(steps, device=device, dtype=dtype)
    true_increment[1:] = wrap_angle(yaw[1:] - yaw[:-1])
    true_increment[first] = 0.0

    ordinal = torch.cumsum(first.long(), dim=0) - 1
    count = int(ordinal[-1].item()) + 1 if steps else 0
    bias0 = torch.randn((count,), **kw) * (GYRO_BIAS_RAD_S * scale)
    scale_error = torch.randn((count,), **kw) * (GYRO_SCALE_ERROR * scale)

    walk = torch.randn((steps, 1), **kw) * (
        GYRO_BIAS_WALK_RAD_S * scale * math.sqrt(DT))
    bias = bias0[ordinal] + _segmented_cumsum(walk, episode).squeeze(1)
    white = torch.randn((steps,), **kw) * (
        GYRO_ARW_RAD_PER_SQRT_S * scale * math.sqrt(DT))

    measured = (true_increment * (1.0 + scale_error[ordinal])
                + bias * DT + white)
    measured[first] = 0.0
    return measured
