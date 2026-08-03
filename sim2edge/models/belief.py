"""Goal-belief navigation policy: lidar and gyro in, no odometry input.

The problem this solves
=======================
The deployed policy is handed a relative goal ``(dx, dy)`` once, at the start of
an episode, and must drive there. The goal is not a physical object -- nothing in
a lidar scan says "the goal is here" -- so knowing when it has *arrived* requires
knowing how far it has moved. Removing the odometry input therefore cannot mean
removing ego-motion estimation. It can only mean moving the estimator inside the
model.

That reframing is the whole design. Everything below is about *where the
integrator lives and what feeds it*, and the answer here is: the integrator is
exact rigid-body composition, the rotation comes from the gyro, and the only
learned part is a correction to the forward translation.

Why the state is the goal vector
================================
Any policy that maintains a goal belief across an episode is recurrent; that is
forced by the task, not chosen. What is chosen is how much structure the
recurrence carries. A plain GRU (option C in the 2026-07-31 log) learns an opaque
hidden state and an arbitrary update rule. This model pins the state to *be* the
goal offset in the current body frame, and pins the update to be the composition
that offset provably obeys:

    g_{t+1} = R(-dtheta_t) . (g_t - dp_t)

Three consequences, all of which mattered more than the flexibility given up:

* the state is two interpretable metres, so it can be supervised directly from
  the true poses already in every shard, monitored on the robot, and reset;
* it quantizes. An opaque recurrent hidden state accumulates INT8 error through
  the recurrence, which is the specific reason the original build ruled out
  recurrence for the deployed graph. Two float32 metres carried outside the
  quantized graph do not have that problem;
* no network capacity is spent relearning a rotation matrix that is exactly
  known.

Why the translation head predicts a *residual*
==============================================
``dp_t`` is not predicted from scratch. It is the commanded displacement plus a
learned correction:

    dp_t = (v_cmd * DT, 0) + correction(lidar history, gyro, previous actions)

The commanded displacement is already a good estimate -- it is exactly what pure
command dead-reckoning (option A in the log) uses, and that option is a
respectable baseline. Making the network predict the residual means:

* **option A is the zero-weights special case of this model.** The baseline is
  nested inside it, so training starts at baseline quality rather than below it,
  and cannot do worse than it by accident;
* **degradation is graceful where it has to be.** In an open area where every
  return is beyond max range, lidar ego-motion is genuinely unobservable -- no
  architecture fixes that. A residual model falls back to dead reckoning there
  on its own, because the correction has nothing to key on and stays near zero;
* the correction head's output is small and zero-centred, which is a far easier
  regression target than an absolute displacement dominated by its own mean.

The lateral output is scaled down hard relative to the forward one. A
differential-drive base cannot translate sideways: lateral motion is wheel slip,
a residual on a residual. Encoding that as a scale rather than a hard zero keeps
slip representable on carpet without inviting the head to explain range noise
with imaginary sideways motion.

Why rotation is taken from the gyro unmodified
==============================================
The gyro's integrated heading error is about 0.45 degrees over a ten-second
episode (1.07 at p95), which is 2.3 cm of lateral error at 3 m of remaining
travel against a 0.25 m goal tolerance. Wheel-odometry yaw, which this design
removes, is worth several degrees. Rotation is simply not the binding constraint
once an IMU is present, and a learned rotation correction would be fitting noise.
This is also the reason the odometry head only has to solve the *easy* half of
ego-motion: lidar is poor at rotation at 2-degree bin width and good at
translation, and the gyro is the other way round.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..preprocess import build_observation, relative_goal_features
from ..spec import (DT, GOAL_MAX_M, GOAL_TOLERANCE_M, HISTORY_FRAMES,
                    LIDAR_BINS, LIDAR_MAX_M, MOTION_HISTORY_FRAMES, V_MAX)
from .nav_policy import NavPolicy, NavPolicyConfig, initialize_layer

__all__ = ["BeliefNavPolicy", "BeliefNavPolicyConfig", "OdometryHead",
           "ScanVote", "compose_belief"]


def compose_belief(belief: torch.Tensor, translation: torch.Tensor,
                   delta_yaw: torch.Tensor) -> torch.Tensor:
    """``g_{t+1} = R(-dtheta) (g_t - dp)``. Exact, and the reason this works.

    ``belief`` and ``translation`` are ``(N, 2)`` in the body frame at time
    ``t``; ``delta_yaw`` is ``(N,)`` radians turned over the tick. The result is
    the goal offset in the body frame at ``t+1``.

    Derivation, because a sign error here is invisible until the robot circles
    the goal: with world goal ``G`` and pose ``(p_t, theta_t)``, the belief is by
    definition ``g_t = R(-theta_t)(G - p_t)``. Substituting
    ``dp = R(-theta_t)(p_{t+1} - p_t)`` and ``dtheta = theta_{t+1} - theta_t``
    into ``g_{t+1} = R(-theta_{t+1})(G - p_{t+1})`` gives the expression above
    with no approximation.
    """
    shifted = belief - translation
    cos, sin = torch.cos(delta_yaw), torch.sin(delta_yaw)
    return torch.stack([cos * shifted[:, 0] + sin * shifted[:, 1],
                        -sin * shifted[:, 0] + cos * shifted[:, 1]], dim=1)


@dataclass(frozen=True)
class BeliefNavPolicyConfig:
    """Actor architecture plus the odometry head's own shape.

    ``actor`` is an ordinary :class:`NavPolicyConfig` and is expected to match a
    pretrained checkpoint exactly -- the warm start described in ``spec.py``
    depends on it.
    """

    actor: NavPolicyConfig = field(default_factory=NavPolicyConfig)
    odometry_channels: tuple[int, ...] = (32, 64)
    odometry_features: int = 96
    odometry_trunk: tuple[int, ...] = (128,)
    raw_lidar_bins: int = 720
    motion_history_frames: int = MOTION_HISTORY_FRAMES
    #: Ceiling on the learned forward correction, as a multiple of the distance
    #: a full-speed tick covers. Slip removes speed far more often than it adds
    #: it, but the bound is symmetric because a systematically low commanded
    #: speed is equally possible and equally correctable.
    forward_correction_scale: float = 0.5
    #: Lateral scale, relative to the forward one. Differential drive cannot
    #: translate sideways; this is the wheel-slip residual only.
    lateral_correction_ratio: float = 0.15
    arrival_sigma: float = 1.96
    #: Relative uncertainty in the forward scale, as a fraction of distance
    #: travelled. This is the *systematic* half of the error budget and the
    #: variance head cannot supply it: that head is calibrated against per-step
    #: squared error, which propagates as a random walk, while wheel slip is a
    #: scale error that accumulates linearly in distance.
    #:
    #: Calibrated by measurement, not by the drift figure: on
    #: ``belief_onpolicy_v15`` the v14 checkpoint drifts 3.56 cm/m, and the
    #: setting that makes the 1.96-sigma bound cover its nominal 95% is 0.05
    #: (measured 94.2%, against 78.3% at 0.03 and 9.5% with no term at all).
    #:
    #: Be clear about what an honest bound costs. At 0.05 the bound widens by
    #: 1.96 * 0.05 * s, which is 49 cm after five metres -- far outside
    #: ``arrival_radius_m``. A calibrated robot therefore *declines to declare
    #: arrival* on a long episode rather than declaring it wrongly. That is the
    #: correct behaviour and it converts false arrivals into timeouts, but it
    #: does not manufacture accuracy: reaching the 85% correct-arrival gate
    #: needs an estimator that actually drifts less, not a wider bound.
    #:
    #: Zero reproduces every existing checkpoint exactly, and is why it is the
    #: default.
    drift_scale_uncertainty: float = 0.0
    #: Belief-space stop radius. Deliberately tighter than the simulator's
    #: 0.25 m truth tolerance so localization error has a safety margin rather
    #: than turning every optimistic estimate into a false arrival.
    arrival_radius_m: float = 0.15
    arrival_confirm_ticks: int = 3
    arrival_speed_m_s: float = 0.08
    #: Carry a bounded forward wheel-scale correction across the episode. Old
    #: checkpoints leave this false and retain their exact stateless behavior;
    #: new estimator runs opt in after training through the recurrence.
    persistent_calibration: bool = False
    calibration_update_rate: float = 0.2
    calibration_min: float = -0.5
    calibration_max: float = 0.25
    #: Optional learned episode memory for motion calibration. Zero preserves
    #: the compact legacy/scalar state; positive values create a GRU whose
    #: explicit hidden state is carried by the runtime alongside the belief.
    motion_state_size: int = 0
    #: Rotate the scan history into the newest frame's bearing grid with the
    #: measured gyro before encoding it. Off by default so every existing
    #: checkpoint keeps its exact numerics; see :meth:`OdometryHead._align_history`
    #: for why a new run wants it on.
    derotate_scan_history: bool = False
    #: Mask the cross-frame range difference to bearings valid in both frames and
    #: scale it by the translation a window can contain rather than by max range.
    #: Off by default for exact checkpoint compatibility.
    scaled_scan_difference: bool = False
    #: Initial bias of the learned confidence gate. The correction is the product
    #: ``confidence * proposal``, so the gate's gradient is proportional to a
    #: proposal that starts near zero: at -2.0 it never leaves sigmoid(-2)=0.12
    #: and the proposal grows ~8x to compensate, leaving the gate a constant
    #: rather than a gate. 0.0 starts it mid-range where it has usable dynamic
    #: range in both directions. The default reproduces existing checkpoints.
    confidence_gate_bias: float = -2.0
    #: Side of the square candidate grid the scan vote scores, or 0 to disable
    #: the branch entirely and keep the pooled-sector encoder alone. See
    #: :class:`ScanVote` for why voting replaces the least-squares family, and
    #: :meth:`OdometryHead._vote_features` for what the head does with it.
    scan_vote_bins: int = 0
    #: Candidate extent as a multiple of a full-speed tick. The vote must bracket
    #: the true displacement; 1.5 leaves margin above ``V_MAX`` without spending
    #: resolution on displacements the base cannot produce.
    scan_vote_extent_scale: float = 1.5
    #: Consensus kernel width in metres. This is the scale at which a bearing
    #: counts as agreeing, so it tracks range noise rather than the signal.
    #: 0.01 measured best at both 0x and 1x sensor noise; widening it to 0.03
    #: admits more bearings as inliers and costs accuracy.
    scan_vote_sigma_m: float = 0.01
    #: Refinement window around the vote's peak, in grid cells. See
    #: :meth:`ScanVote.forward` for why the soft-argmax must not be global.
    scan_vote_radius: int = 3
    #: Frame gaps the vote is computed over. Gap 1 only, and that is measured
    #: rather than assumed: a longer baseline carries proportionally more
    #: displacement against the same range noise, so it looked like free signal,
    #: but the vote degrades sharply with it -- 0.96 cm/m of drift at gap 1,
    #: 10.8 at gap 2, 81 at gap 4. The range-flow model behind the vote is
    #: first-order in displacement and compares equal bearings, and both
    #: assumptions fail once the robot has moved far enough to shift scene
    #: content between bearings and to open and close occlusions.
    scan_vote_gaps: tuple[int, ...] = (1,)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["actor"] = self.actor.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BeliefNavPolicyConfig":
        value = dict(value)
        value["actor"] = NavPolicyConfig.from_dict(value.get("actor", {}))
        for key in ("odometry_channels", "odometry_trunk", "scan_vote_gaps"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)


class ScanVote(nn.Module):
    """Consensus vote for the translation between two scans. No parameters.

    Why voting rather than fitting
    ==============================
    Both geometric solvers tried against this corpus -- a point-to-line solve and
    a full ICP, each validated exact on analytic rooms -- recovered nothing from
    real scans (ICP correlation +0.089 against a true +10.4 cm displacement).
    Both minimize a sum of squared residuals, and roughly a quarter to a third of
    bearings here are gross outliers at depth edges, where a surface visible in
    one frame is occluded in the next. A squared residual is unbounded, so those
    bearings do not merely add noise: each one outweighs a hundred good bearings
    and relocates the minimum. Least squares has a breakdown point of zero, which
    is the wrong estimator for this data no matter how it is tuned.

    Scoring by agreement instead of by squared error inverts that. For each
    candidate translation, every bearing contributes

        exp(-(dr + dx cos theta + dy sin theta)^2 / 2 sigma^2)

    which is bounded in ``[0, 1]``. An outlier's residual is large under *every*
    candidate, so it contributes ~0 to all of them and cannot move the peak -- it
    abstains rather than votes. Inliers agree only near the true displacement, so
    they accumulate there. Measured on ``belief_onpolicy_v15`` at gap 1, this
    recovers the per-tick displacement to 0.51 cm mean error against 8.2 cm for
    least squares on the *identical* residuals, and 1.89 cm for estimating zero.

    Why it holds no parameters
    ==========================
    There is nothing here to learn: the projection geometry is exact and the
    kernel width is the range noise. Gradients through it would have nowhere to
    land, so the whole vote runs under ``no_grad`` and is detached. That is the
    same status ``sensors.deskew_history`` and :meth:`OdometryHead._align_history`
    have -- fixed sensor processing that happens to sit inside the model. The
    learned part is how the head *reads* the surface this produces, which is where
    the remaining judgement actually lives.
    """

    def __init__(self, beams: int, bins: int, extent: float, sigma: float,
                 radius: int = 3, chunk: int = 64):
        super().__init__()
        self.beams, self.bins, self.sigma, self.chunk = beams, bins, sigma, chunk
        self.radius = radius
        # Bin-centre bearings matching ``preprocess.rasterize_lidar``, which
        # floors ``(wrap_angle(angle) + pi) / width``. Half-a-width offset: a
        # constant bearing error rotates every recovered translation, which reads
        # as a null result rather than as a bug.
        width = 2.0 * math.pi / beams
        theta = -math.pi + (torch.arange(beams, dtype=torch.float32) + 0.5) * width
        axis = torch.linspace(-extent, extent, bins, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
        candidates = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        # (C, B): the range change each candidate predicts at each bearing. Fixed
        # geometry, so it is precomputed once rather than per forward pass.
        projection = (candidates[:, :1] * torch.cos(theta)[None]
                      + candidates[:, 1:] * torch.sin(theta)[None])
        self.register_buffer("candidates", candidates, persistent=False)
        self.register_buffer("projection", projection, persistent=False)

    def forward(self, difference: torch.Tensor, weight: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(estimate, surface, sharpness)`` for one pair of scans.

        ``difference`` is ``r_new - r_old_aligned`` per bearing and ``weight``
        masks bearings usable in both frames. ``surface`` is the normalized vote
        map, ``sharpness`` the peak's share of its mass -- an observability
        measure that distinguishes a corridor's ridge from a corner's peak.
        """
        with torch.no_grad():
            n = difference.shape[0]
            total = self.candidates.shape[0]
            score = difference.new_empty((n, total))
            # Chunked over candidates: the (N, C, B) residual tensor is the only
            # large intermediate and nothing downstream needs it, so it is built
            # and discarded a slice at a time. At 720 beams and a 21x21 grid the
            # full tensor would be 1.5 GB on a flattened training batch.
            for start in range(0, total, self.chunk):
                block = self.projection[start:start + self.chunk]
                residual = difference[:, None, :] + block[None]
                support = torch.exp(-0.5 * (residual / self.sigma) ** 2)
                score[:, start:start + self.chunk] = (
                    support * weight[:, None, :]).sum(-1)

            # Normalizing by the peak makes the temperature independent of how
            # many bearings happen to be valid, which varies by a factor of
            # several between an open room and a corridor.
            peak_value = score.amax(1, keepdim=True).clamp_min(1e-9)
            best = score.argmax(1)

            # Refine only inside a window around the peak. A soft-argmax over
            # the whole grid is pulled toward the grid centre by whatever
            # pedestal the surface carries, and because the grid is centred on
            # zero while the robot is really moving ~2 cm forward, that
            # shrinkage is a systematic *underestimate of travel* -- precisely
            # the component that integrates into drift rather than averaging
            # out. Measured on belief_onpolicy_v15: windowing drops drift from
            # 1.30 to 0.96 cm/m, against 2.10 cm/m for command dead reckoning.
            row, column = best // self.bins, best % self.bins
            index = torch.arange(self.bins, device=score.device)
            near = (((index[None] - row[:, None]).abs() <= self.radius)[:, :, None]
                    & ((index[None] - column[:, None]).abs()
                       <= self.radius)[:, None, :]).reshape(n, -1)
            mass = torch.softmax(
                torch.where(near, score / peak_value * 40.0,
                            torch.full_like(score, -1.0e4)), dim=1)
            estimate = mass @ self.candidates

            # Sharpness stays global on purpose: a broad surface is only visible
            # outside the refinement window. What it measures is how much
            # bearing-diverse evidence the scan carries, *not* how corrupted it
            # is -- outliers add a pedestal that dividing by the peak normalizes
            # away, and 30% of them move sharpness by under 2%. It still
            # correlates -0.46 with the vote's error on belief_onpolicy_v15,
            # because sparse returns are the failure that actually occurs.
            spread = torch.softmax(score / peak_value * 40.0, dim=1)
            sharpness = spread.gather(1, best[:, None])
            surface = (score / peak_value).view(n, 1, self.bins, self.bins)

            # With no usable bearings every candidate scores exactly zero, and
            # argmax then returns index 0 -- a *corner* of the grid, which reads
            # as a 3 cm displacement in both axes. Today the overlap gate zeroes
            # the correction before that can reach the belief, so this is
            # hygiene rather than a live bug; it stops being hygiene the moment
            # anyone reads the vote without the gate in front of it.
            informative = (weight.sum(-1, keepdim=True) > 0.0) & (
                peak_value > 0.0)
            estimate = torch.where(informative, estimate,
                                   torch.zeros_like(estimate))
            sharpness = torch.where(informative, sharpness,
                                    torch.zeros_like(sharpness))
            surface = torch.where(informative[:, :, None], surface.view(n, 1, -1),
                                  torch.zeros_like(surface.view(n, 1, -1))
                                  ).view(n, 1, self.bins, self.bins)
        return estimate.detach(), surface.detach(), sharpness.detach()


class OdometryHead(nn.Module):
    """Learn a bounded correction to command dead reckoning.

    Equal-bearing range differences are not a valid scan matcher under
    translation: depth discontinuities and newly occluded surfaces dominated
    the sub-millimetre slip signal by two orders of magnitude. Rather than bake
    that biased geometric estimate into the model, this head encodes the full
    deskewed metric history and learns the residual directly under privileged
    pose supervision. Commands and gyro provide the strong motion prior; lidar
    supplies independent scene motion evidence. A learned confidence gate keeps
    zero correction -- exact command dead reckoning -- as the fallback.
    """

    def __init__(self, config: BeliefNavPolicyConfig):
        super().__init__()
        self.config = config
        frames = config.motion_history_frames
        first, last = config.odometry_channels
        self.conv = nn.Sequential(
            nn.Conv1d(3 * frames, first, 7, stride=2, padding=3,
                      padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(first, last, 5, stride=2, padding=2,
                      padding_mode="circular"),
            nn.ReLU(),
        )
        # Preserve coarse bearing sectors. Global averaging made forward and
        # lateral scan motion indistinguishable; 24 sectors retain orientation
        # while keeping the dense projection small.
        self.scan_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(24), nn.Flatten(),
            nn.Linear(last * 24, config.odometry_features), nn.ReLU(),
        )
        self.motion_encoder = nn.Sequential(
            nn.Linear(3 * frames, 64), nn.ReLU(),
        )
        width = config.odometry_features + 64
        self.vote = None
        if config.scan_vote_bins > 0:
            self.vote = ScanVote(
                config.raw_lidar_bins, config.scan_vote_bins,
                config.scan_vote_extent_scale * V_MAX * DT,
                config.scan_vote_sigma_m, config.scan_vote_radius)
            # The surface is read as a small image because its *shape* is the
            # information a scalar estimate throws away: an isolated peak, a
            # ridge along one axis, and a broad plateau are three different
            # reliabilities that share one argmax, and only the first deserves
            # to move the belief.
            self.vote_encoder = nn.Sequential(
                nn.Conv2d(len(config.scan_vote_gaps), 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(),
            )
            side = -(-config.scan_vote_bins // 4)
            self.vote_projection = nn.Sequential(
                nn.Linear(16 * side * side, 64), nn.ReLU())
            # Per gap: the displacement the vote implies, already expressed as a
            # correction to the command in units of the correction limit, plus
            # the peak sharpness. Handing the head the answer in the units it
            # must output makes a near-identity weight the solution, which is the
            # same reason the model predicts a residual rather than a pose.
            width += 64 + 3 * len(config.scan_vote_gaps)
        trunk: list[nn.Module] = []
        for output_width in config.odometry_trunk:
            trunk.extend((nn.Linear(width, output_width), nn.ReLU()))
            width = output_width
        self.trunk = nn.Sequential(*trunk)
        self.proposal_head = nn.Linear(width, 2)
        # Overlap, plus peak sharpness when the vote branch supplies it.
        observability = 2 if config.scan_vote_bins > 0 else 1
        self.confidence_head = nn.Sequential(
            nn.Linear(width + observability, 32), nn.ReLU(), nn.Linear(32, 1))
        self.variance_head = nn.Linear(width, 2)
        if config.motion_state_size > 0:
            # Encoded scan/motion, normalized proposal x/y, and confidence.
            recurrent_inputs = width + 3
            self.motion_memory = nn.GRUCell(
                recurrent_inputs, config.motion_state_size)
            # Bias-free is a structural fallback guarantee: a zero state can
            # never create motion in the absence of scan evidence.
            self.motion_scale_head = nn.Linear(
                config.motion_state_size, 1, bias=False)
        self.apply(initialize_layer)
        nn.init.uniform_(self.proposal_head.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.proposal_head.bias)
        # Begin close to command dead reckoning without starving the residual
        # branch of gradient as the old -6 gate did.
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.constant_(self.confidence_head[-1].bias,
                          config.confidence_gate_bias)
        nn.init.zeros_(self.variance_head.weight)
        nn.init.constant_(self.variance_head.bias, -12.0)
        if config.motion_state_size > 0:
            # Enabling memory starts from the verified stateless estimator.
            # Training can add a recurrent residual without an initialization
            # jump in the deployed translation.
            nn.init.zeros_(self.motion_scale_head.weight)

    @property
    def head(self) -> nn.Linear:
        """Compatibility handle for dead-reckoning ablations."""
        return self.confidence_head[-1]

    @staticmethod
    def _rotate_scan(scan: torch.Tensor, valid: torch.Tensor,
                     delta_yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a previous circular scan at bearings in the current frame."""
        n, beams = scan.shape
        base = torch.arange(beams, device=scan.device, dtype=scan.dtype)[None]
        position = base + delta_yaw[:, None] * (beams / (2.0 * torch.pi))
        low_float = torch.floor(position)
        fraction = position - low_float
        low = low_float.long().remainder(beams)
        high = (low + 1).remainder(beams)
        lo = torch.gather(scan, 1, low)
        hi = torch.gather(scan, 1, high)
        rotated = lo + fraction * (hi - lo)
        usable = torch.gather(valid, 1, low)
        # At zero rotation the upper neighbour carries no weight, so requiring it
        # would erode the newest frame's mask against itself. Identity in,
        # identity out.
        usable = usable & (torch.gather(valid, 1, high) | (fraction == 0))
        return rotated, usable

    def _align_history(self, metric: torch.Tensor, valid: torch.Tensor,
                       gyro: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate every frame into the newest frame's bearing grid.

        Yaw between the oldest and newest frame of a one-second window shifts
        scan content by 44 of 720 bins on average and 137 at p95, measured on
        ``belief_onpolicy_v15``. The encoder's receptive field is about 28 bins,
        so without this the convolution is stacking *unrelated bearings* in its
        channel dimension whenever the robot turns, and the only cross-frame
        signal that survives is the command -- which is exactly the degenerate
        solution the head was previously found to have learned.

        The gyro measures that rotation directly. Removing it leaves translation
        parallax, which is the quantity being estimated. This is the same
        argument as the within-sweep deskew in :func:`sensors.deskew_history`,
        applied across frames rather than within one revolution, and it declines
        to spend capacity relearning a rotation that is already known.
        """
        n, frames, beams = metric.shape
        # ``gyro[:, i]`` is the increment *ending* at frame ``i`` -- the
        # convention `deskew_history` documents -- so the rotation carrying frame
        # ``i`` into the newest frame is the sum over ``i+1:``, and exactly zero
        # for the newest frame itself.
        trailing = gyro.flip(-1).cumsum(-1).flip(-1)
        delta = torch.cat(
            [trailing[:, 1:], torch.zeros_like(trailing[:, :1])], dim=1)
        rotated, usable = self._rotate_scan(
            metric.reshape(n * frames, beams),
            valid.reshape(n * frames, beams),
            delta.reshape(n * frames))
        return rotated.view(n, frames, beams), usable.view(n, frames, beams)

    def _commanded_displacement(self, previous_action: torch.Tensor,
                                gyro: torch.Tensor, gap: int) -> torch.Tensor:
        """Commanded displacement over the last ``gap`` ticks, in the newest frame.

        The vote reports displacement between two frames that have been rotated
        into a common orientation, so the command it is compared against must be
        composed through the same rotations rather than summed. For ``gap`` 1
        this reduces to ``v * DT`` forward, which is what
        :meth:`BeliefNavPolicy.translation` uses.
        """
        carried = previous_action.new_zeros((previous_action.shape[0], 2))
        for index in range(-gap, 0):
            speed = previous_action[:, index, 0] * V_MAX
            step = torch.stack([speed * DT, torch.zeros_like(speed)], dim=1)
            carried = compose_belief(carried, step, gyro[:, index])
        return -carried

    def _vote_features(self, aligned: torch.Tensor, aligned_valid: torch.Tensor,
                       previous_action: torch.Tensor, gyro: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vote every configured gap. Returns ``(features, sharpness)``.

        ``aligned`` must already sit in the newest frame's bearing grid, which is
        what makes the residual a pure translation question.
        """
        newest = aligned[:, -1]
        newest_valid = aligned_valid[:, -1]
        limit = self.config.forward_correction_scale * V_MAX * DT
        surfaces, scalars, sharpnesses = [], [], []
        for gap in self.config.scan_vote_gaps:
            older = aligned[:, -1 - gap]
            difference = newest - older
            weight = (newest_valid & aligned_valid[:, -1 - gap]).to(aligned.dtype)
            # A max-range fill paired against a real return is a missing
            # measurement, not a range change. Letting it vote is what
            # manufactures a 9 m "closure" out of a bearing that saw nothing.
            weight = weight * (newest < LIDAR_MAX_M - 1.0e-3).to(aligned.dtype)
            estimate, surface, sharpness = self.vote(difference, weight)
            commanded = self._commanded_displacement(previous_action, gyro, gap)

            # Both vectors above live in the *newest* frame, because that is the
            # bearing grid the scans were aligned into. The head's output does
            # not: it is added to ``dead_reckoned`` in :meth:`translation`, which
            # is the body frame at the start of the tick. Rotating the residual
            # back by the yaw accumulated over the gap puts it in the frame the
            # head actually answers in. The gap is at most one tick of yaw here,
            # so the correction is under a degree of a centimetre -- small, and
            # systematic in exactly the turning steps where slip is largest.
            turn = gyro[:, -gap:].sum(1)
            residual = compose_belief(estimate - commanded,
                                      torch.zeros_like(estimate), -turn)
            # In units of the correction limit, and clamped so an unbracketed
            # vote cannot inject a spike.
            residual = (residual / limit).clamp(-4.0, 4.0)
            surfaces.append(surface)
            scalars.append(torch.cat([residual, sharpness], dim=1))
            sharpnesses.append(sharpness)
        encoded = self.vote_projection(
            self.vote_encoder(torch.cat(surfaces, dim=1)))
        features = torch.cat([encoded] + scalars, dim=1)
        return features, torch.cat(sharpnesses, dim=1).mean(1, keepdim=True)

    def _raw_inputs(self, lidar: torch.Tensor,
                    raw_lidar: torch.Tensor | None,
                    raw_valid: torch.Tensor | None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        if raw_lidar is None:
            # Compatibility path for old callers and checkpoints: reconstruct a
            # metric scan from the actor bins. New training always supplies raw.
            metric = lidar * LIDAR_MAX_M
            metric = F.interpolate(metric, size=self.config.raw_lidar_bins,
                                   mode="linear", align_corners=False)
            return metric, torch.ones_like(metric, dtype=torch.bool)
        if raw_lidar.shape[-1] != self.config.raw_lidar_bins:
            raise ValueError(
                f"expected {self.config.raw_lidar_bins} raw lidar bins, got "
                f"{raw_lidar.shape[-1]}")
        valid = (torch.isfinite(raw_lidar) if raw_valid is None
                 else raw_valid.to(torch.bool))
        metric = torch.where(valid, raw_lidar,
                             torch.full_like(raw_lidar, LIDAR_MAX_M))
        return metric, valid

    def _features(self, lidar: torch.Tensor, gyro: torch.Tensor,
                  previous_action: torch.Tensor,
                  raw_lidar: torch.Tensor | None,
                  raw_valid: torch.Tensor | None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one sensor window and return hidden features plus overlap."""
        metric, valid = self._raw_inputs(lidar, raw_lidar, raw_valid)
        frames = self.config.motion_history_frames
        if metric.shape[1] < frames:
            pad = frames - metric.shape[1]
            metric = torch.cat([metric[:, :1].expand(-1, pad, -1), metric], 1)
            valid = torch.cat([valid[:, :1].expand(-1, pad, -1), valid], 1)
        if gyro.shape[1] < frames:
            gyro = F.pad(gyro, (frames - gyro.shape[1], 0))
        if previous_action.shape[1] < frames:
            pad = frames - previous_action.shape[1]
            previous_action = torch.cat([
                previous_action[:, :1].expand(-1, pad, -1), previous_action
            ], 1)
        vote_features = sharpness = None
        if self.vote is not None:
            # The vote is a translation question, so it always reads aligned
            # frames regardless of what the convolution below is given. When
            # ``derotate_scan_history`` is on the alignment is shared rather than
            # repeated.
            if self.config.derotate_scan_history:
                metric, valid = self._align_history(metric, valid, gyro)
                aligned, aligned_valid = metric, valid
            else:
                aligned, aligned_valid = self._align_history(metric, valid, gyro)
            vote_features, sharpness = self._vote_features(
                aligned, aligned_valid, previous_action, gyro)
        elif self.config.derotate_scan_history:
            metric, valid = self._align_history(metric, valid, gyro)
        normalized = metric / LIDAR_MAX_M
        newest = normalized[:, -1:]
        difference = newest - normalized
        if self.config.scaled_scan_difference:
            # Two problems with sharing a convolution between an absolute range
            # channel and a difference channel normalized the same way.
            #
            # First, an invalid return is filled with LIDAR_MAX_M, so pairing a
            # 3 m hit against a no-return manufactures a 9 m "range change" that
            # dwarfs anything motion produces. Only bearings valid in *both*
            # frames carry a difference; the mask channel already says which.
            #
            # Second, scale: the parallax being measured is centimetres, so
            # dividing it by a 12 m range puts the signal three orders of
            # magnitude below the channel beside it and, at standard init,
            # essentially out of the first layer's reach. Normalize instead by
            # the translation one window can physically contain.
            difference = torch.where(valid & valid[:, -1:], difference,
                                     torch.zeros_like(difference))
            difference = difference * (
                LIDAR_MAX_M / (V_MAX * DT * frames))
        encoded = self.conv(torch.cat(
            [normalized, difference, valid.float()], dim=1))
        scan_features = self.scan_projection(encoded)
        motion = torch.cat((gyro, previous_action.flatten(1)), dim=1)
        parts = [scan_features, self.motion_encoder(motion)]
        if vote_features is not None:
            parts.append(vote_features)
        hidden = self.trunk(torch.cat(parts, dim=1))
        overlap = (valid[:, -2] & valid[:, -1]).float().mean(1, keepdim=True)
        if sharpness is not None:
            # The gate's question is "is this scan evidence about my motion?"
            # and peak sharpness answers it geometrically: a corridor's ridge
            # and a corner's peak are the same argmax and different evidence.
            # Unlike the learned gate, this reaches the confidence head without
            # passing through a proposal initialized near zero, so it carries
            # gradient from the first step.
            overlap = torch.cat([overlap, sharpness], dim=1)
        return hidden, overlap

    def _outputs(self, hidden: torch.Tensor, overlap: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode proposal, uncertainty and observability-gated confidence."""

        raw = torch.tanh(self.proposal_head(hidden))
        limit = self.config.forward_correction_scale * V_MAX * DT
        proposal = torch.stack([
            limit * raw[:, 0],
            limit * self.config.lateral_correction_ratio * raw[:, 1],
        ], dim=1)
        confidence = torch.sigmoid(self.confidence_head(torch.cat(
            [hidden, overlap], dim=1)))
        # Observability is a hard constraint, not something training may learn
        # around. With no overlapping returns the result is exactly the command
        # prior even if a confidence bias later drifts upward. Column 0 is the
        # both-frames-valid fraction; a vote checkpoint carries peak sharpness
        # beside it, which the learned gate reads but the hard clamp does not --
        # a hard floor on sharpness would suppress the open-room case the
        # residual design already handles by falling back to the command.
        confidence = confidence * (overlap[:, :1] / 0.5).clamp(0.0, 1.0)
        variance = (F.softplus(self.variance_head(hidden)) + 1.0e-7).clamp(
            1.0e-7, 0.05 ** 2)
        return proposal, variance, confidence.squeeze(1)

    def forward(self, lidar: torch.Tensor, gyro: torch.Tensor,
                previous_action: torch.Tensor,
                raw_lidar: torch.Tensor | None = None,
                raw_valid: torch.Tensor | None = None,
                *, return_uncertainty: bool = False,
                return_proposal: bool = False):
        """Return scan correction, optionally with diagonal motion variance.

        ``lidar`` is ``(N, H, B)``, ``gyro`` is ``(N, H)`` radians turned per
        tick, ``previous_action`` is ``(N, H, 2)`` normalized commands.
        """
        hidden, overlap = self._features(
            lidar, gyro, previous_action, raw_lidar, raw_valid)
        proposal, variance, confidence = self._outputs(hidden, overlap)
        correction = confidence[:, None] * proposal
        if return_uncertainty:
            motion = proposal if return_proposal else correction
            return motion, variance, confidence
        return correction

    def forward_memory(
            self, state: torch.Tensor, lidar: torch.Tensor,
            gyro: torch.Tensor, previous_action: torch.Tensor,
            raw_lidar: torch.Tensor | None = None,
            raw_valid: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                       torch.Tensor, torch.Tensor]:
        """Return a recurrent forward-scale residual and next explicit state."""
        if self.config.motion_state_size <= 0:
            raise ValueError("motion memory is disabled in this checkpoint")
        hidden, overlap = self._features(
            lidar, gyro, previous_action, raw_lidar, raw_valid)
        proposal, variance, confidence = self._outputs(hidden, overlap)
        limit = self.config.forward_correction_scale * V_MAX * DT
        normalized_proposal = proposal / torch.tensor(
            [limit, limit * self.config.lateral_correction_ratio],
            device=proposal.device, dtype=proposal.dtype)
        memory_input = torch.cat(
            [hidden, normalized_proposal, confidence[:, None]], dim=1)
        proposed_state = self.motion_memory(memory_input, state)
        evidence = (confidence / 0.1).clamp(0.0, 1.0)[:, None]
        state = state + evidence * (proposed_state - state)
        recurrent_scale = 0.25 * torch.tanh(
            self.motion_scale_head(state).squeeze(1))
        correction = confidence[:, None] * proposal
        return correction, variance, confidence, state, recurrent_scale


class BeliefNavPolicy(nn.Module):
    """An actor plus the belief it navigates by. No pose input anywhere.

    Two entry points, and they must stay numerically identical:

    * :meth:`step` -- one control tick, taking and returning the belief. This is
      what the exported graph and the C++ node use, and it is a static graph
      with two floats of carried state.
    * :meth:`rollout` -- a whole sequence, for training through the recurrence.

    ``rollout`` is a loop over ``step``, deliberately, rather than a second
    implementation optimized for the training shape. The two diverging is the
    single most expensive class of bug this project has had.
    """

    def __init__(self, config: BeliefNavPolicyConfig | None = None):
        super().__init__()
        self.config = config or BeliefNavPolicyConfig()
        self.actor = NavPolicy(self.config.actor)
        self.odometry = OdometryHead(self.config)

    # ── belief ───────────────────────────────────────────────────────────────

    def translation(self, lidar: torch.Tensor, gyro: torch.Tensor,
                    previous_action: torch.Tensor,
                    raw_lidar: torch.Tensor | None = None,
                    raw_valid: torch.Tensor | None = None) -> torch.Tensor:
        """Commanded displacement plus the learned correction, ``(N, 2)`` metres.

        The commanded speed is read from the newest previous-action frame rather
        than passed in separately. That is the command executed over the tick
        being integrated, and deriving it here means there is exactly one copy
        of it -- a second argument carrying the same number is a desync waiting
        to happen, and a one-tick desync in this term is invisible in the loss
        and fatal to the belief.
        """
        commanded_v = previous_action[:, -1, 0] * V_MAX
        dead_reckoned = torch.stack(
            [commanded_v * DT, torch.zeros_like(commanded_v)], dim=1)
        return dead_reckoned + self.odometry(
            lidar, gyro, previous_action, raw_lidar, raw_valid)

    def motion_estimate(self, lidar: torch.Tensor, gyro: torch.Tensor,
                        previous_action: torch.Tensor,
                        raw_lidar: torch.Tensor | None = None,
                        raw_valid: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Translation, diagonal process variance and scan confidence."""
        commanded_v = previous_action[:, -1, 0] * V_MAX
        dead_reckoned = torch.stack(
            [commanded_v * DT, torch.zeros_like(commanded_v)], dim=1)
        correction, variance, confidence = self.odometry(
            lidar, gyro, previous_action, raw_lidar, raw_valid,
            return_uncertainty=True)
        return dead_reckoned + correction, variance, confidence

    def calibrated_motion_estimate(
            self, calibration: torch.Tensor, lidar: torch.Tensor,
            gyro: torch.Tensor, previous_action: torch.Tensor,
            raw_lidar: torch.Tensor | None = None,
            raw_valid: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate motion and update the episode's forward scale state.

        The lidar head proposes a bounded correction at full speed. Its
        confidence-gated forward component is interpreted as a wheel-scale
        measurement and exponentially filtered across the episode. This lets
        persistent tyre/floor calibration survive beyond the four scan frames
        visible to the head, while zero evidence remains exact command dead
        reckoning.
        """
        if not self.config.persistent_calibration:
            if self.config.motion_state_size > 0:
                (correction, variance, confidence, calibration,
                 recurrent_scale) = self.odometry.forward_memory(
                    calibration, lidar, gyro, previous_action,
                    raw_lidar, raw_valid)
                commanded_v = previous_action[:, -1, 0] * V_MAX
                instant_scale = correction[:, 0] / (V_MAX * DT)
                scale = (instant_scale + recurrent_scale).clamp(
                    self.config.calibration_min,
                    self.config.calibration_max)
                translation = torch.stack([
                    commanded_v * DT * (1.0 + scale), correction[:, 1]
                ], dim=1)
                return translation, variance, confidence, calibration
            translation, variance, confidence = self.motion_estimate(
                lidar, gyro, previous_action, raw_lidar, raw_valid)
            return translation, variance, confidence, calibration

        proposal, variance, confidence = self.odometry(
            lidar, gyro, previous_action, raw_lidar, raw_valid,
            return_uncertainty=True, return_proposal=True)
        gated = confidence[:, None] * proposal
        candidate = gated[:, 0] / (V_MAX * DT)
        commanded_v = previous_action[:, -1, 0] * V_MAX
        moving = (commanded_v.abs() / (0.2 * V_MAX)).clamp(0.0, 1.0)
        alpha = self.config.calibration_update_rate * moving
        calibration = (calibration + alpha * (candidate - calibration)).clamp(
            self.config.calibration_min, self.config.calibration_max)
        translation = torch.stack([
            commanded_v * DT * (1.0 + calibration), gated[:, 1]
        ], dim=1)
        return translation, variance, confidence, calibration

    def update_belief(self, belief: torch.Tensor, lidar: torch.Tensor,
                      gyro: torch.Tensor,
                      previous_action: torch.Tensor,
                      raw_lidar: torch.Tensor | None = None,
                      raw_valid: torch.Tensor | None = None) -> torch.Tensor:
        """Advance the goal belief across the tick that has just elapsed.

        **Everything here looks backward, and that is the point.** The gyro
        increment is the rotation measured over the period that just ended, and
        the commanded speed is the one executed over that same period. Both are
        available to the robot right now. An earlier draft integrated the
        command about to be issued against the rotation just measured, which
        mixes two different ticks; it survived the shape checks, cost 8 cm of
        belief error at initialization, and would have been read as a bad
        odometry head rather than an indexing bug.

        Rotation is the gyro's, unmodified, for the reason given in the module
        docstring. Nothing here reads a pose.
        """
        step = self.translation(lidar, gyro, previous_action,
                                raw_lidar, raw_valid)
        return compose_belief(belief, step, gyro[:, -1])

    @staticmethod
    def _propagate_covariance(covariance: torch.Tensor,
                              motion_variance: torch.Tensor,
                              delta_yaw: torch.Tensor,
                              drift_variance: torch.Tensor | None = None
                              ) -> torch.Tensor:
        """Rotate a 2-D covariance and add translation process variance.

        ``motion_variance`` is the per-step term the variance head declares, and
        summing it down the episode is a random walk. ``drift_variance`` is the
        separate, isotropic contribution of a *systematic* scale error, which
        grows with distance travelled rather than with step count. Both are
        needed: without the second the declared bound falls behind the true
        error by roughly the square root of the episode length. Measured on
        ``belief_onpolicy_v15`` with the random-walk term alone, the true error
        was 8.8x the declared sigma and the 1.96-sigma bound covered 9.5% of
        steps instead of 95%.
        """
        cos, sin = torch.cos(delta_yaw), torch.sin(delta_yaw)
        rotation = torch.stack([
            torch.stack([cos, sin], dim=1),
            torch.stack([-sin, cos], dim=1),
        ], dim=1)
        process = torch.diag_embed(motion_variance)
        rotated = rotation @ covariance @ rotation.transpose(1, 2) + process
        if drift_variance is None:
            return rotated
        return rotated + torch.diag_embed(
            drift_variance[:, None].expand(-1, 2))

    def _drift_variance(self, travelled: torch.Tensor,
                        step_length: torch.Tensor) -> torch.Tensor:
        """Variance a scale error adds over one step, given distance so far.

        A relative scale uncertainty ``eps`` puts the standard deviation at
        ``eps * s`` after travelling ``s``, so the *variance* increment across a
        step is ``eps^2 * ((s + ds)^2 - s^2)``. Writing it as a difference of
        squares rather than as ``eps^2 * ds^2`` is the whole point: the cross
        term ``2 s ds`` is what makes the error accumulate linearly in distance
        instead of as a random walk, and dropping it is exactly the mistake the
        random-walk model makes.
        """
        epsilon = self.config.drift_scale_uncertainty
        advanced = travelled + step_length
        return (epsilon ** 2) * (advanced.square() - travelled.square())

    @staticmethod
    def _largest_covariance_eigenvalue(covariance: torch.Tensor) -> torch.Tensor:
        """Closed-form largest eigenvalue of a symmetric 2x2 matrix."""
        a, d = covariance[:, 0, 0], covariance[:, 1, 1]
        b = 0.5 * (covariance[:, 0, 1] + covariance[:, 1, 0])
        return 0.5 * (a + d + torch.sqrt((a - d).square() + 4.0 * b.square()))

    # ── acting ───────────────────────────────────────────────────────────────

    def observation(self, lidar: torch.Tensor, belief: torch.Tensor,
                    previous_action: torch.Tensor) -> torch.Tensor:
        """The actor's flat input -- the same 731 numbers the old policy read.

        The goal block comes from the belief instead of from a pose subtraction.
        That change of provenance is invisible to the actor by construction,
        which is exactly what makes the pretrained weights transfer.
        """
        return build_observation(lidar, relative_goal_features(belief),
                                 previous_action[:, -self.config.actor.history_frames:])

    def step(self, belief: torch.Tensor, lidar: torch.Tensor,
             gyro: torch.Tensor, previous_action: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """One control tick: ``(action, belief)``, in that causal order.

        **Update first, then act.** On the robot, a tick begins by taking in the
        gyro and lidar produced since the last one; only then is there anything
        new to decide on. Acting on the stale belief and updating afterwards
        computes the same numbers in the wrong order and leaves the actor one
        tick behind the sensors for the whole episode.

        The returned belief is the one the returned action was chosen from, so a
        caller can supervise exactly the quantity the actor consumed.
        """
        belief = self.update_belief(belief, lidar, gyro, previous_action)
        action = self.actor(self.observation(lidar, belief, previous_action))
        return action, belief

    def step_sensors(self, belief: torch.Tensor, lidar: torch.Tensor,
                     raw_lidar: torch.Tensor, raw_valid: torch.Tensor,
                     gyro: torch.Tensor, previous_action: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """One tick using the high-resolution scan-motion input."""
        belief = self.update_belief(belief, lidar, gyro, previous_action,
                                    raw_lidar, raw_valid)
        action = self.actor(self.observation(lidar, belief, previous_action))
        return action, belief

    def step_state(self, belief: torch.Tensor, covariance: torch.Tensor,
                   arrival_count: torch.Tensor, lidar: torch.Tensor,
                   raw_lidar: torch.Tensor, raw_valid: torch.Tensor,
                   gyro: torch.Tensor, previous_action: torch.Tensor,
                   calibration: torch.Tensor | None = None,
                   travelled: torch.Tensor | None = None):
        """Deployable recurrent step with uncertainty-aware arrival.

        Returns ``(command, arrived, belief, covariance, arrival_count,
        scan_confidence)``, followed by ``calibration`` and ``travelled`` when
        those were supplied. All state is explicit; no pose or wheel odometry is
        accepted by this interface.

        ``travelled`` is cumulative path length and is required when the
        checkpoint declares a ``drift_scale_uncertainty``. One float is the
        whole cost of the systematic term in :meth:`_drift_variance`, and it
        stays outside the quantized graph for the same reason the belief does.
        """
        if travelled is None and self.config.drift_scale_uncertainty > 0.0:
            raise ValueError(
                "a drift-aware checkpoint requires carried travelled distance")
        if calibration is None:
            if (self.config.persistent_calibration
                    or self.config.motion_state_size > 0):
                raise ValueError(
                    "recurrent motion checkpoint requires carried state")
            translation, variance, confidence = self.motion_estimate(
                lidar, gyro, previous_action, raw_lidar, raw_valid)
        else:
            translation, variance, confidence, calibration = (
                self.calibrated_motion_estimate(
                    calibration, lidar, gyro, previous_action,
                    raw_lidar, raw_valid))
        delta_yaw = gyro[:, -1]
        belief = compose_belief(belief, translation, delta_yaw)
        step_length = torch.linalg.vector_norm(translation, dim=1)
        drift = None
        if travelled is not None and self.config.drift_scale_uncertainty > 0.0:
            drift = self._drift_variance(travelled, step_length)
        covariance = self._propagate_covariance(
            covariance, variance, delta_yaw, drift)
        if travelled is not None:
            travelled = travelled + step_length
        command = self.actor(self.observation(lidar, belief, previous_action))

        largest_variance = self._largest_covariance_eigenvalue(covariance)
        upper_bound = (torch.linalg.vector_norm(belief, dim=1)
                       + self.config.arrival_sigma
                       * torch.sqrt(largest_variance.clamp_min(0.0)))
        # Arrival must be able to stop a moving actor. Requiring the actor to
        # slow down first creates a circular dependency: the inherited actor
        # has no stop output, so it drives through a correct goal belief and
        # the gate never fires. The uncertainty bound plus consecutive-tick
        # confirmation are the false-positive safeguards; once they agree,
        # stopping is the safe action regardless of current commanded speed.
        candidate = upper_bound <= self.config.arrival_radius_m
        arrival_count = torch.where(candidate, arrival_count + 1,
                                    torch.zeros_like(arrival_count))
        arrived = arrival_count >= self.config.arrival_confirm_ticks
        command = torch.where(arrived[:, None], torch.zeros_like(command), command)
        result = (command, arrived, belief, covariance, arrival_count,
                  confidence)
        if calibration is not None:
            result = result + (calibration,)
        return result if travelled is None else result + (travelled,)

    def rollout(self, lidar: torch.Tensor, gyro: torch.Tensor,
                previous_action: torch.Tensor, belief0: torch.Tensor,
                raw_lidar: torch.Tensor | None = None,
                raw_valid: torch.Tensor | None = None, *,
                return_motion: bool = False):
        """Run a sequence through the recurrence. Gradients flow the whole way.

        Inputs are time-major: ``lidar`` ``(T, N, H, B)``, ``gyro`` ``(T, N, H)``,
        ``previous_action`` ``(T, N, H, 2)``, and ``belief0`` ``(N, 2)``.
        Returns actions ``(T, N, 2)`` and the beliefs they were taken from,
        ``(T, N, 2)``.

        The first step does not integrate. ``belief0`` is by definition already
        correct *at* step zero, so applying an update there would double-count
        the tick that ended just before the window opened -- and because the
        error is a constant offset rather than a divergence, it is precisely the
        kind that trains away into the head as a fake bias.
        """
        steps = lidar.shape[0]
        belief = belief0
        calibration = self.initial_calibration(belief0)
        actions, beliefs = [], []
        translations, variances, confidences = [], [], []
        for t in range(steps):
            if t > 0:
                translation, variance, confidence, calibration = (
                    self.calibrated_motion_estimate(
                        calibration, lidar[t], gyro[t], previous_action[t],
                        None if raw_lidar is None else raw_lidar[t],
                        None if raw_valid is None else raw_valid[t]))
                belief = compose_belief(belief, translation, gyro[t, :, -1])
            else:
                translation = belief.new_zeros(belief.shape)
                variance = belief.new_zeros(belief.shape)
                confidence = belief.new_zeros(belief.shape[0])
            beliefs.append(belief)
            translations.append(translation)
            variances.append(variance)
            confidences.append(confidence)
            actions.append(self.actor(
                self.observation(lidar[t], belief, previous_action[t])))
        result = (torch.stack(actions), torch.stack(beliefs))
        if return_motion:
            return result + (torch.stack(translations),
                             torch.stack(variances),
                             torch.stack(confidences))
        return result

    def motion_rollout(self, lidar: torch.Tensor, gyro: torch.Tensor,
                       previous_action: torch.Tensor,
                       raw_lidar: torch.Tensor, raw_valid: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run only the motion recurrence, for causal counterfactual training."""
        state = self.initial_calibration(lidar.new_zeros(lidar.shape[1], 2))
        translations, variances, confidences = [], [], []
        for t in range(lidar.shape[0]):
            if t == 0:
                translation = lidar.new_zeros(lidar.shape[1], 2)
                variance = lidar.new_zeros(lidar.shape[1], 2)
                confidence = lidar.new_zeros(lidar.shape[1])
            else:
                translation, variance, confidence, state = (
                    self.calibrated_motion_estimate(
                        state, lidar[t], gyro[t], previous_action[t],
                        raw_lidar[t], raw_valid[t]))
            translations.append(translation)
            variances.append(variance)
            confidences.append(confidence)
        return (torch.stack(translations), torch.stack(variances),
                torch.stack(confidences))

    # ── plumbing ─────────────────────────────────────────────────────────────

    def load_actor(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Warm-start the actor from a ``NavPolicy`` checkpoint, strictly.

        Strict on purpose. A silently partial load here would produce a policy
        that trains without complaint and navigates like a randomly initialized
        one, and the symptom would look like a reward-shaping problem.
        """
        self.actor.load_state_dict(state_dict)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def odometry_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.odometry.parameters())

    def initial_belief(self, goal_body: torch.Tensor) -> torch.Tensor:
        """The ``(dx, dy)`` the caller asked for, clamped to the trained range.

        A goal past ``GOAL_MAX_M`` is "far away in that direction" and the
        distance feature saturates there anyway. Clamping the *belief* as well
        keeps the integrator from carrying a magnitude the actor was never
        trained to read, which matters because the belief persists.
        """
        distance = torch.linalg.vector_norm(goal_body, dim=1, keepdim=True)
        excess = (distance / GOAL_MAX_M).clamp_min(1.0)
        return goal_body / excess

    def initial_state(self, goal_body: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize explicit deployment state at an episode reset."""
        belief = self.initial_belief(goal_body)
        covariance = torch.eye(2, device=goal_body.device,
                               dtype=goal_body.dtype)[None].repeat(
                                   goal_body.shape[0], 1, 1) * 1.0e-8
        count = torch.zeros(goal_body.shape[0], dtype=torch.long,
                            device=goal_body.device)
        return belief, covariance, count

    def initial_travel(self, goal_body: torch.Tensor) -> torch.Tensor:
        """Zero path length at reset, which is when the belief is exact."""
        return torch.zeros(goal_body.shape[0], device=goal_body.device,
                           dtype=goal_body.dtype)

    def initial_calibration(self, goal_body: torch.Tensor) -> torch.Tensor:
        """Zero scale correction is exact command dead reckoning at reset."""
        if self.config.motion_state_size > 0:
            return torch.zeros(goal_body.shape[0],
                               self.config.motion_state_size,
                               device=goal_body.device, dtype=goal_body.dtype)
        return torch.zeros(goal_body.shape[0], device=goal_body.device,
                           dtype=goal_body.dtype)

    def example_inputs(self, batch_size: int = 1, device=None):
        """The exact argument tuple :meth:`step` takes, for export tracing."""
        cfg = self.config.actor
        return (
            torch.zeros(batch_size, 2, device=device),
            torch.zeros(batch_size, cfg.history_frames, cfg.lidar_bins,
                        device=device),
            torch.zeros(batch_size, cfg.history_frames, device=device),
            torch.zeros(batch_size, cfg.history_frames, 2, device=device),
        )

    def sensor_example_inputs(self, batch_size: int = 1, device=None):
        """Exact argument tuple for tracing :meth:`step_state`."""
        cfg = self.config.actor
        goal = torch.zeros(batch_size, 2, device=device)
        belief, covariance, count = self.initial_state(goal)
        return (
            belief, covariance, count,
            torch.zeros(batch_size, cfg.history_frames, cfg.lidar_bins,
                        device=device),
            torch.full((batch_size, cfg.history_frames,
                        self.config.raw_lidar_bins), LIDAR_MAX_M, device=device),
            torch.zeros(batch_size, cfg.history_frames,
                        self.config.raw_lidar_bins, dtype=torch.bool,
                        device=device),
            torch.zeros(batch_size, cfg.history_frames, device=device),
            torch.zeros(batch_size, cfg.history_frames, 2, device=device),
        )
