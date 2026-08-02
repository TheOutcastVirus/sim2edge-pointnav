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
           "compose_belief"]


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

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["actor"] = self.actor.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BeliefNavPolicyConfig":
        value = dict(value)
        value["actor"] = NavPolicyConfig.from_dict(value.get("actor", {}))
        for key in ("odometry_channels", "odometry_trunk"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)


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
        trunk: list[nn.Module] = []
        for output_width in config.odometry_trunk:
            trunk.extend((nn.Linear(width, output_width), nn.ReLU()))
            width = output_width
        self.trunk = nn.Sequential(*trunk)
        self.proposal_head = nn.Linear(width, 2)
        self.confidence_head = nn.Sequential(nn.Linear(width + 1, 32), nn.ReLU(),
                                             nn.Linear(32, 1))
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
        if self.config.derotate_scan_history:
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
        hidden = self.trunk(torch.cat(
            (scan_features, self.motion_encoder(motion)), dim=1))
        overlap = (valid[:, -2] & valid[:, -1]).float().mean(1, keepdim=True)
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
        # prior even if a confidence bias later drifts upward.
        confidence = confidence * (overlap / 0.5).clamp(0.0, 1.0)
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
                              delta_yaw: torch.Tensor) -> torch.Tensor:
        """Rotate a 2-D covariance and add translation process variance."""
        cos, sin = torch.cos(delta_yaw), torch.sin(delta_yaw)
        rotation = torch.stack([
            torch.stack([cos, sin], dim=1),
            torch.stack([-sin, cos], dim=1),
        ], dim=1)
        process = torch.diag_embed(motion_variance)
        return rotation @ covariance @ rotation.transpose(1, 2) + process

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
                   calibration: torch.Tensor | None = None):
        """Deployable recurrent step with uncertainty-aware arrival.

        Returns ``(command, arrived, belief, covariance, arrival_count,
        scan_confidence)``. All state is explicit; no pose or wheel odometry is
        accepted by this interface.
        """
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
        covariance = self._propagate_covariance(
            covariance, variance, delta_yaw)
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
        return result if calibration is None else result + (calibration,)

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
