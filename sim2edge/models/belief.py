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

from ..preprocess import build_observation, relative_goal_features
from ..spec import DT, GOAL_MAX_M, HISTORY_FRAMES, LIDAR_BINS, V_MAX
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
    #: Ceiling on the learned forward correction, as a multiple of the distance
    #: a full-speed tick covers. Slip removes speed far more often than it adds
    #: it, but the bound is symmetric because a systematically low commanded
    #: speed is equally possible and equally correctable.
    forward_correction_scale: float = 0.5
    #: Lateral scale, relative to the forward one. Differential drive cannot
    #: translate sideways; this is the wheel-slip residual only.
    lateral_correction_ratio: float = 0.15

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
    """Lidar history + gyro + previous commands -> a translation correction.

    Its own convolutional stack rather than a branch off the actor's. The two
    are asked different questions of the same scan -- the actor wants "where is
    there space", the head wants "how did this scan move since the last one" --
    and sharing a trunk between them made the actor's features drift toward
    whichever objective had the larger gradient.

    The head reads the *binned* 180-bin scans, not the raw 720-beam ones. One
    tick of motion at full speed is 2.3 cm, which is barely above the 2 cm range
    noise on a single beam, so this is never a per-beam measurement: it is an
    aggregate over ~180 bins, where the noise falls by roughly sqrt(180) and the
    two-degree bin width stops being the limiting term. If measured drift turns
    out too high, raising this head's angular resolution is the first knob to
    reach for -- it is deliberately the cheapest thing to change.
    """

    def __init__(self, config: BeliefNavPolicyConfig):
        super().__init__()
        self.config = config
        frames = config.actor.history_frames
        bins = config.actor.lidar_bins

        conv: list[nn.Module] = []
        # Consecutive frames enter as channels, so the very first convolution
        # can difference them -- which is the entire signal this head lives on.
        channels, length = frames, bins
        for index, output_channels in enumerate(config.odometry_channels):
            kernel = 5 if index == 0 else 3
            padding = kernel // 2
            conv.extend((
                nn.Conv1d(channels, output_channels, kernel, stride=2,
                          padding=padding),
                nn.ReLU(),
            ))
            channels = output_channels
            length = (length + 2 * padding - kernel) // 2 + 1
        self.conv = nn.Sequential(*conv)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * length, config.odometry_features),
            nn.ReLU(),
        )
        # Gyro increments and previous commands are what tell the head how much
        # of the observed change is rotation it should already have accounted
        # for, and how much translation to expect before it looks.
        self.motion_encoder = nn.Sequential(
            nn.Linear(3 * frames, 64), nn.ReLU(),
        )
        trunk: list[nn.Module] = []
        width = config.odometry_features + 64
        for output_width in config.odometry_trunk:
            trunk.extend((nn.Linear(width, output_width), nn.ReLU()))
            width = output_width
        self.trunk = nn.Sequential(*trunk)
        self.head = nn.Linear(width, 2)

        self.apply(initialize_layer)
        # Start a hair away from the dead-reckoning baseline, not exactly on it.
        #
        # Exactly zero is the tempting choice and it is wrong: with a zero final
        # weight the gradient reaching the trunk is ``W.T @ grad`` = 0, so the
        # convolutional stack gets no gradient at all on the first optimizer
        # step and only unsticks once the head has drifted off zero by itself.
        # A uniform 1e-3 leaves the initial correction at ~1e-4 m against a
        # 2.3 cm commanded tick -- indistinguishable from dead reckoning -- while
        # keeping the whole head differentiable from step one. Same reasoning,
        # and the same constant, as ``NavPolicy.action_head``.
        nn.init.uniform_(self.head.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, lidar: torch.Tensor, gyro: torch.Tensor,
                previous_action: torch.Tensor) -> torch.Tensor:
        """``(N, 2)`` body-frame translation correction in metres.

        ``lidar`` is ``(N, H, B)``, ``gyro`` is ``(N, H)`` radians turned per
        tick, ``previous_action`` is ``(N, H, 2)`` normalized commands.
        """
        motion = torch.cat((gyro, previous_action.flatten(1)), dim=1)
        fused = torch.cat((self.projection(self.conv(lidar)),
                           self.motion_encoder(motion)), dim=1)
        raw = torch.tanh(self.head(self.trunk(fused)))
        limit = self.config.forward_correction_scale * V_MAX * DT
        return torch.stack([
            limit * raw[:, 0],
            limit * self.config.lateral_correction_ratio * raw[:, 1],
        ], dim=1)


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
                    previous_action: torch.Tensor) -> torch.Tensor:
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
        return dead_reckoned + self.odometry(lidar, gyro, previous_action)

    def update_belief(self, belief: torch.Tensor, lidar: torch.Tensor,
                      gyro: torch.Tensor,
                      previous_action: torch.Tensor) -> torch.Tensor:
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
        step = self.translation(lidar, gyro, previous_action)
        return compose_belief(belief, step, gyro[:, -1])

    # ── acting ───────────────────────────────────────────────────────────────

    def observation(self, lidar: torch.Tensor, belief: torch.Tensor,
                    previous_action: torch.Tensor) -> torch.Tensor:
        """The actor's flat input -- the same 731 numbers the old policy read.

        The goal block comes from the belief instead of from a pose subtraction.
        That change of provenance is invisible to the actor by construction,
        which is exactly what makes the pretrained weights transfer.
        """
        return build_observation(lidar, relative_goal_features(belief),
                                 previous_action)

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

    def rollout(self, lidar: torch.Tensor, gyro: torch.Tensor,
                previous_action: torch.Tensor, belief0: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
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
        actions, beliefs = [], []
        for t in range(steps):
            if t > 0:
                belief = self.update_belief(belief, lidar[t], gyro[t],
                                            previous_action[t])
            beliefs.append(belief)
            actions.append(self.actor(
                self.observation(lidar[t], belief, previous_action[t])))
        return torch.stack(actions), torch.stack(beliefs)

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
