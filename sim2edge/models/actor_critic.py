"""The stochastic wrapper PPO trains, around the deterministic graph we deploy.

Nothing here ships to the robot. The exported artifact is still
:class:`~sim2edge.models.nav_policy.NavPolicy` -- ``NavActorCritic.actor`` *is*
that object, not a copy of it -- so reinforcement learning improves exactly the
weights that get quantized, and the critic and the exploration noise are thrown
away at export.

Three decisions that look like taste and are not:

**The distribution lives before the tanh.** The actor's head emits an unbounded
``raw``; the policy is a diagonal Gaussian over that, and the tanh plus the SI
scaling are treated as part of the environment. This keeps the importance ratio
exact -- a Gaussian over the bounded command would put mass outside the
Create3's envelope, and clipping it back silently invalidates the density the
ratio divides by. The cost is that a saturated actor can wander in ``raw``
without changing what the robot does, which is why the trainer reports tanh
saturation every iteration.

**The critic does not share the actor's encoder.** Shared features are cheaper
and are the usual default, but here the actor arrives carefully fitted from
behaviour cloning and the critic starts from noise; sharing would route that
noise straight through the features BC spent hours on. A second 290k-parameter
tower is nothing next to a 24 GB card, and it never has to be exported.

**The standard deviation is a state-independent parameter.** Two numbers, not a
head. A learned per-state std is one more thing that can collapse in a corner
before the critic is trustworthy, and DAgger already established that a fixed
0.15-of-limit perturbation is a usable exploration scale for this robot.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .nav_policy import (NavPolicy, NavPolicyConfig, build_encoder, encode,
                         initialize_layer, physical_action)

__all__ = ["NavActorCritic", "NavCritic", "gaussian_entropy",
           "gaussian_log_prob"]

#: Initial exploration std, in pre-squash units. Near the origin tanh has unit
#: slope, so this is very nearly a fraction of each axis limit -- and 0.15 is
#: the execution-noise sigma the demonstrations were collected with, which makes
#: the first PPO rollouts explore the same tube of states BC was trained on.
DEFAULT_INIT_STD = 0.15

#: Bounds on log std, applied by the trainer after every optimizer step.
#: The floor keeps a fine-tuning run from going fully deterministic (and its
#: ratios from exploding); the ceiling keeps a diverging run from emitting
#: uniform noise while the environment keeps producing plausible-looking data.
LOG_STD_MIN, LOG_STD_MAX = -4.0, 0.0

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def gaussian_log_prob(value: torch.Tensor, mean: torch.Tensor,
                      log_std: torch.Tensor) -> torch.Tensor:
    """``(N,)`` joint log density of a diagonal Gaussian.

    Written out rather than taken from ``torch.distributions`` so sampling can
    accept an explicit generator -- rollouts must be reproducible from a seed --
    and so the exact expression the ratio depends on is visible here.
    """
    return (-0.5 * ((value - mean) / log_std.exp()).square()
            - log_std - _HALF_LOG_2PI).sum(dim=-1)


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """Differential entropy in nats. State-independent, hence a scalar."""
    return (log_std + _HALF_LOG_2PI + 0.5).sum()


class NavCritic(nn.Module):
    """``(N, OBS_DIM) -> (N,)`` state value, in the reward's own units."""

    def __init__(self, config: NavPolicyConfig | None = None):
        super().__init__()
        self.config = config or NavPolicyConfig()
        (self.lidar_conv, self.lidar_projection, self.goal_encoder,
         self.action_encoder, self.trunk) = build_encoder(self.config)
        self.value_head = nn.Linear(self.config.trunk[-1], 1)

        self.apply(initialize_layer)
        # Unit gain on the output layer. The actor's near-zero head exists to
        # keep tanh unsaturated at step one; a value function has no such
        # boundary to stay away from, and starting it at exactly zero only
        # delays the warmup it is about to run.
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value_head(encode(self, observation)).squeeze(-1)


class NavActorCritic(nn.Module):
    """A deployable actor, a discardable critic, and two exploration scales."""

    def __init__(self, actor: NavPolicy | None = None,
                 config: NavPolicyConfig | None = None,
                 init_std: float = DEFAULT_INIT_STD):
        super().__init__()
        if init_std <= 0.0:
            raise ValueError("init_std must be positive")
        self.actor = actor if actor is not None else NavPolicy(config)
        self.critic = NavCritic(self.actor.config)
        self.log_std = nn.Parameter(
            torch.full((2,), math.log(init_std), dtype=torch.float32))

    @property
    def config(self) -> NavPolicyConfig:
        return self.actor.config

    def actor_parameters(self):
        """Everything the clipped surrogate is allowed to move.

        The exploration scale belongs here and not with the critic: during
        critic warmup the actor's learning rate is exactly zero, and a std that
        kept shrinking through those iterations would quietly narrow the data
        distribution the critic is being fitted on.
        """
        return list(self.actor.parameters()) + [self.log_std]

    def clamp_log_std(self) -> None:
        with torch.no_grad():
            self.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)

    def act(self, observation: torch.Tensor,
            generator: torch.Generator | None = None):
        """Sample one step. Returns ``(raw, action, log_prob, value)``.

        ``raw`` is what the buffer stores and the ratio is computed from;
        ``action`` is the normalized command the environment consumes. Both are
        needed because they are not recoverable from each other once tanh
        saturates.
        """
        mean = self.actor.pre_squash(observation)
        raw = mean + self.log_std.exp() * torch.randn(
            mean.shape, generator=generator, device=mean.device,
            dtype=mean.dtype)
        return (raw, torch.tanh(raw),
                gaussian_log_prob(raw, mean, self.log_std),
                self.critic(observation))

    def evaluate(self, observation: torch.Tensor, raw: torch.Tensor):
        """Re-score stored actions under the current weights.

        Returns ``(log_prob, entropy, value)``. At the first minibatch of an
        update these log probabilities equal the stored ones exactly, which is
        why every PPO ratio starts at 1.
        """
        mean = self.actor.pre_squash(observation)
        return (gaussian_log_prob(raw, mean, self.log_std),
                gaussian_entropy(self.log_std),
                self.critic(observation))

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        """The normalized command that will be deployed: no exploration noise."""
        return torch.tanh(self.actor.pre_squash(observation))

    def physical_action(self, observation: torch.Tensor) -> torch.Tensor:
        """SI ``(v, omega)``; identical to calling the exported actor."""
        return physical_action(self.actor.pre_squash(observation),
                               self.config.v_max, self.config.omega_max)
