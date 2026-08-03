"""Proximal Policy Optimization, and the careful handoff from behaviour cloning.

Everything here is simulator-free and runs on a tensor of numbers, so the
algorithm is testable without Isaac; the rollout loop that produces those
numbers lives in ``scripts/train_ppo.py`` because only it needs the simulator.

Why RL at all, given a 49%-accurate clone: behaviour cloning can imitate the
oracle and nothing else. It has no notion of *reward*, so it never learns that a
collision is bad -- only that the expert did not have one. The measured failure
is exactly that shape: the policy drives too fast into corners because the
uniform imitation objective is dominated by long, easy, open-space trajectories.
A progress-minus-collision return makes those states expensive in the objective
rather than merely rare in the data.

PPO in three lines. The policy gradient increases the probability of actions
that beat the critic's expectation, ``E[grad log pi(a|s) * A(s,a)]``. Taking a
large step invalidates the data, because it was collected under the old policy.
So PPO optimizes ``min(r*A, clip(r, 1-eps, 1+eps)*A)`` with
``r = pi_new(a|s) / pi_old(a|s)``: inside the trust region the gradient is the
ordinary one, outside it the objective flattens and there is nothing to gain by
leaving.

**The handoff is most of this file, and it is not optional.** Naive BC-init plus
PPO reliably destroys a good actor, by a specific mechanism: the critic starts
random, so advantages are noise, so the first few hundred policy-gradient steps
are large random updates to carefully fitted weights. By the time the critic is
useful the actor is no better than scratch. Three mitigations, all load bearing:

1. **Critic warmup with the actor learning rate at exactly zero** -- not small.
   A small rate still accumulates damage over hundreds of iterations of pure
   noise.
2. **An annealed anchor to a frozen BC snapshot.** Frozen matters: anchoring to
   a moving copy of the live policy is a no-op regularizer. The anneal clock
   starts *after* warmup, or the anchor would decay to nothing while the actor
   was still frozen.
3. **A conservative trust region**, clip 0.1 and actor LR 1e-4 against the usual
   0.2 and 3e-4, plus a hard KL early stop. This fine-tunes a good policy; it
   does not train one from scratch.

The two failure signatures to watch, both reported every iteration, are entropy
collapse and rollout success falling below the BC baseline. Either means the
anneal is too fast.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

from ..models import NavActorCritic, NavPolicy
from ..spec import (MOTION_PRIOR_SOURCE, SENSOR_CONTRACT_VERSION, fingerprint,
                    observation_fingerprint)
from .bc import atomic_torch_save

__all__ = ["EpisodeStats", "PPOConfig", "PPOTrainer", "RolloutBuffer",
           "clipped_surrogate", "load_ppo_checkpoint", "ppo_checkpoint",
           "save_ppo_checkpoint"]


@dataclass(frozen=True)
class PPOConfig:
    """Defaults fine-tune a BC checkpoint on a 24 GB 4090."""

    iterations: int = 300
    #: Steps per environment per iteration. With 2,048 environments this is
    #: 65,536 transitions -- two orders of magnitude more than a CPU PPO gets,
    #: which is the entire reason the simulator is on the GPU.
    num_steps: int = 32

    #: 0.99 at 20 Hz is an effective horizon near 100 steps, or five seconds.
    #: That is the timescale over which this policy is being asked to plan.
    gamma: float = 0.99
    #: The standard bias/variance compromise: 0 is the one-step TD error, 1 is
    #: the full Monte-Carlo return minus the baseline.
    gae_lambda: float = 0.95

    clip_range: float = 0.1
    epochs: int = 4
    minibatches: int = 4
    actor_lr: float = 1.0e-4
    #: Ten times the actor's. The critic starts from noise and has no fitted
    #: behaviour to protect, and every iteration it stays wrong is an iteration
    #: of noisy advantages.
    critic_lr: float = 1.0e-3
    value_coef: float = 0.5
    #: Off by default. The annealed anchor and the 0.1 clip are already the
    #: trust region, and at fine-tuning scale an entropy bonus mostly inflates
    #: log_std against them. Raise it if reported entropy collapses.
    entropy_coef: float = 0.0
    max_kl: float = 0.02
    grad_clip: float = 1.0

    #: Iterations of value-only training before the actor moves at all.
    #:
    #: Deliberately not the 500 that a 1,024-transition CPU rollout needed: at
    #: 65,536 transitions per iteration, 25 iterations is 1.6M transitions of
    #: critic fitting, and copying the old number would burn 30M environment
    #: steps before the policy was allowed to learn anything.
    critic_warmup_iters: int = 25
    anchor_coef: float = 1.0
    #: Fraction of the post-warmup run over which the anchor decays to zero.
    anchor_anneal_fraction: float = 0.3

    init_std: float = 0.15
    seed: int = 0
    log_every: int = 1
    checkpoint_every: int = 10

    def __post_init__(self):
        if self.iterations < 1 or self.num_steps < 1:
            raise ValueError("iterations and num_steps must be positive")
        if self.epochs < 1 or self.minibatches < 1:
            raise ValueError("epochs and minibatches must be positive")
        if not 0.0 < self.clip_range < 1.0:
            raise ValueError("clip_range must lie in (0, 1)")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.critic_warmup_iters < 0 or self.anchor_coef < 0.0:
            raise ValueError("warmup and anchor coefficients cannot be negative")
        if not 0.0 < self.anchor_anneal_fraction <= 1.0:
            raise ValueError("anchor_anneal_fraction must lie in (0, 1]")
        if self.max_kl <= 0.0:
            raise ValueError("max_kl must be positive")


class RolloutBuffer:
    """One iteration of on-policy experience, preallocated on the device.

    On-policy means this is discarded after every update: data collected under
    policy *k* is only valid for updating policy *k*. That is why PPO is
    sample-hungry and why thousands of parallel environments help so much.

    Storage is ``(T, N, ...)`` and allocated once. At 32 steps x 2,048
    environments the observations are 192 MB; nothing here grows during a run.
    """

    def __init__(self, num_steps: int, num_envs: int, observation_dim: int,
                 device: torch.device):
        shape = (num_steps, num_envs)
        make = lambda *tail: torch.zeros(shape + tail, device=device)
        self.observation = make(observation_dim)
        #: Pre-squash actions: the coordinates the Gaussian is defined in, and
        #: therefore the only ones whose stored log probability is reusable.
        self.raw_action = make(2)
        self.log_prob = make()
        self.value = make()
        self.reward = make()
        #: 1.0 where the episode ended, terminal or truncated. This is what
        #: keeps value from leaking across an episode boundary into a fresh,
        #: unrelated layout.
        self.done = make()
        self.advantage = make()
        self.returns = make()
        self.num_steps, self.num_envs = num_steps, num_envs
        self.cursor = 0

    def add(self, observation, raw_action, log_prob, value, reward,
            done) -> None:
        index = self.cursor
        if index >= self.num_steps:
            raise RuntimeError("rollout buffer is full; compute and reset it")
        self.observation[index] = observation
        self.raw_action[index] = raw_action
        self.log_prob[index] = log_prob
        self.value[index] = value
        self.reward[index] = reward
        self.done[index] = done.float() if done.dtype == torch.bool else done
        self.cursor = index + 1

    def compute_advantages(self, last_value: torch.Tensor, gamma: float,
                           lam: float) -> None:
        """Generalized Advantage Estimation, walked backwards through time.

        ``delta`` is the TD error -- reward plus discounted next value, minus
        this value -- which reads as "how much better did this step go than the
        critic predicted". GAE exponentially averages those over horizons.
        """
        if self.cursor != self.num_steps:
            raise RuntimeError("rollout is incomplete")
        advantage = torch.zeros_like(last_value)
        for step in reversed(range(self.num_steps)):
            next_value = (last_value if step == self.num_steps - 1
                          else self.value[step + 1])
            not_done = 1.0 - self.done[step]
            delta = (self.reward[step] + gamma * next_value * not_done
                     - self.value[step])
            advantage = delta + gamma * lam * not_done * advantage
            self.advantage[step] = advantage
        # The critic regresses onto the same estimate the actor is using, so a
        # value-function improvement and an advantage improvement cannot
        # disagree about what actually happened.
        self.returns = self.advantage + self.value

    def batches(self, count: int, generator: torch.Generator | None = None
                    ) -> Iterator[tuple[torch.Tensor, ...]]:
        """Shuffled flat views. Advantages are normalized over the whole batch.

        Normalizing makes the effective step size independent of reward scale:
        without it, doubling the progress weight in the environment would
        silently double the policy step. Over the batch rather than per
        minibatch, so the statistics do not themselves become noise.
        """
        total = self.num_steps * self.num_envs
        if count > total:
            raise ValueError("more minibatches than transitions")
        flat = lambda t: t.reshape(total, *t.shape[2:])
        advantage = flat(self.advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1.0e-8)
        tensors = (flat(self.observation), flat(self.raw_action),
                   flat(self.log_prob), advantage, flat(self.returns))
        order = torch.randperm(total, generator=generator,
                               device=self.observation.device)
        for chunk in order.chunk(count):
            yield tuple(tensor[chunk] for tensor in tensors)

    def reset(self) -> None:
        self.cursor = 0


class EpisodeStats:
    """Outcomes of the episodes that *finished* inside one rollout.

    Kept on the device and summarized once per iteration: the point of this
    class is that reading success rate must not cost a GPU synchronization on
    every one of the 32 steps.

    Only completed episodes are counted. The ones still running at the end of
    the rollout are still running at the start of the next one, so nothing is
    lost by ignoring them here.
    """

    def __init__(self, num_envs: int, device: torch.device):
        self._running_return = torch.zeros(num_envs, device=device)
        self._running_length = torch.zeros(num_envs, device=device)
        # episodes, successes, collisions, timeouts, return, length
        self._totals = torch.zeros(6, device=device)

    def update(self, reward: torch.Tensor, terminated: torch.Tensor,
               truncated: torch.Tensor, reached: torch.Tensor,
               collided: torch.Tensor) -> None:
        self._running_return += reward
        self._running_length += 1.0
        done = (terminated | truncated).float()
        self._totals += torch.stack((
            done.sum(),
            (done * reached.float()).sum(),
            (done * collided.float()).sum(),
            (truncated & ~terminated).float().sum(),
            (done * self._running_return).sum(),
            (done * self._running_length).sum(),
        ))
        keep = 1.0 - done
        self._running_return *= keep
        self._running_length *= keep

    def summary(self) -> dict[str, float]:
        totals = self._totals.double().cpu()
        episodes = max(1.0, float(totals[0]))
        return {
            "episodes": float(totals[0]),
            "success_rate": float(totals[1]) / episodes,
            "collision_rate": float(totals[2]) / episodes,
            "timeout_rate": float(totals[3]) / episodes,
            "episode_return": float(totals[4]) / episodes,
            "episode_length": float(totals[5]) / episodes,
        }

    def reset(self) -> None:
        self._totals.zero_()


def clipped_surrogate(log_prob: torch.Tensor, old_log_prob: torch.Tensor,
                      advantage: torch.Tensor, clip_range: float):
    """The PPO objective. Returns ``(loss, clip_fraction, approx_kl)``.

    Negated because optimizers minimize. ``approx_kl`` is the low-variance,
    always-positive k3 estimator rather than the mean log-ratio difference,
    which is noisy enough around zero to trip an early stop by accident.
    """
    ratio = torch.exp(log_prob - old_log_prob)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantage
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        log_ratio = log_prob - old_log_prob
        clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
    return loss, clip_fraction, approx_kl


class PPOTrainer:
    """Owns the optimizer, the warmup schedule and the anchor anneal."""

    def __init__(self, policy: NavActorCritic, config: PPOConfig,
                 anchor: NavPolicy | None = None, *,
                 device: torch.device | None = None):
        self.policy = policy
        self.config = config
        self.device = device or next(policy.parameters()).device
        # Two groups so the actor's rate can be driven to exactly zero during
        # warmup while the critic keeps training. Group 0 is always the actor.
        #
        # Adam, not the AdamW that BC uses: decoupled weight decay pulls the
        # cloned weights toward zero every step, which is precisely the drift
        # the anchor below exists to prevent.
        self.optimizer = torch.optim.Adam([
            {"params": policy.actor_parameters(), "lr": config.actor_lr},
            {"params": list(policy.critic.parameters()), "lr": config.critic_lr},
        ])
        self.anchor = anchor
        if anchor is not None:
            if anchor is policy.actor:
                raise ValueError(
                    "the anchor must be a frozen snapshot, not the live actor; "
                    "anchoring to a moving copy regularizes nothing and freezing "
                    "this one would stop the actor training entirely")
            # A frozen snapshot, never the live actor. Anchoring to something
            # that moves with the policy regularizes nothing at all.
            anchor.eval().requires_grad_(False)
        self.iteration = 0
        self._generator = torch.Generator(device=self.device.type)
        self._generator.manual_seed(config.seed + 1)

    # ── schedules ────────────────────────────────────────────────────────────

    @property
    def warming_up(self) -> bool:
        return self.iteration < self.config.critic_warmup_iters

    def anchor_weight(self) -> float:
        """Linear decay to zero over the first slice of *post-warmup* training."""
        if self.anchor is None or self.config.anchor_coef <= 0.0:
            return 0.0
        if self.warming_up:
            return self.config.anchor_coef
        span = max(1.0, self.config.anchor_anneal_fraction
                   * (self.config.iterations - self.config.critic_warmup_iters))
        progress = (self.iteration - self.config.critic_warmup_iters) / span
        return self.config.anchor_coef * max(0.0, 1.0 - progress)

    # ── the update ───────────────────────────────────────────────────────────

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """One iteration of epochs over the rollout. Returns loggable scalars."""
        config = self.config
        warming_up = self.warming_up
        # Literally zero, not merely small: during warmup the advantages are
        # meaningless and any actor step is pure damage.
        self.optimizer.param_groups[0]["lr"] = (
            0.0 if warming_up else config.actor_lr)
        beta = 0.0 if warming_up else self.anchor_weight()

        # How much of the return's variance the critic accounts for, measured
        # before this update changes it. Raw value loss cannot answer that: it
        # is in squared return units, so a well-fitted critic on a
        # high-variance reward sits at a flat, large-looking number forever.
        # A frozen-actor validation run held value_loss at ~1.6 while the
        # returns it was predicting spanned -10 to +30.
        with torch.no_grad():
            residual = (buffer.returns - buffer.value).var()
            explained = 1.0 - residual / buffer.returns.var().clamp_min(1.0e-8)

        totals = torch.zeros(6, device=self.device)
        updates = 0
        stopped_early = False
        for _ in range(config.epochs):
            for (observation, raw_action, old_log_prob, advantage,
                 returns) in buffer.batches(config.minibatches,
                                            self._generator):
                log_prob, entropy, value = self.policy.evaluate(
                    observation, raw_action)
                policy_loss, clip_fraction, approx_kl = clipped_surrogate(
                    log_prob, old_log_prob, advantage, config.clip_range)
                value_loss = (value - returns).square().mean()

                loss = (policy_loss
                        + config.value_coef * value_loss
                        - config.entropy_coef * entropy)
                anchor_loss = torch.zeros((), device=self.device)
                if beta > 0.0:
                    with torch.no_grad():
                        reference = self.anchor.pre_squash(observation)
                    anchor_loss = (self.policy.actor.pre_squash(observation)
                                   - reference).square().mean()
                    loss = loss + beta * anchor_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         config.grad_clip)
                self.optimizer.step()
                self.policy.clamp_log_std()

                totals += torch.stack((
                    policy_loss.detach(), value_loss.detach(),
                    entropy.detach(), clip_fraction, approx_kl,
                    anchor_loss.detach()))
                updates += 1

            # A backstop for when clipping is not enough. Checked between
            # epochs rather than mid-epoch so every minibatch of a pass gets
            # the same policy, and the surviving statistics stay comparable.
            if float(approx_kl) > config.max_kl:
                stopped_early = True
                break

        self.iteration += 1
        means = (totals / max(1, updates)).double().cpu()
        log_std = self.policy.log_std.detach().double().cpu()
        return {
            "iteration": self.iteration,
            "policy_loss": float(means[0]),
            "value_loss": float(means[1]),
            "explained_variance": float(explained),
            "entropy": float(means[2]),
            "clip_fraction": float(means[3]),
            "approx_kl": float(means[4]),
            "anchor_loss": float(means[5]),
            "anchor_weight": beta,
            "actor_lr": self.optimizer.param_groups[0]["lr"],
            "std_v": float(log_std[0].exp()),
            "std_omega": float(log_std[1].exp()),
            "warming_up": bool(warming_up),
            "kl_early_stop": bool(stopped_early),
        }


def ppo_checkpoint(policy: NavActorCritic, trainer: PPOTrainer,
                   metrics: dict[str, float] | None = None) -> dict[str, Any]:
    """Checkpoint that ``load_policy_checkpoint`` can read.

    The actor's weights sit under ``model`` with the same fingerprints a BC run
    writes, so ``scripts/eval_bc.py``, DAgger collection and the export path all
    accept a PPO checkpoint without knowing one was produced. The critic,
    exploration scale and optimizer ride along under their own keys for resume,
    and are ignored by every consumer that only wants an actor.
    """
    return {
        "format_version": 1,
        "algorithm": "ppo",
        "spec_fingerprint": fingerprint(),
        "sensor_contract_version": SENSOR_CONTRACT_VERSION,
        "motion_prior_source": MOTION_PRIOR_SOURCE,
        "observation_fingerprint": observation_fingerprint(
            policy.config.history_frames),
        "model_config": policy.config.to_dict(),
        "training_config": asdict(trainer.config),
        "model": policy.actor.state_dict(),
        "critic": policy.critic.state_dict(),
        "log_std": policy.log_std.detach().cpu(),
        "optimizer": trainer.optimizer.state_dict(),
        "iteration": trainer.iteration,
        "metrics": dict(metrics or {}),
    }


def save_ppo_checkpoint(path: str | Path, policy: NavActorCritic,
                        trainer: PPOTrainer,
                        metrics: dict[str, float] | None = None) -> None:
    atomic_torch_save(ppo_checkpoint(policy, trainer, metrics), Path(path))


def load_ppo_checkpoint(path: str | Path, policy: NavActorCritic,
                        trainer: PPOTrainer | None = None,
                        device="cpu") -> dict[str, Any]:
    """Restore actor, critic, exploration scale and optimizer for a resume."""
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("sensor_contract_version", "world-fixed-v1") != SENSOR_CONTRACT_VERSION:
        raise ValueError("checkpoint uses a different sensor contract")
    if saved.get("spec_fingerprint") != fingerprint():
        raise ValueError("checkpoint uses a different spec")
    want = observation_fingerprint(policy.config.history_frames)
    if saved.get("observation_fingerprint") != want:
        raise ValueError("checkpoint uses a different observation layout")
    policy.actor.load_state_dict(saved["model"])
    policy.critic.load_state_dict(saved["critic"])
    with torch.no_grad():
        policy.log_std.copy_(saved["log_std"].to(policy.log_std.device))
    if trainer is not None:
        trainer.optimizer.load_state_dict(saved["optimizer"])
        trainer.iteration = int(saved["iteration"])
    return saved
