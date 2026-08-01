"""Sequence training for the goal-belief policy: BPTT over the recurrence.

Separate from ``bc.py`` rather than a flag on it. The two trainers disagree
about what a sample *is* -- ``bc.py`` draws independent rows and can shuffle
them freely, while this one needs episode-contiguous windows and carries state
across them -- and every attempt to express that as an option on one loop ends
with a function whose every second line is an ``if sequence:``.

This is a fine-tuning stage, not a from-scratch run. The actor arrives already
trained (BC, then PPO) and the odometry head arrives at exactly the
dead-reckoning baseline, so the job is to teach one new head while not breaking
an actor that already reaches 94% -- which is why the actor's learning rate
defaults to a tenth of the head's, and why ``freeze_actor`` exists at all.

Three losses, and the balance between them is the main thing to tune:

* **action** -- the ordinary imitation loss, on the oracle's command. Keeps the
  actor honest while the belief underneath it moves.
* **translation** -- per-step supervision on the body-frame displacement. Dense,
  well-conditioned, and lands the gradient on the step that produced the error
  instead of asking the recurrence to assign credit backwards.
* **belief** -- what is actually cared about: the goal offset the actor reads.
  Only this one sees the compounding, so only this one can punish a head whose
  per-step errors are individually small and systematically signed.

Training on the *true* belief trajectory alone would be teacher forcing, and a
teacher-forced belief model is famously fine right up until it runs on its own
predictions. It is avoided here structurally rather than with scheduled
sampling: ``rollout`` integrates the model's own belief for the whole window, so
the belief loss at step 31 is already measured against 31 steps of the model's
own compounding error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn

from ..models import BeliefNavPolicy, BeliefNavPolicyConfig, NavPolicyConfig
from ..spec import (DT, GOAL_TOLERANCE_M, V_MAX, fingerprint,
                    legacy_observation_fingerprint, observation_fingerprint)
from .bc import atomic_torch_save, imitation_loss

__all__ = ["BeliefConfig", "belief_losses", "load_belief_checkpoint",
           "warm_start", "TRANSLATION_SCALE_M"]

#: The distance one full-speed control tick covers. Used only to make the
#: translation loss dimensionless -- see :func:`belief_losses`.
TRANSLATION_SCALE_M = V_MAX * DT


@dataclass(frozen=True)
class BeliefConfig:
    """Defaults sized for a 24 GB RTX 4090 and a warm-started actor."""

    epochs: int = 1
    #: Windows per batch. The effective sample count is this times ``seq_len``.
    batch_size: int = 256
    #: 32 steps is 1.6 s -- long enough that compounding is visible in the loss
    #: and short enough that BPTT memory stays under a gigabyte of activations.
    #: Belief error at this horizon is the number to watch; see the log.
    seq_len: int = 32
    learning_rate: float = 3.0e-4
    #: The actor is already good. Moving it at the head's rate is how a warm
    #: start gets thrown away in the first hundred steps.
    actor_lr_scale: float = 0.1
    freeze_actor: bool = False
    min_learning_rate: float = 3.0e-5
    warmup_steps: int = 200
    weight_decay: float = 1.0e-5
    grad_clip: float = 1.0
    omega_weight: float = 1.5
    action_coef: float = 1.0
    translation_coef: float = 50.0
    belief_coef: float = 1.0
    train_noise_scale: float = 1.0
    validation_noise_scale: float = 1.0
    val_shards: int = 4
    seed: int = 0
    amp: str = "bf16"
    prefetch: bool = True
    prefetch_workers: int = 4
    prefetch_shards: int = 8
    log_every: int = 20
    max_train_shards: int | None = None

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.seq_len < 2:
            raise ValueError(
                "seq_len must be at least 2; a one-step window never "
                "integrates and would train the head on nothing")
        if not 0.0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("min_learning_rate must be in [0, learning_rate]")
        if not 0.0 <= self.actor_lr_scale <= 1.0:
            raise ValueError("actor_lr_scale must be in [0, 1]")
        for name in ("action_coef", "translation_coef", "belief_coef"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.amp not in ("bf16", "off"):
            raise ValueError("amp must be 'bf16' or 'off'")


def belief_losses(policy: BeliefNavPolicy, batch, config: BeliefConfig):
    """Run one window and return ``(total, parts)`` with parts detached.

    ``parts`` carries the three components plus the diagnostics that actually
    say whether this is working: mean belief error in metres at the *end* of the
    window, and the same at the start. The ratio between them is the compounding
    rate, and it is the number that decides whether the horizon needs extending.
    """
    predicted_action, predicted_belief = policy.rollout(
        batch.lidar, batch.gyro, batch.previous_action, batch.initial_belief)

    steps, envs = predicted_action.shape[0], predicted_action.shape[1]
    action_loss, _, _ = imitation_loss(
        predicted_action.reshape(steps * envs, 2),
        batch.action.reshape(steps * envs, 2),
        omega_weight=config.omega_weight)

    # Step 0 never integrates, so it carries no translation prediction to score.
    predicted_translation = policy.translation(
        batch.lidar[1:].reshape(-1, *batch.lidar.shape[2:]),
        batch.gyro[1:].reshape(-1, batch.gyro.shape[-1]),
        batch.previous_action[1:].reshape(-1, *batch.previous_action.shape[2:]),
    )
    # Both geometric losses are divided by a characteristic length before being
    # squared, and that is not cosmetic. In raw metres the translation residual
    # is ~5e-4 m, so its squared loss is ~3e-7 -- against an action loss of
    # ~5e-2. Summed with any sane-looking coefficient the odometry head receives
    # a gradient three thousand times smaller than the actor's and learns
    # nothing, while every printed number looks reasonable. Normalizing makes
    # the coefficients mean what they appear to mean.
    #
    # Translation is scaled by the distance one full-speed tick covers, and the
    # belief by the goal tolerance -- the length at which an error stops being
    # cosmetic and starts losing episodes.
    translation_loss = torch.nn.functional.mse_loss(
        predicted_translation / TRANSLATION_SCALE_M,
        batch.translation[1:].reshape(-1, 2).float() / TRANSLATION_SCALE_M)

    belief_error = (predicted_belief - batch.belief).float()
    belief_loss = (belief_error / GOAL_TOLERANCE_M).square().sum(dim=-1).mean()

    total = (config.action_coef * action_loss
             + config.translation_coef * translation_loss
             + config.belief_coef * belief_loss)

    with torch.no_grad():
        distance = belief_error.norm(dim=-1)
        parts = {
            "loss": total.detach(),
            "action_loss": action_loss.detach(),
            "translation_loss": translation_loss.detach(),
            "belief_loss": belief_loss.detach(),
            "belief_error_m": distance.mean(),
            "belief_error_final_m": distance[-1].mean(),
            "belief_error_final_p95_m": distance[-1].quantile(0.95),
            # Per-metre-travelled drift is the portable number: it is what a
            # published lidar-odometry result quotes, and it does not depend on
            # this window length.
            "drift_per_m": distance[-1].mean() / batch.translation[1:].norm(
                dim=-1).sum(dim=0).mean().clamp_min(1.0e-6),
        }
    return total, parts


def _parameter_groups(policy: BeliefNavPolicy, config: BeliefConfig):
    """Two groups, so the warm-started actor moves slower than the new head."""
    head = list(policy.odometry.parameters())
    actor = list(policy.actor.parameters())
    if config.freeze_actor:
        for parameter in actor:
            parameter.requires_grad_(False)
        return [{"params": head, "lr": config.learning_rate}]
    return [
        {"params": head, "lr": config.learning_rate},
        {"params": actor,
         "lr": config.learning_rate * config.actor_lr_scale},
    ]


def lr_multiplier(step: int, total_steps: int, config: BeliefConfig) -> float:
    """Warmup then cosine, matching ``bc.py`` so schedules stay comparable."""
    minimum = config.min_learning_rate / config.learning_rate
    if step < config.warmup_steps:
        return max(1.0 / max(1, config.warmup_steps),
                   (step + 1) / max(1, config.warmup_steps))
    progress = ((step - config.warmup_steps)
                / max(1, total_steps - config.warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def warm_start(path, device="cpu",
               *, allow_legacy_sensor_model: bool = True) -> BeliefNavPolicy:
    """Build a belief policy around a pretrained ``NavPolicy`` checkpoint.

    Legacy checkpoints are admitted **by default** here, unlike in
    ``load_policy_checkpoint``. That is deliberate and it is the whole point of
    this function: every actor worth warm-starting from predates the spinning
    sensor model, because the sensor model is what this work added. Refusing
    them would leave nothing to warm-start from.

    The mismatch is real -- the actor has never seen a swept scan -- and closing
    it is exactly what the training run does.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    got, want = checkpoint.get("spec_fingerprint"), fingerprint()
    if got != want:
        raise ValueError(f"checkpoint {path} uses spec {got}, this is {want}")
    actor_config = NavPolicyConfig.from_dict(checkpoint["model_config"])
    observed = checkpoint.get("observation_fingerprint")
    legacy = legacy_observation_fingerprint(actor_config.history_frames)
    current = observation_fingerprint(actor_config.history_frames)
    if observed not in (None, legacy, current):
        raise ValueError(
            f"checkpoint {path} has observation contract {observed}, which is "
            f"neither the current {current} nor the pre-sensor-model {legacy}")
    if observed == legacy and not allow_legacy_sensor_model:
        raise ValueError(
            f"checkpoint {path} predates the spinning sensor model; pass "
            f"allow_legacy_sensor_model=True to warm-start from it anyway")
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(actor=actor_config))
    policy.load_actor(checkpoint["model"])
    return policy.to(device)


def belief_checkpoint(policy: BeliefNavPolicy, optimizer, scheduler,
                      config: BeliefConfig, corpus, *, epoch: int,
                      global_step: int, best_validation: float,
                      warm_start_path: str | None) -> dict[str, Any]:
    return {
        "format_version": 1,
        "algorithm": "belief",
        "spec_fingerprint": fingerprint(),
        "observation_fingerprint": observation_fingerprint(
            policy.config.actor.history_frames, belief=True),
        "model_config": policy.config.to_dict(),
        "training_config": asdict(config),
        "warm_start": warm_start_path,
        "dataset_digest": corpus.digest,
        "dataset": corpus.as_dict(),
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation": best_validation,
    }


def load_belief_checkpoint(path, device="cpu") -> BeliefNavPolicy:
    """Load a trained belief policy, refusing a mismatched contract."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    got, want = checkpoint.get("spec_fingerprint"), fingerprint()
    if got != want:
        raise ValueError(f"checkpoint {path} uses spec {got}, this is {want}")
    config = BeliefNavPolicyConfig.from_dict(checkpoint["model_config"])
    observed = checkpoint.get("observation_fingerprint")
    expected = observation_fingerprint(config.actor.history_frames, belief=True)
    if observed is not None and observed != expected:
        raise ValueError(
            f"checkpoint {path} uses observation contract {observed}, this "
            f"process requires {expected}")
    policy = BeliefNavPolicy(config).to(device)
    policy.load_state_dict(checkpoint["model"])
    return policy.eval()
