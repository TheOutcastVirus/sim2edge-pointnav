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
    batch_size: int = 32
    #: 32 steps is 1.6 s -- long enough that compounding is visible in the loss
    #: and short enough that BPTT memory stays under a gigabyte of activations.
    #: Belief error at this horizon is the number to watch; see the log.
    seq_len: int = 128
    learning_rate: float = 3.0e-4
    #: The actor is already good. Moving it at the head's rate is how a warm
    #: start gets thrown away in the first hundred steps.
    actor_lr_scale: float = 0.1
    freeze_actor: bool = True
    #: Actor-only adaptation starts from a trained causal estimator and teaches
    #: the policy on its accumulated belief distribution without moving that
    #: estimator back toward command shortcuts.
    freeze_estimator: bool = False
    min_learning_rate: float = 3.0e-5
    warmup_steps: int = 200
    weight_decay: float = 1.0e-5
    grad_clip: float = 1.0
    omega_weight: float = 1.5
    action_coef: float = 0.0
    translation_coef: float = 20.0
    belief_coef: float = 1.0
    uncertainty_coef: float = 0.01
    #: Counterfactual scan supervision: if lidar histories are assigned to the
    #: wrong robots, the estimator must reject them and return to command dead
    #: reckoning. Without this term the direct residual head can minimize the
    #: ordinary loss from commands alone and shuffled-lidar ablation is exactly
    #: unchanged.
    mismatched_lidar_coef: float = 20.0
    #: Ranking supervision on the confidence gate, using the counterfactual pair
    #: the mismatched-lidar term already builds. Zero reproduces every existing
    #: run; see the note beside ``confidence_loss`` for why the gate gets no
    #: usable gradient without it.
    confidence_coef: float = 0.0
    #: How far a matched scan's gate must outrank the same step's mismatched one
    #: before the hinge stops pushing, as a fraction of the gate's (0, 1) range.
    confidence_margin: float = 0.5
    full_episodes: bool = True
    validation_batches: int = 32
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
        if self.freeze_actor and self.freeze_estimator:
            raise ValueError("actor and estimator cannot both be frozen")
        for name in ("action_coef", "translation_coef", "belief_coef",
                     "uncertainty_coef", "mismatched_lidar_coef",
                     "confidence_coef"):
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
    recurrent_motion = (policy.config.persistent_calibration
                        or policy.config.motion_state_size > 0)
    rollout = policy.rollout(
        batch.lidar, batch.gyro, batch.previous_action, batch.initial_belief,
        batch.raw_lidar, batch.raw_valid,
        return_motion=recurrent_motion)
    predicted_action, predicted_belief = rollout[:2]

    steps, envs = predicted_action.shape[0], predicted_action.shape[1]
    valid = batch.valid_steps.reshape(steps * envs)
    action_loss, _, _ = imitation_loss(
        predicted_action.reshape(steps * envs, 2)[valid],
        batch.action.reshape(steps * envs, 2)[valid],
        omega_weight=config.omega_weight)

    motion_valid = batch.valid_steps[1:].reshape(-1)
    if config.freeze_estimator:
        # The recurrent rollout above still uses the trained estimator to
        # construct every belief consumed by the actor. Its dense auxiliary
        # views cannot affect any trainable parameter in actor-only mode, and
        # materializing them a second time costs several GB on full episodes.
        translation_loss = action_loss.new_zeros(())
        uncertainty_loss = action_loss.new_zeros(())
        mismatched_lidar_loss = action_loss.new_zeros(())
        confidence_loss = action_loss.new_zeros(())
        scan_confidence_mean = action_loss.new_zeros(())
    else:
        # Step 0 never integrates, so it carries no translation prediction to score.
        if recurrent_motion:
            predicted_translation = rollout[2][1:].reshape(-1, 2)
            predicted_variance = rollout[3][1:].reshape(-1, 2)
            scan_confidence = rollout[4][1:].reshape(-1)
        else:
            predicted_translation, predicted_variance, scan_confidence = policy.motion_estimate(
                batch.lidar[1:].reshape(-1, *batch.lidar.shape[2:]),
                batch.gyro[1:].reshape(-1, batch.gyro.shape[-1]),
                batch.previous_action[1:].reshape(-1, *batch.previous_action.shape[2:]),
                batch.raw_lidar[1:].reshape(-1, *batch.raw_lidar.shape[2:]),
                batch.raw_valid[1:].reshape(-1, *batch.raw_valid.shape[2:]),
            )
        predicted_translation = predicted_translation[motion_valid]
        predicted_variance = predicted_variance[motion_valid]
        translation_target = batch.translation[1:].reshape(-1, 2).float()[motion_valid]
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
        translation_error = predicted_translation.float() - translation_target
        translation_loss = torch.nn.functional.smooth_l1_loss(
            predicted_translation / TRANSLATION_SCALE_M,
            translation_target / TRANSLATION_SCALE_M, beta=0.1)
    # Calibrate the declared variance to observed squared error without allowing
    # the variance branch to move the translation target itself. A log-space
    # robust loss stays non-negative and behaves well across millimetres to cm.
        variance_target = translation_error.detach().square() + 1.0e-7
        uncertainty_loss = torch.nn.functional.smooth_l1_loss(
            predicted_variance.float().log(), variance_target.log(), beta=0.5)

    # A second, counterfactual view closes the shortcut where the head predicts
    # the simulator's mean slip entirely from commands and merely carries lidar
    # along as an unused tensor. Roll along the environment axis so every scan
    # remains temporally coherent but belongs to a different robot. Such a scan
    # provides no evidence about this robot's motion; the only safe answer is
    # the exact command prior. A quarter of valid steps is enough to make that
    # distinction while bounding activation memory on full episodes.
        if recurrent_motion:
            counter_env = torch.arange(
                0, envs, 4, device=batch.lidar.device)
            if counter_env.numel() > 1:
                counter_raw = batch.raw_lidar[:, counter_env].roll(1, dims=1)
                counter_valid_raw = batch.raw_valid[:, counter_env].roll(
                    1, dims=1)
            else:
                shift = max(1, batch.raw_lidar.shape[0] // 2)
                counter_raw = batch.raw_lidar[:, counter_env].roll(
                    shift, dims=0)
                counter_valid_raw = batch.raw_valid[:, counter_env].roll(
                    shift, dims=0)
            counter_sequence, _, counter_gate = policy.motion_rollout(
                batch.lidar[:, counter_env], batch.gyro[:, counter_env],
                batch.previous_action[:, counter_env], counter_raw,
                counter_valid_raw)
            counter_mask = batch.valid_steps[1:, counter_env].reshape(-1)
            counter_translation = counter_sequence[1:].reshape(
                -1, 2)[counter_mask]
            counter_confidence = counter_gate[1:].reshape(-1)[counter_mask]
            paired_confidence = rollout[4][1:, counter_env].reshape(
                -1)[counter_mask]
            counter_action = batch.previous_action[1:, counter_env].reshape(
                -1, *batch.previous_action.shape[2:])[counter_mask]
        else:
            counter_source = batch.raw_lidar[1:].roll(1, dims=1).reshape(
                -1, *batch.raw_lidar.shape[2:])
            counter_valid_source = batch.raw_valid[1:].roll(1, dims=1).reshape(
                -1, *batch.raw_valid.shape[2:])
            counter_index = torch.nonzero(
                motion_valid, as_tuple=False).squeeze(1)[::4]
            paired_confidence = scan_confidence[counter_index]
            counter_translation, _, counter_confidence = policy.motion_estimate(
                batch.lidar[1:].reshape(
                    -1, *batch.lidar.shape[2:])[counter_index],
                batch.gyro[1:].reshape(
                    -1, batch.gyro.shape[-1])[counter_index],
                batch.previous_action[1:].reshape(
                    -1, *batch.previous_action.shape[2:])[counter_index],
                counter_source[counter_index],
                counter_valid_source[counter_index],
            )
            counter_action = batch.previous_action[1:].reshape(
                -1, *batch.previous_action.shape[2:])[counter_index]
        counter_v = counter_action[:, -1, 0] * V_MAX
        counter_prior = torch.stack(
            [counter_v * DT, torch.zeros_like(counter_v)], dim=1)
        mismatched_lidar_loss = torch.nn.functional.smooth_l1_loss(
            counter_translation / TRANSLATION_SCALE_M,
            counter_prior / TRANSLATION_SCALE_M, beta=0.1)
        # The gate cannot learn from the losses above. Every one of them reaches
        # it through ``correction = confidence * proposal``, so its gradient is
        # proportional to a proposal deliberately initialized near zero --
        # measured at 3,000x smaller than the proposal head's own gradient, at
        # either gate bias. The optimizer therefore satisfies the loss by growing
        # the proposal instead and the gate never leaves its initialization,
        # which is exactly the constant 0.13 every trained checkpoint reports.
        #
        # The counterfactual already supplies paired labels for the gate's actual
        # question -- is this scan evidence about *my* motion? A hinge on the
        # separation gives an O(1) gradient that does not pass through the
        # proposal, and asserts no absolute confidence value: a genuinely
        # uninformative real scan is still free to score low, it must merely
        # outrank the same step's mismatched scan.
        confidence_loss = torch.nn.functional.relu(
            config.confidence_margin + counter_confidence - paired_confidence
        ).mean()
        scan_confidence_mean = scan_confidence[motion_valid].mean().detach()

    belief_error = (predicted_belief - batch.belief).float()
    belief_distance_loss = (belief_error / GOAL_TOLERANCE_M).square().sum(dim=-1)
    belief_loss = belief_distance_loss[batch.valid_steps].mean()

    total = (config.action_coef * action_loss
             + config.translation_coef * translation_loss
             + config.belief_coef * belief_loss
             + config.uncertainty_coef * uncertainty_loss
             + config.mismatched_lidar_coef * mismatched_lidar_loss
             + config.confidence_coef * confidence_loss)

    with torch.no_grad():
        distance = belief_error.norm(dim=-1)
        last = batch.valid_steps.long().sum(dim=0).clamp_min(1) - 1
        final_distance = distance.gather(0, last[None]).squeeze(0)
        travelled = (batch.translation.norm(dim=-1) * batch.valid_steps).sum(dim=0)
        parts = {
            "loss": total.detach(),
            "action_loss": action_loss.detach(),
            "translation_loss": translation_loss.detach(),
            "belief_loss": belief_loss.detach(),
            "uncertainty_loss": uncertainty_loss.detach(),
            "mismatched_lidar_loss": mismatched_lidar_loss.detach(),
            "confidence_loss": confidence_loss.detach(),
            "scan_confidence": scan_confidence_mean,
            "belief_error_m": distance[batch.valid_steps].mean(),
            "belief_error_final_m": final_distance.mean(),
            "belief_error_final_p95_m": final_distance.quantile(0.95),
            # Per-metre-travelled drift is the portable number: it is what a
            # published lidar-odometry result quotes, and it does not depend on
            # this window length.
            "drift_per_m": final_distance.mean()
            / travelled.mean().clamp_min(1.0e-6),
        }
    return total, parts


def _parameter_groups(policy: BeliefNavPolicy, config: BeliefConfig):
    """Two groups, so the warm-started actor moves slower than the new head."""
    head = list(policy.odometry.parameters())
    actor = list(policy.actor.parameters())
    for parameter in head:
        parameter.requires_grad_(not config.freeze_estimator)
    for parameter in actor:
        parameter.requires_grad_(not config.freeze_actor)
    if config.freeze_actor:
        return [{"params": head, "lr": config.learning_rate}]
    if config.freeze_estimator:
        return [{"params": actor,
                 "lr": config.learning_rate * config.actor_lr_scale}]
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
        "format_version": 2,
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
