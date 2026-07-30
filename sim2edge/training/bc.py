"""Optimized, resumable behaviour cloning over the shard corpus.

There is intentionally no trainer framework here. The hot path is three clear
operations--load a shard, build freshly corrupted observations, imitate the
oracle--and the checkpoint contains every piece of state needed to resume it.
This keeps the training/deployment contract visible and avoids adapting a 104 GB
sequential corpus to an abstraction designed for millions of tiny files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from ..models import NavPolicy, NavPolicyConfig
from ..spec import OMEGA_MAX, V_MAX, fingerprint
from .data import ShardCorpus, ShardInfo

__all__ = ["BCConfig", "imitation_loss", "load_policy_checkpoint",
           "normalize_actions", "train_bc"]


@dataclass(frozen=True)
class BCConfig:
    """Defaults target a 24 GB RTX 4090 and the 124.5M-sample corpus."""

    epochs: int = 2
    batch_size: int = 8192
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    warmup_steps: int = 500
    weight_decay: float = 1.0e-5
    omega_weight: float = 1.5
    grad_clip: float = 1.0
    train_noise_scale: float = 1.0
    validation_noise_scale: float = 1.0
    val_shards: int = 4
    seed: int = 0
    amp: str = "bf16"
    compile_model: bool = True
    prefetch: bool = True
    log_every: int = 100
    checkpoint_every_shards: int = 1
    max_train_shards: int | None = None

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not 0.0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("min_learning_rate must be between zero and learning_rate")
        if self.omega_weight <= 0.0:
            raise ValueError("omega_weight must be positive")
        if self.amp not in ("bf16", "off"):
            raise ValueError("amp must be 'bf16' or 'off'")


def normalize_actions(actions: torch.Tensor) -> torch.Tensor:
    """Physical commands -> two comparable coordinates in ``[-1, 1]``."""
    return torch.cat((
        2.0 * actions[:, :1] / V_MAX - 1.0,
        actions[:, 1:] / OMEGA_MAX,
    ), dim=1)


def imitation_loss(predicted: torch.Tensor, target: torch.Tensor,
                   omega_weight: float = 1.5):
    """Weighted MSE in normalized action space, plus transparent components."""
    error = (normalize_actions(predicted.float())
             - normalize_actions(target.float())).square()
    v_loss = error[:, 0].mean()
    omega_loss = error[:, 1].mean()
    total = 0.5 * (v_loss + omega_weight * omega_loss)
    return total, v_loss, omega_loss


def _batch_totals(predicted, target, loss, v_loss, omega_loss):
    """Eight additive diagnostics, kept on-device until a log boundary."""
    absolute = (predicted.detach().float() - target.float()).abs()
    v_saturated = ((predicted[:, 0] < 0.01 * V_MAX)
                   | (predicted[:, 0] > 0.99 * V_MAX)).float().mean()
    omega_saturated = (predicted[:, 1].abs()
                       > 0.99 * OMEGA_MAX).float().mean()
    count = predicted.new_tensor(predicted.shape[0], dtype=torch.float32)
    means = torch.stack((loss.detach(), v_loss.detach(), omega_loss.detach(),
                         absolute[:, 0].mean(), absolute[:, 1].mean(),
                         v_saturated, omega_saturated,
                         predicted.new_tensor(1.0, dtype=torch.float32)))
    return means * count


def _lr_multiplier(step: int, total_steps: int, config: BCConfig) -> float:
    minimum = config.min_learning_rate / config.learning_rate
    if step < config.warmup_steps:
        return max(1.0 / max(1, config.warmup_steps),
                   (step + 1) / max(1, config.warmup_steps))
    progress = ((step - config.warmup_steps)
                / max(1, total_steps - config.warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _batch_seed(seed: int, epoch: int, info: ShardInfo, tier: int = 0) -> int:
    # Python's hash is salted per process. This small stable hash makes shuffle
    # and corruption reproducible across restarts without serializing RNG state.
    name = sum((index + 1) * byte for index, byte in
               enumerate(info.path.name.encode()))
    return (seed * 1_000_003 + epoch * 10_007 + tier * 101 + name) % (2**63 - 1)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device.type).manual_seed(seed)


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _checkpoint(policy: NavPolicy, optimizer, scheduler, config: BCConfig,
                corpus: ShardCorpus, *, epoch: int, shard_cursor: int,
                global_step: int, best_validation: float) -> dict[str, Any]:
    return {
        "format_version": 1,
        "spec_fingerprint": fingerprint(),
        "model_config": policy.config.to_dict(),
        "training_config": asdict(config),
        "dataset_digest": corpus.digest,
        "dataset": corpus.as_dict(),
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "shard_cursor": shard_cursor,
        "global_step": global_step,
        "best_validation": best_validation,
    }


def load_policy_checkpoint(path: str | Path, device="cpu") -> NavPolicy:
    """Load an inference policy and reject a stale observation contract."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    got, want = checkpoint.get("spec_fingerprint"), fingerprint()
    if got != want:
        raise ValueError(
            f"checkpoint {path} uses spec {got}, this process is {want}"
        )
    config = NavPolicyConfig.from_dict(checkpoint["model_config"])
    policy = NavPolicy(config).to(device)
    policy.load_state_dict(checkpoint["model"])
    return policy.eval()


@torch.inference_mode()
def _validate(policy, corpus: ShardCorpus, config: BCConfig,
              device: torch.device) -> dict[str, float]:
    if not corpus.validation:
        return {}
    policy.eval()
    scales = (0.0, config.validation_noise_scale)
    totals = {scale: torch.zeros(8, device=device) for scale in scales}

    for info, shard in corpus.loaded(corpus.validation, device,
                                     prefetch=config.prefetch):
        for tier, scale in enumerate(scales):
            generator = _generator(
                device, _batch_seed(config.seed, 0, info, tier + 1))
            for observation, action in shard.batches(
                    config.batch_size, noise_scale=scale, shuffle=False,
                    generator=generator):
                with torch.autocast(device_type=device.type,
                                    dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda"
                                             and config.amp == "bf16")):
                    predicted = policy(observation)
                loss, v_loss, omega_loss = imitation_loss(
                    predicted, action, config.omega_weight)
                totals[scale] += _batch_totals(
                    predicted, action, loss, v_loss, omega_loss)

    metrics: dict[str, float] = {}
    for scale, device_values in totals.items():
        values = device_values.double().cpu().numpy()
        suffix = "clean" if scale == 0.0 else "noisy"
        count = max(1.0, values[7])
        metrics[f"validation_{suffix}_loss"] = float(values[0] / count)
        metrics[f"validation_{suffix}_v_mse"] = float(values[1] / count)
        metrics[f"validation_{suffix}_omega_mse"] = float(values[2] / count)
        metrics[f"validation_{suffix}_v_mae_mps"] = float(values[3] / count)
        metrics[f"validation_{suffix}_omega_mae_radps"] = float(values[4] / count)
        metrics[f"validation_{suffix}_v_saturation"] = float(values[5] / count)
        metrics[f"validation_{suffix}_omega_saturation"] = float(values[6] / count)
    return metrics


def train_bc(data: str | Path, output: str | Path,
             config: BCConfig | None = None, *, device: str | None = None,
             resume: str | Path | None = None) -> dict[str, Any]:
    """Train and return the final checkpoint metadata.

    Checkpoints are written only at shard boundaries, so a power loss replays at
    most one shard. Shard order and noise are derived from stable seeds, making
    that replay exact even though training corruption is fresh every epoch.
    """
    config = config or BCConfig()
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available()
                                         else "cpu"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None:
        occupied = [name for name in ("latest.pt", "final.pt", "metrics.jsonl")
                    if (output / name).exists()]
        if occupied:
            raise FileExistsError(
                f"run directory {output} already contains {', '.join(occupied)}; "
                f"pass resume=... or choose a new output directory"
            )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    corpus = ShardCorpus(data, val_shards=config.val_shards, seed=config.seed,
                         max_train_shards=config.max_train_shards)
    steps_per_epoch = sum(info.samples // config.batch_size
                          for info in corpus.train)
    if steps_per_epoch == 0:
        raise ValueError("batch_size is larger than every training shard")
    total_steps = steps_per_epoch * config.epochs

    policy = NavPolicy().to(device_obj)
    optimizer_kwargs = {"lr": config.learning_rate,
                        "weight_decay": config.weight_decay}
    if device_obj.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(policy.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_multiplier(step, total_steps, config))

    start_epoch = start_cursor = global_step = 0
    best_validation = float("inf")
    if resume is not None:
        saved = torch.load(resume, map_location=device_obj, weights_only=False)
        if saved.get("spec_fingerprint") != fingerprint():
            raise ValueError("resume checkpoint uses a different spec")
        if saved.get("dataset_digest") != corpus.digest:
            raise ValueError("resume checkpoint uses a different dataset split")
        if saved.get("training_config") != asdict(config):
            raise ValueError("resume requires the original training configuration")
        policy.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"])
        start_cursor = int(saved["shard_cursor"])
        global_step = int(saved["global_step"])
        best_validation = float(saved["best_validation"])

    if start_epoch >= config.epochs:
        # A completed checkpoint is already a valid final artifact. Treating a
        # harmless repeated resume as an undefined zero-epoch run is surprising.
        _atomic_torch_save(saved, output / "final.pt")
        print(f"[BC] checkpoint already completed {config.epochs} epoch(s)")
        return saved

    runner = policy
    if config.compile_model and hasattr(torch, "compile"):
        runner = torch.compile(policy, dynamic=True)

    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    (output / "dataset.json").write_text(
        json.dumps({**corpus.as_dict(), "digest": corpus.digest},
                   indent=2, sort_keys=True) + "\n")
    metrics_path = output / "metrics.jsonl"

    print(f"[BC] device={device_obj}  parameters={policy.parameter_count:,}  "
          f"batch={config.batch_size:,}  amp={config.amp}  "
          f"compile={config.compile_model}")
    print(f"[BC] train={len(corpus.train)} shards / "
          f"{corpus.train_samples:,} samples  validation="
          f"{len(corpus.validation)} shards / {corpus.validation_samples:,} "
          f"samples")
    print(f"[BC] {steps_per_epoch:,} updates/epoch, {total_steps:,} total")
    if resume is not None:
        print(f"[BC] resumed epoch {start_epoch + 1}, shard {start_cursor}, "
              f"step {global_step:,}")

    latest = output / "latest.pt"
    run_started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs):
        order = corpus.train_order(epoch)
        cursor = start_cursor if epoch == start_epoch else 0
        order = order[cursor:]
        policy.train()
        epoch_sum = torch.zeros(8, device=device_obj)
        window_sum = torch.zeros(8, device=device_obj)
        window_started = time.perf_counter()

        for local_index, (info, shard) in enumerate(
                corpus.loaded(order, device_obj, prefetch=config.prefetch)):
            shard_index = cursor + local_index
            generator = _generator(
                device_obj, _batch_seed(config.seed, epoch, info))
            for observation, action in shard.batches(
                    config.batch_size, noise_scale=config.train_noise_scale,
                    shuffle=True, generator=generator, drop_last=True):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device_obj.type,
                                    dtype=torch.bfloat16,
                                    enabled=(device_obj.type == "cuda"
                                             and config.amp == "bf16")):
                    predicted = runner(observation)
                    loss, v_loss, omega_loss = imitation_loss(
                        predicted, action, config.omega_weight)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                global_step += 1

                values = _batch_totals(
                    predicted, action, loss, v_loss, omega_loss)
                epoch_sum += values
                window_sum += values
                if config.log_every and global_step % config.log_every == 0:
                    elapsed = time.perf_counter() - window_started
                    logged = window_sum.double().cpu().numpy()
                    samples = logged[7]
                    print(f"[BC] epoch {epoch + 1}/{config.epochs}  "
                          f"step {global_step:,}  "
                          f"loss {logged[0] / samples:.5f}  "
                          f"v {logged[1] / samples:.5f}  "
                          f"omega {logged[2] / samples:.5f}  "
                          f"{samples / elapsed:,.0f} samples/s  "
                          f"lr {scheduler.get_last_lr()[0]:.2e}")
                    window_sum.zero_()
                    window_started = time.perf_counter()

            completed = shard_index + 1
            if (config.checkpoint_every_shards > 0
                    and completed % config.checkpoint_every_shards == 0):
                saved = _checkpoint(
                    policy, optimizer, scheduler, config, corpus, epoch=epoch,
                    shard_cursor=completed, global_step=global_step,
                    best_validation=best_validation)
                _atomic_torch_save(saved, latest)

        validation = _validate(runner, corpus, config, device_obj)
        epoch_values = epoch_sum.double().cpu().numpy()
        sample_count = max(1.0, epoch_values[7])
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": float(epoch_values[0] / sample_count),
            "train_v_mse": float(epoch_values[1] / sample_count),
            "train_omega_mse": float(epoch_values[2] / sample_count),
            "train_v_mae_mps": float(epoch_values[3] / sample_count),
            "train_omega_mae_radps": float(epoch_values[4] / sample_count),
            "train_v_saturation": float(epoch_values[5] / sample_count),
            "train_omega_saturation": float(epoch_values[6] / sample_count),
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.perf_counter() - run_started,
            **validation,
        }
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print("[BC] epoch summary  " + "  ".join(
            f"{key}={value:.5f}" for key, value in record.items()
            if key.endswith("loss")))

        clean = float(validation.get(
            "validation_clean_loss", record["train_loss"]))
        if clean < best_validation:
            best_validation = clean
            best = _checkpoint(
                policy, optimizer, scheduler, config, corpus, epoch=epoch + 1,
                shard_cursor=0, global_step=global_step,
                best_validation=best_validation)
            _atomic_torch_save(best, output / "best.pt")

        final = _checkpoint(
            policy, optimizer, scheduler, config, corpus, epoch=epoch + 1,
            shard_cursor=0, global_step=global_step,
            best_validation=best_validation)
        _atomic_torch_save(final, latest)
        start_cursor = 0

    _atomic_torch_save(final, output / "final.pt")
    print(f"[BC] complete in {(time.perf_counter() - run_started) / 3600:.2f} h  "
          f"best clean validation={best_validation:.5f}")
    return final
