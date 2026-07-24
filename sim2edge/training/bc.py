"""Stage 1: behaviour cloning from the oracle, with DAgger aggregation.

The loss is MSE on the **normalized** action, not the physical one. Physical
``v`` spans 0.30 and ``omega`` spans 3.0, so an MSE on raw units weights angular
error ~100x more than linear error and the policy learns to steer beautifully
while barely modulating speed. Normalizing puts both channels on [-1, 1] and
makes the loss weighting an explicit choice rather than an accident of units.

DAgger is what makes BC deployable here. Pure BC fails by drifting into states
the expert never visited and having no idea what to do there; more expert data
on the expert's own distribution does not fix it. One or two rounds of relabel-
what-the-policy-actually-visits is typically worth more than 10x the pure-BC
data, and is cheap in simulation.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..envs.reference_env import EnvConfig
from ..eval.harness import evaluate
from ..eval.torch_policy import TorchPolicy
from ..models.nav_policy import NavPolicy, build_policy
from ..spec import DEFAULT_SPEC, NavSpec
from .data import Demonstrations, DemoDataset, collect_episodes

__all__ = ["BCConfig", "train_bc", "run_bc_pipeline"]


@dataclass
class BCConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-5
    grad_clip: float = 1.0
    val_fraction: float = 0.05

    #: Relative weight of the angular channel in the normalized-action loss.
    #: Slightly above 1 because heading errors compound over a trajectory while
    #: speed errors mostly do not.
    omega_weight: float = 1.5

    device: str = "cpu"
    seed: int = 0
    log_every: int = 1

    #: DAgger schedule: one entry per round, giving the number of episodes to
    #: collect with the current policy driving.
    dagger_rounds: List[int] = field(default_factory=lambda: [2000, 2000])
    #: Probability the oracle drives during a DAgger round. Decayed per round so
    #: later rounds visit states further from the expert's distribution.
    dagger_beta: List[float] = field(default_factory=lambda: [0.5, 0.25])


def _to_torch(batch: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device)
            for k, v in batch.items()}


def _normalized_actions(actions: torch.Tensor, spec: NavSpec) -> torch.Tensor:
    """Physical (v, omega) -> [-1, 1]^2, vectorized to match ActionSpec."""
    a = spec.action
    v = 2.0 * (actions[:, :1] - a.v_min) / (a.v_max - a.v_min) - 1.0
    w = actions[:, 1:] / a.omega_max
    return torch.clamp(torch.cat([v, w], dim=1), -1.0, 1.0)


def _loss(pred_physical: torch.Tensor, target_physical: torch.Tensor,
          spec: NavSpec, omega_weight: float) -> torch.Tensor:
    pred = _normalized_actions(pred_physical, spec)
    target = _normalized_actions(target_physical, spec)
    err = (pred - target) ** 2
    weight = torch.tensor([1.0, omega_weight], device=err.device)
    return (err * weight).mean()


def train_bc(
    policy: NavPolicy,
    demos: Demonstrations,
    config: Optional[BCConfig] = None,
    spec: NavSpec = DEFAULT_SPEC,
    on_epoch=None,
) -> Dict[str, List[float]]:
    """Fit the actor to the demonstrations. Returns the loss history."""
    cfg = config or BCConfig()
    device = cfg.device
    policy = policy.to(device)

    dataset = DemoDataset(demos, spec)
    train_set, val_set = dataset.split(cfg.val_fraction, seed=cfg.seed)

    # Only actor parameters are trained here. The critic has no supervision
    # signal from demonstrations, and letting the optimizer touch it would just
    # apply weight decay to weights that are about to matter in PPO.
    actor_params = [p for n, p in policy.named_parameters()
                    if not n.startswith(("critic", "critic_lidar"))
                    and n != "log_std"]
    opt = torch.optim.AdamW(actor_params, lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    rng = np.random.default_rng(cfg.seed)

    history: Dict[str, List[float]] = {"train": [], "val": []}
    for epoch in range(cfg.epochs):
        policy.train()
        losses = []
        for batch in train_set.batches(cfg.batch_size, rng=rng, drop_last=True):
            t = _to_torch(batch, device)
            pred = policy(t["depth"], t["lidar"], t["goal"], t["last_action"])
            loss = _loss(pred, t["action"], spec, cfg.omega_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor_params, cfg.grad_clip)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()

        policy.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_set.batches(cfg.batch_size, shuffle=False):
                t = _to_torch(batch, device)
                pred = policy(t["depth"], t["lidar"], t["goal"], t["last_action"])
                val_losses.append(float(_loss(pred, t["action"], spec,
                                              cfg.omega_weight)))

        tr = float(np.mean(losses)) if losses else float("nan")
        va = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train"].append(tr)
        history["val"].append(va)
        if cfg.log_every and (epoch + 1) % cfg.log_every == 0:
            print(f"  epoch {epoch + 1:3d}/{cfg.epochs}  train {tr:.5f}  val {va:.5f}")
        if on_epoch is not None:
            on_epoch(epoch, tr, va)

    return history


def run_bc_pipeline(
    n_episodes: int = 10_000,
    config: Optional[BCConfig] = None,
    env_config: Optional[EnvConfig] = None,
    spec: NavSpec = DEFAULT_SPEC,
    out_dir: str | pathlib.Path = "runs/bc",
    eval_episodes: int = 200,
    policy: Optional[NavPolicy] = None,
) -> Dict[str, object]:
    """Full Stage-1 pipeline: collect -> train -> DAgger rounds -> evaluate.

    Checkpoints after every stage. The best BC checkpoint must stay deployable
    at all times: it is the Gate-B fallback, and PPO is allowed to fail.
    """
    cfg = config or BCConfig()
    env_cfg = env_config or EnvConfig(spec=spec)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    policy = policy or build_policy(spec)
    report: Dict[str, object] = {"stages": []}

    print(f"[BC] collecting {n_episodes} oracle episodes "
          f"(depth={'on' if env_cfg.use_depth else 'off'})")
    t0 = time.time()
    demos, stats = collect_episodes(n_episodes, env_cfg, spec, seed_base=0,
                                    progress_every=max(1, n_episodes // 10))
    print(f"[BC] {len(demos)} steps, {stats['megabytes']:.0f} MB, "
          f"expert SR {stats['success_rate']:.1%}, {time.time() - t0:.0f}s")

    all_demos = demos
    for round_idx, (n_eps, beta) in enumerate(
            zip(cfg.dagger_rounds, cfg.dagger_beta)):
        print(f"[BC] training on {len(all_demos)} samples "
              f"(round {round_idx})")
        history = train_bc(policy, all_demos, cfg, spec)
        metrics = evaluate(TorchPolicy(policy, spec, cfg.device), eval_episodes,
                           env_cfg, spec=spec)
        print(f"[BC] round {round_idx} eval: {metrics.summary()}")
        report["stages"].append({"round": round_idx, "samples": len(all_demos),
                                 "history": history,
                                 "metrics": metrics.to_dict()})
        _save(policy, out / f"bc_round{round_idx}.pt", spec, metrics.to_dict())

        if n_eps <= 0:
            continue
        print(f"[BC] DAgger round {round_idx + 1}: {n_eps} episodes, beta={beta}")
        # The policy drives; the oracle labels. Failures are kept here because
        # the recoveries near a failure are the whole reason for the round.
        extra, extra_stats = collect_episodes(
            n_eps, env_cfg, spec, seed_base=1_000_000 * (round_idx + 1),
            driver=TorchPolicy(policy, spec, cfg.device), beta=beta,
            keep_failures=True,
            progress_every=max(1, n_eps // 5))
        print(f"[BC] DAgger collected {len(extra)} steps "
              f"(policy SR {extra_stats['success_rate']:.1%})")
        all_demos = Demonstrations.concat([all_demos, extra])

    print(f"[BC] final training on {len(all_demos)} samples")
    history = train_bc(policy, all_demos, cfg, spec)
    metrics = evaluate(TorchPolicy(policy, spec, cfg.device), eval_episodes,
                       env_cfg, spec=spec)
    print(f"[BC] final eval: {metrics.summary()}")
    report["final_metrics"] = metrics.to_dict()
    report["final_history"] = history
    report["samples"] = len(all_demos)
    _save(policy, out / "bc_final.pt", spec, metrics.to_dict())
    return report


def _save(policy: NavPolicy, path: pathlib.Path, spec: NavSpec,
          metrics: Optional[Dict] = None) -> None:
    """Checkpoint with the spec fingerprint attached.

    The fingerprint travels with the weights so that export, calibration, and
    the ROS node can all verify they are working with a policy trained under the
    preprocessing contract they are about to apply.
    """
    torch.save({
        "model": policy.state_dict(),
        "spec_fingerprint": spec.fingerprint(),
        "spec": spec.to_dict(),
        "metrics": metrics or {},
    }, path)
    print(f"[BC] saved {path}")
