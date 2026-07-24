"""Stage 2: PPO fine-tuning, with a BC handoff that does not destroy the actor.

Naive BC-init + PPO reliably wrecks a good pretrained policy. The mechanism is
specific: the critic starts random, so its value estimates are noise, so the
advantages are noise, so the first few hundred policy-gradient steps apply large
random updates to a carefully fitted actor. By the time the critic is useful the
actor is no better than random.

Three mitigations, all of which are load-bearing:

1. **Critic warmup.** For the first ``critic_warmup_iters`` iterations the actor
   learning rate is zero and only the value function trains. Advantages are
   noise for exactly as long as it takes the critic to stop producing noise, and
   during that window the actor is not listening.
2. **Annealed BC anchor.** An auxiliary ``beta * ||pi(s) - pi_BC(s)||^2`` term
   holds the actor near the cloned policy, with ``beta`` annealed to zero over
   the first ``anchor_anneal_frac`` of training. The anchor policy is a frozen
   copy, not a moving target.
3. **Conservative trust region.** Clip 0.1 and actor LR 1e-4 to start, against
   the usual 0.2 / 3e-4.

The two failure signatures to watch, both reported every iteration, are entropy
collapse and success rate dipping below the BC baseline. Either means the anneal
is too fast. The BC checkpoint stays deployable throughout, so PPO is allowed to
fail without costing the demo.
"""

from __future__ import annotations

import copy
import pathlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..envs.reference_env import EnvConfig, NavEnv
from ..eval.harness import evaluate
from ..eval.torch_policy import TorchPolicy
from ..models.nav_policy import NavPolicy
from ..spec import DEFAULT_SPEC, NavSpec

__all__ = ["PPOConfig", "PPOTrainer"]

_OBS_KEYS = ("depth", "lidar", "goal", "last_action", "privileged")
_ACTOR_KEYS = ("depth", "lidar", "goal", "last_action")


@dataclass
class PPOConfig:
    num_envs: int = 16
    rollout_steps: int = 64
    iterations: int = 1000

    gamma: float = 0.99
    gae_lambda: float = 0.95

    #: Deliberately conservative: this fine-tunes a good policy rather than
    #: training one from scratch.
    clip_range: float = 0.1
    actor_lr: float = 1.0e-4
    critic_lr: float = 1.0e-3
    epochs: int = 4
    minibatches: int = 4
    grad_clip: float = 1.0

    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_kl: Optional[float] = 0.02  # early-stop an epoch if exceeded

    #: Iterations with the actor frozen while the critic catches up.
    critic_warmup_iters: int = 500
    #: Initial BC anchor weight and the fraction of training it decays over.
    anchor_beta: float = 1.0
    anchor_anneal_frac: float = 0.3

    device: str = "cpu"
    seed: int = 0
    log_every: int = 10
    eval_every: int = 50
    eval_episodes: int = 100


class RolloutBuffer:
    """Fixed-size on-policy storage, shaped ``(T, N, ...)``."""

    def __init__(self, cfg: PPOConfig, spec: NavSpec, device: str):
        T, N = cfg.rollout_steps, cfg.num_envs
        shapes = spec.obs.shapes()
        self.obs = {
            k: torch.zeros((T, N) + tuple(shapes[k]), device=device)
            for k in _OBS_KEYS
        }
        self.pre_actions = torch.zeros((T, N, 2), device=device)
        self.log_probs = torch.zeros((T, N), device=device)
        self.values = torch.zeros((T, N), device=device)
        self.rewards = torch.zeros((T, N), device=device)
        self.dones = torch.zeros((T, N), device=device)
        self.advantages = torch.zeros((T, N), device=device)
        self.returns = torch.zeros((T, N), device=device)

    def compute_returns(self, last_values: torch.Tensor, gamma: float,
                        lam: float) -> None:
        """Generalized advantage estimation, backwards through the rollout."""
        T = self.rewards.shape[0]
        gae = torch.zeros_like(last_values)
        for t in reversed(range(T)):
            next_values = last_values if t == T - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = (self.rewards[t] + gamma * next_values * not_done
                     - self.values[t])
            gae = delta + gamma * lam * not_done * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def flat(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor,
                            torch.Tensor, torch.Tensor]:
        obs = {k: v.flatten(0, 1) for k, v in self.obs.items()}
        return (obs, self.pre_actions.flatten(0, 1), self.log_probs.flatten(),
                self.advantages.flatten(), self.returns.flatten())


class PPOTrainer:
    """PPO with the BC handoff described in the module docstring."""

    def __init__(
        self,
        policy: NavPolicy,
        config: Optional[PPOConfig] = None,
        env_config: Optional[EnvConfig] = None,
        spec: NavSpec = DEFAULT_SPEC,
        bc_reference: Optional[NavPolicy] = None,
    ):
        self.cfg = cfg = config or PPOConfig()
        self.spec = spec
        self.env_config = env_config or EnvConfig(spec=spec)
        self.device = cfg.device
        torch.manual_seed(cfg.seed)

        self.policy = policy.to(self.device)
        # BatchNorm statistics must stop moving before RL starts: rollouts are
        # collected in one mode and updated in another, so drifting statistics
        # make the value of a state non-stationary.
        self.policy.freeze_norm()

        # The anchor is a frozen snapshot, never updated. Anchoring to a moving
        # copy of the policy would be a no-op regularizer.
        self.anchor: Optional[NavPolicy] = None
        if cfg.anchor_beta > 0.0:
            src = bc_reference if bc_reference is not None else policy
            self.anchor = copy.deepcopy(src).to(self.device).eval()
            for p in self.anchor.parameters():
                p.requires_grad_(False)

        actor_params, critic_params = [], []
        for name, p in self.policy.named_parameters():
            (critic_params if name.startswith(("critic", "critic_lidar"))
             else actor_params).append(p)
        self.actor_params = actor_params
        self.opt = torch.optim.Adam([
            {"params": actor_params, "lr": cfg.actor_lr},
            {"params": critic_params, "lr": cfg.critic_lr},
        ])

        self.envs = [NavEnv(self.env_config, seed=cfg.seed * 1000 + i)
                     for i in range(cfg.num_envs)]
        self._obs = [e.reset(seed=cfg.seed * 100_000 + i)
                     for i, e in enumerate(self.envs)]
        self.buffer = RolloutBuffer(cfg, spec, self.device)

        self.iteration = 0
        self._ep_returns: List[float] = []
        self._ep_successes: List[float] = []
        self._ep_return_acc = np.zeros(cfg.num_envs, dtype=np.float64)
        self.history: List[Dict[str, float]] = []

    # ── Rollout ──────────────────────────────────────────────────────────────

    def _stack_obs(self) -> Dict[str, torch.Tensor]:
        return {
            k: torch.from_numpy(
                np.stack([o[k] for o in self._obs]).astype(np.float32)
            ).to(self.device)
            for k in _OBS_KEYS
        }

    @torch.no_grad()
    def collect(self) -> None:
        cfg = self.cfg
        self.policy.eval()
        for t in range(cfg.rollout_steps):
            obs_t = self._stack_obs()
            for k in _OBS_KEYS:
                self.buffer.obs[k][t] = obs_t[k]

            action, pre, log_prob = self.policy.act(obs_t, deterministic=False)
            value = self.policy.value(obs_t["lidar"], obs_t["goal"],
                                      obs_t["last_action"], obs_t["privileged"])
            self.buffer.pre_actions[t] = pre
            self.buffer.log_probs[t] = log_prob
            self.buffer.values[t] = value

            acts = action.cpu().numpy()
            for i, env in enumerate(self.envs):
                res = env.step(acts[i])
                self.buffer.rewards[t, i] = res.reward
                done = res.terminated or res.truncated
                # Only *termination* bootstraps to zero. A time-limit truncation
                # is not a real end of the world, and treating it as one teaches
                # the policy that the universe ends at 60 seconds.
                self.buffer.dones[t, i] = float(res.terminated)
                self._ep_return_acc[i] += res.reward
                if done:
                    self._ep_returns.append(float(self._ep_return_acc[i]))
                    self._ep_successes.append(float(res.info["success"]))
                    self._ep_return_acc[i] = 0.0
                    self._obs[i] = env.reset()
                else:
                    self._obs[i] = res.obs

        obs_t = self._stack_obs()
        self._last_values = self.policy.value(
            obs_t["lidar"], obs_t["goal"], obs_t["last_action"],
            obs_t["privileged"])

    # ── Update ───────────────────────────────────────────────────────────────

    def _anchor_weight(self) -> float:
        """Linear anneal of the BC anchor from ``anchor_beta`` to 0."""
        cfg = self.cfg
        if self.anchor is None or cfg.anchor_beta <= 0.0:
            return 0.0
        span = max(1.0, cfg.iterations * cfg.anchor_anneal_frac)
        # The anneal clock starts when the actor does, i.e. after warmup.
        progress = max(0.0, self.iteration - cfg.critic_warmup_iters) / span
        return float(cfg.anchor_beta * max(0.0, 1.0 - progress))

    def update(self) -> Dict[str, float]:
        cfg = self.cfg
        self.buffer.compute_returns(self._last_values, cfg.gamma, cfg.gae_lambda)
        obs, pre_actions, old_log_probs, advantages, returns = self.buffer.flat()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        warming_up = self.iteration < cfg.critic_warmup_iters
        beta = self._anchor_weight()

        if warming_up:
            # Actor LR literally zero, rather than "small": during warmup the
            # advantages are meaningless and any actor step is pure damage.
            self.opt.param_groups[0]["lr"] = 0.0
        else:
            self.opt.param_groups[0]["lr"] = cfg.actor_lr

        n = pre_actions.shape[0]
        mb_size = max(1, n // cfg.minibatches)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "kl": 0.0, "anchor_loss": 0.0, "clip_frac": 0.0}
        count = 0
        stop = False

        self.policy.train()
        self.policy.freeze_norm()  # train() re-enables BN; keep it frozen
        for _ in range(cfg.epochs):
            if stop:
                break
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb_size):
                idx = perm[start:start + mb_size]
                mb_obs = {k: v[idx] for k, v in obs.items()}

                log_prob, entropy, value = self.policy.evaluate(
                    mb_obs, pre_actions[idx])

                ratio = torch.exp(log_prob - old_log_probs[idx])
                adv = advantages[idx]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_range,
                                      1.0 + cfg.clip_range) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = ((value - returns[idx]) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + cfg.value_coef * value_loss
                        + cfg.entropy_coef * entropy_loss)

                anchor_loss = torch.zeros((), device=self.device)
                if beta > 0.0 and not warming_up:
                    with torch.no_grad():
                        ref = self.anchor.pre_squash(
                            *[mb_obs[k] for k in _ACTOR_KEYS])
                    cur = self.policy.pre_squash(
                        *[mb_obs[k] for k in _ACTOR_KEYS])
                    anchor_loss = ((cur - ref) ** 2).mean()
                    loss = loss + beta * anchor_loss

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
                self.opt.step()

                with torch.no_grad():
                    kl = float((old_log_probs[idx] - log_prob).mean())
                    clip_frac = float(((ratio - 1.0).abs()
                                       > cfg.clip_range).float().mean())
                    stats["policy_loss"] += float(policy_loss.detach())
                    stats["value_loss"] += float(value_loss.detach())
                    stats["entropy"] += float(entropy.mean().detach())
                    stats["kl"] += kl
                    stats["anchor_loss"] += float(anchor_loss.detach())
                    stats["clip_frac"] += clip_frac
                count += 1

                if cfg.max_kl is not None and kl > cfg.max_kl and not warming_up:
                    stop = True
                    break

        for k in stats:
            stats[k] /= max(1, count)
        stats["anchor_beta"] = beta
        stats["warmup"] = float(warming_up)
        return stats

    # ── Loop ─────────────────────────────────────────────────────────────────

    def train(self, out_dir: str | pathlib.Path = "runs/ppo",
              bc_baseline_sr: Optional[float] = None) -> List[Dict[str, float]]:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        best_sr = -1.0
        t0 = time.time()

        for it in range(self.cfg.iterations):
            self.iteration = it
            self.collect()
            stats = self.update()

            if self._ep_successes:
                stats["rollout_success"] = float(np.mean(self._ep_successes[-100:]))
            self.history.append(stats)

            if self.cfg.log_every and (it + 1) % self.cfg.log_every == 0:
                print(f"it {it + 1:5d}  "
                      f"{'WARMUP ' if stats['warmup'] else ''}"
                      f"pi {stats['policy_loss']:+.4f}  v {stats['value_loss']:.4f}  "
                      f"H {stats['entropy']:+.3f}  kl {stats['kl']:+.4f}  "
                      f"beta {stats['anchor_beta']:.3f}  "
                      f"SR {stats.get('rollout_success', float('nan')):.1%}  "
                      f"[{time.time() - t0:.0f}s]")

            if self.cfg.eval_every and (it + 1) % self.cfg.eval_every == 0:
                m = evaluate(TorchPolicy(self.policy, self.spec, self.device),
                             self.cfg.eval_episodes, self.env_config,
                             spec=self.spec)
                print(f"    eval: {m.summary()}")
                # Watch for the two documented failure signatures.
                if bc_baseline_sr is not None and m.success_rate < bc_baseline_sr - 0.05:
                    print(f"    WARNING: SR {m.success_rate:.1%} is below the BC "
                          f"baseline {bc_baseline_sr:.1%} — the anchor anneal is "
                          f"likely too fast.")
                if stats["entropy"] < -3.0:
                    print("    WARNING: entropy collapsing; consider raising "
                          "entropy_coef or slowing the anneal.")
                if m.success_rate > best_sr:
                    best_sr = m.success_rate
                    torch.save({"model": self.policy.state_dict(),
                                "spec_fingerprint": self.spec.fingerprint(),
                                "metrics": m.to_dict(), "iteration": it},
                               out / "ppo_best.pt")

        torch.save({"model": self.policy.state_dict(),
                    "spec_fingerprint": self.spec.fingerprint(),
                    "iteration": self.cfg.iterations}, out / "ppo_final.pt")
        return self.history
