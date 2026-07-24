"""End-to-end smoke tests for the training pipelines.

These are deliberately tiny — a few environments, a few iterations. They are not
checking that the policy learns anything (that needs the 4090 and hours); they
are checking that the plumbing runs: shapes line up, GAE terminates, the BC
handoff engages and disengages on schedule, and the optimizer actually moves the
weights it is supposed to move and leaves alone the ones it is not.

That distinction matters. A PPO implementation with a shape bug fails loudly in
the first minute; one with a *silent* bug — a critic that never trains, an anchor
that never anneals, an actor that updates during warmup — looks fine for hours
and produces a policy that is quietly worse than the BC checkpoint it started
from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim2edge.envs.reference_env import EnvConfig  # noqa: E402
from sim2edge.eval.harness import evaluate, oracle_policy  # noqa: E402
from sim2edge.eval.torch_policy import TorchPolicy, random_policy  # noqa: E402
from sim2edge.models.nav_policy import build_policy  # noqa: E402
from sim2edge.spec import DEFAULT_SPEC  # noqa: E402
from sim2edge.training.bc import BCConfig, train_bc  # noqa: E402
from sim2edge.training.data import collect_episodes  # noqa: E402
from sim2edge.training.ppo import PPOConfig, PPOTrainer  # noqa: E402

S = DEFAULT_SPEC
CFG = EnvConfig(use_depth=False)


@pytest.fixture(scope="module")
def tiny_demos():
    demos, stats = collect_episodes(3, CFG, S, seed_base=0)
    assert stats["success_rate"] == 1.0, "the oracle should not be failing"
    return demos


def test_bc_reduces_loss_and_moves_only_the_actor(tiny_demos):
    torch.manual_seed(0)
    policy = build_policy(S)
    critic_before = [p.detach().clone()
                     for n, p in policy.named_parameters()
                     if n.startswith(("critic", "critic_lidar"))]

    history = train_bc(policy, tiny_demos,
                       BCConfig(epochs=2, batch_size=64, lr=1e-3, log_every=0),
                       S)

    assert history["train"][-1] < history["train"][0], "BC loss did not decrease"

    # The critic has no supervision from demonstrations; letting the optimizer
    # touch it would just decay weights that are about to matter in PPO.
    critic_after = [p.detach()
                    for n, p in policy.named_parameters()
                    if n.startswith(("critic", "critic_lidar"))]
    for before, after in zip(critic_before, critic_after):
        assert torch.equal(before, after), "BC modified critic parameters"


def test_ppo_runs_and_respects_the_critic_warmup():
    """During warmup the actor must not move at all, and the critic must."""
    torch.manual_seed(0)
    policy = build_policy(S)
    trainer = PPOTrainer(
        policy,
        PPOConfig(num_envs=2, rollout_steps=8, iterations=2,
                  critic_warmup_iters=99, minibatches=2, epochs=1,
                  anchor_beta=1.0, eval_every=0, log_every=0),
        env_config=CFG, spec=S)

    actor_before = policy.actor_head.weight.detach().clone()
    critic_before = trainer.policy.critic[-1].weight.detach().clone()

    trainer.iteration = 0
    trainer.collect()
    stats = trainer.update()

    assert stats["warmup"] == 1.0
    assert torch.equal(actor_before, policy.actor_head.weight.detach()), \
        "the actor moved during critic warmup, which is exactly what warmup " \
        "exists to prevent"
    assert not torch.equal(critic_before,
                           trainer.policy.critic[-1].weight.detach()), \
        "the critic did not train during warmup"


def test_ppo_updates_the_actor_after_warmup():
    torch.manual_seed(0)
    policy = build_policy(S)
    trainer = PPOTrainer(
        policy,
        PPOConfig(num_envs=2, rollout_steps=8, iterations=4,
                  critic_warmup_iters=0, minibatches=2, epochs=1,
                  anchor_beta=0.0, eval_every=0, log_every=0),
        env_config=CFG, spec=S)

    before = policy.actor_head.weight.detach().clone()
    trainer.iteration = 1
    trainer.collect()
    stats = trainer.update()

    assert stats["warmup"] == 0.0
    assert not torch.equal(before, policy.actor_head.weight.detach())
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])


def test_anchor_anneals_to_zero_and_starts_after_warmup():
    """The anneal clock must start when the actor does, not at iteration 0."""
    policy = build_policy(S)
    trainer = PPOTrainer(
        policy,
        PPOConfig(num_envs=1, rollout_steps=4, iterations=100,
                  critic_warmup_iters=10, anchor_beta=1.0,
                  anchor_anneal_frac=0.5, eval_every=0, log_every=0),
        env_config=CFG, spec=S)

    trainer.iteration = 0
    assert trainer._anchor_weight() == pytest.approx(1.0)
    trainer.iteration = 10  # warmup just ended: full strength
    assert trainer._anchor_weight() == pytest.approx(1.0)
    trainer.iteration = 35  # halfway through a 50-iteration anneal
    assert trainer._anchor_weight() == pytest.approx(0.5, abs=0.05)
    trainer.iteration = 60  # fully annealed
    assert trainer._anchor_weight() == pytest.approx(0.0)


def test_anchor_is_a_frozen_snapshot():
    """Anchoring to a live reference would be a no-op regularizer."""
    policy = build_policy(S)
    trainer = PPOTrainer(
        policy,
        PPOConfig(num_envs=1, rollout_steps=4, iterations=2, anchor_beta=1.0,
                  eval_every=0, log_every=0),
        env_config=CFG, spec=S)
    assert trainer.anchor is not policy
    assert all(not p.requires_grad for p in trainer.anchor.parameters())


def test_gae_bootstraps_only_on_true_termination():
    """A time-limit truncation is not the end of the world, and treating it as
    one teaches the policy that the universe ends after 60 seconds."""
    policy = build_policy(S)
    trainer = PPOTrainer(
        policy,
        PPOConfig(num_envs=2, rollout_steps=4, iterations=1, eval_every=0,
                  log_every=0),
        env_config=EnvConfig(use_depth=False, max_episode_steps=2), spec=S)
    trainer.collect()
    # With a 2-step limit, episodes truncate rather than terminate, so the done
    # mask used for bootstrapping must stay zero.
    assert trainer.buffer.dones.sum() == 0.0


def test_eval_harness_separates_oracle_from_random():
    """A benchmark on which a random policy scores well is telling you the
    layouts are too easy."""
    from sim2edge.envs.reference_env import NavEnv

    env = NavEnv(CFG, seed=0)
    oracle = evaluate(oracle_policy(env, S), 8, env=env, spec=S)
    rand = evaluate(random_policy(seed=0), 8, CFG, spec=S)
    assert oracle.success_rate > 0.9
    assert rand.success_rate < 0.2
    assert oracle.spl > rand.spl


def test_torch_policy_returns_commands_within_limits():
    policy = TorchPolicy(build_policy(S), S)
    from sim2edge.envs.reference_env import NavEnv

    env = NavEnv(CFG, seed=0)
    obs = env.reset(seed=0)
    for _ in range(5):
        v, omega = policy(obs)
        assert S.action.v_min <= v <= S.action.v_max
        assert abs(omega) <= S.action.omega_max
        obs = env.step((v, omega)).obs
