"""PPO: the estimator, the trust region, and the BC handoff that protects the
actor. No simulator -- everything here runs on tensors of numbers, which is why
the algorithm was written to be separable from the rollout in the first place.
"""

import copy
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sim2edge.models import NavActorCritic, NavPolicy
from sim2edge.models.actor_critic import (LOG_STD_MAX, LOG_STD_MIN,
                                          gaussian_entropy, gaussian_log_prob)
from sim2edge.spec import OBS_DIM, V_MAX, fingerprint
from sim2edge.training import load_policy_checkpoint
from sim2edge.training.ppo import (EpisodeStats, PPOConfig, PPOTrainer,
                                   RolloutBuffer, clipped_surrogate,
                                   load_ppo_checkpoint, ppo_checkpoint,
                                   save_ppo_checkpoint)


def _buffer(num_steps=4, num_envs=3, seed=0):
    """A filled rollout buffer of plausible-looking noise."""
    torch.manual_seed(seed)
    buffer = RolloutBuffer(num_steps, num_envs, OBS_DIM, torch.device("cpu"))
    for _ in range(num_steps):
        buffer.add(
            torch.randn(num_envs, OBS_DIM), torch.randn(num_envs, 2),
            torch.randn(num_envs), torch.randn(num_envs),
            torch.randn(num_envs),
            torch.zeros(num_envs, dtype=torch.bool))
    return buffer


# ── the advantage estimator ──────────────────────────────────────────────────

def test_gae_matches_the_closed_form_on_a_short_rollout():
    gamma, lam = 0.9, 0.8
    buffer = RolloutBuffer(3, 1, OBS_DIM, torch.device("cpu"))
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.25, 0.125]
    for reward, value in zip(rewards, values):
        buffer.add(torch.zeros(1, OBS_DIM), torch.zeros(1, 2), torch.zeros(1),
                   torch.tensor([value]), torch.tensor([reward]),
                   torch.zeros(1, dtype=torch.bool))
    last = torch.tensor([0.0625])
    buffer.compute_advantages(last, gamma, lam)

    following = [values[1], values[2], float(last)]
    deltas = [r + gamma * n - v
              for r, n, v in zip(rewards, following, values)]
    expected, accumulated = [], 0.0
    for delta in reversed(deltas):
        accumulated = delta + gamma * lam * accumulated
        expected.insert(0, accumulated)

    assert buffer.advantage.squeeze(1).tolist() == pytest.approx(expected)
    # The critic regresses onto exactly the estimate the actor is using.
    assert buffer.returns.squeeze(1).tolist() == pytest.approx(
        [a + v for a, v in zip(expected, values)])


def test_a_done_step_does_not_bootstrap_across_the_episode_boundary():
    """The next environment step belongs to a fresh layout. Without `not_done`
    the critic's opinion of that unrelated state leaks backwards into this one."""
    gamma, lam = 0.99, 0.95
    buffer = RolloutBuffer(2, 1, OBS_DIM, torch.device("cpu"))
    buffer.add(torch.zeros(1, OBS_DIM), torch.zeros(1, 2), torch.zeros(1),
               torch.tensor([0.0]), torch.tensor([1.0]),
               torch.ones(1, dtype=torch.bool))          # episode ends here
    buffer.add(torch.zeros(1, OBS_DIM), torch.zeros(1, 2), torch.zeros(1),
               torch.tensor([0.0]), torch.tensor([0.0]),
               torch.zeros(1, dtype=torch.bool))
    buffer.compute_advantages(torch.tensor([1000.0]), gamma, lam)
    # The terminal step sees only its own reward, whatever the next value is.
    assert float(buffer.advantage[0]) == pytest.approx(1.0)


def test_advantages_are_normalized_over_the_whole_batch():
    """Normalizing decouples step size from reward scale: doubling every reward
    coefficient must not double the policy step."""
    buffer = _buffer(num_steps=8, num_envs=16)
    buffer.compute_advantages(torch.zeros(16), 0.99, 0.95)
    scaled = copy.deepcopy(buffer)
    scaled.advantage *= 7.5

    generator = torch.Generator().manual_seed(0)
    plain = torch.cat([batch[3] for batch in buffer.batches(2, generator)])
    generator = torch.Generator().manual_seed(0)
    loud = torch.cat([batch[3] for batch in scaled.batches(2, generator)])
    assert torch.allclose(plain, loud, atol=1e-5)
    assert float(plain.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(plain.std()) == pytest.approx(1.0, abs=1e-3)


def test_every_transition_appears_exactly_once_across_minibatches():
    buffer = _buffer(num_steps=5, num_envs=7)
    buffer.compute_advantages(torch.zeros(7), 0.99, 0.95)
    seen = torch.cat([batch[1] for batch in buffer.batches(3)])
    assert seen.shape == (35, 2)
    assert torch.allclose(seen.sum(0), buffer.raw_action.reshape(-1, 2).sum(0),
                          atol=1e-4)


def test_an_incomplete_rollout_is_refused_rather_than_padded():
    buffer = RolloutBuffer(3, 2, OBS_DIM, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="incomplete"):
        buffer.compute_advantages(torch.zeros(2), 0.99, 0.95)


# ── the clipped surrogate ────────────────────────────────────────────────────

def test_clipping_removes_the_gradient_once_the_ratio_leaves_the_region():
    """That is the whole PPO innovation: outside the trust region the objective
    is flat, so a step that has already moved too far earns nothing more."""
    advantage = torch.tensor([1.0])
    old = torch.zeros(1)
    inside = torch.tensor([math.log(1.05)], requires_grad=True)
    outside = torch.tensor([math.log(1.5)], requires_grad=True)

    for log_prob, expected_gradient in ((inside, True), (outside, False)):
        loss, _, _ = clipped_surrogate(log_prob, old, advantage, 0.1)
        loss.backward()
        assert (float(log_prob.grad.abs()) > 0.0) is expected_gradient


def test_a_negative_advantage_clips_on_the_other_side():
    """Pushing an action that turned out badly further down must also stop."""
    old, advantage = torch.zeros(1), torch.tensor([-1.0])
    log_prob = torch.tensor([math.log(0.5)], requires_grad=True)
    loss, clip_fraction, _ = clipped_surrogate(log_prob, old, advantage, 0.1)
    loss.backward()
    assert float(log_prob.grad.abs()) == pytest.approx(0.0)
    assert float(clip_fraction) == 1.0


def test_the_ratio_is_one_and_the_kl_zero_before_any_update():
    log_prob = torch.randn(64)
    loss, clip_fraction, approx_kl = clipped_surrogate(
        log_prob, log_prob.clone(), torch.randn(64), 0.1)
    assert float(clip_fraction) == 0.0
    assert float(approx_kl) == pytest.approx(0.0, abs=1e-6)


def test_the_kl_estimator_stays_positive_on_noise():
    """The naive mean log-ratio wanders negative and would trip the early stop
    by accident; the k3 estimator used here cannot."""
    torch.manual_seed(3)
    for _ in range(20):
        old = torch.randn(256)
        _, _, approx_kl = clipped_surrogate(
            old + 0.05 * torch.randn(256), old, torch.randn(256), 0.1)
        assert float(approx_kl) >= 0.0


# ── the stochastic wrapper ───────────────────────────────────────────────────

def test_gaussian_log_prob_agrees_with_torch_distributions():
    mean, log_std = torch.randn(5, 2), torch.tensor([-1.5, -0.5])
    value = torch.randn(5, 2)
    reference = torch.distributions.Normal(mean, log_std.exp())
    assert torch.allclose(gaussian_log_prob(value, mean, log_std),
                          reference.log_prob(value).sum(-1), atol=1e-6)
    assert torch.allclose(gaussian_entropy(log_std),
                          reference.entropy()[0].sum(), atol=1e-6)


def test_the_actor_is_the_deployable_policy_and_nothing_else():
    """The exported artifact must be the object RL improved, not a copy of it."""
    actor = NavPolicy()
    policy = NavActorCritic(actor=actor)
    assert policy.actor is actor
    observation = torch.randn(4, OBS_DIM)
    assert torch.allclose(policy.physical_action(observation),
                          actor(observation), atol=1e-6)
    # tanh(raw) is exactly the normalized command the environment de-normalizes.
    physical = actor(observation)
    normalized = policy.deterministic_action(observation)
    assert torch.allclose(0.5 * V_MAX * (normalized[:, 0] + 1.0),
                          physical[:, 0], atol=1e-6)


def test_sampling_is_reproducible_and_scored_consistently():
    policy = NavActorCritic()
    observation = torch.randn(6, OBS_DIM)
    first = policy.act(observation, torch.Generator().manual_seed(11))
    second = policy.act(observation, torch.Generator().manual_seed(11))
    assert torch.allclose(first[0], second[0])
    # Re-scoring an unchanged policy must reproduce the stored log probability,
    # which is what makes every PPO ratio start at exactly one.
    log_prob, _, _ = policy.evaluate(observation, first[0])
    assert torch.allclose(log_prob, first[2], atol=1e-6)


def test_the_critic_does_not_share_the_actors_weights():
    """A random critic's value gradients must not reach features BC fitted."""
    policy = NavActorCritic()
    actor_ids = {id(p) for p in policy.actor.parameters()}
    assert not actor_ids & {id(p) for p in policy.critic.parameters()}

    before = policy.actor.action_head.weight.detach().clone()
    policy.critic(torch.randn(3, OBS_DIM)).sum().backward()
    assert all(p.grad is None for p in policy.actor.parameters())
    assert torch.equal(policy.actor.action_head.weight.detach(), before)


def test_exploration_scale_stays_inside_its_bounds():
    policy = NavActorCritic()
    with torch.no_grad():
        policy.log_std.fill_(12.0)
    policy.clamp_log_std()
    assert float(policy.log_std.detach().max()) == pytest.approx(LOG_STD_MAX)
    with torch.no_grad():
        policy.log_std.fill_(-40.0)
    policy.clamp_log_std()
    assert float(policy.log_std.detach().min()) == pytest.approx(LOG_STD_MIN)


# ── the BC handoff ───────────────────────────────────────────────────────────

def _trainer(config=None, anchor=None, policy=None):
    config = config or PPOConfig(iterations=10, minibatches=2, epochs=1)
    policy = policy or NavActorCritic()
    return PPOTrainer(policy, config, anchor, device=torch.device("cpu")), policy


def test_the_actor_is_completely_frozen_during_critic_warmup():
    """Not a small learning rate: a small one still accumulates damage over
    hundreds of iterations of meaningless advantages."""
    trainer, policy = _trainer(PPOConfig(iterations=10, epochs=1, minibatches=2,
                                         critic_warmup_iters=5))
    before = [p.detach().clone() for p in policy.actor_parameters()]
    critic_before = policy.critic.value_head.weight.detach().clone()

    buffer = _buffer(num_steps=4, num_envs=8)
    buffer.compute_advantages(torch.zeros(8), 0.99, 0.95)
    record = trainer.update(buffer)

    assert record["warming_up"] and record["actor_lr"] == 0.0
    for parameter, original in zip(policy.actor_parameters(), before):
        assert torch.equal(parameter.detach(), original)
    # The critic, meanwhile, is the entire point of the warmup.
    assert not torch.equal(policy.critic.value_head.weight.detach(),
                           critic_before)


def test_the_actor_moves_once_warmup_is_over():
    trainer, policy = _trainer(PPOConfig(iterations=10, epochs=1, minibatches=2,
                                         critic_warmup_iters=0))
    before = policy.actor.action_head.weight.detach().clone()
    buffer = _buffer(num_steps=4, num_envs=8)
    buffer.compute_advantages(torch.zeros(8), 0.99, 0.95)
    record = trainer.update(buffer)
    assert not record["warming_up"] and record["actor_lr"] > 0.0
    assert not torch.equal(policy.actor.action_head.weight.detach(), before)


def test_the_anchor_is_frozen_and_its_weight_anneals_after_warmup():
    """Annealing from iteration zero would decay the anchor to nothing while
    the actor was still frozen, leaving it unprotected exactly when it starts
    moving."""
    config = PPOConfig(iterations=100, critic_warmup_iters=10,
                       anchor_anneal_fraction=0.5, anchor_coef=1.0)
    anchor = NavPolicy()
    trainer, _ = _trainer(config, anchor=anchor)

    assert not any(p.requires_grad for p in anchor.parameters())
    assert trainer.anchor_weight() == pytest.approx(1.0)   # iteration 0, warmup
    trainer.iteration = 10
    assert trainer.anchor_weight() == pytest.approx(1.0)   # anneal starts here
    trainer.iteration = 10 + 45 // 2
    assert 0.0 < trainer.anchor_weight() < 1.0
    trainer.iteration = 55
    assert trainer.anchor_weight() == pytest.approx(0.0)


def test_the_anchor_holds_the_actor_nearer_the_cloned_policy():
    """Two identical runs, one anchored. Adam normalizes its step size, so the
    anchor buys direction rather than magnitude -- which is exactly why the
    claim worth testing is comparative and not an absolute drift bound."""
    torch.manual_seed(5)
    reference = NavActorCritic()
    observation = torch.randn(16, OBS_DIM)
    original = reference.actor.pre_squash(observation).detach()

    drifts = []
    for anchor_coef in (0.0, 1000.0):
        policy = copy.deepcopy(reference)
        anchor = copy.deepcopy(reference.actor)
        config = PPOConfig(iterations=10, epochs=1, minibatches=1,
                           critic_warmup_iters=0, anchor_coef=anchor_coef,
                           anchor_anneal_fraction=1.0, actor_lr=1e-3)
        trainer, _ = _trainer(config, anchor=anchor, policy=policy)
        # Several updates, because at the anchor point the anchor gradient is
        # exactly zero: the term restrains accumulated drift, not one step.
        for step in range(5):
            buffer = _buffer(num_steps=2, num_envs=8, seed=step)
            buffer.compute_advantages(torch.zeros(8), 0.99, 0.95)
            trainer.update(buffer)
        drifts.append(float((policy.actor.pre_squash(observation).detach()
                             - original).abs().mean()))
        # The anchor itself never moves: it is a snapshot, not a second policy.
        assert all(p.grad is None for p in anchor.parameters())
        assert torch.equal(anchor.action_head.weight.detach(),
                           reference.actor.action_head.weight.detach())

    unanchored, anchored = drifts
    assert anchored < unanchored


def test_anchoring_to_the_live_actor_is_refused():
    """It would freeze the actor outright, and the run would look like a very
    stable fine-tune rather than a policy that never moved."""
    policy = NavActorCritic()
    with pytest.raises(ValueError, match="frozen snapshot"):
        PPOTrainer(policy, PPOConfig(), policy.actor,
                   device=torch.device("cpu"))


def test_a_large_policy_change_stops_the_update_early():
    config = PPOConfig(iterations=10, epochs=8, minibatches=2,
                       critic_warmup_iters=0, actor_lr=0.5, max_kl=1e-9)
    trainer, _ = _trainer(config)
    buffer = _buffer(num_steps=4, num_envs=16)
    buffer.compute_advantages(torch.zeros(16), 0.99, 0.95)
    assert trainer.update(buffer)["kl_early_stop"]


def test_an_update_reports_every_statistic_the_runbook_asks_to_watch():
    trainer, _ = _trainer(PPOConfig(iterations=10, epochs=2, minibatches=2,
                                    critic_warmup_iters=0))
    buffer = _buffer(num_steps=4, num_envs=8)
    buffer.compute_advantages(torch.zeros(8), 0.99, 0.95)
    record = trainer.update(buffer)
    for key in ("entropy", "approx_kl", "clip_fraction", "value_loss",
                "explained_variance", "policy_loss", "anchor_weight",
                "std_v", "std_omega"):
        assert key in record and np.isfinite(record[key])
    assert trainer.iteration == 1


def test_explained_variance_reads_the_critic_at_any_reward_scale():
    """The metric raw value loss cannot supply: a critic that is doing well on
    a high-variance return sits at a large, flat value loss forever."""
    buffer = _buffer(num_steps=4, num_envs=32)
    buffer.compute_advantages(torch.zeros(32), 0.99, 0.95)
    trainer, _ = _trainer(PPOConfig(iterations=10, epochs=1, minibatches=2,
                                    critic_warmup_iters=0))

    # A perfect critic explains everything, whatever the returns are scaled to.
    for scale in (1.0, 100.0):
        buffer.returns = buffer.value * scale
        buffer.value = buffer.returns.clone()
        assert trainer.update(buffer)["explained_variance"] == pytest.approx(
            1.0, abs=1e-4)

    # A critic that predicts a constant explains none of it.
    buffer.returns = torch.randn_like(buffer.returns) * 50.0
    buffer.value = torch.zeros_like(buffer.value)
    assert trainer.update(buffer)["explained_variance"] < 0.1


# ── episode bookkeeping ──────────────────────────────────────────────────────

def test_episode_statistics_count_only_finished_episodes():
    stats = EpisodeStats(3, torch.device("cpu"))
    reward = torch.tensor([1.0, 1.0, 1.0])
    none = torch.zeros(3, dtype=torch.bool)
    for _ in range(4):
        stats.update(reward, none, none, none, none)
    assert stats.summary()["episodes"] == 0.0

    reached = torch.tensor([True, False, False])
    collided = torch.tensor([False, True, False])
    stats.update(reward, reached | collided, none, reached, collided)
    summary = stats.summary()
    assert summary["episodes"] == 2.0
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["collision_rate"] == pytest.approx(0.5)
    assert summary["episode_length"] == pytest.approx(5.0)
    assert summary["episode_return"] == pytest.approx(5.0)


def test_a_finished_environment_starts_its_next_episode_from_zero():
    stats = EpisodeStats(1, torch.device("cpu"))
    done = torch.ones(1, dtype=torch.bool)
    none = torch.zeros(1, dtype=torch.bool)
    stats.update(torch.tensor([5.0]), done, none, done, none)
    stats.reset()
    stats.update(torch.tensor([2.0]), done, none, done, none)
    assert stats.summary()["episode_return"] == pytest.approx(2.0)


def test_a_timeout_is_reported_separately_from_a_termination():
    stats = EpisodeStats(2, torch.device("cpu"))
    none = torch.zeros(2, dtype=torch.bool)
    stats.update(torch.zeros(2), torch.tensor([True, False]),
                 torch.tensor([False, True]), torch.tensor([True, False]), none)
    summary = stats.summary()
    assert summary["episodes"] == 2.0
    assert summary["timeout_rate"] == pytest.approx(0.5)
    assert summary["success_rate"] == pytest.approx(0.5)


# ── the loop, end to end ─────────────────────────────────────────────────────

def test_a_full_loop_learns_a_reward_a_navigation_policy_cannot_fake():
    """The rollout in ``scripts/train_ppo.py`` is this, with Isaac in place of
    the toy. Reward here depends only on the commanded forward speed, so a
    policy that improves it can only have done so through the parts under test:
    sampling, the advantage estimate, the ratio and the optimizer step."""
    target = 0.5                      # in normalized command units
    num_envs = 64
    torch.manual_seed(0)
    policy = NavActorCritic()
    config = PPOConfig(iterations=40, num_steps=8, epochs=4, minibatches=4,
                       critic_warmup_iters=2, anchor_coef=0.0,
                       actor_lr=3.0e-3, critic_lr=3.0e-3, clip_range=0.2,
                       max_kl=1.0)
    trainer, _ = _trainer(config, policy=policy)
    buffer = RolloutBuffer(config.num_steps, num_envs, OBS_DIM,
                           torch.device("cpu"))
    generator = torch.Generator().manual_seed(1)

    observation = torch.randn(num_envs, OBS_DIM)
    with torch.no_grad():
        initial = float(policy.deterministic_action(observation)[:, 0].mean())
    for _ in range(config.iterations):
        buffer.reset()
        for _ in range(config.num_steps):
            with torch.no_grad():
                raw, action, log_prob, value = policy.act(observation, generator)
            reward = -(action[:, 0] - target).square()
            buffer.add(observation, raw, log_prob, value, reward,
                       torch.zeros(num_envs, dtype=torch.bool))
            observation = torch.randn(num_envs, OBS_DIM)
        with torch.no_grad():
            last_value = policy.critic(observation)
        buffer.compute_advantages(last_value, config.gamma, config.gae_lambda)
        record = trainer.update(buffer)
        assert np.isfinite(record["policy_loss"])

    with torch.no_grad():
        final = float(policy.deterministic_action(observation)[:, 0].mean())
    assert abs(final - target) < abs(initial - target)
    assert abs(final - target) < 0.2


# ── checkpoints ──────────────────────────────────────────────────────────────

def test_a_ppo_checkpoint_loads_through_the_ordinary_policy_path(tmp_path):
    """Evaluation, DAgger collection and export must not care which algorithm
    produced a checkpoint."""
    trainer, policy = _trainer()
    path = tmp_path / "final.pt"
    save_ppo_checkpoint(path, policy, trainer, {"success_rate": 0.5})

    loaded = load_policy_checkpoint(path)
    observation = torch.randn(3, OBS_DIM)
    assert torch.allclose(loaded(observation), policy.actor(observation),
                          atol=1e-6)
    saved = torch.load(path, weights_only=False)
    assert saved["spec_fingerprint"] == fingerprint()
    assert saved["algorithm"] == "ppo"


def test_resuming_restores_the_critic_and_the_exploration_scale(tmp_path):
    trainer, policy = _trainer()
    with torch.no_grad():
        policy.log_std.fill_(-2.5)
    trainer.iteration = 7
    path = tmp_path / "latest.pt"
    save_ppo_checkpoint(path, policy, trainer)

    fresh_trainer, fresh = _trainer()
    load_ppo_checkpoint(path, fresh, fresh_trainer)
    assert fresh_trainer.iteration == 7
    assert float(fresh.log_std.detach()[0]) == pytest.approx(-2.5)
    observation = torch.randn(3, OBS_DIM)
    assert torch.allclose(fresh.critic(observation), policy.critic(observation),
                          atol=1e-6)


def test_a_checkpoint_from_a_different_observation_layout_is_refused(tmp_path):
    trainer, policy = _trainer()
    path = tmp_path / "stale.pt"
    saved = ppo_checkpoint(policy, trainer)
    saved["observation_fingerprint"] = "0000000000000000"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="observation layout"):
        load_ppo_checkpoint(path, policy, trainer)


# ── the reward's distance field ──────────────────────────────────────────────

def test_the_distance_field_is_finite_and_falls_toward_the_goal():
    from sim2edge.oracle import LayoutPool
    from sim2edge.rooms import sample_room

    rooms = [sample_room(np.random.default_rng(seed)) for seed in range(3)]
    pool = LayoutPool(rooms)
    assert torch.isfinite(pool.distance).all()

    layout = torch.arange(len(rooms))
    goal = torch.tensor(np.stack([room.goal for room in rooms]))
    at_goal = pool.distance_to_goal(layout, goal)
    assert torch.all(at_goal < rooms[0].res * 2)

    # A start is sampled at 2-8 geodesic metres, and the field must say so.
    starts = torch.tensor(np.stack([room.start[:2] for room in rooms]))
    assert torch.all(pool.distance_to_goal(layout, starts) >= 2.0)


def test_no_query_can_be_paid_for_leaving_the_arena():
    """Blocked cells and the pool's padding used to hold INF. One step from a
    finite cell into one of those pays a billion metres of progress."""
    from sim2edge.oracle import LayoutPool
    from sim2edge.rooms import sample_room

    rooms = [sample_room(np.random.default_rng(0)),
             sample_room(np.random.default_rng(1))]
    pool = LayoutPool(rooms)
    assert torch.isfinite(pool.distance).all()
    assert float(pool.distance.max()) < 1e3

    # Deep inside the padding of the smaller room, and far outside every arena.
    far = torch.tensor([[999.0, 999.0], [-999.0, -999.0]])
    outside = pool.distance_to_goal(torch.tensor([0, 1]), far)
    assert torch.all(outside > 0.0) and torch.isfinite(outside).all()


def test_filling_an_unreachable_region_keeps_the_walled_off_cells_far():
    from sim2edge.grids import INF, fill_unreachable

    free = np.array([[True, False, True]] * 3)
    field = np.array([[0.0, INF, INF]] * 3, dtype=np.float32)
    filled = fill_unreachable(field, free)
    assert np.isfinite(filled).all()
    # The walled-off column takes the value of the nearest usable cell rather
    # than a sentinel, so a step into it is worth nothing, not everything.
    assert filled[0, 2] == pytest.approx(0.0)
