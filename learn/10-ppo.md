# 10 — PPO

Behaviour cloning can only ever imitate the oracle. It cannot exceed it, and it
has no notion of the *reward* — it never learns that collisions are bad, only that
the expert did not have them.

Reinforcement learning optimizes return directly. `sim2edge/training/ppo.py`
(378 lines) implements Proximal Policy Optimization, plus a careful handoff from
the BC checkpoint that is most of the file's complexity.

## PPO in one page

The policy gradient says: increase the probability of actions that turned out
better than expected.

```
∇J = E[ ∇log π(a|s) · A(s,a) ]
```

where `A` is the **advantage** — how much better this action was than the
critic's expectation. Positive advantage, make it more likely; negative, less.

The problem with the naive version is step size. Too large and you destroy a good
policy in one update; the data was collected under the *old* policy and stops
being valid the moment you change it.

PPO's fix is the clipped surrogate objective. Let `r = π_new(a|s) / π_old(a|s)`:

```
L = min( r·A,  clip(r, 1-ε, 1+ε)·A )
```

If the update would move the probability ratio more than `ε` in the direction the
advantage wants, the objective flattens and the gradient vanishes. The policy is
free to move within a trust region and gets no reward for leaving it.

`ppo.py:278-283`:

```python
ratio = torch.exp(log_prob - old_log_probs[idx])
adv = advantages[idx]
unclipped = ratio * adv
clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv
policy_loss = -torch.min(unclipped, clipped).mean()
```

`torch.min` of the two, then negated because optimizers minimize. That is the
entire PPO innovation, in four lines.

## GAE — estimating the advantage

`RolloutBuffer.compute_returns` at `ppo.py:105`:

```python
gae = torch.zeros_like(last_values)
for t in reversed(range(T)):
    next_values = last_values if t == T - 1 else self.values[t + 1]
    not_done = 1.0 - self.dones[t]
    delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
    gae = delta + gamma * lam * not_done * gae
    self.advantages[t] = gae
self.returns = self.advantages + self.values
```

`delta` is the **TD error**: reward plus discounted next value, minus current
value. "How much better did this step go than the critic predicted?"

Generalized Advantage Estimation exponentially averages TD errors over horizons,
controlled by `λ`:

- `λ = 0` → advantage is just the one-step TD error. Low variance, high bias.
- `λ = 1` → advantage is the full Monte-Carlo return minus the baseline. Unbiased,
  high variance.
- `λ = 0.95` (`ppo.py:59`) → the standard compromise.

`not_done` is what makes episode boundaries correct: at a terminal step the
bootstrap is zeroed, so value does not leak across the boundary into an unrelated
episode.

`γ = 0.99` at 20 Hz gives an effective horizon of about 100 steps — 5 seconds.
That is the timescale over which this policy plans.

## Advantage normalization

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

`ppo.py:246`. Normalizing per batch makes the effective learning rate independent
of reward scale. Without it, changing `progress` from 3.0 to 6.0 in `RewardSpec`
would silently double your policy step size.

## The rollout

`PPOConfig` defaults (`ppo.py:52`): 16 environments × 64 steps = 1024 transitions
per iteration, 4 epochs × 4 minibatches over them.

That is small. `REBUILD_PLAN.md:112` proposes 4096 environments on the GPU, which
is two orders of magnitude more data per iteration, and is what Isaac Lab exists
to provide (chapter 12). The numpy PPO here is for pipeline testing, not for
training the policy that ships.

**On-policy** means the buffer is thrown away after each iteration. Data collected
under policy `k` is only valid for updating policy `k`. This is why PPO is
sample-hungry and why massive parallelism helps so much.

## The BC handoff — the actual content of this file

Here is the problem, stated at `ppo.py:3-7`:

> Naive BC-init + PPO reliably wrecks a good pretrained policy. The mechanism is
> specific: the critic starts random, so its value estimates are noise, so the
> advantages are noise, so the first few hundred policy-gradient steps apply large
> random updates to a carefully fitted actor. By the time the critic is useful the
> actor is no better than random.

Read that chain again: **random critic → noisy advantages → random actor
updates**. You spent an hour on BC and the first minute of PPO throws it away.

Three mitigations, all load-bearing.

### 1. Critic warmup

```python
warming_up = self.iteration < cfg.critic_warmup_iters
if warming_up:
    # Actor LR literally zero, rather than "small": during warmup the
    # advantages are meaningless and any actor step is pure damage.
    self.opt.param_groups[0]["lr"] = 0.0
else:
    self.opt.param_groups[0]["lr"] = cfg.actor_lr
```

`ppo.py:248-256`. For the first 500 iterations only the value function trains. The
comment is emphatic about *literally zero* — a small-but-nonzero LR still
accumulates damage over hundreds of iterations of pure noise.

### 2. Annealed BC anchor

```python
if beta > 0.0 and not warming_up:
    with torch.no_grad():
        ref = self.anchor.pre_squash(*[mb_obs[k] for k in _ACTOR_KEYS])
    cur = self.policy.pre_squash(*[mb_obs[k] for k in _ACTOR_KEYS])
    anchor_loss = ((cur - ref) ** 2).mean()
    loss = loss + beta * anchor_loss
```

`ppo.py:291-299`. An auxiliary term pulls the actor toward the cloned policy, with
`β` annealed to zero over the first 30% of training (`_anchor_weight`, `:231`).

Two details:

- **The anchor is a frozen copy**, not the live policy. `scripts/train.py:98-100`:
  "anchoring to a moving copy of the policy is a no-op regularizer." Obvious once
  said, easy to get wrong.
- **The anneal clock starts after warmup** (`ppo.py:237-238`), not at iteration 0.
  Otherwise the anchor would decay to nothing while the actor was still frozen.

### 3. Conservative trust region

`clip_range = 0.1` and `actor_lr = 1e-4`, against the usual 0.2 and 3e-4
(`ppo.py:61-64`):

> Deliberately conservative: this fine-tunes a good policy rather than training
> one from scratch.

Plus a KL early-stop at `:318-320`: if the mean KL divergence between old and new
policy exceeds `max_kl = 0.02`, the epoch stops immediately. A hard backstop for
when the clipping is not enough.

## What to watch

`ppo.py:26-28` names the two failure signatures, both reported every iteration:

> The two failure signatures to watch, both reported every iteration, are entropy
> collapse and success rate dipping below the BC baseline. Either means the anneal
> is too fast.

| Statistic | Healthy | Trouble |
|---|---|---|
| `entropy` | slowly decreasing | sudden drop → premature determinism, exploration dead |
| `kl` | ≪ `max_kl` | at the cap every iteration → steps too big |
| `clip_frac` | 0.1 – 0.3 | near 1.0 → the update is fighting the clip |
| `value_loss` | falling during warmup | flat → critic not learning, warmup pointless |
| success rate | ≥ BC baseline | below it → the anchor released too early |

`train()` at `:330` takes `bc_baseline_sr` precisely so it can report that last
comparison.

## Run it

```bash
ssh vkommera@dronelab12-4
cd ~/Documents/Qualcomm/sim2edge-ventuno

# Fine-tune the BC checkpoint from chapter 09
.venv/bin/python scripts/train.py ppo \
    --init runs/tutorial_bc/bc_final.pt \
    --iterations 200 --critic-warmup 50 \
    --num-envs 16 --lidar-only \
    --out runs/tutorial_ppo
```

The fingerprint check fires first (`scripts/train.py:73-78`) — if the checkpoint
was trained under a different spec, this exits rather than training something
subtly wrong.

Be realistic about what 200 iterations of 16 environments buys you: 200 × 1024 =
205 k transitions, about 1.6 environment-hours. That is a smoke test that the loop
runs and the statistics look sane, not a training run. Real training is chapter 12.

## An honest note on the numpy PPO

`sim2edge/training/ppo.py` runs 16 environments in a Python loop. The GPU sits
idle. It exists so the whole pipeline is testable on a laptop, and so a result on
the reference environment transfers to the Isaac Lab run whose hyperparameters
mirror it (`isaaclab_env/train_rsl_rl.py:14-16`).

`REBUILD_PLAN.md:26` proposes deleting it along with BC:

> PPO from `rsl_rl` solves point-goal nav with a dense progress reward.

If you never bootstrap from BC, all three mitigations above become unnecessary,
because there is no carefully fitted actor to protect. The critic warms up
alongside a random actor and nothing is lost. That is a genuinely simpler design —
and it is only available if PPO-from-scratch works, which is untested here.

Notice how much of this chapter is complexity that exists *because* of chapter 09.
That is the clearest argument in the whole repository for the rebuild plan's
position.

## Check yourself

- Why does the ratio `r` equal 1 at the start of every update epoch?
- What is `not_done` protecting against in the GAE loop?
- The anchor is a frozen copy of the BC policy. What happens if you anchor to
  `self.policy` instead?
- Advantages are normalized per batch. What breaks if you skip that and then
  double every reward coefficient?
- Entropy collapses at iteration 60 and success rate falls below the BC baseline.
  Which knob do you turn first?

→ [11 — Evaluation](11-evaluation.md)
