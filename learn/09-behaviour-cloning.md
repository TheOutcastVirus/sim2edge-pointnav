# 09 — Behaviour cloning

You have an oracle that solves the task perfectly (chapter 05) and a network
(chapter 08). The obvious move is supervised learning: record what the oracle
does, fit the network to it. That is **behaviour cloning**.

This chapter shows it working, then shows it failing — with real numbers from a
run on the training host — and then shows the fix.

## The setup

```
for each episode:
    reset the environment
    plan a path with the oracle
    at each step:
        record (observation, oracle_action)
        step with the oracle's action
```

The observation is what the *robot* would see (noisy lidar, drifting odometry);
the label is what the *oracle* does with ground truth. The network learns a
mapping from partial observation to expert action.

`collect_episodes` at `sim2edge/training/data.py:145` does this.

## Storage: quantized demonstrations

125,852 steps × a 96×160 float32 depth image is 7.7 GB. `Demonstrations` at
`data.py:53` compresses:

```python
def _q8(x):  ...   # data.py:36  — depth and lidar to uint8
def _q16(x): ...   # data.py:44  — goal and actions to int16
```

Depth and lidar are already normalized to `[0, 1]`, so uint8 gives 256 levels —
about 3 cm of resolution over an 8 m lidar range, well below the noise floor
domain randomization adds. Actions get int16, because a coarse action label would
put a quantization floor under the BC loss itself.

`dequantized(idx)` at `data.py:119` converts back on the fly during training.

There is a bug worth knowing about hiding in that function historically — see
chapter 15. The short version: the depth placeholder for lidar-only mode has to
dequantize to the *same* value the live path produces for absent data, and once
it did not.

## Loss: on normalized actions, with a weighted `ω`

`_loss` at `bc.py:76`:

```python
pred   = _normalized_actions(pred_physical, spec)
target = _normalized_actions(target_physical, spec)
err = (pred - target) ** 2
weight = torch.tensor([1.0, omega_weight])   # omega_weight = 1.5
return (err * weight).mean()
```

Two decisions:

**Normalize before comparing.** `v` spans `[0, 0.30]` and `ω` spans `[-1.5, 1.5]`.
A raw MSE on physical units would weight angular error 25× higher purely because
of unit choice. Normalizing to `[-1, 1]` makes the weighting a decision instead of
an accident.

**Then deliberately weight `ω` at 1.5** (`bc.py:46-49`):

> Slightly above 1 because heading errors compound over a trajectory while speed
> errors mostly do not.

Drive 10% too slow for a second and you are 3 cm behind. Turn 10% too little for a
second and you are pointed somewhere else, and every subsequent decision is made
from the wrong place.

**Only the actor trains** (`bc.py:100-105`): the critic has no supervision signal
from demonstrations, and letting the optimizer touch it would just apply weight
decay to weights that are about to matter in PPO.

## Run it

On the training host:

```bash
ssh vkommera@dronelab12-4
cd ~/Documents/Qualcomm/sim2edge-ventuno
.venv/bin/python scripts/train.py bc \
    --episodes 300 --epochs 6 --lidar-only \
    --dagger-rounds 0 --eval-episodes 100 \
    --out runs/tutorial_bc
```

Real output:

```
spec fingerprint da7943e44905b1c5  device cuda  lidar-only
[BC] collecting 300 oracle episodes (depth=off)
  [30/300] kept 12772 steps, expert SR 100.0%
  ...
  [300/300] kept 125852 steps, expert SR 99.0%
[BC] 125852 steps, 53 MB, expert SR 99.0%, 156s
[BC] final training on 125852 samples
  epoch   1/6  train 0.02025  val 0.00168
  epoch   2/6  train 0.00125  val 0.00096
  epoch   3/6  train 0.00088  val 0.00088
  epoch   4/6  train 0.00077  val 0.00078
  epoch   5/6  train 0.00070  val 0.00075
  epoch   6/6  train 0.00064  val 0.00071
[BC] final eval: SR  55.0%  SPL 0.523  coll 42.0%  timeout  3.0%  t_goal  18.0s  jerk  0.45  interv  0.00
[BC] saved runs/tutorial_bc/bc_final.pt
```

Note the fingerprint printed at startup, and 53 MB for 125 k steps — that is the
quantized storage working (uncompressed float32 would be ~90 MB for lidar-only,
and 7.7 GB with depth on).

## Now look at those numbers again

| | |
|---|---|
| Validation loss | **0.00071** — the network fits the expert almost exactly |
| Success rate | **55%** |
| Collision rate | **42%** |
| Oracle success rate | **100%** |

The network predicts the expert's action to within 0.0007 in normalized units, and
crashes into things in nearly half of all episodes.

**This is not a bug, and it is not underfitting. It is the fundamental failure
mode of behaviour cloning**, and reproducing it on purpose is the most useful
thing in this chapter.

## Why: compounding error under distribution shift

Supervised learning assumes training and test data come from the same
distribution. In BC that assumption is false in a specific, vicious way.

The training data is states the **oracle** visits. The oracle never drives near a
wall — it plans through inflated free space with a 6 cm margin. So there are
almost no examples of "you are 15 cm from a wall, angled at it" in the dataset.

At test time the *policy* drives. It makes a tiny error — 0.0007 worth. That puts
it slightly off the expert's trajectory, which is a slightly less familiar state,
so the next error is a little larger. Which puts it further off. Within a few
hundred steps it is somewhere the oracle never went, the network has no idea what
to do there, and it drives into a wall.

The classic result: BC error compounds as **O(T²)** in the episode horizon, not
O(T). At T = 1200 steps that quadratic is brutal.

Notice the diagnostic signature — and learn it, because you will see it again:

> **Low validation loss, high failure rate.** Any time those two coexist, suspect
> distribution shift rather than model capacity. Adding data from the same
> distribution will not help; adding epochs will not help; a bigger network will
> not help.

## The fix: DAgger

**Da**taset **Agg**regation. The insight is that you need labels *in the states
the policy actually visits*, and you can get them because the oracle can label any
state, not just states it chose.

```
1. Train BC on oracle data.
2. Let the POLICY drive. Record the states it visits.
3. Ask the ORACLE what it would have done in each of those states.
4. Add those (policy state, oracle action) pairs to the dataset.
5. Retrain. Repeat.
```

Round 2 teaches exactly the recoveries missing from round 1: "you are 15 cm from a
wall, angled at it — here is what the expert would do."

`run_bc_pipeline` at `bc.py:147` implements the loop. The key call at `:197-201`:

```python
extra, extra_stats = collect_episodes(
    n_eps, env_cfg, spec, seed_base=1_000_000 * (round_idx + 1),
    driver=TorchPolicy(policy, spec, cfg.device), beta=beta,
    keep_failures=True, ...)
```

Three details, each load-bearing:

**`driver=` the policy.** `collect_episodes` docstring at `data.py:160-164`:

> Passing a trained policy makes this a DAgger round: the states visited are the
> *policy's* state distribution while the labels remain the oracle's actions,
> which is the entire point.

**`beta`** mixes the two drivers (`data.py:166-167`): with probability `beta` the
oracle drives this step. `scripts/train.py:54` schedules `β = 0.5, 0.25` — later
rounds let the policy wander further from the expert's distribution.

**`keep_failures=True`.** Failures are *kept* in DAgger rounds, unlike the initial
collection. `bc.py:195-196`:

> the recoveries near a failure are the whole reason for the round.

For the initial BC set the default is off, because — per `data.py:169-172` —
imitating a trajectory that ends in a wall is actively harmful. The polarity flips
once the policy is the one driving.

**Different seed base.** `1_000_000 * (round_idx + 1)` keeps DAgger rounds on
disjoint layouts from the initial collection.

## Run DAgger

```bash
.venv/bin/python scripts/train.py bc \
    --episodes 300 --epochs 6 --lidar-only \
    --dagger-rounds 2 --dagger-episodes 150 --eval-episodes 100 \
    --out runs/tutorial_dagger
```

It evaluates after every round, so you can watch what the on-policy data does.
Here is what actually happened:

```
round 0  samples  125852  SR  55.0%  SPL 0.527  coll  45.0%  val 0.00071
round 1  samples  190615  SR  35.0%  SPL 0.345  coll  65.0%  val 0.00099
final    samples  258681  SR  48.0%  SPL 0.478  coll  52.0%  val 0.00138
```

**DAgger made it worse.** Round 0 reproduces the baseline — 55.0% success, and a
collision rate within a few points of the earlier run's 42% (the difference is
seed noise in the data ordering, on 100 eval episodes). Then it drops to 35%, and
recovers only to 48%. Worse than where it started.

This is a real result from a real run, not a contrived one, and it is more useful
than a clean win would have been.

### Diagnosing it

Look at what moved together:

| | round 0 | final |
|---|---|---|
| samples | 125,852 | 258,681 |
| **train** loss | 0.00064 | 0.00108 |
| **val** loss | 0.00071 | 0.00138 |
| epochs | 6 | 6 |

Validation loss doubled — which is *expected* and even desirable: the dataset got
harder, because it now contains near-wall recovery states the oracle set never
had. Predicted correctly.

But **training** loss also nearly doubled. That is the tell. If the model had
fitted the harder dataset, train loss would be low and val loss high. Both being
high means the model has not fitted the data at all — it is **underfit**.

The cause is a budget bug in the experiment, not in DAgger: `--epochs 6` is a
fixed constant, while the dataset grows 2× across rounds. Round 0 gets 6 epochs
over 126 k samples; the final round gets 6 epochs over 259 k samples of a harder
distribution. Same gradient steps per sample would need ~12 epochs, and a harder
distribution needs more than that.

So the honest conclusion is: **at this budget, DAgger costs more than it pays.**
Not "DAgger doesn't work" — the mechanism is sound and well-established — but "you
cannot bolt it on without scaling the training budget with the dataset."

### Fixing it

Things to try, cheapest first:

1. **Scale epochs with dataset size.** `--epochs 15` or more for the later rounds.
   The single most likely fix.
2. **More episodes per round.** 150 episodes of on-policy data against 300 of
   oracle data is a small correction; the DAgger literature uses comparable or
   larger.
3. **Check β.** `scripts/train.py:54` schedules `β = 0.5, 0.25`. At `β = 0.5` the
   oracle drives half the steps, so the visited states are only half-shifted
   toward the policy's own distribution.

On that last point, be careful reading the log line:

```
[BC] DAgger collected 68066 steps (policy SR 98.7%)
```

That 98.7% is **not** the policy's success rate. It is the success rate of the
*mixed* driver — policy plus oracle at probability `β`. With `β = 0.5` the oracle
is doing half the driving, which is why the number looks close to the oracle's
99% and nothing like the policy's actual 48%. The label in `bc.py:202-203` is
optimistic; the real measurement is the held-out eval.

### What to take from this

Two things, and the second is the more important:

- Adding data can make a supervised model worse if the training budget does not
  grow with it. Watch **train** loss, not just validation loss, to tell underfitting
  from distribution shift.
- **Reporting a negative result is the job.** The tempting move here is to tune
  until DAgger wins and show only that. The run above took eight minutes and told
  you something true about the configuration in this repository. If you rebuild
  this project, you now know that `dagger_rounds` with a fixed `epochs` is a trap.

## Where BC sits in the plan

The implementation plan treats BC as the guaranteed demo (`IMPLEMENTATION_PLAN.md`
Gate B): a BC+DAgger policy is usually demo-grade on its own, and PPO is an
upgrade that is *allowed to fail*. `bc.py:158-159`:

> The best BC checkpoint must stay deployable at all times: it is the Gate-B
> fallback, and PPO is allowed to fail.

Every checkpoint carries the spec fingerprint (`_save`, `bc.py:218-231`), so
export, calibration, and the ROS node can all verify they are working with a
policy trained under the preprocessing contract they are about to apply.

## The counter-argument

`REBUILD_PLAN.md:26` calls cutting BC+DAgger "the single biggest cut," and the
reasoning is worth engaging with seriously:

> PPO from `rsl_rl` solves point-goal nav with a dense progress reward. BC
> bootstrapping is for sparse-reward, long-horizon nav (Habitat/PointNav at 500+
> steps). Yours is ≤60 s episodes in a 10 m room with a 360° lidar — dense
> progress reward is enough.

That is a real and probably correct argument. BC exists here to make the *first*
policy cheap, and to give a fallback if PPO misbehaves. If PPO reliably solves the
task from scratch — which nobody in this repository has tested (chapter 15) —
then chapter 05's oracle, this chapter's BC, and chapter 10's careful handoff
machinery all become unnecessary at once.

But you now understand *why* they exist, which is what makes deleting them a
decision instead of a guess.

## Check yourself

- Validation loss 0.0007, success rate 55%. Why does more data from the same
  distribution not fix this?
- Why does DAgger keep failed episodes when plain BC discards them?
- `β` decays from 0.5 to 0.25 across rounds. What would `β = 0` in round 1 risk?
  What would `β = 1` accomplish?
- BC labels come from an oracle with ground-truth pose, but observations have
  drifting odometry. Is that a bug or a feature?

→ [10 — PPO](10-ppo.md)
