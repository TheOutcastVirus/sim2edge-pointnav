# 11 — Evaluation

A success rate on its own is close to meaningless. It does not say whether the
policy took a sensible route, whether it survives noise, or whether it only
succeeded because a watchdog kept rescuing it. `sim2edge/eval/harness.py`
(233 lines) reports all of that.

## The metrics

`EvalMetrics` at `harness.py:40`:

| Metric | Meaning |
|---|---|
| `success_rate` | fraction reaching within 0.35 m |
| `spl` | Success weighted by Path Length — see below |
| `collision_rate` | fraction ending in a collision |
| `timeout_rate` | fraction hitting the 1200-step cap |
| `mean_time_to_goal` | seconds, **successful episodes only** |
| `mean_jerk` | mean ‖Δa‖/dt in normalized action units |
| `mean_return` | mean episode reward |
| `mean_interventions` | safety-layer activations per episode |
| `intervention_episode_rate` | fraction of episodes where safety acted at all |

### SPL

`_spl` at `reference_env.py:364`:

```python
def _spl(optimal: float, actual: float) -> float:
    if actual <= 1e-6:
        return 1.0 if optimal <= 1e-6 else 0.0
    return float(min(1.0, optimal / max(actual, optimal)))
```

and it is **0 for a failed episode by definition** (`harness.py:153`).

SPL is the standard point-goal navigation metric. It answers "did you get there,
and did you take a reasonable route?" in one number:

- Reach the goal along the geodesic optimum → 1.0
- Reach it by driving twice as far → 0.5
- Fail → 0.0

The denominator is `Room.optimal_path_length` — the geodesic distance from
chapter 04. This is the fourth consumer of that field, after the reward, the
planner, and the feasibility test.

Comparing chapter 09's numbers:

| Policy | SR | SPL |
|---|---|---|
| Oracle | 100.0% | 1.000 |
| BC (300 episodes, no DAgger) | 55.0% | 0.523 |

SPL 0.523 against SR 0.55 tells you something specific: **the episodes it wins, it
wins efficiently.** If it were succeeding by wandering, SPL would be far below the
success rate. Here `0.523 / 0.55 ≈ 0.95`, so successful runs are within about 5%
of optimal path length. The policy is not a bad driver — it is a driver that
crashes 42% of the time. Those are different problems with different fixes.

### Jerk

`harness.py:129-133`:

```python
norm = np.array(spec.action.normalize(float(action[0]), float(action[1])))
total_jerk += float(np.linalg.norm(norm - prev_norm)) / dt
```

Mean action change per second. This is not an aesthetic concern: a jerky command
stream on a real Create3 means wheel slip, motor wear, and odometry that degrades
faster than the drift model assumes. It is also the metric the `smoothness` reward
term (`spec.py:246`) exists to move.

For scale, from a run in chapter 05 and 09: the oracle scores 0.41, BC scores 0.45,
and a random policy scores about 20.8.

### Interventions

`harness.py:52-55`:

> Safety-layer interventions per episode; 0 when the layer is disabled.
> Fraction of episodes in which the safety layer acted at all.

`scripts/train.py:119-120` explains why this is first-class:

> interventions are a first-class metric: a run that succeeds only because the
> watchdog kept saving it is a finding, not a success

A policy at 95% success with 40 interventions per episode is not a 95% policy. It
is a safety layer with a neural network attached. Counting interventions is what
keeps that distinction visible.

## Held-out seeds

`harness.py:31-34`:

```python
HELD_OUT_SEED_BASE = 10_000_000
```

> Evaluation seeds live in a disjoint range from training seeds. Training uses
> [0, 1e6); evaluation starts here. Overlapping the two is the single easiest way
> to report a success rate that does not survive contact with new layouts.

Because rooms are procedurally generated from a seed, the seed *is* the layout.
Evaluating on training seeds measures memorization. Note that DAgger rounds use
`1_000_000 * (round_idx + 1)` (`bc.py:198`), staying well below the eval base.

## Noise tiers

`evaluate_noise_tiers` at `:183` runs the same policy at three randomization
scales:

```python
tiers: Sequence[float] = (0.0, 1.0, 2.0)
```

> The 2x tier is a robustness margin, not a target. A policy that scores well at
> 1.0 but falls apart at 2.0 has fit the randomization ranges rather than become
> robust, and the ranges were only ever estimates from a few rosbags.

That last clause is the real argument. The ranges in `spec.py` are estimates. If
your policy is only good at exactly those values, and the real robot is 30% noisier
than your rosbag suggested, you have nothing. A policy that degrades *gracefully*
from 0× to 2× will probably survive being wrong about the ranges.

Crucially, `:208-209`:

> Same seeds at every tier: the layouts are identical and only the noise differs,
> so the comparison isolates robustness.

Different layouts across tiers would confound layout difficulty with noise
robustness.

### Reading the shape

| Pattern across 0× / 1× / 2× | Diagnosis |
|---|---|
| 95 / 93 / 90 | healthy — graceful degradation |
| 95 / 94 / 62 | overfit to the randomization ranges |
| 95 / 70 / 68 | the 1× ranges are too wide, or DR was off during training |
| 60 / 58 / 57 | noise is not the problem; the policy is just weak |

## Run it

```bash
ssh vkommera@dronelab12-4
cd ~/Documents/Qualcomm/sim2edge-ventuno

.venv/bin/python scripts/train.py eval \
    --init runs/tutorial_bc/bc_final.pt \
    --eval-episodes 100 --lidar-only \
    --out runs/tutorial_eval
```

`cmd_eval` at `scripts/train.py:106` runs all three tiers and then a fourth
evaluation with the safety layer enabled, writing everything to `eval.json`.

### Baselines you should always have

```bash
.venv/bin/python -c "
from sim2edge.envs.reference_env import NavEnv, EnvConfig
from sim2edge.eval.harness import evaluate, oracle_policy
from sim2edge.eval.torch_policy import random_policy

cfg = EnvConfig(use_depth=False)
env = NavEnv(cfg, seed=0)
print('oracle:', evaluate(oracle_policy(env), 50, env=env).summary())
print('random:', evaluate(random_policy(seed=0), 50, cfg).summary())
"
```

```
oracle: SR 100.0%  SPL 1.000  coll  0.0%  timeout  0.0%  t_goal  18.7s  jerk  0.39  interv  0.00
random: SR   0.0%  SPL 0.000  coll 100.0%  timeout  0.0%  t_goal   nans  jerk 20.80  interv  0.00
```

The oracle is the ceiling; random is the floor. A policy that beats neither has not
learned.

Look at the jerk column: 0.39 for the oracle, **20.80** for random — a 50× gap.
Jerk separates "driving" from "twitching" far more sharply than success rate does,
which makes it a useful early signal during training, before success rate has moved
off zero.

`t_goal nans` for random is correct, not a bug: `mean_time_to_goal` averages
successful episodes, and there are none.

## Policy adapters

`sim2edge/eval/torch_policy.py` (108 lines) provides three implementations of the
same `PolicyFn` interface — `obs dict → (v, ω)`:

| Class | Line | Runs |
|---|---|---|
| `TorchPolicy` | `:21` | a PyTorch checkpoint |
| `ExecuTorchPolicy` | `:64` | a lowered `.pte` |
| `random_policy` | `:96` | uniform random actions |

Because they share an interface, the *same* `evaluate()` call measures the FP32
model and the INT8 `.pte`. That is what makes the quantization gate in chapter 13
meaningful: you compare success rates, not just tensor differences.

`TorchPolicy.from_checkpoint` at `:37` loads and — importantly — the checkpoint
carries the spec fingerprint from `bc.py:218-231`.

## The evaluation protocol

`IMPLEMENTATION_PLAN.md` §5 specifies what a real run reports:

- 500 held-out episodes
- Success Rate, SPL, collision rate, mean time-to-goal, mean jerk
- at three noise tiers
- plus a quantization parity gate before any robot run: FP32 vs INT8 on the same
  1000 recorded observations, requiring **sim SR drop < 2%** when the quantized
  actions are replayed through the sim eval

That last item is the one people skip and regret. A small tensor-level difference
between FP32 and INT8 says nothing about whether the *behaviour* survived. Replaying
through the evaluation harness does.

## A caution about the numbers in this repository

Every evaluation number you can produce today comes from
`sim2edge/envs/reference_env.py` — a kinematic model with no rigid-body physics
and no contacts. A 95% success rate there is evidence that the *pipeline* works.
It is weak evidence about the robot.

The Isaac Lab environment that would give stronger evidence has never been run
(chapter 15). Keep the distinction sharp when reporting results, because it is
exactly the kind of thing that is easy to blur and embarrassing to have blurred.

## Check yourself

- A policy scores SR 0.80 and SPL 0.40. What is it doing?
- Why is SPL defined as 0 for failures rather than left undefined?
- Why must all three noise tiers use the same seeds?
- Your policy gets 92% success with `mean_interventions = 35`. Is it a good
  policy? What single experiment settles it?
- `mean_time_to_goal` averages successful episodes only. Why would including
  failures make it misleading?

→ [12 — Isaac Lab](12-isaac-lab.md)
