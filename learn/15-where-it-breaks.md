# 15 — Where it breaks

You now know how every piece works. This chapter is about what is actually true of
this repository today, what has already gone wrong, and what to do next.

## Honest status

| Component | State |
|---|---|
| Spec, preprocessing, C++/Python parity | Complete; parity **bit-exact** |
| Rooms, sensors, oracle | Complete; oracle **100% SR** with randomization on |
| Policy network (2.72 M shipped params) | Complete; `torch.export` verified with 4 inputs, all ops on the proven list |
| Reference env + eval harness | Complete, tested |
| BC + DAgger, PPO | Implemented; BC verified end to end |
| Export / calibration / parity / benchmark tooling | Complete — **untested against real QNN hardware** |
| ROS 2 node + safety layer | Complete — **compiles only the parity path; never built against ExecuTorch, never run on the board** |
| Isaac Lab task | Written against the shared spec — **never executed.** Four known defects |

Two sentences that matter more than the table:

1. **No policy from this repository has ever run on the target hardware.**
2. **The Isaac Lab environment — the one that would train the policy that ships —
   has never been started.**

Everything you ran in chapters 01–11 was the numpy pipeline. It works, and it is
real, and it is evidence about the *pipeline*, not about the robot.

## Failures already paid for

These are documented in `docs/architecture.md` and `docs/training_host_setup.md`
because they cost real debugging time. They are the most valuable content in the
repository.

### The depth placeholder bug

In lidar-only mode the depth tensor must still exist, so the exported graph's
signature never changes. It should be filled with the value that means "no
return" — `invalid_fill = 1.0` after normalization.

**Two components filled it with `0.0` instead.** From `docs/architecture.md:60-66`:

> Zero is not a neutral value: it denormalizes to `min_m`, an obstacle 0.3 m dead
> ahead filling the entire frame. The offenders were `Demonstrations.dequantized`
> (BC trained on one constant and rolled out on another: **validation loss fell
> fourfold at a 0% success rate**) and `isaaclab_env/nav_task.py::_get_observations`.

Look at that signature again: **validation loss fell fourfold at a 0% success
rate.** The training metric improved while the policy became useless, because
training and rollout disagreed about what "no camera" looks like.

You can see the artifact of this in the repo. `runs/demo.log`:

```
BC        : SR   0.0%  SPL 0.000  coll 20.0%  timeout 80.0%  jerk  0.22
random    : SR   0.0%  SPL 0.000  coll 100.0%  timeout  0.0%  jerk 20.80
val loss  : 0.00454 -> 0.00179
```

Fixed in commit `e84a9f6` ("Fix depth placeholder mismatch"). The stated lesson is
narrower and better than "share constants" — the constant *was* shared, and read
from the spec, nearly everywhere it appeared:

> **a placeholder for absent data must be the same value the live path produces
> for absent data**, and the way to enforce that is a test that reads both rather
> than asserting a literal.

Pinned by `test_lidar_only_placeholder_matches_the_env`.

### The DAgger budget trap (found while writing chapter 09)

Running the documented DAgger recipe end to end made the policy **worse**: 55% →
48% success, with collisions rising from 42% to 52%. Both training and validation
loss rose, which is the signature of underfitting rather than distribution shift.

Cause: `--epochs` is a fixed constant while the dataset doubles each round, so
later rounds get half the gradient steps per sample on a harder distribution.
`scripts/train.py:53-54` builds the DAgger schedule without touching `epochs`.

This is not a defect in DAgger, and it is not fatal — but the defaults in this
repository do not work as configured, and nothing warned about it. Full numbers and
diagnosis in chapter 09.

### The gap that is still open

The same constant must agree across three implementations — the numpy env, the
Isaac Lab adapter, and the C++ node — and **only two are covered by an automated
check**. `_get_observations` in Isaac Lab reimplements the spec's normalization in
torch, because its data is a GPU batch while `preprocess.py` is numpy. Nothing
verifies the two agree. `nav_task.py` names `tests/test_isaac_obs_parity.py` in a
comment; the file does not exist.

**If you run Isaac Lab, write that test first.** It is the same bug class that has
already cost this project two debugging sessions.

## The failure modes you should expect

| Symptom | Likely cause | Where to look |
|---|---|---|
| Low val loss, high failure rate | BC distribution shift | ch. 09 — needs DAgger or RL, not more data |
| Both train *and* val loss rise after adding data | underfit: fixed epoch budget on a growing dataset | ch. 09 — the DAgger trap, reproduced |
| Policy fine in sim, wrong on robot | preprocessing/calibration mismatch | ch. 13 — check the sidecar fingerprint first |
| Reward spikes to ~1e6 | `UNREACHABLE` sentinel differenced | `reference_env.py:227-233` |
| Policy drives into concave obstacles | Euclidean instead of geodesic progress | ch. 06 |
| PPO destroys a good BC policy | random critic → noisy advantages | ch. 10 — critic warmup |
| Robot turns the wrong way | reversed wheel joint order | `training_host_setup.md` §4.2 |
| Lidar sees nothing in Isaac | warp mesh snapshot is static | ch. 12 — test in the first hour |
| Policy confidently wrong at edges | quantization calibration too narrow | ch. 13 |
| Success rate collapses at 2× noise | fit the randomization ranges, not robustness | ch. 11 |
| Robot stuck 25 cm from a wall | safety layer freezing rotation too | ch. 14 |

## Known limitations, by design

From `README.md:227-244`:

- **Odometry drift.** The goal is anchored in the Create3's odom frame, which
  drifts 1–2%/m with no SLAM correction. Keep demo goals ≤ 8 m. The policy is
  trained against odometry-drift randomization for exactly this reason, but drift
  is **tolerated, not eliminated**.
- **The policy is memoryless.** No recurrence in v1 — it complicates export,
  quantization, and the node. If corner oscillation appears, the planned fix is a
  4-frame stack of lidar + last action before reaching for a GRU.
- **Lidar padding is zeros, not circular.** Chapter 08.
- **The reference env is kinematic**, not rigid-body.
- **The two `/opt/executorch` patches are fragile.** A fake `QCS8300` chipset enum
  mapping to HTP V75, and a `sed` enabling the QNN AOT pybind on aarch64. Pin the
  working commit and do not upgrade mid-project.

## The size problem

~7,200 lines of Python plus 1,600 of C++, none of which has run on the target.
`REBUILD_PLAN.md` argues the same demo fits in roughly 800–1,000 lines, and having
read the code you are now qualified to judge it.

| Cut | Lines | The argument | My read |
|---|---|---|---|
| ManiSkill adapter | 218 | two simulators = two task definitions to sync | **Already done.** Uncontroversial |
| Oracle | 620 | only existed to generate BC data | Contingent — it is also the reward, the SPL denominator, and the feasibility test |
| BC + DAgger + PPO | 955 | dense progress reward is enough for 60 s episodes with 360° lidar | Probably right, untested |
| Geodesic reward | — | Euclidean is nearly identical in convex-ish rooms | Weakest cut; ch. 04 measured a 0.8 m gap on one seed |
| C++ node + parity apparatus | ~830 | write it in rclpy, import the same function | **Strongest cut in the plan** |
| `rooms.py` | 369 | randomize obstacle poses on reset | Right for Isaac Lab; the numpy env needs the grids |
| `spec.py` schema layer | ~400 of 546 | keep the numbers, drop the dataclass machinery | Right — keep the fingerprint |

The one contradiction worth resolving yourself: `rooms.py:11-23` argues that low
and overhanging obstacles are the *entire* reason a depth camera exists, while
`REBUILD_PLAN.md:161-164` says depth adds little navigation capability over a 360°
lidar and is mainly a better demo. Both cannot be right. If you want depth to be
defensible rather than decorative, keep at least one obstacle class the lidar
structurally cannot see — that is ~10 lines of spawn logic, not a module.

## What I would actually do next

In this order. Each step is cheap and each one de-risks the next.

**1. Get Isaac Lab running on a stock task.** (½ day)

```bash
ssh vkommera@dronelab12-4
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0 --headless
```

Rising reward curve in two minutes, or stop and fix the install. Do not debug your
env on top of a broken one.

**2. Empty-room point-goal, no sensors.** (1 day) `REBUILD_PLAN.md` phase 1: a
capsule driven by `write_root_velocity_to_sim`, 5 observation floats, Euclidean
progress, stock rsl_rl. Target >95% in five minutes.

This is the highest-value experiment available, because it answers the question
the whole rebuild debate turns on: **does PPO alone solve this?** If yes, chapters
05, 09, and most of 10 become optional. If no, you know why the oracle exists.

**3. Test the raycaster in the first hour of phase 2.** Move an obstacle, confirm
the scan changes. If it does not, write the analytic ray–circle version — ~40
lines, exact, faster.

**4. Write `tests/test_isaac_obs_parity.py` before training anything real.** The
one open instance of the bug class that has already cost this project twice.

**5. Only then, export.** Chapter 13's smoke test runs today with `--skip-qnn` —
use it after every model change.

## A closing note on the two habits

`REBUILD_PLAN.md:271-277` ends with two rules that are worth more than the plan
itself:

> 1. **One milestone per commit, and write the commit message before the code.**
>    If you can't state what "working" means, you don't understand the phase yet.
> 2. **When you're stuck, ask for the concept, not the file.** "How does Isaac Lab
>    handle per-env obstacle randomization?" leaves you with knowledge; "write my
>    `_reset_idx`" leaves you where you are now.

The second one is the reason this curriculum exists. You have now read the
concepts. The code that comes next should be yours.

## Check yourself — the whole thing

- Trace a single lidar beam from `simulate_lidar` to a `/cmd_vel` message. Name
  every file it passes through.
- Name three independent mechanisms that would catch a changed `LidarSpec.max_m`
  before it reached the robot.
- You have 55% success and 42% collisions. Give three distinct hypotheses and the
  cheapest experiment for each.
- Which of the seven cuts in the table above would you make first, and what
  evidence would change your mind?

← [00 — Start here](00-start-here.md)
