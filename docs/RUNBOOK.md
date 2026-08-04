# Runbook

How to run things. Rationale and history live in `docs/daily/`.

---

## Setup

Two Python environments, deliberately separate — Isaac Sim pins Python 3.11 and
torch 2.7, and the rest of the project has no reason to.

| | Path | Python | Used for |
|---|---|---|---|
| Project | `.venv` | 3.12 | Tests, layout pools, dataset, training |
| Isaac | `~/isaac-venv` | 3.11 | Anything importing `isaaclab` |

```bash
uv sync --extra dev                      # project venv
~/isaac-venv/bin/python -m pip install -e .   # make sim2edge importable from Isaac
sudo apt install ros-jazzy-turtlebot4-description ros-jazzy-xacro
```

Isaac Sim 5.1.0 / Isaac Lab 0.54.4, source install at
`~/Documents/Qualcomm/IsaacLab`. Moving that directory breaks all six editable
packages — re-run `./isaaclab.sh --install` from the new location, or symlink
instead.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

311 pass, 3 skipped without Isaac. No GPU or simulator required — keep it that way.

---

# Phase 1 — Behaviour cloning data

## 1. Build the robot asset (one-time)

```bash
~/isaac-venv/bin/python tools/convert_tb4.py --output assets/turtlebot4_lite.usd
```

~90 s, mostly Isaac booting. Skip if `assets/turtlebot4_lite.usd` exists. Expect
`Unresolved reference .../wheel_drop_*/visuals` warnings; they are cosmetic.

Add `--urdf-only` to expand the xacro without Isaac (fast, and activates two more
tests).

## 2. Warm the warp cache (one-time per warp version)

```bash
ls ~/.cache/warp/     # want a dir matching: python -c 'import warp; print(warp.config.version)'
```

If absent:

```bash
~/isaac-venv/bin/python scripts/collect_bc.py --num_envs 4 --num_layouts 8 --check
```

> Several minutes, **no output**, ~100% CPU on one core. This is warp compiling
> ray-cast kernels, not a hang — `SIGINT` gives no traceback because it is inside
> native code. **Do not interrupt:** a killed compile is discarded and the next
> run starts over.

## 3. Bring-up check

```bash
~/isaac-venv/bin/python scripts/collect_bc.py --num_envs 4 --gui --check
```

Asserts rays hit geometry, bins carry returns, robots spawn in free space,
layouts differ per environment, and the oracle is not pinned at max spin. It
also commands a fixed `1.0 rad/s` turn and `0.3 m/s` straight drive for 50 ticks
each, and requires the measured base response to track both within 20%.
The drive-response probe also runs automatically before every collection and
must pass before the collector allocates buffers or writes any samples.

The adapter integrates the commanded planar Twist directly at the 200 Hz
physics rate. Wheel joints are still animated, but the imported cylinder
contacts are not used as a tire model: they track straight commands yet scrub
unpredictably under opposing wheel speeds. Collision termination continues to
use the same body-radius occupancy grid as the oracle.

GUI: right-drag orbits, middle-drag pans, `F` frames a selected prim. Add
`--dump-scene` for env origins, camera pose and prim world positions when the
viewport is unreadable.

## 4. Collect

```bash
~/isaac-venv/bin/python scripts/collect_bc.py \
    --num_envs 8192 \
    --episodes 65536 \
    --num_layouts 8192 \
    --starts_per_layout 32 \
    --validation_fraction 0.05 \
    --seed 13579 \
    --sigma 0.15 \
    --out data/bc_diverse_v1
```

This creates `data/bc_diverse_v1/train/` and `validation/`. The split owns whole
layout IDs, so offline validation measures layout generalization instead of
seeing new episodes from training geometry. Each layout supplies 32 feasible
start positions and reset samples a fresh yaw, producing up to 262,144 distinct
layout/start tasks before yaw variation. The earlier 500k-episode corpus reused
only 512 fixed layout/start/yaw templates and should not be used for the new
baseline.

Collection appends after existing `bc_*.npz` files in each split directory.
Use a clean output directory when changing the seed or collection parameters;
the collector never overwrites prior shards. Pre-collision prefixes are retained
by default; add `--filter-failures` only for the legacy progress filter. `Ctrl-C`
writes buffered episodes and exits cleanly; a second `Ctrl-C` abandons them.

Progress line:

```
step  1200  kept  4120 (full 1840 trunc 1990 part 290)  dropped 31  | 33.4 ms/step (sim 30.7  rec 2.5  writer-wait 0.2)  | 7665 env-steps/s
```

- `full` reached the goal, `trunc` was cut before a graze, `part` ran out of clock
- `sim` is simulation/oracle work, `rec` is the host recorder, and
  `writer-wait` is time actually blocked behind the asynchronous shard writer
- shard concatenation, compression, and disk writes run on one background CPU
  thread; `--writer_queue` bounds its decompressed RAM backlog
- **yield is only meaningful once `kept + dropped` is several times `num_envs`** —
  below that only fast episodes have finished and it always reads high

Watch at startup: the recorder buffer ceiling is printed and RSS should plateau
there; `ms/step` should stay flat; no `foundLostPairsCapacity` error in the first
30 s.

### Parameters worth knowing

| Flag | Default | Notes |
|---|---|---|
| `--num_envs` | 256 | Raises throughput until simulation or recording saturates. The recommended 4090 run uses 8192. Do not run multiple collectors on one GPU. |
| `--episodes` | 2000 | Retained episode prefixes, not attempts. Overshoots by at most one episode. |
| `--num_layouts` | 512 | Geometry/goal pool size. Use 8192 for the diverse baseline; layouts are built once. |
| `--starts_per_layout` | 32 | Feasible start candidates per layout; yaw remains fresh on every reset. |
| `--sigma` | 0.15 | Execution noise as a fraction of each axis limit. `0` collects pure on-policy expert data — the distribution BC fails on. |
| `--shard_steps` | 200000 | ~160 MB per shard. |
| `--validation_fraction` | 0.05 | Fraction of layout IDs reserved in `validation/`; `0` writes the legacy flat format. |
| `--writer_queue` | 2 | Maximum outstanding async shards, roughly 300 MB decompressed each. Increase only if `writer-wait` is nonzero and RAM is available. |
| `--filter-failures` | off | Legacy mode: drops short or low-progress prefixes. |

Memory scales as `num_envs × 900 × 720 × 2 B`: **5.4 GB at 4096, 10.9 GB at
8192**, plus a transient spike when many environments finish on the same step.

If PhysX reports `foundLostPairsCapacity`, raise
`gpu_found_lost_pairs_capacity` in `NavEnvCfg.sim` — the message states the
number it wants. Do not ignore it: dropped contacts mean robots pass through
geometry while the run still looks healthy.

## 5. Verify before training

```bash
.venv/bin/python scripts/verify_shard.py data/bc_diverse_v1/train/bc_000.npz
.venv/bin/python scripts/verify_shard.py data/bc_diverse_v1/validation/bc_000.npz
```

Runs without Isaac. Checks structure, that storage is raw metres and not
pre-binned, that labels sit inside the Create3's envelope, that perturbation is
live, that observations assemble correctly, and that noise is fresh per epoch.

Progress is reported as a quality warning rather than a hard failure: collision-truncated prefixes still contain valid oracle labels, but a high warning count means the drive dynamics need more work.

---

# Phase 2 — BC training

The trainer streams one shard to the GPU while four background workers inflate
an ordered eight-shard queue into host RAM; it never concatenates the corpus.
The queue is bounded at roughly 2.4 GB on the 128 GB host. New corpora use the
collector's disjoint layout split automatically; `--val-shards` applies only to
legacy flat corpora.

## 1. Pipeline smoke test

```bash
.venv/bin/python scripts/train_bc.py data/bc_diverse_v1 \
    --out runs/bc_smoke --epochs 1 --max-train-shards 2 \
    --val-shards 1 --no-compile
```

This exercises real NPZ decompression, GPU preprocessing, fresh corruption,
backpropagation, validation and checkpoint reload. Delete or retain the smoke
directory as desired; a full run uses a different output directory.

## 2. Full 4090 run

```bash
.venv/bin/python scripts/train_bc.py data/bc_diverse_v1 \
    --out runs/bc_diverse_v1 --epochs 8
```

Hardware defaults are for the 24 GB RTX 4090: batch 8192, BF16 autocast,
compiled model, fused AdamW, and four-worker/eight-shard prefetch. Eight passes
are recommended for the smaller, more diverse corpus. Training uses fresh 1x
sensor and odometry corruption; validation reports both exact-clean and
deterministic 1x losses.

`--prefetch-workers` controls parallel NPZ decompression and
`--prefetch-shards` bounds the ready queue. The defaults use about 2.4 GB of RAM.
Each epoch records `train_input_wait_seconds`; if it is already near zero, more
workers cannot improve throughput. These loader-only options may be changed when
resuming because they do not affect sample order or optimizer state.

The run directory is deliberately self-describing:

| File | Meaning |
|---|---|
| `config.json` | Every training hyperparameter. |
| `dataset.json` | Exact train/validation shard split and manifest digest. |
| `metrics.jsonl` | One machine-readable record per completed epoch. |
| `latest.pt` | Resumable model, optimizer, scheduler and shard cursor. |
| `best.pt` | Lowest clean offline validation loss. |
| `final.pt` | Final completed epoch. |

Resume after an interruption (at most one shard is replayed):

```bash
.venv/bin/python scripts/train_bc.py data/bc_diverse_v1 \
    --out runs/bc_diverse_v1 --epochs 8 \
    --resume runs/bc_diverse_v1/latest.pt
```

Resume refuses a changed spec, dataset split or training configuration rather
than silently changing the run. Use a new output directory for a new experiment.

## 3. Closed-loop evaluation

Offline loss does not establish navigation quality. Run the best candidate in
Isaac on layouts generated from a seed disjoint from collection:

```bash
~/isaac-venv/bin/python scripts/eval_bc.py runs/bc_diverse_v1/best.pt \
    --episodes 512 --layout-seed 10000000 \
    --out runs/bc_diverse_v1/eval.json
```

Each environment contributes exactly one episode, so the 0x, 1x and 2x noise
tiers see the identical initial layout vector. The report includes success rate,
SPL, collision rate, timeout rate, time-to-goal, command jerk and inference
latency. Use these closed-loop metrics—not the smallest MSE—to select the policy.

## Implementation map

```text
sim2edge/models/nav_policy.py  fixed-shape deployable actor
sim2edge/training/data.py      corpus manifest, split and prefetch
sim2edge/training/bc.py        normalized loss, trainer and checkpoints
scripts/train_bc.py            simulator-free training CLI
scripts/eval_bc.py             held-out closed-loop Isaac evaluation
```

The explicit validation split measures held-out layouts from the same collection
seed. The entirely new `layout_seed` in closed-loop evaluation remains the
strongest generalization test.

---

# Phase 3 — PPO fine-tune

BC can only imitate the oracle; it has no notion of reward, so it never learns
that a collision is bad — only that the expert did not have one. PPO optimizes
return directly, starting from the BC checkpoint.

## 1. The reward

Implemented in `Sim2EdgeNavEnv._get_rewards`, per control step:

```text
  progress_reward   * (d_{t-1} - d_t)      geodesic metres, clamped at 0.5 m
+ goal_reward       * reached
- collision_penalty * collided
- time_penalty
- smoothness_penalty * ||a_t - a_{t-1}||²  normalized axes
```

Defaults are 3.0 / 10.0 / 10.0 / 0.005 / 0.05, and they live on `NavEnvCfg` so
an experiment changes a config field, not the environment.

`d` is **geodesic**, read from the same distance field that sets episode
difficulty and the SPL denominator. Euclidean progress pays the robot to drive
into the concave face of an obstacle. The field is filled from the nearest
reachable cell (`grids.fill_unreachable`), so it is finite everywhere and no
step into blocked space can be paid as a billion metres of progress.

Three arithmetic properties are asserted in `tests/test_isaac_adapter.py`: one
step of travel outearns the time penalty, reaching the goal outearns waiting out
the clock, and the progress clamp exceeds one grid cell.

## 2. Sanity check the loop before committing a run

```bash
~/isaac-venv/bin/python scripts/train_ppo.py \
    --init runs/bc_stack_dagger_v2_e4_final/best.pt \
    --num_envs 256 --num-steps 16 --iterations 5 --critic-warmup 2 \
    --out runs/ppo_smoke
```

**This is too short to measure `SR`, by construction.** 5 × 16 = 80 control
steps is 4 s of simulated time, and a successful episode on an 8 m geodesic
takes 200–550 steps at ~0.29 m/s. Only fast failures can finish, so the smoke
run reports ~100% collision no matter how good the policy is — the same
survivorship bias the collector's yield figure has, inverted. Read it for
crashes, throughput and reward scale only.

To measure `SR`, pin the actor and run long enough for episodes to complete:

```bash
~/isaac-venv/bin/python scripts/train_ppo.py \
    --init runs/bc_stack_dagger_v2_e4_final/best.pt \
    --num_envs 512 --num-steps 32 --iterations 40 --critic-warmup 40 \
    --out runs/ppo_frozen
```

`--critic-warmup` ≥ `--iterations` freezes the actor for the whole run, so
whatever `SR` comes out is the BC policy's own behaviour reproduced through the
PPO rollout — a direct test of observation assembly, the live noise, the action
convention and termination, with no learning involved. Expect a few points
below the checkpoint's deterministic held-out number, which is what sampling at
`--init-std` costs. Measured 2026-07-30: **55.0% success, 44.8% collision,
0.14% timeout over 2,900 episodes**, against a 58.4% BC baseline at 1× noise.
Per-iteration `SR` swings ±7 points because each iteration completes only ~75
episodes; pool across iterations before believing any single one.

## 3. Fine-tune

```bash
~/isaac-venv/bin/python scripts/train_ppo.py \
    --init runs/bc_stack_dagger_v2_e4_final/best.pt \
    --num_envs 2048 --iterations 300 --out runs/ppo_v1
```

2,048 environments × 32 steps = 65,536 transitions per iteration. Layouts come
from `--layout-seed 24680`; keep it disjoint from the evaluation seed.
`--noise-scale 1.0` applies the deployment-level sensor and odometry corruption
live, because on-policy data is consumed once and has no loader to add it later.

### The BC handoff, and why it is three separate mechanisms

Naive BC-init + PPO destroys a good actor: the critic starts random → advantages
are noise → the first policy-gradient steps are large random updates to
carefully fitted weights.

| Mechanism | Flag | Default |
|---|---|---|
| Critic warmup, actor LR **exactly** zero | `--critic-warmup` | 25 iterations |
| Annealed anchor to a frozen BC snapshot | `--anchor-coef` | 1.0, → 0 over 30% of the run |
| Conservative trust region | `--clip-range`, `--actor-lr` | 0.1, 1e-4 |

The warmup default is 25 and not the 500 an earlier CPU implementation used:
at 65,536 transitions per iteration, 500 would burn 30M environment steps before
the policy was permitted to learn anything.

### What to watch, every iteration

| Statistic | Healthy | Trouble |
|---|---|---|
| `SR` | ≥ the BC baseline from the start | below it → the anchor released too early |
| `std` | slowly decreasing | sudden drop → exploration dead |
| `kl` | well under `max_kl` (0.02) | at the cap every iteration → steps too big |
| `clip` | 0.1 – 0.3 | near 1.0 → the update is fighting the clip |
| `ev` (explained variance) | climbing toward 1 during warmup, then ≳0.9 | near 0 or negative → the critic is not learning and the warmup bought nothing |

Read `ev`, never the raw value loss. Measured on a frozen-actor run at 512
envs × 32 steps, the critic improved monotonically — `ev` 0.02 → 0.76 over
eight iterations — while `value_loss` *rose* 0.06 → 2.77, because its targets
were growing as terminal rewards began landing (0 → 88 completed episodes per
iteration). The raw loss is in squared return units and misleads in both
directions. `ev` reached 0.76 on 16k transitions per iteration, so the 25-iteration
warmup default has ample margin at the production 65k.
| `timeout_rate` | under ~1% | climbing → truncation-as-termination stops being a safe approximation |

Checkpoints: `best.pt` on rollout success rate (never on a loss — Phase 2 showed
offline loss improving while navigation did not), `latest.pt` every 10
iterations, `final.pt` at the end. All three carry the actor under the same keys
a BC checkpoint uses, so `scripts/eval_bc.py`, DAgger collection and export
accept them without knowing which algorithm produced them.

Resume:

```bash
~/isaac-venv/bin/python scripts/train_ppo.py --resume runs/ppo_v1/latest.pt \
    --init runs/bc_stack_dagger_v2_e4_final/best.pt --out runs/ppo_v1
```

## 4. Evaluate the same way as Phase 2

```bash
~/isaac-venv/bin/python scripts/eval_bc.py runs/ppo_v1/best.pt \
    --episodes 512 --layout-seed 10000000 --out runs/ppo_v1/eval.json
```

Rollout success during training is measured on *training* layouts with
exploration noise still active; it is a progress signal, not a result. The
held-out closed-loop number remains the one that counts.

## Implementation map

```text
sim2edge/models/actor_critic.py  Gaussian wrapper + critic around the actor
sim2edge/training/ppo.py         buffer, GAE, clipped surrogate, handoff
isaaclab_env/nav_task.py         the reward and the live observation noise
scripts/train_ppo.py             the rollout loop, and only that
```

---

# Phase 4 — Odometry-free lidar/IMU point navigation

The deployment goal is supplied once as `(x_forward, y_left)` in the reset base
frame. Runtime inputs are the RPLIDAR sweep, gyro Z, and previously issued
commands; wheel odometry and pose are not accepted by the model interface.

## 1. Collect the current motion distribution

The production corpus was collected with the command in Phase 1 §4, changing
only the output to `data/overnight_v3`. Its 65,587 complete/truncated episodes
span 8,192 layouts with a whole-layout 5% validation split. Wheel calibration
is two-sided while random slip and turn scrub are loss-only; newly collected
pose labels already contain those live dynamics and do not receive the legacy
load-time slip adapter.

The 8,192-environment collection needs about 11 GB just for the recorder. When
another GPU process is resident, start at 4,096 environments; doubling is safe
only if `nvidia-smi` leaves room for the transient completion spike.

## 2. Train an actor on the actual spinning-lidar observation

```bash
.venv/bin/python scripts/train_bc.py data/overnight_v3 \
  --out runs/bc_swept_v3 --epochs 2 --batch-size 4096 --no-compile
```

The loader constructs a free-running RPLIDAR sweep from raw scan history and
applies gyro/command deskew before binning. It measured about 155k samples/s on
the 4090 and completed 15.27M samples × 2 epochs in 218 s. The low validation
loss did not by itself produce a good closed-loop actor (40.6% pose success), so
keep the old PPO actor as the motion-estimator warm start until PPO adaptation
beats it.

## 3. Train and causally check the motion belief

```bash
.venv/bin/python scripts/train_belief.py \
  --data data/overnight_v3 \
  --warm-start runs/ppo_v1/best.pt \
  --out runs/belief_motion_v4 \
  --epochs 1 --batch-size 16 \
  --mismatched-lidar-coef 20

.venv/bin/python scripts/eval_belief_offline.py \
  runs/belief_motion_v4/best.pt \
  --data data/overnight_v3 \
  --out runs/belief_motion_v4/offline_ablation.json
```

Complete episodes are the default. The goal belief is initialized only at an
episode boundary, the actor is frozen, action imitation is off, and `best.pt` is
selected on held-out validation episodes. The verified v4 checkpoint scored
7.73 cm validation final error. On the paired 256-episode ablation it scored
6.15 cm versus 7.01 cm for dead reckoning; shuffling lidar degraded it to
6.25 cm and zeroing lidar to 7.01 cm. This is a narrow but positive causal lidar
result. Do not start expensive PPO if `lidar_causal` is false.

Resume at the next shard after the most recent periodic checkpoint:

```bash
.venv/bin/python scripts/train_belief.py \
  --data data/overnight_v3 --out runs/belief_motion_v4 \
  --epochs 1 --batch-size 16 --mismatched-lidar-coef 20 \
  --resume runs/belief_motion_v4/latest.pt
```

### Encoder flags for the scan history

Three options change how the odometry head reads the scan history. All three are
off by default so every existing checkpoint keeps its exact numerics, and all
three are initialization- or input-level changes, so they need `--reset-odometry`
to be meaningful and cannot be turned on with `--resume`.

```bash
.venv/bin/python scripts/train_belief.py \
  --data data/belief_onpolicy_v15 \
  --belief-start runs/belief_ppo_biasfix_v14/iter_1130.pt \
  --out runs/belief_gate_v21 \
  --epochs 1 --batch-size 8 --learning-rate 3e-4 \
  --derotate-scan-history --scaled-scan-difference \
  --confidence-coef 20 --confidence-gate-bias 0.0 \
  --reset-odometry --checkpoint-every 200 --seed 0
```

| flag | what it does |
|---|---|
| `--derotate-scan-history` | gyro-aligns every frame into the newest frame's bearing grid before encoding. Measured: **no material gain** (8.53 vs 8.56 cm paired, 8.51 vs 8.76 cm on the trainer's own validation). Correct and cheap, but not the constraint. |
| `--scaled-scan-difference` | masks the cross-frame range difference to bearings valid in both frames and scales it by per-window travel instead of `LIDAR_MAX_M`. **No result yet.** |
| `--confidence-coef` | ranks the confidence gate on matched vs mismatched scans. Without it the gate's gradient is ~740× below the proposal head's and it never leaves its initialization. Run: `belief_gate_v21`, **8.95 cm** trainer validation against 8.76 (v19 control) and 8.51 (v19 derotated). The diagnosis of the starvation stands; fixing it did not move the number. |
| `--confidence-gate-bias` | initial gate bias, default −2.0 (= 0.119). Does not fix the starvation on its own. |
| `--scan-vote-bins` | the consensus scan vote, on a BINS×BINS candidate grid. This is the replacement for the 24-sector pooling; see below. Needs `--reset-odometry` because it widens the trunk input. |
| `--scan-vote-sigma` | consensus kernel width, default 0.01 m. Measured best at both 0× and 1× sensor noise; 0.03 admits too many bearings as inliers and costs accuracy. |

### The scan vote

The 2026-08-02 log left the 24-sector average pooling as the prime suspect and
`scripts/probe_scan_vote.py` settled it. On `belief_onpolicy_v15` at 1× noise,
per-tick, against ground truth:

| estimator | drift | scatter | mean error |
|---|---:|---:|---:|
| command dead reckoning | 2.10 cm/m | 0.10 cm | 0.08 cm |
| **consensus vote** | **0.96 cm/m** | 0.57 cm | 0.44 cm |
| least squares, same residuals | 89.3 cm/m | 11.0 cm | 7.65 cm |
| estimating zero | 100.0 cm/m | 0.73 cm | 1.90 cm |

Least squares reproduces the ICP failure the 08-02 log recorded (correlation
0.084 here, +0.089 there) **on the identical residuals the vote succeeds on**,
which locates the problem in the estimator rather than in the data. Roughly a
quarter to a third of bearings are gross outliers at depth edges; a squared
residual is unbounded so each one relocates the minimum, while a bounded
agreement kernel lets them abstain.

Two things that matter and are easy to get wrong:

* **refine the peak locally.** A soft-argmax over the whole grid is pulled
  toward the grid centre by the surface's pedestal, and because the grid is
  centred on zero while the robot is really moving forward, that shrinkage is a
  systematic *underestimate of travel* — the component that integrates into
  drift rather than averaging out. Windowing cut drift from 1.30 to 0.96 cm/m.
* **do not lengthen the baseline.** It looks like free signal, since
  displacement grows with the gap while range noise does not. Measured: 0.96
  cm/m at gap 1, 10.8 at gap 2, 81 at gap 4. The range-flow model is
  first-order in displacement and compares equal bearings, and both assumptions
  fail once the robot has moved far enough to shift content between bearings.

Grid resolution is a mild knob: 1.10 cm/m at 21 bins, 1.04 at 31, 0.91 at 41,
costing 0.42 / 0.86 / 1.43 ms of single-threaded CPU.

### What the vote is worth, and why the loss has to change first

`belief_vote_v22` scored **worse** on the mean than every checkpoint without the
vote (9.38 cm against 8.53) while scoring **best** on lidar causality (+0.96 cm
shuffled degradation against 0.67) with a gate that finally discriminates (0.757
matched / 0.045 shuffled, against v14's 0.133 / 0.083). The head is using the
scan more and doing worse, and that is arithmetic rather than a training
failure:

| estimator, over a 220-step episode | systematic | random | total |
|---|---:|---:|---:|
| command only | 8.77 cm | 1.52 cm | **8.90 cm** |
| vote only, per tick | 3.24 cm | 8.40 cm | **9.00 cm** |
| optimal blend, α ≈ 0.5 | 2.81 cm | 4.24 cm | **5.08 cm** |

The vote's per-tick scatter (0.566 cm) is 5.5× the command's and consumes
exactly the bias advantage it buys. **Filtering does not help**: averaging cuts
per-sample variance but adds correlation that compensates in the sum, and belief
error *is* the sum — so `--persistent-calibration` and `--motion-state-size` are
not the fix, despite being the obvious next thing to reach for.

The fix is the objective. `translation_coef` 20 against `belief_coef` 1
evaluates to about 1.0 against 0.014 at the operating point — the per-tick term
outweighs the integral it stands in for by **69×** — and per tick, inverse
variance weighting gives the scan about 3% of the weight. Gradient descent has
been correctly ignoring the lidar for four runs. `--translation-coef` and
`--belief-coef` now exist to invert that; `belief_integral_v23` is the first run
using them.

Inverting the weighting works, on the axis it targets. `belief_integral_v23`
(`--translation-coef 2 --belief-coef 20`) scores 8.73 cm with **+1.80 cm of
shuffled-lidar degradation** — against 0.96 for v22 and 0.67 for the best
checkpoint without the vote. Removing the lidar sends it 8.73 → 10.53 cm, 91% of
the way back to dead reckoning, so the estimator is finally lidar-driven rather
than a command rescaling. The mean still does not beat the 8.53 cm baseline.

### How much scan ego-motion this sensor supports — measured

Before building anything else on longer baselines, read this.
`scripts/probe_correlative_match.py` is a correlative scan matcher: Cartesian,
no linearization, no bearing correspondence, robust by construction. On analytic
rooms it recovers 30 cm displacements to **0.3 cm** including under 1.2 rad of
rotation. On corpus scans it fails outright, and the reason is not the matcher:

| baseline | travel | truth outscores zero displacement | median score ratio |
|---|---:|---:|---:|
| 1 tick | 2 cm | **97%** of windows | 1.12 |
| 4 ticks | 9 cm | 43% | 0.95 |
| 9 ticks | 21 cm | 41% | 0.88–0.95 |
| 19 ticks | 40 cm | 44% | 0.88 |

At one tick the motion is unambiguously in the scans. By four ticks a *static*
alignment explains the newer scan better than the true one does. That is
measured on the score surface with no search involved, so it bounds every
frame-to-frame estimator, not just this one.

**Consequence:** the `sqrt(S/L)` route to lower belief error is closed. Per-tick
is the only regime available, so the per-tick scatter of 0.566 cm is a floor,
and an optimally weighted blend near 5 cm is the ceiling of frame-to-frame scan
matching on this sensor model. Grid resolution is not the limit — scatter is
0.56 cm at 21, 31 and 41 bins alike.

**This is now confirmed and is the binding constraint on the whole phase.** See
"The sensor model is the ceiling" below before doing any further estimator work.

The suspect for the decay is sensor reconstruction, not estimation:
`sweep_scan` stitches each frame from five stored ticks and quantizes each
beam's acquisition time to the nearest tick, which its docstring costs at 25 ms
— 1.15 cm of unmodelled per-beam position error at full speed, the same size as
the per-tick signal. `LIDAR_SPIN_HZ = 5.5` also remains unverified against the
A1. Look there, and at local-map accumulation, before trying another estimator.

### The sensor model is the ceiling — read this before changing the estimator

Place one reconstructed frame into another using the **true** relative pose and
measure the misalignment that remains. A correct reconstruction answers zero.

| | 0x sensor noise | 1x sensor noise |
|---|---:|---:|
| baseline 1 tick | **1.095 cm** | 0.862 cm |
| baseline 4 ticks | **8.560 cm** | 9.051 cm |
| baseline 9 ticks | **23.045 cm** | 23.672 cm |

against a matcher floor of 0.21-0.27 cm measured on analytic rooms at the same
settings, so this is the reconstruction and not the instrument.

Sensor noise is irrelevant — the two columns agree, so this is the
reconstruction disagreeing with itself. And the 4-tick figure is the sweep
period: at `LIDAR_SPIN_HZ = 5.5` a revolution is 0.182 s = 3.6 ticks, and full
speed covers **8.36 cm** in one. Frames more than a revolution apart are stitched
from disjoint sub-scan sets whose distortions do not cancel.

The slip residual the estimator must measure is **0.078 cm per tick**. The noise
floor of its own input is 8x that at one tick and 100x at four. Nothing
downstream can work, and that single fact accounts for every negative result in
Phase 4 — the three-architecture plateau, the ICP null, derotation, the scaled
difference, the gate fix, the vote's 0.03 slip correlation, the baseline decay,
and scan-to-map.

`sweep_scan` quantizes each beam's acquisition time to the nearest stored tick
(25 ms, or 1.15 cm at full speed — the 1-tick figure above sits inside that) and
the deskew then compensates each frame with the *commanded* speed. **A real
RPLIDAR A1 timestamps every return**, so hardware has strictly better
information than this simulator supplies. The conclusion is not that lidar
cannot measure ego-motion here; it is that this corpus cannot answer the
question.

**Sub-tick beam placement was tried and does not fix it.**
`SIM2EDGE_SUBTICK_SWEEP=1` interpolates both the range and the motion
correction at each beam's true acquisition time, kept consistent with each
other (`sensors.sweep_scan(subtick=)`, `sensors.deskew_history(subtick=)`). It
measures 1.151 / 9.251 / 22.056 cm — no change. Beam-age quantization is a real
approximation and is not the binding one. The flag stays, off by default and
tested, on the same footing as `--derotate-scan-history`.

Ruled out so far: sensor noise (0x and 1x agree), beam-age quantization
(measured), and search failure in the instrument (the fixed +/-10 cm extent in
the probe's first version, now scaled to the baseline).

Next candidate, with arithmetic but not yet a measurement: `rasterize_lidar`
snaps every deskewed return back onto a fixed 720-bin grid. At the corpus's
median valid range of 1.23 m one 0.5 deg bin subtends 1.08 cm of arc, so uniform
quantization contributes 0.311 cm RMS -- a third of the measured floor, and
already 4x the slip signal on its own. That is a representation fix rather than
a sensor fix: keep the deskewed `(range, exact bearing)` pairs for the
estimator instead of re-binning, since only the actor's 180-bin observation
needs a grid.

Run the acceptance test before and after any change here:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/probe_sensor_consistency.py \
    --data data/belief_onpolicy_v15
```

### Honest arrival uncertainty

`scripts/probe_arrival_honesty.py` measures whether the covariance describes the
belief error. On `iter_1130.pt` it did not, by a wide margin: true error 8.8×
the declared 1-sigma, and the 1.96-sigma bound the arrival test uses covered
**9.5%** of steps instead of 95%. That is the mechanism behind declaring arrival
at a true 31–34 cm.

The cause is a model-form error, not a training one. `belief_losses` calibrates
the variance head against *per-step* squared error and `_propagate_covariance`
sums those independently, which is a random walk; the dominant error is a
systematic scale error that grows linearly in distance. `drift_scale_uncertainty`
adds that term, carrying cumulative path length as one more float of explicit
state:

| `drift_scale_uncertainty` | error / declared σ | 1.96σ coverage |
|---|---:|---:|
| 0.0 (existing checkpoints) | 8.76 | 9.5% |
| 0.03 | 1.46 | 78.3% |
| **0.05** | **0.96** | **94.2%** |

0.05 is the calibrated setting for the current estimator. Note what it costs: the
bound widens by `1.96 * 0.05 * s`, or 49 cm after five metres, so a calibrated
robot **declines to declare arrival** on a long episode rather than declaring it
wrongly. That converts false arrivals into timeouts, which is the right trade for
hardware, but it does not manufacture accuracy — the 85% correct-arrival gate
needs an estimator that drifts less, not a wider bound.

One run holds ~16 GB, so do not run two on a 24 GB card. About 1,900 updates at
roughly 2 steps/s on `belief_onpolicy_v15` at batch 8.

## 4. Optional actor-only offline adaptation

```bash
.venv/bin/python scripts/train_belief.py \
  --data data/overnight_v3 \
  --belief-start runs/belief_motion_v4/best.pt \
  --out runs/belief_actor_v6 \
  --train-actor --freeze-estimator --batch-size 8
```

Eight is the safe batch size with roughly 5 GB free GPU memory. The measured
probe took 77 s for 200 updates and regressed deterministic pose success to
37.5%, so this branch is diagnostic rather than the current production path.
Validation selects on dimensionless action loss and prints belief error
separately.

## 5. Adapt the actor in the closed loop

```bash
~/isaac-venv/bin/python scripts/train_belief_ppo.py \
  --init runs/belief_motion_v4/best.pt \
  --out runs/belief_ppo_v4 \
  --num-envs 256 --num-layouts 1024 \
  --iterations 2400 --freeze-estimator
```

This keeps policy gradients and privileged pose labels out of the validated
motion estimator. With the competing 17–18 GB GPU process, 256 environments is
the tested safe setting; without it, scale environments and layouts together.
At 256 × 32, 2,400 PPO iterations provide the same 19.66M transitions as the
earlier 300-iteration, 2,048-environment run.

Resume preserves model, critic, exploration scale, optimizer, and iteration:

```bash
~/isaac-venv/bin/python scripts/train_belief_ppo.py \
  --init runs/belief_motion_v4/best.pt \
  --resume runs/belief_ppo_v4/latest.pt \
  --out runs/belief_ppo_v4 \
  --num-envs 256 --num-layouts 1024 \
  --iterations 2400 --freeze-estimator
```

## 6. Evaluate and enforce the gates

```bash
~/isaac-venv/bin/python scripts/eval_belief.py \
  runs/belief_ppo_v4/final.pt \
  --reference runs/ppo_v1/best.pt --allow-legacy-reference \
  --modes reference pose dead-reckoning belief lidar-shuffled lidar-zeroed \
  --episodes 512 --num-layouts 512 --layout-seed 10000001 \
  --noise-scales 0 1 2 --enforce-gates \
  --out runs/belief_ppo_v4/eval.json
```

The evaluator disables privileged true-goal termination for belief modes, then
ends only on model-declared arrival, collision, or timeout. Layouts and noise
draws are paired across oracle-pose, command dead-reckoning, learned belief,
shuffled-lidar, and zeroed-lidar modes. The gate requires at 1×: at least 85%
correct declared arrivals, at most 5% false arrivals, at most 10% collisions,
mean/p95 final belief error at most 5/12.5 cm, drift at most 2 cm/m, and lower
final error than dead reckoning at 0×/1×/2×.

The current best actor is
`runs/belief_ppo_biasfix_v14/iter_1130.pt`.  It reaches 85.2% pose-fed success,
8.6% collision, and 0.788 SPL on 128 held-out layouts.  The full paired result
is `runs/belief_ppo_biasfix_v14/eval_full_128.json`: learned belief beats command
dead reckoning at all three noise tiers, but the gate is still **false**.  At
1× it has 75.8% navigation success, 72.7% correct declared arrival, 7.8% false
arrival, 10.2% collision, 11.5/29.0 cm mean/p95 final error, and 2.94 cm/m drift.
The actor is now adequate; recurrent localization on belief-policy trajectories
is the binding constraint.  Treat every checkpoint as a development artifact
until a full held-out evaluation writes `"passed": true`.

Collect that distribution with the complete recurrent driver, not the
pose-fed `--driver-checkpoint` compatibility path:

```bash
~/isaac-venv/bin/python scripts/collect_bc.py \
  --num_envs 4096 --episodes 16384 \
  --num_layouts 4096 --starts_per_layout 32 --seed 1501 \
  --out data/belief_onpolicy_v15 --shard_steps 200000 \
  --validation_fraction 0.05 --writer_queue 2 \
  --belief-driver-checkpoint runs/belief_ppo_biasfix_v14/iter_1130.pt
```

The completed v15 corpus has 3.61M samples. One full-episode epoch at a constant
3e-5 LR preserved lidar causality but changed held-out mean error only from
9.49 to 9.46 cm. Do not repeat that branch with more epochs.

The limitation is **not** observability, despite what the 2026-08-01 handoff
concluded. Measured on this corpus at `noise_scale 0.0`, the median range change
between consecutive reconstructed frames tracks distance travelled at a ratio of
about 1.0 (2.13 cm median for 1.99 cm of travel), so the frames are consistent
rigid views and the motion is in them. A point-to-line solver and a full ICP,
both validated exact against analytic rooms and precise to ~0.07 cm on a 0.08 cm
one-tick slip, nevertheless recover nothing from corpus scans. The reason is that
the *mean* frame difference is 6× the median: roughly a third of bearings are
gross outliers at depth edges in cluttered rooms, and no estimator tried so far —
learned or geometric — is robust to that. See `docs/daily/2026_08_02_log.md` §4.

No external hardware pose reference is currently available. Robot trials can
verify timestamps, coordinate signs, reset behavior, stationary drift, and
qualitative navigation, but simulator acceptance must not be described as a
measured sim-to-real result.

---

# Phase 5 — Export and deploy

The graph half is built and runs on the training host. INT8 calibration, the
C++ node and the on-board lowering are not.

## 1. Export the deployable step and check it

```bash
.venv/bin/python scripts/export_belief.py runs/belief_vote_v22/final.pt \
    --out runs/belief_vote_v22/export
```

`torch.export` is the exact front end `to_edge_transform_and_lower` consumes, so
signature and operator problems surface here rather than on the board. The
script writes `belief_step.pt2` plus an `export_report.json` carrying the full
operator histogram, and it **fails** if the exported program does not reproduce
the eager module on adversarial state — non-trivial belief, inflated covariance,
a partly elapsed arrival count and a scan with 25% missing returns.

Measured on `belief_ppo_biasfix_v14/iter_1130.pt`: 41 distinct operators,
491,159 parameters, parity exactly 0.0, and 0.69 ms median host-CPU latency
against a 50 ms control period. The latency figure is a proxy for the budget,
not a board measurement.

### The scan vote stays out of the graph

For a vote checkpoint the script computes the vote first, via
`OdometryHead.compute_vote`, and traces the network with its outputs as ordinary
inputs. `--include-vote` traces it in instead, and exists to measure what that
costs. On the same checkpoint:

| | operators | host latency |
|---|---:|---:|
| vote outside the graph (default) | 50 | 1.17 ms |
| `--include-vote` | 56 | 2.46 ms |

Tracing it in adds `gather`, `cumsum`, `flip`, `remainder`, `floor` and a
`set_grad_enabled` higher-order wrapper, none of which the reference models
proved delegated. It is also an argmax over a surface measuring centimetres,
which is not a thing to quantize.

**The seam is the parameter boundary, not the vote branch.** `ScanVote` and
`_vote_surfaces` are fixed geometry and hold no weights — the same status the
deskew and the angular binning have, and those already live in the C++ node's
preprocessing translation unit. `_encode_vote`, the convolution that reads the
vote surface, *does* have weights and stays in the graph; the surface crosses
the boundary as a small image. Putting it on the preprocessing side instead
would run learned weights in float on the CPU and split the model across the
quantization boundary.

A test requires the precomputed path to be bit-identical to computing the vote
inline. A divergence there would appear only as an unexplained sim-to-real gap.

The vote itself costs 0.42 ms single-threaded at 21 bins and 0.86 ms at 31.

## 2. Lower to the NPU, on the board

The AOT pybind only exists on the board, so lowering runs there. `ssh
ubuntu@ventuno`; the working copy is `~/sim2edge-deploy`.

```bash
rsync -az --exclude __pycache__ sim2edge/ ubuntu@ventuno:~/sim2edge-deploy/sim2edge/
rsync -az scripts/lower_qnn.py scripts/dump_calibration.py \
    runs/body_v2_actor_prod/best.pt ubuntu@ventuno:~/sim2edge-deploy/
```

Build the calibration set on the host, where the corpus lives:

```bash
.venv/bin/python scripts/dump_calibration.py \
    --data data/body_frame_v2_prod --samples 1024 --noise-scale 1.0 \
    --out runs/body_v2_actor_prod/export/calibration.npz
```

Then on the board — **all four exports are required**:

```bash
export PYTHONPATH=/opt:$HOME/sim2edge-deploy
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.48.0.260626
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2:$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4
export ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v75/unsigned;/usr/lib/rfsa/adsp;/dsp"

~/.venv/executorch/bin/python lower_qnn.py best.pt \
    --calibration calibration.npz --quant-dtype use_16a8w --per-channel \
    --out out/actor_qnn.pte
```

Measured: 373 KiB, **one delegate call and no CPU fallback**, 11.5 s to lower,
and quantization error of 0.42%/1.73% (mean/max of limit) on omega and
0.20%/2.44% on v.

### The environment is the whole difficulty — four facts

| | |
|---|---|
| `PYTHONPATH=/opt` | ExecuTorch is built at `/opt/executorch` but not installed into `~/.venv/executorch`. Nothing needs rebuilding. |
| `aarch64-oe-linux-gcc11.2` **first** | The obvious `aarch64-ubuntu-gcc9.4` does not support this SoC and ships no V75 stub. Symptom: `Unsupported SoC model (SnapdragonModel): 82`, then `Invalid dsp arch. Cannot determine stub`. The `82` is hardware auto-detection — changing `--soc-model` does not change it. |
| `ADSP_LIBRARY_PATH` | Semicolon-separated, and unset by default on this board. |
| `--soc-model QCS8300` | The default, and not a typo. The chip is a QCS8275 (`soc_id` 675); QNN has no QCS8275 entry and the patched QCS8300 enum maps to HTP V75, which is what the board reports. |

Validate the runtime before trusting any number — an HTP architecture mismatch
produces garbage output, not an error:

```bash
$QNN_SDK_ROOT/bin/aarch64-ubuntu-gcc9.4/qnn-platform-validator \
    --backend dsp --coreVersion --testBackend
```

Expect `Hexagon Architecture V75` and `Unit Test ... Passed`. The trailing
`Error in saving the results` is only a CSV write into a read-only directory.

### Circular padding is rewritten for export, and must stay that way

`padding_mode="circular"` lowers to a scatter into an uninitialized buffer, and
`QnnPartitioner` raises `KeyError: dim_order_ops._empty_dim_order.default`
rather than declining the node. `rewrite_circular_convolutions` writes the same
padding as a wrap-around `cat` — bit-identical, deep-copied so training is
untouched, and asserted in `tests/test_model.py` and by the lowering script
itself. `--keep-circular-padding` reproduces the crash on purpose.

### Use the true min/max activation observer

The default moving-average observer averages per-batch extremes and clips the
pre-tanh logit, collapsing every saturated turn onto one command: 37% of the
omega limit in worst-case error, and **it does not improve with more bits**.
`MinMaxObserver` is the default here and takes that to 1.7%.
`--moving-average-observer` restores the old behaviour for comparison.

## 3. Run and benchmark on the board

```bash
/opt/executorch/build-x86/executor_runner --model_path out/actor_qnn.pte \
    --num_executions 1000 --print_output none
```

`build-x86` is a native aarch64 build; the name is misleading. Measured on the
Ventuno against a 50 ms control period:

| | latency | share of period |
|---|---:|---:|
| actor, INT8 on HTP V75 | 0.15–0.18 ms | 0.4% |
| actor, portable CPU kernels | 5.15 ms | 10.3% |
| CPU localizer, 128 scans / stride 2 | 21–27 ms | 42–53% |

Model load is a one-time ~200 ms. **The NPU is not the constraint and never
was** — the CPU scan-to-submap localizer costs about 150× the actor. Benchmark
it with real scans, at production settings, single-threaded:

```bash
.venv/bin/python scripts/bench_localizer.py --scans scans.npz \
    --max-submap-scans 128 --point-stride 2
```

Cost is near-linear in submap size and in `1/point_stride`; the full table is in
`docs/daily/2026_08_04_log.md`. `point_stride 4` halves the dominant cost in the
loop and has **not** been evaluated for accuracy.

## 4. The quantization gate is still to run

The gate is a **sim success-rate drop under 2 points**, not an action MSE: a
small action error can still cost success at exactly the moments that matter.
Nothing should be built on the current `.pte` before it passes. If it fails,
escalate in order — calibration data first. `data/body_frame_v2_prod` is
`driver = "oracle"`, so the deployed policy's own state distribution is the
stronger source, and `dump_calibration.py` warns whenever the corpus is not
policy-driven.

The action half of it has now been measured twice, independently, and the two
agree: 0.42% mean / 1.71–1.73% max of the omega limit, once from `lower_qnn.py`
in Python and once from the C++ node path on the board. That is necessary and
not sufficient.

---

# Phase 6 — The ROS 2 stack

`ros2_ws/` is the deployed robot. It builds from this same repository on the
training host and on the Ventuno; `docs/daily/2026_08_04_ros2_log.md` records
how it was measured.

## Build

On the board:

```bash
cd ~/sim2edge-ventuno/ros2_ws
source src/sim2edge_bringup/scripts/ventuno_env.sh   # QNN + ExecuTorch paths
source /opt/ros/jazzy/setup.bash
colcon build --cmake-args -DEXECUTORCH_INSTALL_DIR=/opt/executorch/build-x86
```

On the training host the same `colcon build` works with no arguments. Without
ExecuTorch the preprocessing, deskew and localizer paths still build, which is
everything the parity gate compares.

`ventuno_env.sh` carries the facts that are not guessable — the
`aarch64-oe-linux-gcc11.2` ordering, `ADSP_LIBRARY_PATH`, `PYTHONPATH=/opt`, and
the ExecuTorch shared-library directory. Read its comments before changing any
of them.

## Pin the navigator to a big core. This is not tuning.

The QCS8275 is 4× Cortex-A78C (`cpu0-3`) plus 4× Cortex-A55 (`cpu4-7`). The
localizer costs **6.98 ms on `cpu2` and 53.57 ms on `cpu6`** — 7.7×, against a
clock ratio of 1.2×, because scan-to-submap matching is memory-bound
nearest-neighbour work and an in-order A55 is close to its worst case. On an A55
the localizer alone exceeds the entire 50 ms control period.

Both launch files default `nav_cpus:=2,3` and apply it with `taskset`. Unpinned,
35 of ~2,200 ticks missed the period; pinned, 0 of 2,114 did.

## Test it without a robot

```bash
# On the host: record held-out episodes plus what Python produced from them.
.venv/bin/python tools/export_replay.py \
    --data data/body_frame_v2_prod/validation --episodes 12 \
    --checkpoint runs/body_v2_actor_prod/best.pt \
    --out ros2_ws/replays/validation_12.s2e

# On the board: numeric agreement, per stage, plus latency.
./build/sim2edge_core/sim2edge_parity \
    --replay replays/validation_12.s2e --model models/actor_qnn.pte \
    --backend qnn --qnn-lib-dir $QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2

# The localizer knobs, swept.
./build/sim2edge_core/sim2edge_bench --replay replays/validation_12.s2e

# The real node graph, driven from the same recording.
ros2 launch sim2edge_bringup replay_test.launch.py \
    replay:=$PWD/replays/validation_12.s2e model:=$PWD/models/actor_qnn.pte \
    qnn_lib_dir:=$QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2
```

`--checkpoint` matters. Without it the replay records the corpus's stored oracle
command as its reference, and the parity report's action row then measures how
far the policy is from its teacher rather than quantization error. The file
records which reference it holds and the report says which.

**The two tools answer different questions.** `sim2edge_parity` answers "do the
numbers agree" and runs on a laptop. The launch answers "does the graph come up,
hold 20 Hz under real DDS load, and deliver a command inside the period".
Neither answers whether the robot navigates: the corpus was oracle-driven, so
the recorded pose does not respond to what the policy commands.

## Attach the sensors

One-time, or the lidar node dies on open with a permission error that names no
permission:

```bash
sudo usermod -aG dialout ubuntu     # then open a NEW login shell
```

`/dev/ttyUSB0` is `root:dialout`, and the board's `ubuntu` user was not in the
group. The A1 shows up as a `10c4:ea60 CP210x`; the OAK-D as a
`03e7:2485 Movidius MyriadX` — check with `lsusb` before blaming software.

What the physical A1 measures on this unit, all of it different from `spec.py`
and from what the driver reports about itself: **7.32 Hz** (not 5.5 nominal, and
not the 10.0 the driver logs for the Sensitivity mode), **1080 bins**
per revolution of which **~855 carry a return** (the sim's sweep was 720, all
valid). `time_increment` is nonzero, so deskew uses real per-beam timing and
`LIDAR_SPIN_HZ` is inert at runtime — it still matters to training data.
See `docs/daily/2026_08_04_hardware_log.md`.

## Run the robot

Every shell that talks to the stack needs all three of these first:

```bash
cd ~/sim2edge-ventuno/ros2_ws
source /opt/ros/jazzy/setup.bash
source src/sim2edge_bringup/scripts/ventuno_env.sh
source install/setup.bash
```

### Sensors only, no base — this is what has actually been run

```bash
ros2 launch sim2edge_bringup robot.launch.py \
    model:=$PWD/models/actor_qnn.pte \
    qnn_lib_dir:=$QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2 \
    require_odom:=false
```

Brings up `sllidar`, `oak_imu`, `sim2edge_navigator` (pinned to `cpu2,3`) and
`sim2edge_safety`. Nothing moves: no base is subscribed to `/cmd_vel`. The
motion prior has no velocity term without `/odom`, so this is a latency and
numerics run and **not** evidence about navigation.

### With the Create 3

Just drop `require_odom:=false`. The DDS environment is already exported from
the board's `~/.bashrc` by the sibling `ventuno-object-tracking` project —
domain 0, `rmw_fastrtps_cpp`, and a `usb0` profile byte-identical to the one
this repo ships. The commented block at the bottom of `ventuno_env.sh` is the
self-contained equivalent; uncomment it if you would rather not depend on
another project's path.

The profile whitelists loopback *and* `192.168.186.3`, so `ros2` commands run
fine in an ssh shell **on the board**. What it does cut off is `ros2` on a
*different* machine reaching the board over Wi-Fi.

Verify the link before launching — `docs/create3_connection.md` in
`ventuno-object-tracking` is the authority, and the gadget is fragile across USB
bounces:

```bash
cat /sys/class/udc/*/state          # want: configured
ping -c3 192.168.186.2              # want: 0% loss
ros2 node list                      # want: /motion_control, /_internal/mobility, ...
ros2 topic info /cmd_vel            # want: Subscription count >= 1
```

Two failure modes, and they chain — fix them in this order:

1. **Base's multicast arrives but `ping` fails and `ip neigh` is `FAILED`.** The
   gadget's TX endpoint has wedged; the link is one-way. No amount of DDS
   configuration helps, because discovery needs both directions. Re-bind the
   UDC: `sudo ~/Documents/ventuno-object-tracking/scripts/create3_usb_gadget.sh`
2. **Base pingable and its web UI answers, but no SPDP.** Its ROS app has gone
   quiet. Restart *only* the app — **never the Reboot button**, which re-wedges
   the gadget and restarts the loop:
   `curl -X POST http://192.168.186.2/api/restart-app`, then allow ~30 s.

   Re-binding the gadget causes this, and so does **rebooting the board** —
   which is the common case, because the board is the USB *device*, so its
   reboot yanks the base's USB peer out from under it. The link itself comes
   back clean on its own (the `create3-usb-gadget` service is enabled, and
   `ping` shows 0% loss), which is what makes this confusing: everything at the
   network layer looks healthy and there are simply no ROS topics. **After any
   board reboot, expect to restart the base's app.**

### Give it a goal

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
    '{header: {frame_id: "base_link"}, pose: {position: {x: 2.0, y: 0.0}}}'
```

A body-frame offset, consumed once. **The policy commands close to `V_MAX` the
moment a goal lands** — a bench run reached 0.44 m/s of a 0.46 limit on the
first tick after the goal. Set a goal only when the robot is somewhere it can
safely move, or with the wheels off the ground.

### Watch it

```bash
ros2 topic echo /nav/status        # per-stage cost, belief, hold reason
ros2 topic echo /safety/reason     # why the safety layer is limiting
```

### Stop it

`Ctrl-C` on the launch. `sim2edge_safety` is a separate process holding
`/cmd_vel` with a 0.30 s watchdog, so if the navigator dies the base is
commanded zero rather than continuing on its last command — but the safety node
dies with the launch too, so **`Ctrl-C` is not an e-stop**. Keep the base's own
button reachable.

### Confirm the gyro's yaw axis whenever the OAK-D is re-mounted

```bash
python3 tools/probe_yaw_axis.py --seconds 120     # on the board, /imu running
```

Turn the robot a quarter turn **left** (counter-clockwise from above), pause,
then turn it back **right**. Flat — don't tilt or lift it. No timing to hit: the
script segments on gyro magnitude and integrates each burst separately.

On the current mount the answer is **y**, not `z`: 100.9° on y against 3.3° on
z, with the out-and-back closing to 0.35° over 100°. Set `gyro_yaw_axis` and
`gyro_yaw_sign` in `navigator.yaml` from what it reports; the navigator logs
which channel it is using at startup.

Insist on the out-and-back cancelling. One turn shows which channel moved; only
the return leg separates rotation from drift. Getting this wrong does not fail
— it under-corrects deskew and presents as a localization failure.

Three inputs reach the policy and no others: the lidar sweep, the OAK-D gyro,
and the Create 3's velocity. Point at a goal once with `/goal_pose` as a
body-frame offset; nothing else consults an external frame.

`sim2edge_safety` owns `/cmd_vel` and is a separate process on purpose, so that
it keeps stopping the robot when the navigator crashes. The navigator publishes
a proposal on `nav/command` and never touches the motors.

**No hardware has been attached to any of this.** The board has no device on
`/dev/ttyUSB*` and no OAK-D connected, so `serial_port`, the A1's real spin rate
and the gyro's sign are all unverified.

## Regenerate the C++ spec header after touching `spec.py`

```bash
.venv/bin/python tools/gen_spec_header.py            # rewrite
.venv/bin/python tools/gen_spec_header.py --check    # fail if stale
```

The generated header carries the observation fingerprint, and the actor loader,
the parity harness and the replay node all compare against it. A graph exported
under a different preprocessing contract returns plausible-looking garbage
rather than an error, which is why that comparison is load-bearing.

---

# Driving a policy by hand

```bash
~/isaac-venv/bin/python scripts/interactive_nav.py                     # actor only
~/isaac-venv/bin/python scripts/interactive_nav.py --driver localized  # full stack
```

One rendered environment on a held-out layout. **Left-click the floor to send
the robot there**; the panel resets it, moves it elsewhere in the same room,
draws a new room, pauses, and switches the noise tier live. A click is snapped
to the nearest legal robot centre within 60 cm and refused beyond that, so a
click inside an obstacle is rejected rather than driven at.

This is a demonstration, not a measurement. One episode at a time is not a
success rate, and nothing here writes a report. Use `eval_bc.py`,
`eval_localized.py` or `eval_belief.py` for numbers.

| | `--driver pose` (default) | `--driver localized` |
|---|---|---|
| goal features from | simulator truth | the robot's own estimate |
| motion prior | — | Create3-style body-frame odometry increment + gyro |
| correction | — | CPU exact-bearing scan-to-submap tracker |
| arrival decided by | true distance | the robot's own uncertainty bound, 3 ticks |
| answers | "is the actor good" | "would the robot do this" |
| loop rate | ~12 Hz | ~9 Hz (the scan match is the cost) |

Only `localized` is the deployable stack, and the orange disc it draws is where
the robot *believes* the goal is — the gap between the two discs is the
localization error, at the scale a person can see.

**One input there is not end-to-end, and it is not conservative.** Under the
default `--prior-source create3-ekf` the odometry increment is synthesized from
achieved simulated motion plus encoder noise, exactly as in `eval_localized.py`,
because the simulator has no wheel-encoder model. No ground-truth pose reaches
the actor or the tracker — but an increment measured from achieved displacement
*sees through* the wheel slip the simulator deliberately injects, where a real
encoder counts rotations and would report the commanded travel instead. The
prior therefore carries additive noise and none of the drivetrain scale error
that `LocalPoseTracker.motion_scale` exists to estimate.

`--prior-source command-gyro` is the fully self-contained configuration, and its
three inputs are worth naming exactly: the **lidar** sweep, the **gyro** yaw
increment, and the robot's **own commanded speed**. The last is not a
measurement — the tracker integrates `commanded_speed * dt * motion_scale` — so
it is blind to wheel slip in the same way a wheel encoder is, and `motion_scale`
is the only thing that recovers it. That comes from the scan match, so here the
lidar carries the whole translation correction by itself. This is the arm that
missed the localization gate, which is why the evidence-gated fallback to
Create3 odometry was exercised. Closing it properly needs either a wheel-encoder
model in the simulator or the real Create3 `/odom`.

Truth is read at one more point in both modes, and only one: the instant you
click, to turn the clicked world point into an offset in the robot's frame. It
is then stored in the tracker's frame and never consulted again — the same
status `initial_goal` has in `eval_localized.py`. Somebody has to point at the
map; nothing drives on it.

The viewer loads `assets/turtlebot4_lite_visual.usd`, not the task's default
`assets/turtlebot4_lite.usd`. The latter predates the `package://` fix in
`tools/convert_tb4.py` and contains only the URDF's primitive wheel visuals, so
the robot renders as a pair of wheels. Nothing else notices — collision is
primitives and the lidar never sees the robot — and a viewer is the one consumer
that does. `--robot-usd` overrides it.

Neither collision nor arrival ends an episode here: the environment reports both
and keeps rendering, and only a button or a click moves the robot. Wall-clock
pacing holds the 20 Hz control period so the motion reads the way it will on
hardware; `--speed 0` runs free.

---

## Common failures

| Symptom | Fix |
|---|---|
| Silent, minutes, ~100% CPU on one core | Warp JIT. One-time per version; let it finish. |
| `No module named 'pxr'` | Something under `isaaclab.sim` imported before `AppLauncher` ran. Import `isaaclab.app`, launch, *then* the rest. |
| `cannot import name 'AppLauncher' ... circular import` | A local file shadows a stdlib module Isaac imports — e.g. a scratch `inspect.py`. Rename it. |
| `PxGpuDynamicsMemoryConfig::foundLostPairsCapacity` | Raise `gpu_found_lost_pairs_capacity` in `NavEnvCfg.sim` to the number reported. |
| RSS climbs and `ms/step` creeps up | Recorder buffers should be a fixed ceiling printed at startup. If not, something reintroduced per-step allocation. |
| Every lidar bin reads 1.0 | Rays hit nothing — sensor targets not matching obstacle prims, or obstacles parked below the floor. |
| Viewport looks empty | Rooms span `(0,0)`–`(size)` from the env origin, so the camera must aim at the middle. `--gui` frames env 0 after reset; `--dump-scene` gives coordinates. |
| The robot renders as two floating wheels | The asset predates the `package://` fix and has no visual meshes. Use `assets/turtlebot4_lite_visual.usd`, or rebuild with `tools/convert_tb4.py`. |
| `ValueError: shard ... was collected under spec X` | `spec.py` changed after collection. Recollect, or check out the spec that produced the shard. |
| Yield looks great early, then collapses | Expected. Fast episodes finish first; wait until `kept + dropped` ≫ `num_envs`. |

## Repository layout

```
sim2edge/        simulator-free core: spec, grids, rooms, oracle,
                 preprocess, noise, dataset, localization
ros2_ws/         the deployed robot: C++ ROS 2 workspace, builds on the
                 training host and on the Ventuno from this same tree
isaaclab_env/    Isaac Lab adapter -- owns no task definition;
                 nav_task.py (batch) and interactive.py (viewer)
scripts/         collect_bc.py, verify_shard.py, train_bc.py,
                 eval_bc.py, train_ppo.py, train_belief.py,
                 train_belief_ppo.py, eval_belief.py,
                 eval_localized.py, interactive_nav.py
tools/           convert_tb4.py
tests/           311 passing tests, no GPU or simulator required
docs/daily/      engineering log
assets/, data/   generated; not in git
```

`archive/v1-numpy-pipeline` holds the previous build — numpy reference env, PPO,
eval harness, ExecuTorch export, ROS 2 C++ node — worth consulting when those
phases come back.
