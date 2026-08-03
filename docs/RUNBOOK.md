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

268 pass, 2 skipped without Isaac. No GPU or simulator required — keep it that way.

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

`--include-vote` exists to measure what including it costs, not to ship it.
`ScanVote` holds no parameters — it is fixed geometry, the same status the
deskew and the angular binning already have, and those live in the C++ node's
preprocessing translation unit. Tracing it in adds `gather`, `cumsum`, `flip`,
`remainder`, `floor` and a `set_grad_enabled` higher-order wrapper (41 → 56
operators), none of which the reference models proved delegated. It is also an
argmax over a surface measuring centimetres, which is not a thing to quantize.

It costs 0.42 ms single-threaded at 21 bins and 0.86 ms at 31, so it fits the
preprocessing budget comfortably.

## 2. Still to build

INT8 calibration from the final policy's own state distribution, the QNN
lowering on the board, the C++ node, and the preprocessing parity test against
it. `archive/v1-numpy-pipeline` holds a working version of all four —
`tools/export_qnn.py`, `tools/parity_check.py`,
`ros_ws/src/nav_policy_node/src/preprocess_parity.cpp` — and
`docs/deployment.md` on that branch carries the board-specific facts (QNN SDK
2.48, the two load-bearing ExecuTorch patches, `--soc-model QCS8300`, the
semicolon-separated `ADSP_LIBRARY_PATH`). The quantization gate there is a
**sim success-rate drop under 2 points**, not an action MSE.

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
| `ValueError: shard ... was collected under spec X` | `spec.py` changed after collection. Recollect, or check out the spec that produced the shard. |
| Yield looks great early, then collapses | Expected. Fast episodes finish first; wait until `kept + dropped` ≫ `num_envs`. |

## Repository layout

```
sim2edge/        simulator-free core: spec, grids, rooms, oracle,
                 preprocess, noise, dataset
isaaclab_env/    Isaac Lab adapter -- owns no task definition
scripts/         collect_bc.py, verify_shard.py, train_bc.py,
                 eval_bc.py, train_ppo.py, train_belief.py,
                 train_belief_ppo.py, eval_belief.py
tools/           convert_tb4.py
tests/           245 passing tests, no GPU or simulator required
docs/daily/      engineering log
assets/, data/   generated; not in git
```

`archive/v1-numpy-pipeline` holds the previous build — numpy reference env, PPO,
eval harness, ExecuTorch export, ROS 2 C++ node — worth consulting when those
phases come back.
