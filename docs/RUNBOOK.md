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

175 pass, 2 skipped without Isaac. No GPU or simulator required — keep it that way.

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

# Phase 4 — Export and deploy

*(Not yet built. ExecuTorch export, INT8 calibration, C++ node, and the
preprocessing parity test.)*

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
                 eval_bc.py, train_ppo.py
tools/           convert_tb4.py
tests/           175 tests, no GPU or simulator required
docs/daily/      engineering log
assets/, data/   generated; not in git
```

`archive/v1-numpy-pipeline` holds the previous build — numpy reference env, PPO,
eval harness, ExecuTorch export, ROS 2 C++ node — worth consulting when those
phases come back.
