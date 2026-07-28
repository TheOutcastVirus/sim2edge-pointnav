# Getting this running on an NVIDIA GPU host

This is the **Track A** machine from §8 of the implementation plan: the box that
trains the policy that ships. Everything in this repo except `isaaclab_env/` runs
without a GPU, so nothing here is required to use the pipeline — it is required
to train the *deployed* policy, because the reference environment
(`sim2edge/envs/reference_env.py`) is kinematic and will not close the sim-to-real
gap on its own.

Read `docs/deployment.md` for the board half. The two halves meet at the spec
fingerprint and nowhere else.

> **Status honesty.** None of the Isaac Lab steps below have been executed. They
> were written against the Isaac Lab 2.x API off-host, which is exactly the
> situation §4 of the plan warns about, so treat every code path in
> `isaaclab_env/` as a first draft to be corrected against the installed
> version, not as working code. The four known defects are listed in
> [Known gaps](#known-gaps-in-isaaclab_env) and they are the actual work.

---

## 1. Hardware and OS

| | Minimum | Plan's assumption |
|---|---|---|
| GPU | RTX 3070, 8 GB | RTX 4090, 24 GB |
| VRAM | 8 GB (≈512 envs lidar-only) | 24 GB (2–4k envs lidar-only) |
| RAM | 32 GB | 64 GB |
| Disk | 60 GB free (Isaac Sim is ~30 GB) | — |
| OS | Ubuntu 22.04 or 24.04 | Ubuntu 22.04 |
| Driver | ≥ 535 | — |

Isaac Sim needs an RTX card — the tiled depth renderer uses RT cores. A GTX card
will not work for the depth configuration and is not worth fighting.

```bash
nvidia-smi   # must print a driver >= 535 before going further
```

---

## 2. What to install, in order

The dependency graph matters: Isaac Sim pins its own torch build, so installing
this repo's `requirements.txt` into Isaac's Python **after** Isaac is set up will
happily downgrade torch and break it. Use two environments.

### 2a. A plain venv for everything except Isaac Lab

This is what runs the tests, the oracle, BC on the reference env, the export
tooling, and the parity gate. It is the environment you will spend most of your
time in.

```bash
git clone <this repo> sim2edge-ventuno && cd sim2edge-ventuno
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 -m pytest          # expect 75 passed
python3 tools/gen_spec_header.py --check
```

If `pytest` reports fewer than 75 or any failure, stop — the shared code is
broken and Isaac Lab will inherit the breakage. Note that the suite is
CPU-only by design; a GPU changes nothing about it.

### 2b. Isaac Sim + Isaac Lab

Follow NVIDIA's current instructions rather than anything pinned here; the
install story has changed twice and will change again. As of writing, the pip
route is:

```bash
python3 -m venv ~/isaac-venv && source ~/isaac-venv/bin/activate
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' \
    --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && ./isaaclab.sh --install
```

Verify before touching this repo's code:

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

The first run downloads shaders and takes several minutes. If that tutorial does
not run, no amount of work on `isaaclab_env/` will help.

**Record the version.** `pip show isaaclab rsl_rl` — you will need it for step 5,
and a version bump is the most likely cause of a signature mismatch.

### 2c. Make this repo importable from Isaac's Python

`isaaclab_env/` imports `sim2edge`, which needs only numpy and torch — both
already present in the Isaac venv. Do **not** `pip install -r requirements.txt`
there. Add the repo to the path instead:

```bash
source ~/isaac-venv/bin/activate
cd ~/IsaacLab
echo /path/to/sim2edge-ventuno > \
    $(python3 -c 'import site; print(site.getsitepackages()[0])')/sim2edge.pth

python3 -c 'from sim2edge.spec import DEFAULT_SPEC as s; print(s.fingerprint())'
# must print the same fingerprint the venv in 2a prints
```

That fingerprint check is the whole point. If the two environments disagree, they
are running different task contracts and any checkpoint moved between them is
silently wrong.

`scipy` is optional — `sim2edge/oracle/grid.py` falls back to exact pure-numpy
implementations of Dijkstra and the EDT. The fallbacks are ~60× slower on the
EDT, which matters when generating thousands of rooms, so install scipy in the
Isaac venv if room generation becomes the bottleneck. Check that it actually
imports (`python3 -c 'import scipy.ndimage'`) — a NumPy-ABI-mismatched scipy
raises `AttributeError`, not `ImportError`, and the fallback catches it.

---

## 3. Confirm the shared core before writing Isaac code

Everything in `sim2edge/` is used unchanged by Isaac Lab. Bugs found here are
bugs you would otherwise debug at 2048 environments with a 40 GB simulator in the
way.

```bash
source .venv/bin/activate

# The oracle must be near-perfect: it is BC's ceiling.
python3 - <<'EOF'
from sim2edge.envs.reference_env import NavEnv, EnvConfig
from sim2edge.eval.harness import evaluate, oracle_policy
env = NavEnv(EnvConfig(use_depth=False), seed=0)
print(evaluate(oracle_policy(env), 100, env=env).summary())
EOF
# Expect success_rate >= 0.99. Anything lower is a geometry bug in
# rooms.py/oracle/ and it will cap BC quality in Isaac Lab identically.

# The full BC path, small. ~15 min on CPU.
python3 scripts/train.py bc --episodes 300 --lidar-only --epochs 6 \
    --dagger-rounds 0 --eval-episodes 100 --out runs/smoke
```

Then the export path, which does not need the board:

```bash
python3 tools/smoke_export.py
# Stages 1, 2, 5 must pass (torch.export with 4 inputs, HTP-proven operator
# set, eager latency). Stages 3-4 fail without ExecuTorch installed; that is
# expected on a training host and is not a problem to fix here.
```

---

## 4. Known gaps in `isaaclab_env/`

These are the real work. Fix them in this order — each gates the next.

### 4.1 The lidar sees no obstacles

`nav_task.py` configures the raycaster with:

```python
mesh_prim_paths=["/World/ground"]
```

but `_spawn_layouts` creates obstacles at
`/World/envs/env_{i}/obstacle_{j}`. As written the lidar returns only ground-plane
hits — it will look like a working sensor while being blind to every obstacle.

This is the single highest-priority fix, because it fails *quietly*. Either spawn
obstacles into the ground mesh hierarchy or extend `mesh_prim_paths`.

### 4.2 `robot: ArticulationCfg = MISSING`

Isaac Lab does not ship a Create3. Options, cheapest first:

1. Convert the TB4 URDF from ROS's `turtlebot4_description` with Isaac Lab's
   URDF importer.
2. Start from a two-wheel differential-drive template and set the geometry to
   match `RobotSpec`: `wheel_base_m=0.235`, `radius_m=0.175`, wheel radius
   0.036 m.

`_apply_wheel_velocities` assumes exactly two velocity-controlled joints ordered
`[left, right]`. **Verify the joint order from the imported asset.** Reversed
joints produce a robot that turns the wrong way, which reads as a broken policy.

Sensor mounting must match `RobotSpec` too: lidar at 0.175 m, camera at 0.19 m
with a 1.204 rad HFOV. §8 Day 1–2 of the plan says to check these against real
rosbags; do that rather than trusting the numbers.

### 4.3 `_spawn_layouts` conflicts with environment cloning

`_setup_scene` calls `clone_environments(copy_from_source=False)`, which shares
source geometry between clones — incompatible with per-environment distinct
layouts. With `replicate_physics=False` and a Python loop spawning every obstacle
prim individually, startup at 2048 environments will also be very slow.

Expect to rework this. A reasonable approach is a fixed pool of pre-authored
layout prims selected per environment, rather than authoring 2048 unique rooms at
startup.

### 4.4 `_load_actor_into_runner` raises `NotImplementedError`

`train_rsl_rl.py:158`. rsl_rl's default `ActorCritic` is a flat MLP;
`NavPolicy` is a conv depth encoder plus a 1D conv lidar encoder plus a trunk.
Loading a BC checkpoint needs `NavPolicy` registered as a custom rsl_rl policy
class.

I left this unimplemented deliberately: rsl_rl's `ActorCritic.__init__` signature
has changed across releases, and a wrong guess loads weights into the wrong
tensors *without error*. Write it against the installed version.

Two things to get right:

- **The observation arrives flattened.** `_get_observations` concatenates
  depth + lidar + goal + last_action into one 15,545-element vector. Your policy
  class must unflatten in `ObservationSpec.actor_inputs` order — the same order
  the exported graph binds. A slice-offset bug here trains fine and deploys
  wrong. Unit-test the split against `spec.actor_input_shapes` before training.
- **`freeze_norm()` must stay in effect.** `train_rsl_rl.py` calls it, but any
  `.train()` inside rsl_rl re-enables BatchNorm. `sim2edge/training/ppo.py`
  re-applies it after every `train()` for this reason; do the same here.

### 4.5 Write the observation parity test

`nav_task.py` names `tests/test_isaac_obs_parity.py` in a comment. It does not
exist. `_get_observations` reimplements the spec's normalization in torch,
because the data is a GPU batch and `sim2edge/preprocess.py` is numpy — so there
are now two implementations of the same arithmetic and nothing checks they agree.

That is the same defect class that has already cost this project two debugging
sessions (see [Failures already paid for](#failures-already-paid-for)). Write the
test so it feeds identical values to both and asserts agreement to 1e-5. Most of
it can run **without Isaac Lab** if you exercise the formulas on plain tensors,
which means it belongs in the normal `pytest` run.

---

## 5. Training, once the gaps are closed

Bring it up in this order. Do not start at 2048 environments.

```bash
source ~/isaac-venv/bin/activate && cd ~/IsaacLab

# 1. Four environments, random actions. Goal: it steps without crashing.
./isaaclab.sh -p /path/to/sim2edge-ventuno/isaaclab_env/train_rsl_rl.py \
    --task Sim2Edge-Nav-v0 --num_envs 4 --headless --iterations 2

# 2. Cross-check the sensors against the numpy reference on the SAME room.
#    Both read sim2edge.rooms, so any disagreement is an adapter bug by
#    construction. Compare lidar range profiles before trusting a training run.

# 3. Scale up, lidar-only, and measure throughput.
./isaaclab.sh -p .../train_rsl_rl.py --task Sim2Edge-Nav-v0 \
    --num_envs 2048 --headless --init /path/to/runs/bc/bc_final.pt

# 4. Depth last. It is the throughput bottleneck: expect 256-1024 envs.
./isaaclab.sh -p .../train_rsl_rl.py --task Sim2Edge-Nav-v0 \
    --num_envs 512 --headless --use-depth --init runs/bc/bc_final.pt
```

### Where BC data comes from

BC does not need Isaac Sim. `collect_episodes` runs the oracle in the reference
environment on CPU and produces demonstrations under the same spec, so you can
generate the dataset while Isaac Lab is still being fixed:

```bash
source .venv/bin/activate
python3 scripts/train.py bc --episodes 10000 --lidar-only --out runs/bc
```

Measured on this project's CPU: ~2 s/episode for collection, so 10k episodes is
roughly 5–6 hours; run it overnight or shard it. Storage is quantized (§ README),
so 10k lidar-only episodes is a few GB, not 200.

The resulting checkpoint carries the spec fingerprint, so `train_rsl_rl.py`
will refuse it if the contracts disagree. That refusal is the feature.

Whether reference-env BC transfers well enough to be worth initializing Isaac
PPO from is **an open question this project has not answered.** The kinematics
differ. If the BC init turns out to hurt, train PPO from scratch in Isaac Lab and
keep BC as the reference-env baseline.

### Watch these two signals

Both are documented failure modes of the BC→PPO handoff and both are printed
every iteration:

- **Success rate dipping below the BC baseline** → the anchor anneal is too fast.
- **Entropy collapsing** (below −3.0) → raise `entropy_coef` or slow the anneal.

`train_rsl_rl.py` prints a `[handoff]` line every 10 iterations showing whether
the actor is frozen and the current `anchor_beta`. During warmup the actor LR is
literally 0.0, not merely small.

---

## 6. Handing a checkpoint to the board

The training host does not need ExecuTorch. It produces two artifacts:

```bash
# 1. The checkpoint. Carries spec_fingerprint + spec dict inside it.
runs/isaac_ppo/ppo_best.pt

# 2. The calibration set, built from the FINAL policy's own rollouts.
python3 tools/calib_dataset.py --weights runs/isaac_ppo/ppo_best.pt \
    --output calib/nav_calib.npz
```

Copy both to the board. Everything after that is `docs/deployment.md`.

The calibration set must come from the policy you are actually shipping, through
`tools/calib_dataset.py`, which guarantees the samples went through the same
preprocessing the C++ node performs. Observations from anywhere else defeat the
purpose — quantization ranges calibrated on the wrong distribution are the
project's #1 documented risk.

Ideally the calibration set also includes real rosbag observations
(`--bag-npz`), since sim activation ranges are not deployment activation ranges.
See `docs/domain_randomization.md`.

---

## Failures already paid for

Two bugs found by running this pipeline, both of the same class, both of which
would have surfaced on an NVIDIA host as "PPO won't learn":

1. **`Demonstrations.dequantized` synthesized the lidar-only depth placeholder as
   `0.0`** while the env emits `invalid_fill = 1.0`. Zero normalizes back to
   `min_m` — an obstacle 0.3 m dead ahead filling the frame — i.e. the *opposite*
   of "no return". BC's validation loss fell fourfold at a 0% success rate.
   Now pinned by `test_lidar_only_placeholder_matches_the_env`.
2. **The same bug in `isaaclab_env/nav_task.py`**, `_get_observations`, filling
   `torch.zeros` in lidar-only mode. Fixed, but *not* covered by a test, because
   testing it needs Isaac Lab — which is the argument for §4.5.

The pattern: a constant that must agree across three implementations (numpy env,
Isaac, C++ node) and disagreed in two of them. When you touch
`_get_observations`, assume this class of bug is present until a test says
otherwise.

---

## Quick reference

| Task | Environment | GPU? |
|---|---|---|
| `pytest`, parity gate, spec header | `.venv` | no |
| Oracle, BC, DAgger, reference-env PPO, eval | `.venv` | no |
| `smoke_export.py` stages 1/2/5, `calib_dataset.py` | `.venv` | no |
| Isaac Lab task, rsl_rl PPO at scale | `~/isaac-venv` | **yes, RTX** |
| ExecuTorch lowering, QNN export, robot | the board | no (NPU) |
