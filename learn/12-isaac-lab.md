# 12 — Isaac Lab

Everything so far has run one environment at a time in numpy. That is fine for
understanding the pipeline and hopeless for training the policy that ships: PPO is
on-policy, so it needs enormous amounts of fresh experience, and 16 Python
environments deliver a trickle.

Isaac Lab runs thousands of environments on the GPU, with physics and rendering,
all in parallel tensors. `isaaclab_env/nav_task.py` (406 lines) is this project's
task definition for it.

> **Status: this file has never been executed.** It is a careful draft with four
> known defects, documented in `docs/training_host_setup.md` §4. Read it as a
> design to audit, not code to run. This chapter is honest about which parts are
> real.

## The scale argument

| | Environments | Transitions/iteration |
|---|---|---|
| `sim2edge/training/ppo.py` | 16 | 1,024 |
| Isaac Lab, lidar-only | 2,048–4,096 | ~130,000–260,000 |
| Isaac Lab, with depth | 256–1,024 | rendering-bound |

Two orders of magnitude. This is why a task that takes days in a serial simulator
takes an hour here — and why `REBUILD_PLAN.md:114` expects >95% success on an
empty plane "within ~5 minutes of wall clock."

Note the depth row. `REBUILD_PLAN.md:171-172`:

> Expect **256–1024 envs**, not 4096. Rendering is the throughput wall.

Tiled rendering is the single most expensive thing in the loop, which is why
`nav_task.py:126-131` renders at exactly the policy's 96×160 rather than rendering
large and downsampling.

## `DirectRLEnv`

Isaac Lab offers two workflows. The **manager-based** one composes an environment
from configurable observation/reward/termination term objects. The **direct** one
has you write `_get_observations`, `_get_rewards`, `_get_dones` as methods.

This project uses direct, and `REBUILD_PLAN.md:70-72` agrees:

> `DirectRLEnv` is the right workflow here — the manager-based one adds config
> indirection you don't need for a single task.

The lifecycle:

```
_setup_scene()      once   — spawn ground, robot, sensors, obstacles
_pre_physics_step() 20 Hz  — take the action, convert to actuator commands
_apply_action()   200 Hz   — write to the simulator each physics substep
_get_observations() 20 Hz  — build the observation tensor
_get_rewards()      20 Hz  — compute the reward tensor
_get_dones()        20 Hz  — termination and truncation masks
_reset_idx(ids)     varies — reset only the environments that finished
```

`_reset_idx` taking a *tensor of indices* is the key idea. Environments finish at
different times; you reset only those, while the rest keep stepping.

## Configuration reads the spec

`NavEnvCfg` at `nav_task.py:66`:

```python
decimation = 10  # physics at 200 Hz, control at the spec's 20 Hz
episode_length_s = SPEC.reward.max_episode_steps * SPEC.action.dt
action_space = SPEC.action.shape[0]
observation_space = int(
    np.prod(SPEC.obs.depth.shape) + SPEC.obs.lidar.bins
    + SPEC.obs.goal.shape[0] + SPEC.obs.last_action.shape[0])
state_space = SPEC.obs.privileged.dim
```

Not a literal in sight. `decimation = 10` means physics steps ten times per
control step: 200 Hz physics, 20 Hz control, matching `ActionSpec.control_rate_hz`.

`state_space` is Isaac Lab's name for the privileged critic observation — the
asymmetric actor-critic from chapter 08, supported natively.

### The flattening seam

`nav_task.py:74-77`:

> rsl_rl consumes a flat vector; the policy's unflattening back into the four
> named tensors happens in `Sim2EdgePolicyWrapper` below, using the same order the
> exported graph binds — so a reordering bug cannot survive to deployment
> unnoticed.

The observation is 15,545 floats: `96×160 + 180 + 3 + 2`. rsl_rl wants one flat
vector; the policy wants four named tensors. Somebody has to slice.

This is a genuine risk. A slice-offset bug trains perfectly well — the network
just learns a permuted input — and then deploys wrong, because the C++ node binds
tensors in the *correct* order. `docs/training_host_setup.md:221-225` says to
unit-test the split against `spec.actor_input_shapes` before training. Do that.

## Sensors

### Lidar via ray casting

```python
lidar: RayCasterCfg = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, SPEC.robot.lidar_height_m)),
    attach_yaw_only=True,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(-180.0, 180.0),
        horizontal_res=1.0,
    ),
    max_distance=SPEC.obs.lidar.max_m,
    mesh_prim_paths=["/World/ground"],
)
```

`horizontal_res=1.0` gives 360 beams, matching `LIDAR_BEAMS` in chapter 07. The
comment at `:104-107` repeats the rule:

> Simulate the real beam count and let preprocessing min-pool to the spec's bins,
> exactly as on the robot. Ray-casting straight to 180 bins would skip the pooling
> step that the deployed pipeline actually performs.

`attach_yaw_only=True` keeps the scan plane horizontal regardless of chassis
pitch/roll — correct for a 2D lidar on a flat floor.

**This is the biggest technical risk in the file.** `REBUILD_PLAN.md:128-131`:

> `RayCasterCfg` with `patterns.LidarPatternCfg` ray-casts against a *warp mesh
> snapshot* of the prims in `mesh_prim_paths`. Whether pose changes to those prims
> are reflected after the snapshot is version-dependent. Budget one hour to test
> it: move an obstacle, confirm the scan changes.

If the snapshot is static, your lidar sees the ground plane and nothing else, and
the policy trains against an empty world while obstacle collisions still fire. It
would look like a reward bug for a long time. The listed defect at
`docs/training_host_setup.md` §4 is exactly this: "a raycaster that sees no
obstacles."

The recommended fallback (`REBUILD_PLAN.md:134-139`) is to compute the scan
analytically in torch — closed-form ray–circle intersection over
`(num_envs, n_beams, n_obstacles)`, plus ray–segment for the four walls. About 40
lines, exact, and faster than ray casting. If you rebuild, write this first.

### Depth via tiled camera

```python
data_types=["distance_to_image_plane"],   # forward depth, as OAK-D
width=SPEC.obs.depth.width,
height=SPEC.obs.depth.height,
```

`distance_to_image_plane` is forward distance (a Z-buffer), not radial range —
which is what a stereo camera reports and what `preprocess_depth` expects. Picking
`distance_to_camera` instead would give you radial distance and a subtly warped
image, worst at the frame edges.

## Known defects

`docs/training_host_setup.md` §4 documents four. They are worth reading before you
touch this file, because each is the kind of bug that costs a day:

| # | Defect | Symptom |
|---|---|---|
| 4.2 | No Create3 asset; `robot: ArticulationCfg = MISSING` | won't start |
| 4.2 | `_apply_wheel_velocities` assumes joint order `[left, right]` | robot turns the wrong way — "reads as a broken policy" |
| 4.3 | `_spawn_layouts` conflicts with `clone_environments` | per-env layouts don't work; startup very slow at 2048 |
| 4.4 | `_load_actor_into_runner` raises `NotImplementedError` | cannot load a BC checkpoint |
| 4.5 | No observation parity test | Isaac's torch normalization vs `preprocess.py` — unverified |

Two deserve extra attention.

**4.4 was left unimplemented deliberately** (`training_host_setup.md:215-217`):

> rsl_rl's `ActorCritic.__init__` signature has changed across releases, and a
> wrong guess loads weights into the wrong tensors *without error*. Write it
> against the installed version.

That is the right call. A confidently-wrong weight load is worse than a
`NotImplementedError`.

**4.5 is the same bug class this whole repository is organized against.**
`_get_observations` reimplements the spec's normalization in torch, because its
data is a GPU batch while `sim2edge/preprocess.py` is numpy. So there are now two
implementations of the same arithmetic, and — unlike the Python/C++ pair — nothing
checks that they agree. `nav_task.py` names `tests/test_isaac_obs_parity.py` in a
comment. It does not exist.

## Getting there

Do not debug this environment on top of a broken install. `REBUILD_PLAN.md:62-69`:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0 --headless
```

> If this doesn't produce a rising reward curve in 2 minutes, stop and fix the
> install.

Then, the intended entry point for this task:

```bash
./isaaclab.sh -p isaaclab_env/train_rsl_rl.py --task Sim2Edge-Nav-v0 \
    --num_envs 2048 --headless --init runs/tutorial_bc/bc_final.pt
```

`isaaclab_env/train_rsl_rl.py` (203 lines) installs the chapter-10 critic-warmup
and BC-anchor mechanisms as rsl_rl callbacks, because — per its docstring at
`:22-24` — rsl_rl has no built-in equivalent, so both would otherwise be silently
skipped.

The training host has an RTX 4090 and an `~/isaac-venv`:

```bash
ssh vkommera@dronelab12-4
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# NVIDIA GeForce RTX 4090, 24564 MiB
```

Full setup instructions are in `docs/training_host_setup.md`.

## The staged alternative

`REBUILD_PLAN.md` argues for building this up rather than debugging it whole, and
the argument is strong. Its phase 1 (`:78-116`) is:

- a capsule `RigidObject` driven by `write_root_velocity_to_sim`, **not** a Create3
  articulation with wheel IK — which deletes defect 4.2 entirely, and is defensible
  because the real Create3 accepts `/cmd_vel` and closes its own wheel loop;
- 5 observation floats, no sensors at all;
- an empty plane;
- Euclidean progress reward;
- stock rsl_rl PPO, no tuned hyperparameters.

Target: >95% success in about five minutes. And the reasoning at `:115-116`:

> If this doesn't work, nothing downstream will — the bug is in your reward or
> your frame math, and it is much cheaper to find here.

That is the correct instinct. The current `nav_task.py` tries to be correct about
articulations, per-env layouts, ray-cast lidar, tiled depth, geodesic rewards, and
privileged observations *simultaneously*, having never once produced a rising
reward curve. When it fails, you will not know which of the six is wrong.

## Check yourself

- Why simulate 360 lidar beams when the policy consumes 180 bins?
- `decimation = 10`. What breaks if you set it to 1?
- Why does a slice-offset bug in the observation flattening train fine but deploy
  wrong?
- The raycaster may not see moved obstacles. Design a five-line test that would
  catch it in one minute.
- Why does driving a capsule with `write_root_velocity_to_sim` count as *more*
  faithful to deployment, not less?

→ [13 — Export and quantization](13-export-and-quantization.md)
