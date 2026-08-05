# Runbook

Commands to run things. Rationale and history live in `docs/daily/`.

---

## Setup

Two Python environments — Isaac Sim pins Python 3.11 / torch 2.7.

| | Path | Python | Used for |
|---|---|---|---|
| Project | `.venv` | 3.12 | Tests, dataset, training |
| Isaac | `~/isaac-venv` | 3.11 | Anything importing `isaaclab` |

```bash
uv sync --extra dev
~/isaac-venv/bin/python -m pip install -e .
sudo apt install ros-jazzy-turtlebot4-description ros-jazzy-xacro
```

Isaac Sim 5.1.0 / Isaac Lab 0.54.4, source install at `~/Documents/Qualcomm/IsaacLab`.
Moving that directory breaks the editable install — re-run `./isaaclab.sh --install`
from the new location, or symlink instead.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

No GPU or simulator required — keep it that way.

---

# Phase 1 — Behaviour cloning data

## 1. Build the robot asset (one-time)

```bash
~/isaac-venv/bin/python tools/convert_tb4.py --output assets/turtlebot4_lite.usd
```

Skip if `assets/turtlebot4_lite.usd` exists. `--urdf-only` expands the xacro without Isaac.

## 2. Warm the warp cache (one-time per warp version)

```bash
ls ~/.cache/warp/     # want a dir matching: python -c 'import warp; print(warp.config.version)'
```

If absent:

```bash
~/isaac-venv/bin/python scripts/collect_bc.py --num_envs 4 --num_layouts 8 --check
```

**Do not interrupt** — several minutes of no output is warp JIT-compiling, not a hang.
A killed compile is discarded and the next run starts over.

## 3. Bring-up check

```bash
~/isaac-venv/bin/python scripts/collect_bc.py --num_envs 4 --gui --check
```

Add `--dump-scene` for env origins, camera pose and prim world positions.

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

Do not run multiple collectors on one GPU. Collection appends to existing
`bc_*.npz` shards in each split directory — use a clean output directory when
changing seed or parameters. `Ctrl-C` writes buffered episodes and exits
cleanly; a second `Ctrl-C` abandons them.

Memory scales as `num_envs × 900 × 720 × 2 B` (5.4 GB at 4096, 10.9 GB at 8192).
If PhysX reports `foundLostPairsCapacity`, raise `gpu_found_lost_pairs_capacity`
in `NavEnvCfg.sim` to the number it names.

| Flag | Default | Meaning |
|---|---|---|
| `--num_envs` | 256 | Parallel environments |
| `--episodes` | 2000 | Retained episode prefixes |
| `--num_layouts` | 512 | Geometry/goal pool size |
| `--starts_per_layout` | 32 | Feasible start candidates per layout |
| `--sigma` | 0.15 | Execution noise fraction; `0` = pure expert |
| `--shard_steps` | 200000 | ~160 MB per shard |
| `--validation_fraction` | 0.05 | Fraction of layout IDs reserved for validation |
| `--writer_queue` | 2 | Max outstanding async shards |
| `--filter-failures` | off | Legacy: drop short/low-progress prefixes |

## 5. Verify before training

```bash
.venv/bin/python scripts/verify_shard.py data/bc_diverse_v1/train/bc_000.npz
.venv/bin/python scripts/verify_shard.py data/bc_diverse_v1/validation/bc_000.npz
```

---

# Phase 2 — BC training

## 1. Pipeline smoke test

```bash
.venv/bin/python scripts/train_bc.py data/bc_diverse_v1 \
    --out runs/bc_smoke --epochs 1 --max-train-shards 2 \
    --val-shards 1 --no-compile
```

## 2. Full 4090 run

```bash
.venv/bin/python scripts/train_bc.py data/bc_diverse_v1 \
    --out runs/bc_diverse_v1 --epochs 8
```

Defaults target a 24 GB RTX 4090: batch 8192, BF16 autocast, compiled model,
fused AdamW, four-worker/eight-shard prefetch.

Resume after an interruption:

```bash
.venv/bin/python scripts/train_bc.py data/bc_diverse_v1 \
    --out runs/bc_diverse_v1 --epochs 8 \
    --resume runs/bc_diverse_v1/latest.pt
```

Resume refuses a changed spec, dataset split, or config — use a new output
directory for a new experiment.

## 3. Closed-loop evaluation

```bash
~/isaac-venv/bin/python scripts/eval_bc.py runs/bc_diverse_v1/best.pt \
    --episodes 512 --layout-seed 10000000 \
    --out runs/bc_diverse_v1/eval.json
```

Use these closed-loop metrics (success rate, SPL, collision, timeout) to
select the policy — not offline loss.

---

# Phase 3 — PPO fine-tune

## 1. Sanity-check the loop

```bash
~/isaac-venv/bin/python scripts/train_ppo.py \
    --init runs/bc_stack_dagger_v2_e4_final/best.pt \
    --num_envs 256 --num-steps 16 --iterations 5 --critic-warmup 2 \
    --out runs/ppo_smoke
```

Too short to measure `SR` — read it for crashes, throughput, and reward scale only.

To measure `SR` with the actor pinned (no learning):

```bash
~/isaac-venv/bin/python scripts/train_ppo.py \
    --init runs/bc_stack_dagger_v2_e4_final/best.pt \
    --num_envs 512 --num-steps 32 --iterations 40 --critic-warmup 40 \
    --out runs/ppo_frozen
```

## 2. Fine-tune

```bash
~/isaac-venv/bin/python scripts/train_ppo.py \
    --init runs/bc_stack_dagger_v2_e4_final/best.pt \
    --num_envs 2048 --iterations 300 --out runs/ppo_v1
```

`--layout-seed` should stay disjoint from the evaluation seed.

| Mechanism | Flag | Default |
|---|---|---|
| Critic warmup, actor LR zero | `--critic-warmup` | 25 iterations |
| Annealed anchor to frozen BC snapshot | `--anchor-coef` | 1.0 → 0 over 30% of run |
| Trust region | `--clip-range`, `--actor-lr` | 0.1, 1e-4 |

Watch per iteration: `SR` (should stay ≥ BC baseline), `std` (slowly decreasing),
`kl` (well under `max_kl` 0.02), `clip` (0.1–0.3), `ev` (climbing toward 1),
`timeout_rate` (under ~1%).

Checkpoints: `best.pt` on rollout success rate, `latest.pt` every 10
iterations, `final.pt` at the end.

Resume:

```bash
~/isaac-venv/bin/python scripts/train_ppo.py --resume runs/ppo_v1/latest.pt \
    --init runs/bc_stack_dagger_v2_e4_final/best.pt --out runs/ppo_v1
```

## 3. Evaluate

```bash
~/isaac-venv/bin/python scripts/eval_bc.py runs/ppo_v1/best.pt \
    --episodes 512 --layout-seed 10000000 --out runs/ppo_v1/eval.json
```

---

# Phase 4 — Odometry-free lidar/IMU point navigation

## 1. Collect the motion corpus

Same command as Phase 1 §4, output to `data/overnight_v3`.

## 2. Train an actor on the spinning-lidar observation

```bash
.venv/bin/python scripts/train_bc.py data/overnight_v3 \
  --out runs/bc_swept_v3 --epochs 2 --batch-size 4096 --no-compile
```

## 3. Train and check the motion belief

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

Resume at the next shard:

```bash
.venv/bin/python scripts/train_belief.py \
  --data data/overnight_v3 --out runs/belief_motion_v4 \
  --epochs 1 --batch-size 16 --mismatched-lidar-coef 20 \
  --resume runs/belief_motion_v4/latest.pt
```

Encoder options (`--derotate-scan-history`, `--scaled-scan-difference`,
`--confidence-coef`, `--confidence-gate-bias`, `--scan-vote-bins`,
`--scan-vote-sigma`) all need `--reset-odometry` and cannot be combined with
`--resume`. Example with the scan vote:

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

Sensor-reconstruction acceptance test, run before/after touching
`sensors.sweep_scan` or `deskew_history`:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/probe_sensor_consistency.py \
    --data data/belief_onpolicy_v15
```

Arrival-uncertainty calibration check:

```bash
.venv/bin/python scripts/probe_arrival_honesty.py
```

One run holds ~16 GB — do not run two on a 24 GB card.

## 4. Optional actor-only offline adaptation

```bash
.venv/bin/python scripts/train_belief.py \
  --data data/overnight_v3 \
  --belief-start runs/belief_motion_v4/best.pt \
  --out runs/belief_actor_v6 \
  --train-actor --freeze-estimator --batch-size 8
```

## 5. Adapt the actor in the closed loop

```bash
~/isaac-venv/bin/python scripts/train_belief_ppo.py \
  --init runs/belief_motion_v4/best.pt \
  --out runs/belief_ppo_v4 \
  --num-envs 256 --num-layouts 1024 \
  --iterations 2400 --freeze-estimator
```

Resume:

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

Current best actor: `runs/belief_ppo_biasfix_v14/iter_1130.pt`. Treat every
checkpoint as a development artifact until a full held-out evaluation writes
`"passed": true`.

Collect on-policy data with the full recurrent driver:

```bash
~/isaac-venv/bin/python scripts/collect_bc.py \
  --num_envs 4096 --episodes 16384 \
  --num_layouts 4096 --starts_per_layout 32 --seed 1501 \
  --out data/belief_onpolicy_v15 --shard_steps 200000 \
  --validation_fraction 0.05 --writer_queue 2 \
  --belief-driver-checkpoint runs/belief_ppo_biasfix_v14/iter_1130.pt
```

No external hardware pose reference is currently available. Robot trials can
verify timestamps, coordinate signs, reset behavior, and qualitative
navigation, but simulator acceptance must not be described as a measured
sim-to-real result.

---

# Phase 5 — Export and deploy

## 1. Export the deployable step and check it

The production stack (`body-frame-v2` actor + classical CPU localizer) exports
with `export_actor.py`:

```bash
.venv/bin/python scripts/export_actor.py runs/body_v2_actor_prod/best.pt \
    --out runs/body_v2_actor_prod/export
```

Fails if the exported program does not reproduce the eager module. Writes
`actor.pt2` plus `export_report.json` (operators, parameter count, parity,
latency).

`scripts/export_belief.py` is the equivalent for a belief-network checkpoint
(the earlier learned-estimator design, superseded by the Create3/EKF prior
above but still buildable):

```bash
.venv/bin/python scripts/export_belief.py runs/belief_vote_v22/final.pt \
    --out runs/belief_vote_v22/export
```

Fails if the exported program does not reproduce the eager module on
adversarial state. `--include-vote` traces the scan vote into the graph
instead of computing it outside (measurement only — default is outside).

## 2. Lower to the NPU, on the board

`ssh ubuntu@ventuno`; working copy is `~/sim2edge-deploy`.

```bash
rsync -az --exclude __pycache__ sim2edge/ ubuntu@ventuno:~/sim2edge-deploy/sim2edge/
rsync -az scripts/lower_qnn.py scripts/dump_calibration.py \
    runs/body_v2_actor_prod/best.pt ubuntu@ventuno:~/sim2edge-deploy/
```

Build the calibration set on the host:

```bash
.venv/bin/python scripts/dump_calibration.py \
    --data data/body_frame_v2_prod --samples 1024 --noise-scale 1.0 \
    --out runs/body_v2_actor_prod/export/calibration.npz
```

On the board — all four exports below are required, in this order
(`aarch64-oe-linux-gcc11.2` must come before `aarch64-ubuntu-gcc9.4` in
`LD_LIBRARY_PATH`, and `ADSP_LIBRARY_PATH` is unset by default):

```bash
export PYTHONPATH=/opt:$HOME/sim2edge-deploy
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.48.0.260626
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2:$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4
export ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v75/unsigned;/usr/lib/rfsa/adsp;/dsp"

~/.venv/executorch/bin/python lower_qnn.py best.pt \
    --calibration calibration.npz --quant-dtype use_16a8w --per-channel \
    --out out/actor_qnn.pte
```

`--soc-model QCS8300` is the default and correct for this board (the QCS8275
has no native QNN entry; the patched QCS8300 enum maps to the right HTP V75
target) — do not change it. Validate the runtime before trusting any number:

```bash
$QNN_SDK_ROOT/bin/aarch64-ubuntu-gcc9.4/qnn-platform-validator \
    --backend dsp --coreVersion --testBackend
```

Expect `Hexagon Architecture V75` and `Unit Test ... Passed`.

`rewrite_circular_convolutions` must stay on (default) — `--keep-circular-padding`
reproduces a lowering crash on purpose, for testing only. Leave
`MinMaxObserver` as the activation observer; `--moving-average-observer` is
for comparison only and degrades quantized accuracy.

## 3. Run and benchmark on the board

```bash
/opt/executorch/build-x86/executor_runner --model_path out/actor_qnn.pte \
    --num_executions 1000 --print_output none
```

`build-x86` is a native aarch64 build — the name is misleading.

```bash
.venv/bin/python scripts/bench_localizer.py --scans scans.npz \
    --max-submap-scans 128 --point-stride 2
```

## 4. The quantization gate

The gate is a sim success-rate drop under 2 points, not action MSE. Nothing
should be built on a `.pte` before it passes. `data/body_frame_v2_prod` is
`driver = "oracle"` — the deployed policy's own state distribution is the
stronger calibration source; `dump_calibration.py` warns if the corpus is not
policy-driven.

---

# Phase 6 — The ROS 2 stack

## Build

On the board:

```bash
cd ~/sim2edge-ventuno/ros2_ws
source src/sim2edge_bringup/scripts/ventuno_env.sh   # QNN + ExecuTorch paths
source /opt/ros/jazzy/setup.bash
colcon build --cmake-args -DEXECUTORCH_INSTALL_DIR=/opt/executorch/build-x86
```

On the training host, `colcon build` with no arguments (ExecuTorch paths not needed).

Read `ventuno_env.sh`'s comments before changing any of its exports.

## Pin the navigator to a big core — required, not tuning

Both launch files default `nav_cpus:=2,3` and apply it with `taskset`. Do not
run unpinned — an A55 core (`cpu4-7`) cannot hold the 50 ms control period for
the localizer.

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

Always pass `--checkpoint`, or the replay records the corpus's oracle command
as its reference instead of the policy's.

## Attach the sensors

One-time:

```bash
sudo usermod -aG dialout ubuntu     # then open a NEW login shell
```

Check `lsusb` before blaming software: the A1 is `10c4:ea60 CP210x`, the
OAK-D is `03e7:2485 Movidius MyriadX`.

## Run the robot

Every shell needs, in order:

```bash
cd ~/sim2edge-ventuno/ros2_ws
source /opt/ros/jazzy/setup.bash
source src/sim2edge_bringup/scripts/ventuno_env.sh
source install/setup.bash
```

Sensors only, no base:

```bash
ros2 launch sim2edge_bringup robot.launch.py \
    model:=$PWD/models/actor_qnn.pte \
    qnn_lib_dir:=$QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2 \
    require_odom:=false
```

With the Create 3, drop `require_odom:=false`.

### Verify the Create 3 link before launching

```bash
cat /sys/class/udc/*/state          # want: configured
ping -c3 192.168.186.2              # want: 0% loss
ros2 node list                      # want: /motion_control, /_internal/mobility, ...
ros2 topic info /cmd_vel            # want: Subscription count >= 1
```

Two failure modes, fix in this order:

1. **Ping fails, `ip neigh` shows `FAILED`** — gadget TX endpoint wedged:
   `sudo ~/Documents/ventuno-object-tracking/scripts/create3_usb_gadget.sh`
2. **Pingable but no SPDP** — base's ROS app is quiet (this follows any board
   reboot, since the board is the USB device). Restart the app, never the
   Reboot button:
   `curl -X POST http://192.168.186.2/api/restart-app`, then wait ~30 s.

### Give it a goal

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
    '{header: {frame_id: "base_link"}, pose: {position: {x: 2.0, y: 0.0}}}'
```

**The policy commands close to `V_MAX` immediately on a new goal** — only set
one when the robot can safely move, or with wheels off the ground.

### Watch / stop it

```bash
ros2 topic echo /nav/status        # per-stage cost, belief, hold reason
ros2 topic echo /safety/reason     # why the safety layer is limiting
```

`Ctrl-C` on the launch stops the navigator, but `sim2edge_safety` dies with
it too — **this is not an e-stop**. Keep the base's own button reachable.

### Confirm the gyro's yaw axis whenever the OAK-D is re-mounted

```bash
python3 tools/probe_yaw_axis.py --seconds 120     # on the board, /imu running
```

Turn the robot a quarter turn left, pause, then turn it back right, flat.
Set `gyro_yaw_axis` / `gyro_yaw_sign` in `navigator.yaml` from what it reports.

## Regenerate the C++ spec header after touching `spec.py`

```bash
.venv/bin/python tools/gen_spec_header.py            # rewrite
.venv/bin/python tools/gen_spec_header.py --check    # fail if stale
```

---

# Driving a policy by hand

```bash
~/isaac-venv/bin/python scripts/interactive_nav.py                     # actor only
~/isaac-venv/bin/python scripts/interactive_nav.py --driver localized  # full stack
```

Left-click the floor to send the robot there. `--driver localized` is the
deployable stack; the orange disc is the robot's belief of the goal. This is
a demonstration, not a measurement — use `eval_bc.py`, `eval_localized.py`,
or `eval_belief.py` for numbers.

---

## Common failures

| Symptom | Fix |
|---|---|
| Silent, minutes, ~100% CPU on one core | Warp JIT. One-time per version; let it finish. |
| `No module named 'pxr'` | Import `isaaclab.app`, launch, *then* the rest. |
| `cannot import name 'AppLauncher' ... circular import` | A local file shadows a stdlib module Isaac imports (e.g. `inspect.py`). Rename it. |
| `PxGpuDynamicsMemoryConfig::foundLostPairsCapacity` | Raise `gpu_found_lost_pairs_capacity` in `NavEnvCfg.sim` to the number reported. |
| RSS climbs and `ms/step` creeps up | Recorder buffers should be a fixed ceiling printed at startup; if not, something reintroduced per-step allocation. |
| Every lidar bin reads 1.0 | Rays hit nothing — sensor targets not matching obstacle prims. |
| Viewport looks empty | `--gui` frames env 0 after reset; `--dump-scene` gives coordinates. |
| Robot renders as two floating wheels | Use `assets/turtlebot4_lite_visual.usd`, or rebuild with `tools/convert_tb4.py`. |
| `ValueError: shard ... was collected under spec X` | `spec.py` changed after collection. Recollect, or check out the spec that produced the shard. |
| Yield looks great early, then collapses | Expected — wait until `kept + dropped` ≫ `num_envs`. |

## Repository layout

```
sim2edge/        simulator-free core: spec, grids, rooms, oracle,
                 preprocess, noise, dataset, localization
ros2_ws/         the deployed robot: C++ ROS 2 workspace, builds on the
                 training host and on the Ventuno from this same tree
isaaclab_env/    Isaac Lab adapter: nav_task.py (batch), interactive.py (viewer)
scripts/         collect_bc.py, verify_shard.py, train_bc.py,
                 eval_bc.py, train_ppo.py, train_belief.py,
                 train_belief_ppo.py, eval_belief.py,
                 eval_localized.py, interactive_nav.py
tools/           convert_tb4.py
tests/           no GPU or simulator required
docs/daily/      engineering log
assets/, data/   generated; not in git
```
