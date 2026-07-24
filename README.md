# sim2edge-ventuno

End-to-end PyTorch deployment pipeline: train a goal-conditioned navigation
policy in simulation, quantize it to INT8, and run it on a Turtlebot 4 Lite
whose compute has been replaced with an **Arduino Ventuno Q** (QCS8275, Hexagon
HTP V75 NPU), through ExecuTorch's QNN backend.

**Task.** Navigate to a coordinate specified relative to the robot's starting
pose while avoiding obstacles, using onboard sensing only: 2D lidar, an OAK-D
Lite depth camera, and wheel odometry.

See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the full design
rationale and the two-week schedule.

---

## The one thing to understand first

Both sibling projects on this board (`../ventuno-collision-avoidance`,
`../ventuno-object-tracking`) lost the most time to the same failure: the
preprocessing used to calibrate the quantized model did not exactly match the
preprocessing the C++ node performed at runtime. This does not crash and does
not log. It produces a confident, plausible, wrong action.

The entire repo is structured around making that impossible:

```
sim2edge/spec.py                 ← the ONLY place constants are written down
   │
   ├── sim2edge/preprocess.py    ← Python implementation (sim, calibration, tests)
   ├── tools/gen_spec_header.py  ← generates ↓
   │      spec_generated.hpp     ← compiled into the C++ node
   │      └── preprocess.cpp     ← C++ implementation (robot)
   │
   └── fingerprint()             ← hash of the contract, written into every
                                   .pte sidecar and checked at model load
```

Three mechanisms enforce it, and none of them rely on anyone remembering:

1. **`tools/parity_check.py`** runs both implementations on the same raw sensor
   data — including NaN depth, all-invalid scans, and out-of-range values — and
   requires agreement to 1e-5. It currently agrees **bit-exactly** on depth and
   lidar. Also runs as `tests/test_cpp_parity.py`.
2. **The spec fingerprint** is written into each `.pte`'s JSON sidecar at export
   and verified by the ROS node at load. A model exported under one contract
   cannot silently run under another.
3. **`tools/gen_spec_header.py --check`** fails CI if the generated C++ header
   is stale, so a contract change cannot land in training and be forgotten on
   the robot.

---

## Repository layout

```
sim2edge/                   platform-agnostic core (numpy + torch only)
├── spec.py                 obs/action/reward/randomization contract — stdlib only
├── preprocess.py           Python reference preprocessing
├── rooms.py                procedural layout generator (shared by all 3 sims)
├── sensors.py              raycast lidar + 2.5D depth renderer
├── randomization.py        per-episode domain randomization
├── safety.py               classical safety layer (Python twin of nav_safety)
├── models/nav_policy.py    dependency-free torch model (train/calibrate/export)
├── oracle/                 occupancy grid, geodesic field, pure-pursuit expert
├── envs/reference_env.py   portable numpy environment
├── training/               BC + DAgger, PPO with the BC handoff
└── eval/                   evaluation harness and policy adapters

isaaclab_env/               PRIMARY training env (Isaac Lab, RTX 4090)
maniskill_env/              secondary env (ManiSkill/ROCm) — interop story
tools/                      export, calibration, parity, benchmarking
ros_ws/src/nav_policy_node/ C++ inference node (ExecuTorch, QNN + XNNPACK)
ros_ws/src/nav_safety/      classical safety layer — owns /cmd_vel
tests/                      pytest suite (66 tests, no GPU or ROS required)
```

### Three simulators, one task

`sim2edge/spec.py` and `sim2edge/rooms.py` are shared by all three environments,
so they are three renderings of one task rather than three similar tasks.

| Environment | Role | Requirements |
|---|---|---|
| `sim2edge/envs/reference_env.py` | Portable reference; runs the whole pipeline in CI | numpy only |
| `isaaclab_env/` | **Primary** — trains the deployed policy, 2–4k parallel envs | Isaac Lab, CUDA |
| `maniskill_env/` | Secondary — AMD/ROCm iteration, interop claim | ManiSkill |

---

## Quickstart

```bash
pip install -r requirements.txt

# Run the test suite (no GPU, no ROS, no simulator needed)
python3 -m pytest

# Verify the preprocessing contract end to end
python3 tools/parity_check.py --emit-inputs /tmp/in.bin
g++ -std=c++17 -O2 -I ros_ws/src/nav_policy_node/include \
    ros_ws/src/nav_policy_node/src/{preprocess,preprocess_parity}.cpp -o /tmp/parity
/tmp/parity /tmp/in.bin /tmp/cpp.bin
python3 tools/parity_check.py --emit-inputs /tmp/in.bin --cpp-outputs /tmp/cpp.bin
```

### Train

```bash
# Stage 1 — BC + DAgger. This is the Gate-B guaranteed demo.
python3 scripts/train.py bc --episodes 10000 --lidar-only --out runs/bc

# Stage 2 — PPO fine-tuning, with critic warmup + annealed BC anchor
python3 scripts/train.py ppo --init runs/bc/bc_final.pt --iterations 3000

# Evaluate at three noise tiers, plus a safety-layer run
python3 scripts/train.py eval --init runs/bc/bc_final.pt
```

### Deploy (on the board)

```bash
# 1. Build the calibration set — from the FINAL policy, plus real rosbag data
python3 tools/calib_dataset.py --weights runs/bc/bc_final.pt \
    --bag-npz calib/real_obs.npz --output calib/nav_calib.npz

# 2. Lower to INT8 for the HTP  (QCS8300 = patched stand-in for QCS8275/V75)
python3 tools/export_qnn.py --weights runs/bc/bc_final.pt \
    --calibration calib/nav_calib.npz --output models/nav_policy_qnn.pte

# 3. CPU fallback, always built
python3 tools/export_cpu.py --weights runs/bc/bc_final.pt \
    --output models/nav_policy_xnnpack.pte

# 4. THE GATE — no robot run before this passes
python3 tools/parity_check.py --pte models/nav_policy_qnn.pte \
    --weights runs/bc/bc_final.pt --sim-episodes 200

# 5. Latency against the 50 ms budget
python3 tools/bench_latency.py --pte models/nav_policy_qnn.pte

# 6. Run
colcon build --cmake-args -DEXECUTORCH_INSTALL_DIR=/opt/executorch
ros2 launch nav_policy_node nav_policy.launch.py backend:=npu goal_x:=3.0
```

---

## Design decisions worth knowing

**The policy never touches the motors.** `nav_policy_node` publishes to
`~/cmd_raw`; `nav_safety` owns `/cmd_vel`. They are separate processes so that a
crash, stall, or ExecuTorch deadlock in the inference node is exactly the
condition the watchdog detects — rather than taking the watchdog down with it
and leaving the robot driving on its last command. Nothing learned or quantized
has stopping authority.

**Obstacles have height, and that is the point.** The layout generator produces
tall obstacles, obstacles *below* the lidar plane, and overhangs the robot
drives under. A 2D lidar is structurally unable to handle the latter two, which
is the entire justification for the depth channel. If the simulator only made
tall boxes, the Gate-C depth-vs-lidar comparison would measure nothing.

**Progress reward is geodesic, never Euclidean.** Euclidean progress pays the
robot to drive into the concave side of an obstacle. The geodesic field is
computed once per layout and reused by the oracle, the reward, and the
feasibility check, so "optimal path" means one thing everywhere.

**The oracle is limited to the real robot's dynamics.** Its speed, turn rate,
and accelerations are read from the same `ActionSpec` the simulator and the C++
node use. An expert demonstrating commands a Create3 cannot track produces a
policy that faithfully imitates commands a Create3 cannot track — and the
resulting gap looks like a perception problem while actually being a dynamics
problem.

**BC → PPO uses an explicit handoff.** Naive BC-init + PPO destroys the actor: a
random critic gives noisy advantages, which apply large random updates to a
carefully fitted policy. Mitigated with a critic-warmup phase (actor LR exactly
zero), an annealed BC anchor against a *frozen* snapshot, and a conservative
trust region. BatchNorm statistics are frozen before RL starts.

**Demonstrations are stored quantized** (depth `uint8`, lidar `uint16`). A 10k-
episode float32 dataset would be ~200 GB. The precision loss is smaller than the
depth sensor's own noise, and smaller than what INT8 inference does to the same
tensor anyway.

---

## Status

| Component | State |
|---|---|
| Spec, preprocessing, C++/Python parity | Complete, parity **bit-exact** |
| Rooms, sensors, geodesic oracle | Complete — oracle **100% SR** with randomization on |
| Policy network (2.72 M shipped params) | Complete; `torch.export` verified with 4 inputs, all ops HTP-proven |
| Reference env + eval harness | Complete |
| BC + DAgger, PPO | Implemented; BC verified end to end on the reference env |
| Export / calibration / parity / benchmark tooling | Complete — **untested against real QNN hardware** |
| ROS 2 node + safety layer | Complete — **compiles only the parity path here; not yet built against ExecuTorch or run on the board** |
| Isaac Lab adapter | Written against the shared spec — **requires validation on the training host** |
| ManiSkill adapter | Written against the shared spec — **not yet run** |

Anything marked in bold needs hardware or a simulator install that this
development machine does not have.

---

## Known limitations

- **Odometry drift.** The goal is anchored in the Create3's odom frame, which
  drifts 1–2%/m with no SLAM correction. Keep demo goals ≤ 8 m. The policy is
  trained against odometry-drift randomization for exactly this reason, but
  drift is not eliminated — it is tolerated.
- **The policy is memoryless.** No recurrence in v1, since it complicates
  export, quantization, and the ROS node. If corner oscillation appears, the
  planned fix is a 4-frame stack of lidar + last action before reaching for a
  GRU.
- **Lidar encoder padding is zeros, not circular.** Circular padding is the
  geometrically correct choice for a 360° scan, but it lowers to an operator
  not among those proven delegated on HTP V75.
- **The reference env is kinematic**, not a rigid-body simulation. It is a good
  match for a Create3 on flat floor and is not intended to replace Isaac Lab.
- **The two `/opt/executorch` patches are fragile.** A fake `QCS8300` chipset
  enum mapping to HTP V75, and a `sed` enabling the QNN AOT pybind on aarch64.
  Pin the working ExecuTorch commit and do not upgrade mid-project.
