# Deployment on the Arduino Ventuno Q

Everything here is inherited from `../ventuno-collision-avoidance` and
`../ventuno-object-tracking`, which already solved the deployment half of this
problem on the same board. **Do not re-derive any of it.**

The authoritative runbook is
`../ventuno-collision-avoidance/.claude/skills/ventuno-setup/references/executorch-qnn.md`.
This document covers only what is specific to this project.

For the *other* end of the pipeline — the NVIDIA host that trains the policy this
document deploys — see [`training_host_setup.md`](training_host_setup.md). The two
halves meet at the spec fingerprint and nowhere else.

---

## Inherited facts

- QNN SDK **2.48.0.260626** at `/opt/qcom/aistack/qairt/`
- ROS 2 **Jazzy**
- **Everything builds natively on the board.** There is no cross-compilation
  step; do not try to add one.
- ResNet18@224 and YOLOX-Tiny@416 both lower fully to HTP at INT8 with no
  unsupported ops. Our policy uses a strict subset of their operators.

### The ~5 Hz figure in the reference repos is wrong

Both sibling projects documented roughly 5 Hz inference. That was
**pipeline-limited, not NPU-limited** — the NPU was considerably faster. Our
policy is about a tenth of ResNet18's size, so a 20 Hz control loop has ample
headroom. If `tools/bench_latency.py` reports anything near 5 Hz, suspect the
harness or a partially-delegated graph, not the hardware.

---

## Board bring-up

Follow the staged, idempotent installer pattern in
`../ventuno-collision-avoidance/scripts/install_ventuno_deps.sh`.

### The two load-bearing ExecuTorch patches

`install_ventuno_deps.sh:531-610` patches `/opt/executorch` in two places:

1. **A fake `QCS8300` chipset enum mapping to HTP V75.** ExecuTorch has no
   `QCS8275` entry. This is why every export command in this repo passes
   `--soc-model QCS8300`. It is not a typo and not a different chip.
2. **A `sed` enabling the QNN AOT pybind on aarch64**, without which lowering
   cannot run on the board at all.

**Both break on an ExecuTorch upgrade.** Pin the exact commit that works and do
not upgrade during the project. Record the commit hash in the run log alongside
the exported `.pte`.

### Build ExecuTorch with XNNPACK ON

The reference board build used `-DBUILD_XNNPACK_BACKEND=OFF`, which silently
makes `backend:=cpu` a dead parameter value. This project needs the CPU path to
be a **live fallback** (see Gate A below), so build with:

```bash
EXECUTORCH_BUILD_XNNPACK=ON
```

`nav_policy_node/CMakeLists.txt` hard-fails at configure time if the XNNPACK
targets are missing, rather than letting you discover it on the robot.

---

## Environment gotchas

These will bite otherwise. All inherited, all verified.

| Issue | Detail |
|---|---|
| `ADSP_LIBRARY_PATH` | **Semicolon-separated**, not colon. A FastRPC quirk. `install_ventuno_deps.sh:381`. |
| FastRPC packages | `qcom-fastrpc1` and `qcom-fastrpc-dev` must be installed or DSP loading fails. |
| Validate before trusting | Run `qnn-platform-validator --backend dsp --coreVersion --testBackend`. An HTP architecture mismatch produces **garbage output, not an error**. |
| DDS buffers | Set the UDP buffer sysctl in `/etc/sysctl.d/60-ros2-dds.conf` or raw image topics silently throttle. **Restart nodes after changing it.** |
| OAK-D resolution | Keep at 640×360-class (USB2 bandwidth). The policy input is 96×160 anyway. |
| `find_package(executorch)` | Needs the tokenizer-target workaround; already present in `nav_policy_node/CMakeLists.txt`. |
| Create3 namespace | **Check `ros2 topic list` on Day 1.** The two reference projects saw different namespaces, one needing `create3_republisher`. The symptom of getting this wrong is "everything looks healthy but the robot doesn't move" — check this before debugging anything else. |

---

## Day-1 smoke test

The plan flags one genuine unknown: the reference exports were all
single-tensor, and this policy takes **four** input tensors. Resolve it on Day 1
with an untrained network, before any training exists.

```bash
# Untrained model — the point is the export path, not the policy
python3 tools/export_qnn.py --output models/smoke_qnn.pte
python3 tools/bench_latency.py --pte models/smoke_qnn.pte
```

`torch.export` with the 4-input signature is already verified in
`tests/test_model_env_safety.py::test_multi_input_torch_export_succeeds`, and
the exported graph is checked against the operator set the reference models
proved delegated. What remains unverified off-board is whether
`build_executorch_binary` accepts a 4-tuple example input.

**If the helper flow fights the multi-input signature**, switch flows rather
than reshaping the model — a model contorted to fit a helper function no longer
matches the C++ node:

```bash
python3 tools/export_qnn.py --flow explicit --output models/smoke_qnn.pte
```

The explicit flow does `torch.export` → `prepare_pt2e` → calibrate →
`convert_pt2e` → `to_edge_transform_and_lower` with `QnnPartitioner` by hand.

---

## The parity gate

**No robot run happens before this passes.** Budget half a day; the reference
projects lost considerably more to skipping the equivalent.

Step 1 needs neither the board nor ExecuTorch — it compiles the preprocessing
translation unit with plain `g++` and is also run by `tests/test_cpp_parity.py`.
Run it on the training host before copying anything.

```bash
# 1. Preprocessing: Python vs C++, on adversarial raw sensor data
python3 tools/parity_check.py --emit-inputs /tmp/in.bin
ros2 run nav_policy_node preprocess_parity /tmp/in.bin /tmp/cpp.bin
python3 tools/parity_check.py --emit-inputs /tmp/in.bin --cpp-outputs /tmp/cpp.bin

# 2. Quantization: INT8 .pte vs FP32, including a full sim benchmark
python3 tools/parity_check.py --pte models/nav_policy_qnn.pte \
    --weights runs/bc/bc_final.pt --sim-episodes 200
```

The gate is **sim success-rate drop < 2 percentage points**, not an action MSE.
A small action error can still cost success at exactly the moments that matter.

If the quantization gate fails, escalate in this order — the first is by far the
most common cause:

1. **Fix the calibration data.** Is it from the *final* policy (not the oracle,
   whose state distribution differs)? Does it include real rosbag observations?
2. Per-channel weight observers.
3. QAT.

---

## Running

```bash
colcon build --cmake-args -DEXECUTORCH_INSTALL_DIR=/opt/executorch
source install/setup.bash

# Dry run first on any new layout: computes commands, publishes nothing
ros2 launch nav_policy_node nav_policy.launch.py \
    backend:=npu goal_x:=3.0 goal_y:=0.0 publish_cmd_vel:=false

# Live
ros2 launch nav_policy_node nav_policy.launch.py backend:=npu goal_x:=3.0
```

### Topic map

| Topic | Direction | Note |
|---|---|---|
| `/scan`, depth, `/odom` | in | staleness-rejected per `SafetySpec` |
| `/nav_policy/cmd_raw` | policy → safety | the policy's unfiltered output |
| `/cmd_vel` | safety → robot | **only `nav_safety` publishes here** |
| `/nav_safety/interventions` | out | running count; an evaluation metric |
| `/nav_policy/debug` | out | built only when subscribed |

### Monitoring

```bash
ros2 topic echo /nav_safety/interventions    # should stay near-flat
ros2 topic echo /nav_policy/debug            # obs summary + inference time
```

A rising intervention count means the safety layer is doing the driving. Record
it — per §5 of the plan, a run that succeeds only because the watchdog kept
rescuing it is a finding, not a success.

---

## Fallbacks

| Gate | If behind | Fallback |
|---|---|---|
| A (Day 2) | QNN lowering fights back | Ship the XNNPACK `.pte`; the policy is small enough that CPU may hold 20 Hz. Debug QNN in parallel — do not block training on it. |
| B (Day 5) | PPO not converging | Ship BC-only. With DAgger it is usually demo-grade. |
| C (Day 9) | Depth policy underperforms | Demo lidar-only; present depth as validated in sim. |

Cut order if the schedule slips: RGB → DR ablations → PPO (keep BC+DAgger) →
depth. The lidar-only BC policy on the NPU is the minimum viable capstone.
