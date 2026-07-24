# Sim2Edge: End-to-End PyTorch Deployment Pipeline for Arduino Ventuno Q

**Goal:** Train a goal-conditioned navigation policy in simulation (Isaac Lab primary, ManiSkill-compatible) and deploy it end-to-end on a Turtlebot 4 Lite whose compute has been replaced with an Arduino Ventuno Q (QCS8275, HTP V75 NPU), running inference through ExecuTorch's QNN backend.

**Task:** Navigate to a coordinate specified relative to the robot's starting pose while avoiding obstacles, using onboard sensing only (2D lidar + OAK-D Lite + odometry).

**Timeline:** 2 weeks. The plan is sequenced so that a deployable, demoable system exists from ~Day 4 onward, and every subsequent step upgrades it. Scope gates are explicit — if a step slips, the fallback at each gate is still a complete story.

---

## 0. Guiding decisions (and what changed from PLAN.md v1)

| Decision | Choice | Rationale |
|---|---|---|
| Export path | `torch.export` → ExecuTorch QNN directly | **ONNX and TensorRT are dropped.** ExecuTorch consumes `torch.export` output natively; ONNX added a conversion/debugging surface with zero benefit for this target. TensorRT is Jetson-only. The proven flow already exists in the reference repos (see §6). |
| Distributed RL | Dropped (was DD-PPO/Habitat) | Habitat isn't in the stack. Isaac Lab gives 2–4k parallel envs on the single 4090 — single-GPU PPO is more than enough throughput for this task. |
| Observation space | Depth image + 2D lidar + goal vector + last action; RGB as a stretch add-on | You chose full vision + lidar. Within "vision," **depth is the primary visual channel**: it closes most of the sim-to-real visual gap for free (geometry transfers; textures/lighting don't), and it's what catches obstacles the lidar plane misses. RGB adds texture cues but brings the hardest sim-to-real problem in the project — it is staged last and cuttable (§9). |
| Sim split | Isaac Lab primary, ManiSkill secondary via a thin interop layer | Training runs happen on the 4090. ManiSkill port shares the task/obs/action spec so you can iterate on the AMD 6900 XT (§4). |
| BC→PPO handoff | Explicit stabilization phase | Naive BC-init + PPO destroys the pretrained actor (random critic → garbage advantages). Fixed with critic warmup + annealed BC auxiliary loss (§3.3). |
| Quantization | PTQ INT8 (8a8w), same as reference repos | QAT only if PTQ measurably degrades the policy (§6.2). |

---

## 1. Prior-art foundation (read these first)

Two sibling repos already solved the deployment half of this project on the same board. **Do not re-derive any of this.**

- **The runbook:** `../ventuno-collision-avoidance/.claude/skills/ventuno-setup/references/executorch-qnn.md` — complete, verified bring-up + debugging guide for ExecuTorch QNN on the Ventuno Q.
- **Board bring-up:** `../ventuno-collision-avoidance/scripts/install_ventuno_deps.sh` — idempotent, staged installer. Includes two load-bearing patches to `/opt/executorch` (lines 531–610): a fake `QCS8300` chipset enum mapping to HTP V75 (ExecuTorch has no QCS8275 entry), and a `sed` enabling the QNN AOT pybind on aarch64. These will break on ExecuTorch upgrades — pin the version that works.
- **Export scripts:** `../ventuno-collision-avoidance/tools/export_resnet18_qnn.py` (NPU) and `export_resnet18_cpu.py` (XNNPACK fallback).
- **C++ inference node pattern:** `../ventuno-collision-avoidance/src/collision_classifier/src/` — backend strategy pattern (`qnn_backend.cpp` / `cpu_backend.cpp` behind an interface), manual `dlopen` of QNN libs with `RTLD_GLOBAL`, `Module(path, LoadMode::MmapUseMlock)`.
- **Multi-sensor + control pattern:** `../ventuno-object-tracking/src/object_tracker/object_tracker/robot_tracker.py` — latest-message caching with staleness rejection instead of `message_filters`, decoupled control timer, stop-on-loss, zero-Twist on shutdown.
- **Lessons-learned:** `../ventuno-*/docs/projecthub_description.md` — "what was hard" sections.

Key facts inherited: QNN SDK 2.48.0.260626 at `/opt/qcom/aistack/qairt/`; everything builds **natively on the board** (no cross-compilation); ROS 2 Jazzy; ResNet18@224 and YOLOX-Tiny@416 both fully delegated to HTP at INT8 with no unsupported ops. Note: the ~5 Hz inference rate documented in those repos was **incorrect/pipeline-limited** — the NPU achieved more than that. Our policy network is far smaller than either model, so a 20 Hz control loop has ample headroom; still, measure actual `.pte` latency on Day 1 (§8).

---

## 2. Policy architecture

Designed backward from the NPU: every op below is already proven on HTP V75 at 8a8w by the reference models (Conv2d incl. depthwise, BatchNorm, ReLU/SiLU, MaxPool, cat, Linear).

**Observations (all in robot frame):**

| Input | Shape | Notes |
|---|---|---|
| Depth image (OAK-D Lite) | 1×96×160 (or 1×120×160) | Downsampled, clipped to [0.3, 6.0] m, normalized, invalid→max. Small on purpose. |
| 2D lidar scan | 180 bins | Min-pooled from raw scan, clipped/normalized. TB4 Lite's RPLIDAR gives 360°. |
| Goal vector | 3 | (distance clipped/normalized, sin θ, cos θ) to goal in robot frame, from odometry. |
| Last action | 2 | (v, ω) previous command — needed for the smoothness penalty to be learnable. |
| (Stretch) RGB | 3×96×160 | Only if Gate C passes (§9). |

**Network (~1–3 M params total):**

- **Depth encoder:** 4-layer CNN (32→64→96→128 ch, stride-2 convs, BN+ReLU) → flatten → FC 256. No pretrained backbone needed at this resolution; keeps export trivial.
- **Lidar encoder:** 3× Conv1d over the 180-bin scan (or plain MLP 180→256→128) → FC 128.
- **Fusion:** concat(depth 256, lidar 128, goal-MLP 32, last-action 16) → MLP 512→256.
- **Actor head:** FC → 2 outputs, `tanh`-squashed to (v ∈ [0, v_max], ω ∈ [−ω_max, ω_max]). Gaussian head during training; **only the mean path is exported**.
- **Critic (training only, never exported):** separate trunk with **privileged observations** — ground-truth geodesic distance field value, nearest-obstacle distance, true pose. Asymmetric actor-critic is free accuracy since the critic never ships.
- **Memoryless first.** No GRU in v1 — recurrent state complicates export, quantization, and the ROS node. If corner-oscillation shows up (it did in the object-tracking project), add a 4-frame stack of lidar+last-action before reaching for a GRU.

**Rules carried over from the reference repos:**
- Write the model as a **dependency-free, inlined torch-only module** (no torchvision/isaaclab imports in the model file) — the board's export venv can't import torchvision (`../ventuno-collision-avoidance/tools/collision_model.py:8-15`). One file, `models/nav_policy.py`, used identically for training, calibration, and export.
- **Nothing but the network in the graph.** Normalization constants documented in one config; action squash/scaling done in-graph (tanh is fine on HTP) but any clamping/safety logic in C++.

---

## 3. Training pipeline (Isaac Lab, RTX 4090)

### 3.1 Environment

- Isaac Lab task: procedurally generated rooms — random rectangular arena (4–10 m), 5–20 obstacles (boxes, cylinders, walls; vary size/aspect), random start pose, goal sampled 2–8 m away with guaranteed feasibility (check via the same distance field the oracle uses).
- Robot: Create3 base kinematics (differential drive) with TB4 Lite sensor rig — simulated 2D lidar (ray-cast), depth camera at the OAK-D mount pose. Isaac Sim has Create3/TB4 assets; validate sensor poses against the real robot early.
- Action space: (v, ω) velocity commands at **20 Hz decision rate**, matching deployment. Sim physics substeps finer.
- 2048–4096 parallel envs for the lidar-only config; camera envs are the throughput bottleneck — use Isaac Lab tiled rendering, expect ~256–1024 envs with depth.

### 3.2 Stage 1 — Imitation learning from an oracle

- **Oracle:** A* (or FMM) over the ground-truth occupancy grid → smoothed path → pure-pursuit + velocity profile → (v, ω) expert actions. The oracle must respect the same v_max/ω_max/accel limits as the real Create3, or BC learns commands the robot can't track.
- **Data:** 10k+ successful episodes, recorded as (obs, expert action) at 20 Hz, with domain randomization **on during collection** (§3.4).
- **Loss:** MSE as in v1, plus: 
  - **DAgger-style aggregation (1–2 rounds):** roll out the BC policy, relabel visited states with oracle actions, retrain. Cheap in sim, directly attacks covariate shift, and typically worth more than 10× more pure-BC data.
- **Exit criterion:** ≥80% success rate (SR) in sim on held-out layouts. BC alone often plateaus around 70–90% here; that's fine — PPO closes the gap.

### 3.3 Stage 2 — PPO fine-tuning

Framework: `rsl_rl` or `skrl` (both first-class in Isaac Lab; pick `rsl_rl` for simplicity).

**Reward (per step):**

```
R_t =  λ_goal    · 𝟙(reached)                       (terminal, e.g. +10)
     + λ_prog    · (d_{t−1} − d_t)                   (geodesic distance, not Euclidean)
     − λ_coll    · 𝟙(collision)                      (terminal or heavy penalty, e.g. −10)
     − λ_smooth  · ‖a_t − a_{t−1}‖²
     − λ_time    · 1                                  (small per-step cost; v1 was missing this —
                                                       without it, safe-but-slow loitering is optimal)
```

Use **geodesic** distance (from the oracle's distance field) for the progress term — Euclidean progress rewards driving into concave obstacles.

**BC→PPO handoff (the step v1 was missing):**
1. Initialize actor from BC weights; critic from scratch.
2. **Critic warmup:** ~500 iterations updating only the critic (actor LR = 0) so advantages aren't noise.
3. **Anchored fine-tune:** add auxiliary loss `β·‖π_θ(s) − π_BC(s)‖²` (or a KL term), annealing β→0 over the first ~30% of training. Conservative clip range (0.1) and low actor LR (1e-4) initially.
4. Watch for entropy collapse and for SR dipping below the BC baseline — either means the anneal is too fast.

**Exit criterion:** ≥95% SR, collision rate <2%, on held-out randomized layouts, **with sensor noise at deployment levels**.

### 3.4 Domain randomization (applied in both stages)

| Category | Randomization |
|---|---|
| Lidar | Range noise σ ∈ [0, 3 cm], per-beam dropout up to 5%, angular offset ±1°, min/max range clipping |
| Depth | Noise ∝ distance², holes/speckle (random invalid patches), slight extrinsic perturbation of camera pose, exposure-like scale jitter |
| Dynamics | Wheel-slip / velocity tracking error ±10%, mass/inertia ±20%, motor response modeled as first-order lag with τ ∈ [50, 150] ms |
| Latency | **Action delay of 1–3 control steps (50–150 ms), randomized per episode.** This is the single most important randomization for sim-to-real on a real ROS stack — sensor→cmd_vel latency on the robot will be ≥1 control period. |
| Odometry | Goal-vector corruption: slow drift + noise on the relative-goal input, since the real goal estimate comes from drifting Create3 odometry |
| Layout | Obstacle shapes/density, arena size, floor friction |

**Measure, don't guess:** on Day 1–2, record real lidar + depth + odom from the robot (rosbag), and set the noise ranges to bracket what you actually observe.

---

## 4. Isaac Lab ↔ ManiSkill interop layer

Keep this thin or it eats the schedule:

- `sim2edge/spec.py` — single source of truth: observation spec (names, shapes, dtypes, normalization constants), action spec (limits, rate), reward coefficients, randomization ranges. Plain dataclasses, no sim imports.
- Each platform implements one adapter that maps its native env to the spec (Isaac Lab `DirectRLEnv` subclass; ManiSkill `BaseEnv` subclass with the same procedural room generator, which is pure numpy and shared).
- Training code (BC dataset writer, PPO configs, eval harness) consumes only the spec — never a platform API.
- **ManiSkill's role:** iteration on the AMD 6900 XT (ROCm PyTorch) and the interoperability story for the writeup. It is **not** on the critical path; the deployed policy trains in Isaac Lab. Port it in the Week-2 parallel track (§8), cut it first if behind (§9).

---

## 5. Evaluation protocol

**In sim (automated, run on every checkpoint):** 500 held-out episodes; report Success Rate (goal within 0.35 m), **SPL**, collision rate, mean time-to-goal, mean |jerk| (smoothness). Evaluate at three noise tiers: clean / deployment-level / 2× deployment-level (robustness margin).

**Quantization parity (before any robot run):** run the FP32 checkpoint and the INT8 `.pte` (via ExecuTorch host runtime or the x86 QNN simulator path already enabled in the reference build) on the same 1k recorded observation batch; require action MSE small relative to action range and **sim SR drop <2%** when the quantized actions are replayed through the sim eval.

**On robot:** 3 obstacle layouts × 5 goals × 3 trials. Record: success (0.5 m radius, judged by measuring tape or overhead reference — odometry alone can't judge itself), collisions, safety-layer intervention count, wall-clock time. Interventions are a first-class metric — a run that succeeds only because the watchdog kept saving it is a finding, not a success.

---

## 6. Deployment pipeline

### 6.1 Export (on the Ventuno, in the pinned ExecuTorch venv)

1. Strip to inference module: actor mean path only, dependency-free `nav_policy.py`, `model.eval()`.
2. **QNN/HTP path** — mirror `export_resnet18_qnn.py`:
   ```python
   from executorch.backends.qualcomm.export_utils import QnnConfig, build_executorch_binary, make_quantizer
   quantizer = make_quantizer(quant_dtype=QuantDtype.use_8a8w,
                              backend=QnnExecuTorchBackendType.kHtpBackend, soc_model="QCS8300")
   qnn_config = QnnConfig(soc_model="QCS8300", backend="htp",
                          target="aarch64-oe-linux", build_folder=f"{EXECUTORCH_ROOT}/build-x86", ...)
   build_executorch_binary(model=actor, qnn_config=qnn_config,
                           dataset=calibration_generator, custom_quantizer=quantizer)
   ```
   (`QCS8300` is the patched-in stand-in for QCS8275/HTP V75 — see §1.)
   Multi-input note: the reference exports were single-tensor; the policy takes 4 tensors. Verify `build_executorch_binary` handles a tuple example input early (Day 1 smoke test, §8) — if it fights you, fall back to the explicit `torch.export` → `to_edge_transform_and_lower` flow with the QNN partitioner.
3. **XNNPACK CPU fallback** `.pte` too (as in `export_resnet18_cpu.py`, with `_check_ir_validity=False`). Note the reference board build used `-DBUILD_XNNPACK_BACKEND=OFF` — flip it ON for this project so `backend:=cpu` actually works as a live fallback.

### 6.2 Quantization

- **PTQ INT8 (8a8w)**, calibration = 500–1000 observation tuples drawn from **sim rollouts of the final policy + the real rosbags**, passed through the *identical* preprocessing that the C++ node performs.
- **The #1 documented failure mode in both reference projects is calibration/runtime preprocessing mismatch — it produces silent garbage, not errors.** Mitigation is structural: one `preprocess_spec` (constants in the shared spec file), a Python reference implementation, a C++ implementation, and a **parity test** that runs both on the same raw rosbag messages and asserts near-equality *before* the first robot trial. Budget half a day for this; the reference projects lost more.
- QAT only if the §5 parity gate fails after fixing preprocessing. (Escalation order: fix calibration data → per-channel weight observers → QAT.)

### 6.3 ROS 2 inference node (C++)

Clone the reference node structure (`collision_classifier`), don't reinvent:

- **Backend strategy pattern:** `qnn_backend.cpp` / `cpu_backend.cpp` behind one interface, `backend` param selects; QNN libs `dlopen`'d with `RTLD_NOW | RTLD_GLOBAL` before Module load; `Module(path, LoadMode::MmapUseMlock)`.
- **Sensor handling — follow `robot_tracker.py`, not `message_filters`:** cache latest `/scan`, latest depth image, latest odom (queue depth 1 each); the 20 Hz control timer grabs current caches and rejects any input staler than a per-sensor `max_age` (lidar ~150 ms, depth ~200 ms). Stale input ⇒ treat as sensor loss.
- **Goal handling:** goal is set once (param or topic) as (x, y) relative to start; each tick, transform to current robot frame using odometry. Accept that Create3 odom drifts ~1–2%/m — keep demo goals ≤8 m and note this limitation honestly in the writeup. (SLAM-corrected goals are out of scope; the policy was trained against odom-noise randomization for exactly this reason.)
- **Safety layer (non-negotiable, classical, and *outside* the network):**
  - Lidar bubble: any return < d_stop (~0.30 m) in the forward sector ⇒ v := 0 (rotation allowed); < d_slow (~0.6 m) ⇒ scale v down.
  - Command clamps to Create3 limits + rate limiting.
  - Sensor-loss / policy-timeout watchdog (0.5 s) ⇒ zero Twist. Zero Twist on shutdown. (Both patterns verbatim from the reference controllers.)
  - Log every intervention (it's an eval metric).
- Publish a debug topic (obs summary + raw vs. safety-filtered action) — but only when subscribed, per the reference pattern.
- **Namespace check on Day 1:** the two reference projects saw different Create3 topic namespaces (one needed `create3_republisher`). "Everything looks healthy but the robot doesn't move" is the symptom; check `ros2 topic list` before debugging anything else.

### 6.4 Board environment gotchas (inherited, will bite otherwise)

- `ADSP_LIBRARY_PATH` is **semicolon**-separated (FastRPC quirk) — `install_ventuno_deps.sh:381`.
- `qcom-fastrpc1`/`qcom-fastrpc-dev` apt packages required or DSP loading fails.
- Run `qnn-platform-validator --backend dsp --coreVersion --testBackend` before trusting anything — HTP arch mismatch = garbage output, not an error.
- DDS UDP buffer sysctl (`/etc/sysctl.d/60-ros2-dds.conf`) or raw image topics silently throttle; restart nodes after changing.
- Keep OAK-D at 640×360-class resolution (USB2 bandwidth); the policy input is tiny anyway.
- `find_package(executorch)` needs the tokenizer-target workaround from `collision_classifier/CMakeLists.txt:29-39`.

---

## 7. Repo layout

```
sim2edge-ventuno/
├── sim2edge/                  # platform-agnostic core
│   ├── spec.py                # obs/action/preprocess/reward spec — single source of truth
│   ├── models/nav_policy.py   # dependency-free model (train + calibrate + export)
│   ├── oracle/                # occupancy grid, A*/FMM, pure-pursuit expert
│   ├── rooms.py               # procedural layout generator (pure numpy, shared by both sims)
│   └── eval/                  # sim eval harness, parity tests
├── isaaclab_env/              # Isaac Lab task + BC collection + rsl_rl configs
├── maniskill_env/             # ManiSkill adapter (secondary)
├── tools/                     # export_qnn.py, export_cpu.py, calib_dataset.py, parity_check.py
├── ros_ws/src/
│   ├── nav_policy_node/       # C++ inference node (cloned pattern from collision_classifier)
│   └── nav_safety/            # watchdog/safety layer + launch files
└── docs/
```

---

## 8. Two-week schedule

Two parallel tracks — **Track A (training, lab 4090)** and **Track B (deployment, Ventuno + robot)** — meeting at integration gates. Track B starts Day 1; deployment is *not* a Week-2 activity.

**Days 1–2 — Walking skeleton + ground truth**
- B: Board bring-up via `install_ventuno_deps.sh` pattern (pin the working ExecuTorch commit). **Smoke test: export an untrained 4-input policy net → QNN `.pte` → run on NPU from a stub C++ node; measure latency.** Verify Create3 namespace; record rosbags of real lidar/depth/odom (feeds §3.4 noise ranges and §6.2 calibration).
- A: Isaac Lab env with Create3/TB4 asset, procedural rooms, lidar-only obs first; oracle planner working; verify sim sensor poses/FOVs against the Day-1 rosbags.
- **Gate A (end Day 2):** `.pte` runs on NPU; sim env steps with oracle reaching goals.

**Days 3–5 — Lidar-only policy, end-to-end**
- A: BC data collection (10k episodes) + training; 1 DAgger round; start PPO fine-tune (lidar-only trains fast at 2–4k envs).
- B: Full C++ node (sensor caching, preprocessing, safety layer); preprocessing parity test; calibration pipeline.
- **Gate B (end Day 5): lidar-only INT8 policy navigating the real robot to goals with obstacle avoidance.** This is the guaranteed demo. Everything after this improves it.

**Days 6–9 — Add depth (the "vision" upgrade)**
- A: Enable tiled-rendered depth camera in the env; retrain BC→PPO with depth+lidar (init what transfers from the lidar-only run); depth randomization tuned against real OAK-D bags.
- B: Depth preprocessing in C++ + parity test; re-export/re-calibrate; on-robot A/B vs. lidar-only, especially on low/overhanging obstacles the lidar plane misses (the reason depth is in the obs).
- Parallel (AMD desktop): ManiSkill adapter over the shared spec + rooms generator; sanity-train lidar-only BC there for the interop claim.
- **Gate C (end Day 9):** depth+lidar policy ≥ lidar-only on the robot benchmark.

**Days 10–12 — Hardening + evaluation**
- Full eval protocol in sim and on robot (§5); domain-randomization ablation (one run each: no-latency-DR, no-noise-DR) for the writeup; fix the worst real-world failure mode found.
- Stretch, only if Gate C passed early: RGB channel + heavy visual DR.

**Days 13–14 — Buffer + writeup**
- Demo video, metrics tables, honest limitations (odometry drift, memoryless policy, safety-intervention counts), pipeline diagram. The buffer *will* be consumed by sim-to-real surprises; that's what it's for.

---

## 9. Scope gates & fallbacks

| If behind at… | Cut / fallback |
|---|---|
| Gate A | Fall back to XNNPACK CPU `.pte` (policy is small; CPU may hit 20 Hz anyway) and debug QNN in parallel — don't block training on the NPU. |
| Gate B | Ship BC-only (skip PPO) — with DAgger, BC alone is usually demo-grade. |
| Gate C | Demo lidar-only as primary, show depth policy in sim; frame depth as "validated in sim, deployment in progress." |
| Any point | Cut order: RGB → ManiSkill parity → DR ablations → PPO (keep BC+DAgger) → depth. The lidar-only BC policy on the NPU is the minimum viable capstone and it exists by Day 5. |

## 10. Top risks

| Risk | Mitigation |
|---|---|
| Calibration/preprocess mismatch → silent garbage on NPU (bit both reference projects) | Structural: shared spec constants, mandatory parity test before robot trials (§6.2). |
| Multi-input export path unproven (references were single-tensor) | Day-1 smoke test with untrained net; explicit lowering flow as fallback (§6.1). |
| Sim-to-real latency gap | Action-delay randomization (§3.4); measure real sensor→cmd latency Day 1 and bracket it. |
| Camera-env training throughput on one 4090 | Small depth res (96×160), tiled rendering, lidar-only policy as the safety net. |
| PPO destroys BC init | Critic warmup + annealed BC anchor (§3.3); keep the best BC checkpoint deployable at all times. |
| ExecuTorch upgrade breaks the two `/opt/executorch` patches | Pin the exact commit the reference projects used; don't upgrade during these 2 weeks. |
| Odometry drift breaks goal tracking | Odom-noise DR on the goal input; goals ≤8 m; report as a stated limitation. |
