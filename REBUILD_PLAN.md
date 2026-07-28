# Rebuild plan — Isaac Lab only, written by you

The current repo is ~7,100 lines across three simulators, a numpy reference env,
an oracle planner, BC+DAgger, a hand-rolled PPO, a C++ preprocessing twin, and a
parity harness. Almost none of that is needed to demonstrate sim→edge.

This plan rebuilds the same demo in **roughly 800–1,000 lines you write yourself**,
one working milestone at a time.

**Rule for the whole rebuild: do not start phase N+1 until phase N trains and you
can explain every line of it.** That constraint is the entire point.

First: `git checkout -b archive/ai-version && git checkout -b rebuild main` and
delete everything except `configs/spec.json` and `docs/`. Old code you can peek at
becomes old code you paste.

---

## What gets cut, and why

| Cut | Reason |
|---|---|
| `maniskill_env/` | Two simulators = two task definitions to keep in sync forever. You have a 4090; use it. Isaac Lab only. |
| `sim2edge/envs/reference_env.py` (368 lines) | A numpy re-implementation of the task so tests run on a laptop. You're rebuilding to understand one env, not three. |
| `sim2edge/oracle/` (620 lines: A*, geodesic field, pure-pursuit) | Only existed to generate BC data. See below. |
| `sim2edge/training/` BC + DAgger + PPO (955 lines) | PPO from `rsl_rl` solves point-goal nav with a dense progress reward. BC bootstrapping is for sparse-reward, long-horizon nav (Habitat/PointNav at 500+ steps). Yours is ≤60 s episodes in a 10 m room with a 360° lidar — dense progress reward is enough. **This is the single biggest cut.** |
| Geodesic-distance reward + per-env geodesic grids | Requires the oracle. Euclidean progress is nearly identical in convex-ish rooms and needs zero infrastructure. Add geodesic later *only if* you observe the policy getting stuck behind concave obstacles. |
| C++ ROS node, `preprocess.cpp`, `gen_spec_header.py`, `parity_check.py`, `test_cpp_parity.py` (~700 lines + the whole `ros_ws`) | This machinery exists to keep two implementations of preprocessing in sync. Write the robot node in **rclpy** and import the *same Python function* the sim calls. The class of bug disappears instead of being tested for. A 4-layer CNN at 20 Hz does not need C++. |
| `sim2edge/rooms.py` procedural generator + fixed-per-env layouts | Randomize obstacle poses on `_reset_idx` instead of baking one room per env. Fewer lines, more layout diversity. |
| `configs/spec.json` + `spec.py` (546 lines of dataclasses) | Keep the *numbers* — they're reasonable. Throw away the schema layer. A dataclass or a plain module of constants is fine. |

**Kept from the old design, because it earns its keep:** the export sidecar. When
you export the `.pte`, write a JSON next to it with the obs layout and the
normalization constants, and have the robot node assert it matches at load. ~20
lines, and it catches the failure mode the README is right to be scared of.

---

## Target layout

```
sim2edge/
├── spec.py              ~80    constants: obs sizes, action limits, reward weights
├── env.py               ~250   the Isaac Lab DirectRLEnv
├── policy.py            ~90    actor/critic, torch-only, export-safe
├── preprocess.py        ~80    raw sensor msg -> obs vector (numpy; used by sim + robot)
├── train.py             ~80    rsl_rl entry point
├── export.py            ~100   checkpoint -> .pte + sidecar JSON
└── nav_node.py          ~150   rclpy node: subscribe, preprocess, infer, safety, publish
```

Seven files. No package sub-tree, no `__init__` re-export layers.

---

## Phase 0 — Get Isaac Lab running (½ day)

The AMD 6900 XT cannot run Isaac Sim. All simulation happens on the 4090; treat
the AMD box as a code-editing and export machine only.

1. Install Isaac Sim 5.x + Isaac Lab per the official docs (pip install path is
   the least painful now).
2. Sanity check with a stock task before touching your own code:
   ```
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
       --task Isaac-Cartpole-Direct-v0 --headless
   ```
   If this doesn't produce a rising reward curve in 2 minutes, stop and fix the
   install. Do not debug your env on top of a broken install.
3. Copy `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/` as your starting
   skeleton. `DirectRLEnv` is the right workflow here — the manager-based one adds
   config indirection you don't need for a single task.

**Milestone:** stock Cartpole trains.

---

## Phase 1 — Empty-room point-goal, no sensors (1 day)

The smallest thing that can learn anything.

**Robot.** Do *not* start with the Create3 USD articulation and wheel inverse
kinematics. Spawn a capsule/cylinder `RigidObject` and drive it by writing root
velocity directly from the action:

```python
# action = (v, omega) in SI units, already squashed by the policy's tanh
lin = torch.stack([v * cos_yaw, v * sin_yaw, zeros], dim=-1)
self.robot.write_root_velocity_to_sim(torch.cat([lin, ang], dim=-1))
```

This is not a shortcut you have to apologize for: the real Create3 accepts
`/cmd_vel` and closes its own wheel loop, so commanding body velocity *is* the
deployed interface. You add a first-order lag in Phase 4 to model the tracking
error. Skipping the articulation saves you a USD asset, joint drive tuning, and a
whole class of "why is my robot vibrating" debugging.

**Observation.** 5 floats: `[goal_dist_norm, goal_sin, goal_cos, last_v, last_omega]`.
Goal in the robot frame, distance normalized by 8 m, direction as sin/cos (never
feed a raw angle to a network — it wraps).

**Reward.**
```
r = 3.0 * (prev_dist - dist)          # progress, the term that does the work
  + 10.0 * reached                    # terminal bonus
  - 0.005                             # time penalty
  - 0.05 * ||a_t - a_{t-1}||^2        # smoothness; keeps the real robot from jerking
```

**Terminations:** reached goal (< 0.35 m), or episode timeout (60 s @ 20 Hz = 1200 steps).

**Train:** `rsl_rl` PPO, 4096 envs, default hyperparameters. Do not tune anything.

**Milestone:** >95% success on an empty plane, within ~5 minutes of wall clock.
If this doesn't work, nothing downstream will — the bug is in your reward or your
frame math, and it is much cheaper to find here.

---

## Phase 2 — Obstacles + lidar (2–3 days, the real work)

**Obstacles.** Fixed count per env (start with 8 cylinders — cylinders make the
lidar math trivial if you need the fallback below). Spawn them once as
`RigidObject` with kinematic bodies, then randomize their XY poses in `_reset_idx`
with `write_root_pose_to_sim`. Rejection-sample positions so nothing lands within
`robot_radius + margin` of the spawn or the goal.

**Lidar — read this before you start, it's the one real technical risk.**
`RayCasterCfg` with `patterns.LidarPatternCfg` ray-casts against a *warp mesh
snapshot* of the prims in `mesh_prim_paths`. Whether pose changes to those prims
are reflected after the snapshot is version-dependent. Budget one hour to test it:
move an obstacle, confirm the scan changes. Two outcomes:

- *It updates* → use `RayCaster`, done.
- *It doesn't* → **compute the scan analytically in torch.** With circular
  obstacles at known positions this is closed-form ray–circle intersection,
  vectorizes to `(num_envs, n_beams, n_obstacles)`, is exact, is faster than
  ray-casting, and is about 40 lines. Add the four walls as ray–segment. This is a
  legitimate engineering choice, not a hack — you own the geometry either way.

Either way: 180 bins over 360°, clamp to `[0.15, 8.0] m`, normalize to `[0,1]`,
and fill invalid returns with `1.0` (far), **never 0.0** — zero normalizes back to
"obstacle 15 cm dead ahead."

**Collision detection.** Skip `ContactSensor`. You know every obstacle position:
`min_dist_to_obstacle < robot_radius` in torch is one line, and it also gives you
the clearance signal for free. Terminate with a −10 penalty.

**Encoder.** 1D conv over the 180 bins (3 layers, stride 2, 32→64→64 channels →
Linear to 128), concatenated with the 5-dim goal/action vector, then a 2-layer
MLP trunk. Use zero padding, not circular — circular padding lowers to an op that
may not delegate cleanly to the HTP, and the wrap-around seam costs you nothing.

**Milestone:** >85% success with 8 obstacles, 2048–4096 envs, ~30–60 min of
training. Record a video (`--video` flag in the rsl_rl script). **This is the
point where you have a defensible project even if everything after it slips.**

---

## Phase 3 — Depth camera (2 days) — do this *after* Phase 5

Honest assessment: with a 360° lidar, depth adds little navigation capability. Its
real value here is the *demo* — a conv stack on the Hexagon NPU is a far more
interesting deployment story than an MLP, and it's what your sibling projects
already proved delegates at 8a8w.

So: get lidar-only running on the actual robot first, *then* come back and add
depth as the second act.

- `TiledCameraCfg`, `data_types=["distance_to_image_plane"]`, at the policy
  resolution (96×160 or smaller — don't render big and downsample).
- Expect **256–1024 envs**, not 4096. Rendering is the throughput wall.
- Same normalization pattern: clamp to `[0.3, 6.0]`, normalize, invalid → 1.0.
- 4-layer strided CNN (32→64→96→128) → Linear 256, concat into the existing trunk.
- Keep the graph static: no adaptive pooling, no dynamic shapes.

**Milestone:** matches or beats lidar-only success rate. If it doesn't, that's a
finding worth reporting, not a failure.

---

## Phase 4 — Domain randomization, thin (½ day)

Five things, applied in `_reset_idx`. Resist adding more.

1. **Action delay** — 1–3 control steps of latency via a small rolling buffer.
   The most important one by far for a real ROS stack.
2. **Velocity tracking** — gain in `[0.9, 1.1]` and a first-order lag
   `tau ∈ [0.05, 0.15] s` on the commanded velocity.
3. **Lidar noise** — additive `N(0, ≤0.03 m)` and 0–5% beam dropout (dropped beams
   get the invalid fill, same as real).
4. **Odometry drift** — 0–2% error accumulating on the goal vector, since the goal
   is computed from odom on the robot.
5. **Depth** (Phase 3 only) — 0–12% hole fraction, mild scale jitter, small pitch
   offset.

The old `spec.json` has sane ranges for all of these; reuse the numbers.

Mass/friction randomization is pointless when you command body velocity directly —
skip it.

**Milestone:** success rate drops on enable, recovers to within ~5% of the clean
number after retraining.

---

## Phase 5 — Export (1–2 days)

1. Make sure the exported module is **only the deterministic actor mean path** —
   no Gaussian head, no critic, no normalizer wrapper. If `rsl_rl` wraps your
   policy with an observation normalizer, either fold its running mean/std into the
   first layer or disable it and normalize in `preprocess.py`. Pick one; a
   mismatched normalizer is the exact silent-wrong-action failure the old README
   warns about.
2. Export → quantize (PTQ, 8a8w) → lower to QNN with ExecuTorch. Reuse
   `docs/deployment.md` and whatever your sibling projects did; that's institutional
   knowledge, not AI slop.
3. **Calibration data must come from the same `preprocess.py` the robot runs.**
   Dump a few thousand observations from a sim rollout and calibrate on those.
4. **Write the sidecar.** Next to `policy.pte`, a `policy.json`:
   ```json
   {"obs": ["lidar:180", "goal:3", "last_action:2"],
    "lidar": {"min_m": 0.15, "max_m": 8.0, "bins": 180, "invalid_fill": 1.0},
    "action": {"v_max": 0.3, "omega_max": 1.5},
    "sha": "<hash of the above>"}
   ```
   The node refuses to load a `.pte` whose sidecar hash doesn't match its compiled
   expectations.
5. Check delegation: confirm the whole graph lands on HTP, and measure latency.
   Budget is 50 ms at 20 Hz; you'll likely see single-digit ms.

**Milestone:** `.pte` runs on the board, output matches the PyTorch model on the
same input to a few LSBs.

---

## Phase 6 — On the robot (2–3 days)

One `rclpy` node, ~150 lines:

- Subscribe `/scan`, `/odom`; goal from a parameter or a topic.
- Preprocess with the **imported `preprocess.py`** — same file the sim uses.
- Infer at 20 Hz on a timer (not in the subscriber callback).
- **Safety layer** (~40 lines, and non-negotiable): publish zero velocity if any
  input is stale (>0.15 s lidar, >0.2 s odom), if inference hasn't returned in
  0.5 s, or if the minimum range in a ±35° forward sector is below 0.3 m. Scale
  linear velocity down linearly between 0.6 m and 0.3 m.
- Publish `/cmd_vel`.

Port to C++ only if you measure a real 20 Hz miss. You almost certainly won't.

**Milestone:** robot drives to a goal 3 m away with a box in the path, and
e-stops when you step in front of it.

---

## Schedule

| | |
|---|---|
| Phase 0–1 | 1.5 days |
| Phase 2 | 3 days |
| Phase 4 | 0.5 day |
| Phase 5 | 2 days |
| Phase 6 | 3 days |
| Phase 3 (depth, second act) | 2 days |
| **Total** | **~12 working days**, with lidar-only running on hardware by day 10 |

---

## Two habits that keep this yours

1. **One milestone per commit, and write the commit message before the code.** If
   you can't state what "working" means, you don't understand the phase yet.
2. **When you're stuck, ask for the concept, not the file.** "How does Isaac Lab
   handle per-env obstacle randomization?" leaves you with knowledge; "write my
   `_reset_idx`" leaves you where you are now.
