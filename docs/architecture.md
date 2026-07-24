# Architecture and design decisions

This document records *why* things are the way they are. The what is in the
code; the why is what gets lost.

---

## 1. The contract, and why it is a single file

`sim2edge/spec.py` holds every constant that has to be agreed on by simulation,
quantization calibration, and the robot. It is stdlib-only — no numpy, no torch,
no simulator — because it must be importable from four very different
environments: the Isaac Lab CUDA stack, a ROCm host, the board's minimal
ExecuTorch export venv, and a code generator.

Everything else is an implementation of that contract:

```
                       sim2edge/spec.py
                              │
        ┌─────────────┬───────┴────────┬──────────────────┐
        ▼             ▼                ▼                  ▼
   preprocess.py  rooms.py      gen_spec_header.py   nav_policy.py
   (Python impl)  (layouts)            │             (sizes only)
        │             │                ▼
        │             │        spec_generated.hpp
        │             │                │
        │             │                ▼
        │             │          preprocess.cpp  ← C++ impl
        │             │                │
        └─────────────┴────────────────┘
                      │
              parity_check.py  ← proves the two implementations agree
```

### The fingerprint

`NavSpec.fingerprint()` is a SHA-256 of the contract, truncated to 16 hex
characters. It travels with every artifact:

- written into training checkpoints,
- written into calibration `.npz` files,
- written into each `.pte`'s JSON sidecar,
- compiled into the C++ node via `spec_generated.hpp`.

Each consumer verifies it. A policy trained under one preprocessing contract
*cannot* be silently calibrated or deployed under another.

Randomization ranges are deliberately excluded from the hash: they describe how
a policy was trained, not how its inputs are constructed at runtime, and the
evaluation harness legitimately varies them.

---

## 2. Why the preprocessing algorithms are what they are

Every choice here is constrained by "must be transcribable to C++ literally,
with no library-behaviour dependence."

**Downsampling is explicit min-pooling over integer index blocks, not
`cv::resize`.** Two reasons. Interpolating across a depth discontinuity invents
a surface that exists in neither the foreground nor the background. And
min-pooling is the only reduction that cannot delete a thin obstacle — a
1-pixel-high bar in a 480×640 frame survives to the 96×160 policy input.

**Invalid pixels are normalized *before* pooling.** "Invalid" and "far" both map
to 1.0, so the pooling needs no separate mask, and there is no second data
structure to keep synchronized between the two implementations.

**Invalid means far, not near.** A stereo hole is treated as free space. A policy
that brakes for every hole never moves in a real room; the lidar and the
classical bubble cover the pessimistic case.

**Lidar beams are binned by angle, not by index.** The RPLIDAR's start angle is
not stable across driver versions, and index binning would silently rotate the
policy's entire view of the world. Binning by wrapped angle makes the result
independent of scan direction, start angle, and beam count.

**The goal bearing is sin/cos, not an angle.** A raw angle is discontinuous at
±π — exactly where the robot needs to turn around. Quantizing a wrapping input
to INT8 is a silent-failure generator.

**Python's `%` and C's `fmod` differ for negative operands**, and the bin index
depends on the wrap. `preprocess.cpp:wrapAngle` reproduces Python's semantics
deliberately; getting this wrong is an off-by-one-hemisphere error.

---

## 3. The observation, and why depth is in it

| Input | Shape | Rationale |
|---|---|---|
| Depth | 1×96×160 | Geometry transfers sim-to-real for free; texture and lighting do not. Small on purpose. |
| Lidar | 180 | Min-pooled from the raw scan. 360° — reversing out of a trap needs to see behind. |
| Goal | 3 | (normalized distance, sin, cos) in the robot frame. |
| Last action | 2 | Required for the smoothness penalty to be *learnable* at all. |

The last-action input is not decoration. Without it, the smoothness penalty
depends on information not present in the observation, so the optimal action is
not representable by the policy.

**Depth exists for the cases a 2D lidar structurally cannot handle:** obstacles
below the scan plane, and overhangs that are not obstacles at all. The layout
generator produces all three height classes, and the simulated lidar genuinely
cannot see the first two. If the world only contained tall boxes, the Gate-C
depth-vs-lidar comparison would be measuring noise.

---

## 4. The network, designed backward from the NPU

Every operator is one the reference models already proved fully delegated on HTP
V75 at 8a8w: `conv1d`, `conv2d`, `batch_norm`, `linear`, `relu`, `cat`,
`flatten`, `tanh`. No adaptive pooling, no dynamic shape, no recurrence, no
data-dependent control flow. `tests/test_model_env_safety.py` asserts the
exported graph stays inside that set — an unsupported op silently splits the
graph and falls back to CPU per-op, which costs far more than it looks like.

**Lidar encoding is convolutional, not an MLP**, so it is translation-equivariant
in bearing: an obstacle 30° left and the same obstacle 30° right are the same
local pattern. Padding is zeros rather than circular, which is geometrically
wrong for a 360° scan but avoids a pad operator outside the proven set. The
wrap-around seam is the price.

**The critic is asymmetric and never ships.** It gets ground-truth geodesic
distance, obstacle clearance, and true pose. It is free accuracy because it does
not exist at inference. It has its own lidar encoder but deliberately *no* depth
encoder: the privileged vector already contains what a value function mostly
needs, and a second depth CNN would nearly double the cost of every training
step to re-derive information the critic is simply handed.

**Only the actor mean path is exported.** `ActorMean` holds references to the
actor submodules only, so the critic and `log_std` are unreachable from the
exported graph and cannot be pulled in by accident.

**The squash is in-graph.** `tanh` is cheap and proven on HTP, and doing the
scaling inside the graph means the C++ node receives SI units directly and
cannot apply a different scale than training did. Everything else — clamps, rate
limits, safety — is in C++ where it can be unit-tested.

---

## 5. Training

### BC → PPO handoff

Naive BC-init + PPO reliably destroys a good pretrained policy, and the
mechanism is specific: the critic starts random → value estimates are noise →
advantages are noise → the first few hundred policy-gradient steps apply large
random updates to a carefully fitted actor. By the time the critic is useful,
the actor is no better than random.

Three mitigations, all load-bearing:

1. **Critic warmup** — actor learning rate *exactly zero* for N iterations, not
   merely small. While advantages are noise, any actor step is pure damage.
2. **Annealed BC anchor** — `β·‖π(s) − π_BC(s)‖²` against a *frozen snapshot*,
   annealed to zero over the first 30%. Anchoring to a moving copy of the policy
   would be a no-op regularizer.
3. **Conservative trust region** — clip 0.1 and actor LR 1e-4.

Two failure signatures are reported every iteration: entropy collapse, and
success rate dipping below the BC baseline. Either means the anneal is too fast.

**BatchNorm is frozen before RL starts.** BN and PPO interact badly: rollouts are
collected in one mode and the update runs in another, so running statistics
drift with the changing policy and a state's value stops being stationary. BN is
active during BC, where it helps, and frozen afterwards. The frozen affine
parameters fold into the preceding convolution at export, so this costs nothing
at inference.

### The Gaussian lives in pre-squash space

The distribution is over the unbounded actor output, with no change-of-variables
correction — the same convention rsl_rl uses. The cost is that log-probabilities
are of `u`, not of the physical action. The benefit is that the entropy term
stays well-conditioned instead of collapsing as `tanh` saturates, which matters
a great deal for a fine-tune that starts near-deterministic.

### DAgger

Pure BC fails by drifting into states the expert never visited and having no
idea what to do there. More expert data *on the expert's own distribution* does
not fix this. `collect_episodes` takes the policy that drives and always labels
with the oracle, so BC and DAgger are the same code path with a different
argument.

---

## 6. Deployment

### Two processes, not one

`nav_policy_node` publishes to `~/cmd_raw`. `nav_safety` owns `/cmd_vel`.

If the safety logic lived inside the inference node, a crash, stall, or
ExecuTorch deadlock there would take the watchdog down with it and leave the
robot driving on its last command. As separate processes, the policy node going
silent is exactly the condition `nav_safety` is built to detect, and it responds
by publishing a zero Twist.

Nothing learned, and nothing quantized, has stopping authority.

### The bubble stops translation but permits rotation

A robot stopped 25 cm from a wall must still be able to turn away from it.
Freezing both axes converts a recoverable state into a deadlock.

### Sensor handling: caching, not `message_filters`

Following `robot_tracker.py` from the object-tracking project: every
subscription keeps only the latest message (queue depth 1), and a decoupled
20 Hz timer takes whatever is current, rejecting anything older than a
per-sensor `max_age`. An approximate-time synchronizer stalls the control loop
whenever one sensor hiccups, which for a moving robot is worse than acting on a
slightly stale scan — and much worse than deliberately stopping, which is what
staleness triggers here.

### Interventions are a metric, not a log line

`/nav_safety/interventions` publishes a running count. Per §5 of the plan, a run
that succeeds only because the watchdog kept rescuing it is a finding, not a
success. The simulator's `SafetyLayer` applies the same thresholds from the same
generated constants, so sim and hardware intervention counts are comparable.

---

## 7. Deliberate non-goals

- **ONNX and TensorRT are not used.** ExecuTorch consumes `torch.export` output
  natively; ONNX would add a conversion and debugging surface with no benefit
  for this target, and TensorRT is Jetson-only.
- **No SLAM.** The goal is anchored in the drifting odom frame. This is a stated
  limitation, bounded by keeping goals ≤ 8 m, and the reason odometry-drift
  randomization is in the training distribution.
- **No recurrence in v1.** It complicates export, quantization, and the node.
  The planned escalation if corner oscillation appears is a 4-frame stack of
  lidar + last action, before reaching for a GRU.
- **The reference env is not a physics simulator.** It is a kinematic
  differential-drive model with first-order motor lag — a good match for a
  Create3 on flat floor, and not a replacement for Isaac Lab.
