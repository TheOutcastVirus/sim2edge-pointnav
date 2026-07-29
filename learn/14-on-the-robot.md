# 14 — On the robot

The last hop. A `.pte` on a board, ROS 2 topics, and a physical robot that can hurt
itself and other things.

Two nodes, deliberately separate:

| Node | Lines | Job |
|---|---|---|
| `ros_ws/src/nav_policy_node/` | ~950 C++ | subscribe, preprocess, infer, publish a *proposed* command |
| `ros_ws/src/nav_safety/` | 284 C++ | own `/cmd_vel`, veto anything unsafe |

## The architecture decision that matters most

```
/scan ──┐
/odom ──┼──> nav_policy_node ──> /nav_policy/cmd_raw ──> nav_safety ──> /cmd_vel
/goal ──┘         (NPU)                                  (classical)
```

The policy node **does not publish to `/cmd_vel`**. `nav_policy_node.hpp:32-34`:

> This node does NOT publish to /cmd_vel. It publishes its raw policy output and
> lets nav_safety own the motors, so the safety layer keeps running (and keeps
> stopping the robot) even if this node crashes or stalls.

And from `nav_safety_node.cpp:3-11`:

> This is a SEPARATE NODE from nav_policy_node on purpose. If the safety logic
> lived inside the inference node, then a crash, a stall, or an ExecuTorch
> deadlock in that node would take the watchdog down with it and leave the robot
> driving on its last command. Here, the policy node going silent is precisely the
> condition this node is built to detect, and it responds by publishing a zero
> Twist.
>
> Everything in this file is classical and deterministic. Nothing learned, and
> nothing quantized, has stopping authority.

That last sentence is the design principle of the whole deployment. You have a
network that was trained in a simulator, quantized to 8 bits, and is running on an
accelerator through a partially-patched vendor SDK. It is not the thing that
should decide whether to stop.

Process isolation makes "the policy hung" a *detectable* condition rather than an
undefined one. Same-process safety code shares the fate of the code it guards.

## Sensor handling: latest-message, not synchronized

`nav_policy_node.hpp:24-30`:

> each subscription caches only the latest message (queue depth 1), and a
> decoupled control timer takes whatever is current and rejects anything older
> than a per-sensor `max_age`. An approximate-time synchronizer would stall the
> control loop whenever one sensor hiccuped, which for a moving robot is strictly
> worse than acting on a slightly stale scan — and worse still than deliberately
> stopping.

The ROS-idiomatic choice is `message_filters::ApproximateTime`, which waits for
matching timestamps across topics. That is right for offline processing and wrong
for a control loop: if the depth camera drops a frame, the synchronizer holds the
lidar message waiting for a partner that never comes, and your 20 Hz loop stops
producing. The robot keeps rolling on its last command.

Three-tier response instead:

1. Fresh data → run the policy.
2. Slightly stale → still run; a 60 ms-old scan on a robot moving 0.3 m/s is 2 cm
   of error.
3. Too stale (`lidar_max_age_s = 0.15`, `odom_max_age_s = 0.20`) → **stop
   deliberately.**

Deliberate stopping is a decision. Waiting is an absence of one.

Also note the control timer is decoupled from the callbacks — inference runs on a
timer at 20 Hz, not inside the subscriber callback. `REBUILD_PLAN.md:243`
recommends the same. Inference inside a callback couples your control rate to your
sensor's publishing jitter.

## The safety layer

`sim2edge/safety.py` is the Python twin; `nav_safety_node.cpp` is what runs.
Constants come from `spec_generated.hpp`, generated from `spec.py`, "so that the
simulator's SafetyLayer and this node apply the same thresholds, which is what
makes intervention counts comparable between sim and hardware."

`SafetySpec` (`spec.py:405`):

| Constant | Value |
|---|---|
| `forward_sector_rad` | 0.61 (±35°) |
| `stop_distance_m` | 0.30 |
| `slow_distance_m` | 0.60 |
| `policy_timeout_s` | 0.5 |
| `lidar_max_age_s` | 0.15 |
| `odom_max_age_s` | 0.20 |

`SafetyLayer.filter` (`safety.py:70`) in order:

```python
# 1. hard clamps to the base's limits, first
v = float(np.clip(v, a.v_min, a.v_max))
omega = float(np.clip(omega, -a.omega_max, a.omega_max))

# 2. sensor loss -> full stop
if not sensor_ok:
    v, omega = 0.0, 0.0
elif ranges is not None:
    d = forward_min_range(ranges, angle_min, angle_increment, s.forward_sector_rad)
    if not math.isfinite(d):
        v = 0.0                      # no valid forward returns: do not drive blind
    elif d < s.stop_distance_m:
        v = 0.0                      # rotation deliberately still allowed
    elif d < s.slow_distance_m:
        span = max(1e-6, s.slow_distance_m - s.stop_distance_m)
        v *= (d - s.stop_distance_m) / span

# 3. rate limiting LAST
```

### The asymmetry

`safety.py:16-19`:

> Note the asymmetry: the bubble zeroes *linear* velocity but never blocks
> rotation. A robot that has stopped 25 cm from a wall must still be able to turn
> away from it, and a safety layer that freezes both axes turns a recoverable
> situation into a deadlock.

Stop translating, keep rotating. Freeze both and the robot is stuck 25 cm from a
wall forever, needing a human. The one exception is sensor loss (`:95-97`): without
sensors there is no basis for choosing a direction to turn, so rotation stops too.

### Ordering

Rate limiting comes **last** (`:110-111`): "so the bubble's decision is what gets
ramped rather than being ramped away." Rate-limit first and the bubble's `v = 0`
becomes `v = previous − 0.03`, which is not a stop.

And `forward_min_range` (`:32`) returns `inf` when the forward sector has no valid
return at all — "the caller must treat that as sensor loss rather than as clear
space." An empty scan is not an empty room.

### See it

```bash
python -c "
import numpy as np, math
from sim2edge.safety import SafetyLayer, forward_min_range
from sim2edge.spec import DEFAULT_SPEC as S

inc = 2*math.pi/360
blocked = np.full(360, 5.0, np.float32); blocked[178:183] = 0.25
clear   = np.full(360, 5.0, np.float32)
print('forward min (blocked):', forward_min_range(blocked, -math.pi, inc, S.safety.forward_sector_rad))

s = SafetyLayer(S)
print('obstacle at 0.25 m, commanding (0.30, 0.50):')
for i in range(3):
    print('   tick', i, s.filter(0.30, 0.50, blocked, -math.pi, inc))
s.reset()
print('clear path, commanding (0.30, 0.50):')
for i in range(3):
    print('   tick', i, s.filter(0.30, 0.50, clear, -math.pi, inc))
"
```

```
forward min (blocked): 0.25
obstacle at 0.25 m, commanding (0.30, 0.50):
   tick 0 (0.0, 0.15000000000000002, True)
   tick 1 (0.0, 0.30000000000000004, True)
   tick 2 (0.0, 0.45000000000000007, True)
clear path, commanding (0.30, 0.50):
   tick 0 (0.03, 0.15000000000000002, True)
   tick 1 (0.06, 0.30000000000000004, True)
   tick 2 (0.09, 0.45000000000000007, True)
```

Read carefully:

- **Blocked:** `v = 0` immediately (obstacle inside 0.30 m), but `ω` still ramps
  0.15 → 0.30 → 0.45. The robot can turn away. Exactly the asymmetry.
- **Clear:** `v` ramps 0.03 per tick — `accel_max × dt = 0.6 × 0.05`. Even with a
  clear path it takes 10 ticks (half a second) to reach 0.30 m/s.
- **`intervened = True` in both cases**, including the clear one, because rate
  limiting changed the command. Worth knowing when you read intervention counts:
  they include ramping, not just emergencies.

## The parity gate

The C++ preprocessing must match the Python that produced the calibration set.
`tools/parity_check.py` (316 lines) enforces it:

```
make_samples(n, spec)        :108  — including NaN depth, all-invalid scans,
                                     out-of-range values
     │
     ├──> python_reference()  :165 — sim2edge/preprocess.py
     └──> write_inputs() :59 -> preprocess_parity (C++) -> read_outputs() :76
                │
          check_preprocessing() :184 — require agreement to 1e-5
```

The adversarial inputs are the point. Any two implementations agree on a clean
4 m scan. They disagree on `NaN`, on `inf`, on a scan where every return is
invalid, on a value exactly at `min_m`.

```bash
ssh vkommera@dronelab12-4
cd ~/Documents/Qualcomm/sim2edge-ventuno
.venv/bin/python -m pytest tests/test_cpp_parity.py -q
```

The README reports current agreement is **bit-exact** on both depth and lidar.

## The contract check at load

`verifyModelContract` (`nav_policy_node.hpp:50`) reads the sidecar JSON next to the
`.pte` and compares its fingerprint against the one compiled into the binary from
`spec_generated.hpp`. Mismatch → refuse to load.

This is the third and final enforcement point of the chapter-02 contract:

1. training checkpoints carry the fingerprint (`bc.py:228`)
2. exports write it to the sidecar (`export_common.py:160`)
3. the node verifies it at load

## Bring-up order

`docs/deployment.md` has the full procedure. The order that matters:

1. **`.pte` loads and the fingerprint matches** — before anything moves.
2. **`nav_safety` alone, robot on blocks.** Publish commands to
   `/nav_policy/cmd_raw` by hand; confirm `/cmd_vel` is vetoed when you put your
   hand in front of the lidar. **Verify the watchdog before trusting the policy.**
3. **Policy node, wheels off the ground.** Watch `/nav_policy/cmd_raw` for sane
   values and check the 20 Hz timer holds.
4. **On the floor, short goal, hand near the e-stop.**

`REBUILD_PLAN.md:253-254` states the milestone well:

> robot drives to a goal 3 m away with a box in the path, and e-stops when you
> step in front of it.

Both halves. A robot that navigates but does not stop is not a success.

## The rclpy argument

`REBUILD_PLAN.md:28` proposes deleting all 1,617 lines of C++:

> Write the robot node in **rclpy** and import the *same Python function* the sim
> calls. The class of bug disappears instead of being tested for. A 4-layer CNN at
> 20 Hz does not need C++.

The chain that disappears: `preprocess.cpp` (188), `spec_generated.hpp` (70),
`gen_spec_header.py` (147), `parity_check.py` (316), `test_cpp_parity.py` (107) —
roughly 830 lines whose only job is keeping two implementations in agreement. Import
one function and there is nothing to keep in agreement.

The performance objection is weak. Note also that the **~5 Hz NPU inference figure
quoted in the sibling repos is wrong** — the NPU achieves considerably more than
that, so it cannot justify a C++ rewrite either.

What you would keep either way: the two-node split, the safety layer, the sidecar
check. `REBUILD_PLAN.md:239-248` keeps all three, in `rclpy`, in about 150 lines.

## Status

- ROS 2 node + safety layer: **complete, but compile-tested only on the parity
  path.** Never built against ExecuTorch, never run on the board.
- No `.pte` from this repository has executed on a Hexagon NPU.
- The safety layer's Python twin is tested; the C++ is not.

## Check yourself

- Why does the safety node run in a separate process rather than a separate thread?
- Why does the bubble stop translation but not rotation — and why is sensor loss
  the exception?
- Why is rate limiting applied after the bubble instead of before?
- `forward_min_range` returns `inf` for an empty forward sector. What would go
  wrong if it returned `max_range`?
- The parity test uses NaN and all-invalid scans. Why are clean inputs nearly
  worthless for this test?

→ [15 — Where it breaks](15-where-it-breaks.md)
