# 07 — Sensors and randomization

Two files, one idea: **the simulator's job is not to be realistic, it is to be
realistic in the ways that matter and randomized in the ways that don't.**

- `sim2edge/sensors.py` (221 lines) — simulated lidar and depth
- `sim2edge/randomization.py` (181 lines) — per-episode perturbations

## Part 1 — Sensors

### The ordering rule

`sensors.py:3-9` states the design constraint that makes the rest of the project
safe:

> Both sensors produce **raw** output in the same units and conventions the real
> drivers publish — metres, `LaserScan`-style range arrays, forward-distance
> depth images — and are then fed through the *same* `sim2edge.preprocess`
> functions the C++ node uses. Nothing here produces a policy input directly.
> That ordering is deliberate: it means the preprocessing code path is exercised
> by every single training step, so a preprocessing bug shows up as a training
> failure rather than as a surprise on the robot.

A simulator that emitted 180 normalized bins directly would be faster and would
bypass the exact code you most need tested. Instead it emits 360 raw ranges and
lets `preprocess_lidar` do its min-pooling — millions of times per training run.

### Ray marching

`march_rays` at `sensors.py:35` casts a fan of rays against an occupancy grid by
stepping along each ray in fixed increments and looking up cells:

```python
step = resolution * step_scale          # 0.05 * 0.5 = 2.5 cm
dists = np.arange(n_steps) * step       # (S,)
xs = origin[0] + cos_a * dists          # (N, S)
ys = origin[1] + sin_a * dists
cols = np.clip((xs / resolution).astype(np.int64), 0, w - 1)
rows = np.clip((ys / resolution).astype(np.int64), 0, h - 1)
```

Every ray, every step, in two array operations. The docstring at `:45-49`
justifies the approximation:

> Marching uses a fixed step of `step_scale` cells rather than an exact DDA: it
> vectorizes cleanly over all rays at once, and the resulting range error is
> bounded by half a cell (2.5 cm at the default resolution), which is smaller
> than the lidar noise the domain randomization injects anyway.

That is the right way to reason about simulator fidelity: an error smaller than
the noise you are deliberately adding is an error that cannot matter.

`_first_hit` at `:70` collapses `(n_rays, n_steps)` hit masks to a range per ray
using `argmax` on a boolean array — `argmax` returns the first `True`, and the
`any_hit` mask handles the no-hit case.

### The lidar

`simulate_lidar` at `:85` returns `(ranges, angle_min, angle_increment)` — the
three `LaserScan` fields preprocessing consumes, "so sim and robot enter
preprocessing through an identical interface."

Two lines carry all the chapter-04 height logic:

```python
occ = room.lidar_occ.occupied     # only obstacles crossing the scan plane
...
ranges = np.where(ranges < spec.obs.lidar.min_m, np.float32(np.inf), ranges)
```

The first: a box 10 cm tall is *simply not there*, exactly as on the real robot
(`:99-101`). This is how the simulator represents the failure mode the depth
camera is supposed to fix.

The second: returns closer than the device minimum are reported as `inf`, which is
what the driver does. Preprocessing then maps them to "far", and — per `:113-115`
— the safety bubble, not the policy, is responsible for that regime.

`LIDAR_BEAMS = 360` (`:32`) matches the real RPLIDAR A1 rather than the policy's
180 bins, so "the min-pooling in preprocessing does the same work in sim as on the
robot."

### The depth camera

`render_depth` at `:140` is a 2.5D renderer: obstacles are extruded prisms, and a
ray hits one only where it passes through that prism's vertical extent. That is
what lets the simulator represent overhangs and low obstacles.

`_pinhole_angles` at `:125` has a note worth internalizing (`:128-131`):

> Angles are `atan` of a linear image coordinate, not linear in the pixel index.
> Using linear angles instead would distort the geometry toward the image edges by
> several degrees at a 69-degree FOV.

A pinhole camera's pixels are uniform on the *image plane*, not uniform in angle.
Getting this backwards is a classic bug that produces a systematically warped
world at the frame edges — precisely where obstacles first appear.

### Try it

```bash
python -c "
import numpy as np, math
from sim2edge.rooms import sample_room
from sim2edge.sensors import simulate_lidar
from sim2edge.spec import DEFAULT_SPEC as S

room = sample_room(np.random.default_rng(0), S)
x, y, yaw = room.start
ranges, amin, ainc = simulate_lidar(room, x, y, yaw, S)
print('beams:', ranges.shape, ' angle_min: %.3f' % amin, ' inc: %.5f' % ainc)
finite = np.isfinite(ranges)
print('valid: %d/%d   min %.2f m   max %.2f m' % (finite.sum(), len(ranges), ranges[finite].min(), ranges[finite].max()))
# beam 180 is angle 0 in the robot frame = straight ahead
print('straight ahead: %.2f m' % ranges[180])
"
```

```
beams: (360,)  angle_min: -3.142  inc: 0.01745
valid: 360/360   min 0.47 m   max 6.62 m
straight ahead: 1.38 m
```

Then rotate the robot and watch the scan rotate with it:

```bash
python -c "
import numpy as np, math
from sim2edge.rooms import sample_room
from sim2edge.sensors import simulate_lidar
from sim2edge.spec import DEFAULT_SPEC as S
room = sample_room(np.random.default_rng(0), S)
x, y, _ = room.start
for yaw in (0.0, math.pi/2, math.pi):
    r, _, _ = simulate_lidar(room, x, y, yaw, S)
    print(f'yaw={yaw:.2f}  forward={r[180]:.2f} m  left={r[270]:.2f} m  right={r[90]:.2f} m')
"
```

```
yaw=0.00  forward=6.45 m  left=1.75 m  right=0.47 m
yaw=1.57  forward=1.75 m  left=1.18 m  right=6.45 m
yaw=3.14  forward=1.18 m  left=0.47 m  right=1.75 m
```

Read the columns, not the rows: the 6.45 m opening moves forward → right → (behind)
as the robot turns left by 90° each time, and the 1.75 m reading moves left →
forward → right. The scan is rigidly attached to the robot frame, exactly as it
should be. If you ever see these columns *not* permute consistently when you rotate,
you have a frame bug — and it is much cheaper to find here than in a training curve.

## Part 2 — Domain randomization

### Why it exists

A policy trained in a perfect simulator learns to exploit the simulator. It will
assume zero latency, exact velocity tracking, and noiseless range readings —
because in the simulator, those are true. On the robot, none of them are, and the
policy fails in ways that look like it never learned anything.

Domain randomization fixes this by making the *simulator itself* a distribution.
If latency varies between 1 and 3 steps every episode, the policy cannot learn a
strategy that depends on latency being exactly 1. It has to learn something robust.

### Two rules, both structural

**Rule 1: sample per episode, hold fixed within it.** `randomization.py:3-6`:

> Sampled once per episode and held fixed for its duration, which is what makes it
> learnable: a policy can adapt to a robot that is consistently sluggish this
> episode, but not to one whose latency changes every step.

This is the difference between *system identification* — which a policy can do
implicitly from a few steps of history — and unlearnable noise.

**Rule 2: perturb raw sensor output, before preprocessing.** `randomization.py:8-12`:

> Injecting it after preprocessing would be a different — and wrong —
> distribution: the min-pooling in preprocessing interacts strongly with per-beam
> noise and dropout, biasing pooled values downward, and that interaction is
> exactly what the real robot will do too.

Concretely: min-pooling 2 noisy beams takes the *minimum* of two noisy values,
which is biased low. That bias is real on the robot. If you add noise after
pooling, you get an unbiased perturbation the robot will never produce.

### What gets randomized

`EpisodeRandomization` at `randomization.py:31`, sampled by `sample_randomization`
at `:75` from the ranges in `RandomizationSpec` (`spec.py:264`):

| Group | Parameters | Range |
|---|---|---|
| Lidar | per-beam σ, dropout, angular offset | 0–3 cm, 0–5%, ±1° |
| Depth | noise `σ = c·d²`, holes, speckle, scale, pitch, height | 0–1%·d², 0–12%, 0–2%, ±3%, ±2° |
| Dynamics | velocity gain, motor τ, wheel slip, mass scale | 0.9–1.1, 0.05–0.15 s, 0–5%, 0.8–1.2 |
| Latency | action delay | **1–3 control steps** |
| Odometry | drift rate, yaw drift, noise | 0–2%/m, 0–2%/rad, 0–2 cm |

Two notes on the modelling:

- **Depth noise grows as `d²`** (`spec.py:281-282`). That is the physics of stereo
  disparity: error is inversely proportional to disparity, which is inversely
  proportional to distance. A constant-σ depth noise model is wrong in a way that
  matters at the far end of the range.
- **Action delay is called out as the most important one** (`spec.py:298-300`):

  > The single most important randomization for sim-to-real on a real ROS stack:
  > sensor -> cmd_vel latency on the robot is never less than one control period.

  `REBUILD_PLAN.md:185-187` independently reaches the same conclusion. If you
  randomize one thing, randomize latency.

### The corruption functions

`corrupt_lidar` at `:115` and `corrupt_depth` at `:138` apply the sampled
parameters to raw sensor arrays. Dropped beams become invalid — the *same* invalid
the real driver emits — so preprocessing maps them to `1.0` by the same path.
There is no separate "this is a simulated dropout" code path, which is exactly
right.

### Try it

```bash
python -c "
import numpy as np
from sim2edge.randomization import sample_randomization, EpisodeRandomization
from sim2edge.spec import DEFAULT_SPEC as S

print('none():', EpisodeRandomization.none())
print()
for seed in (1, 2):
    dr = sample_randomization(np.random.default_rng(seed), S)
    print(f'seed {seed}: delay={dr.action_delay_steps}  gain={dr.velocity_gain:.3f}  '
          f'tau={dr.motor_tau_s:.3f}  lidar_sigma={dr.lidar_sigma_m:.4f}  '
          f'dropout={dr.lidar_dropout:.3f}  odom_drift={dr.odom_drift_rate:.4f}')
"
```

Real output for seed 1:

```
delay=2  gain=1.051  tau=0.104  lidar_sigma=0.0285  dropout=0.007  odom_drift=0.0061
```

That is one episode's robot: 100 ms of control latency, tracking 5% fast, 2.9 cm
of range noise, drifting 6 mm per metre driven. The next episode is a different
robot.

### See noise degrade the lidar

```bash
python -c "
import numpy as np, math
from sim2edge.rooms import sample_room
from sim2edge.sensors import simulate_lidar
from sim2edge.randomization import sample_randomization, corrupt_lidar
from sim2edge.preprocess import preprocess_lidar
from sim2edge.spec import DEFAULT_SPEC as S

room = sample_room(np.random.default_rng(0), S)
rng = np.random.default_rng(5)
clean_raw, amin, ainc = simulate_lidar(room, *room.start, S)
clean = preprocess_lidar(clean_raw, amin, ainc, S)
for scale in (0.0, 1.0, 2.0):
    dr = sample_randomization(rng, S, S.randomization.at_scale(scale))
    noisy_raw = corrupt_lidar(clean_raw.copy(), dr, rng)
    noisy = preprocess_lidar(noisy_raw, amin + dr.lidar_angle_offset, ainc, S)
    print(f'scale {scale}: mean |delta| = {np.abs(noisy-clean).mean():.5f}   '
          f'max |delta| = {np.abs(noisy-clean).max():.5f}')
"
```

Note that the angular offset is applied to `angle_min`, not to the ranges —
matching what a miscalibrated mount actually reports (chapter 06, `reference_env.py:289-291`).

## What is deliberately *not* randomized

No textures, no lighting, no visual appearance. That is a consequence of using
**depth**, not RGB: geometry transfers from sim to real for free, textures and
lighting do not. `spec.py:49-51` makes the argument.

`REBUILD_PLAN.md:200-201` adds one more cut worth knowing:

> Mass/friction randomization is pointless when you command body velocity
> directly — skip it.

That is a fair hit on `mass_scale` and `friction` here. In this env `mass_scale`
only stretches the motor time constant (`reference_env.py:179`), which
`motor_tau_s` already randomizes directly. Two knobs, one effect.

## Measure, don't guess

`randomization.py:14-15` and `IMPLEMENTATION_PLAN.md:121`:

> **Measure, don't guess:** on Day 1–2, record real lidar + depth + odom from the
> robot (rosbag), and set the noise ranges to bracket what you actually observe.

`tools/bag_to_obs.py` (204 lines) exists to extract real sensor statistics from a
rosbag. The ranges currently in `spec.py` are plausible defaults, not measured
ones — treat them as placeholders until a bag says otherwise. See
`docs/domain_randomization.md`.

## Check yourself

- Why must lidar noise be applied before min-pooling rather than after?
- Why does depth noise scale as `d²` while lidar noise is constant?
- The angular offset is added to `angle_min` rather than rotating the range array.
  What does that simulate, and why is it more faithful?
- Randomization is fixed within an episode. What could a policy learn if you
  resampled every step? What could it learn if you never randomized at all?

→ [08 — The policy network](08-the-policy-network.md)
