# 06 — The environment

Everything so far has been static: a contract, a preprocessing function, a room, a
planner. This chapter puts them in a loop.

`sim2edge/envs/reference_env.py` (368 lines) is a pure-numpy navigation
environment. It runs on a laptop with no GPU and no simulator install, which is
why the whole pipeline — oracle, BC, PPO, eval, export, parity — is testable
without a 40 GB dependency. It is *not* the environment the deployed policy trains
in; Isaac Lab is (chapter 12).

## The loop

```python
env = NavEnv(EnvConfig(use_depth=False), seed=0)
obs = env.reset(seed=0)
while True:
    action = policy(obs)              # (v, omega), physical units
    result = env.step(action)
    obs = result.obs
    if result.terminated or result.truncated:
        break
```

`StepResult` (`:74`) is a dataclass of `obs, reward, terminated, truncated, info`
— the Gymnasium five-tuple, without the Gymnasium dependency.

The **terminated vs truncated** distinction matters for RL and is not cosmetic:

- `terminated` — the episode genuinely ended (reached the goal, or crashed). The
  value of the next state is zero.
- `truncated` — we ran out of time at 1200 steps. The state still has value; the
  clock just expired.

Bootstrapping through a truncation but not through a termination is what keeps the
value function honest. Chapter 10 shows where PPO uses it.

## `reset` — building an episode

`reset` at `:112`:

```python
self.room = room if room is not None else sample_room(self.rng, self.spec)
self.dr = sample_randomization(self.rng, self.spec, self.config.randomization)
self.x, self.y, self.yaw = self.room.start
...
self._action_queue = [(0.0, 0.0)] * self.dr.action_delay_steps
self._prev_geodesic = self._geodesic_now()
```

Three things happen: a new room, a new set of randomization parameters
(chapter 07), and a pre-filled action-delay queue. That last line has a comment
worth quoting (`:133-134`):

> Pre-fill the delay queue with zero commands: the real robot is stationary and
> has issued nothing when the policy's first tick lands.

Getting this wrong — filling the queue with the first action, say — would give the
policy a free, latency-less first step that the real robot never gets.

## `step` — order of operations

`step` at `:143` does five things, and the order is the design:

```
1. clamp the action to the base's limits          :145-148
2. optionally apply the safety layer              :151-155
3. push through the latency queue                 :158-160
4. integrate the dynamics                         :162
5. record last_action, compute reward             :167-171
```

The subtlety is at step 5 (`:164-166`):

> `last_action` is what the *policy* commanded, not what the safety layer or the
> motor lag produced: it is the policy's own previous output, and the smoothness
> penalty is defined on the policy's action sequence.

If you stored the post-safety, post-lag velocity instead, the smoothness penalty
would punish the policy for the safety layer's interventions and for the motor's
sluggishness — things it does not control. The policy would learn to compensate
for its own governor.

### Safety off by default

`EnvConfig.apply_safety = False` (`:65`), with the reason at `:62-64`:

> Off during training (a policy trained behind a safety net learns to rely on it);
> on for deployment-parity evaluation, where intervention count is a metric.

This is a real phenomenon. Train with a bubble that stops you at 0.3 m and the
policy discovers it can drive straight at walls for free. Then you deploy with a
slightly different bubble and it drives into one.

## The dynamics

`_integrate` at `:173` is a unicycle model with first-order motor lag:

```python
tau = max(1e-3, dr.motor_tau_s * dr.mass_scale)
alpha = 1.0 - math.exp(-dt / tau)
self.v += (v_cmd * dr.velocity_gain - self.v) * alpha
self.omega += (w_cmd * dr.velocity_gain - self.omega) * alpha
```

The robot does not instantly reach commanded velocity; it approaches it
exponentially with time constant `τ ∈ [0.05, 0.15] s`. `mass_scale` stretches
`τ` — a heavier robot responds more sluggishly.

Then wheel slip, applied asymmetrically (`:184-186`):

> Slip removes commanded translation without removing rotation, which is what a
> wheel losing traction on a smooth floor actually does.

Then the kinematics, and here is a detail worth stealing (`:189-200`):

```python
if abs(self.omega) < 1e-6:
    self.x += v_eff * math.cos(self.yaw) * dt
    self.y += v_eff * math.sin(self.yaw) * dt
else:
    # Exact arc integration
    r = v_eff / self.omega
    yaw2 = self.yaw + self.omega * dt
    self.x += r * (math.sin(yaw2) - math.sin(self.yaw))
    self.y -= r * (math.cos(yaw2) - math.cos(self.yaw))
```

Euler integration (`x += v·cos(yaw)·dt` regardless of `ω`) visibly under-turns at
`ω_max` and 20 Hz. The comment at `:193-195` names the consequence: you would
misattribute the resulting tracking error to the policy. Exact arc integration is
three extra lines and removes an entire category of phantom bug.

### Odometry drift

`:205-213` accumulates drift proportional to distance travelled and rotation
performed — the way wheel-encoder odometry actually degrades. Crucially:

> It corrupts only the goal vector the policy sees; ground truth is untouched.

So `env.x, env.y` remain exact for reward and collision purposes, while
`odom_pose()` (`:275`) returns what the robot *believes*. That split is what makes
odometry drift a learnable perception problem rather than a corrupted reward.

## The reward

`_finish_step` at `:215`. In full:

```python
reward = (rw.progress * progress          # 3.0 per metre of geodesic progress
          - rw.smoothness * smooth        # 0.05 * ||a_t - a_{t-1}||^2
          - rw.time_penalty)              # 0.005 per step
if reached:  reward += rw.goal_bonus      # +10
if collided: reward -= rw.collision_penalty  # -10
```

### Progress is geodesic, never Euclidean

`spec.py:238-240`:

> Progress uses **geodesic** distance from the oracle's distance field, never
> Euclidean: Euclidean progress pays the robot to drive into the concave side of
> an obstacle, which is exactly where local navigation policies get stuck.

Picture a U-shaped obstacle between robot and goal. Euclidean progress rewards
every metre driven into the mouth of the U, then charges the robot to back out.
Geodesic progress correctly reports that entering the U makes things worse.

Chapter 04 measured the gap on seed 0: Euclidean 5.098 m, geodesic 5.918 m.

### The unreachable guard

```python
if geodesic >= UNREACHABLE * 0.5 or self._prev_geodesic >= UNREACHABLE * 0.5:
    progress = 0.0
```

`:227-228` explains: differencing the `1e6` sentinel against a finite value would
produce a six-figure reward spike. One such transition in a batch destroys the
value function. This is the kind of guard you only write after it has bitten you.

### Progress rewards are telescoping

Worth understanding: summed over an episode, `Σ (dₜ₋₁ − dₜ)` collapses to
`d_start − d_end`. Total progress reward depends only on where you started and
finished, not the route. So progress does not bias *which* path is taken — it just
provides dense per-step feedback instead of one sparse signal at the end. The
time penalty is what makes shorter routes better.

## The observation

`_observe` at `:281` is the chapter-03 preprocessing wired to simulated sensors:

```python
ranges, angle_min, angle_inc = simulate_lidar(self.room, self.x, self.y, self.yaw, spec)
ranges = corrupt_lidar(ranges, self.dr, self.rng)
...
ox, oy, oyaw = self.odom_pose()          # drifting, not ground truth
goal_robot = world_goal_to_robot_frame(self.room.goal, (ox, oy), oyaw)
obs = self.pre.build(depth_m=depth, scan_ranges=..., goal_xy_robot=goal_robot, ...)
obs["privileged"] = self._privileged()
```

Two details:

**The angular offset is applied to the declared start angle** (`:289-291`), not to
the ranges, "exactly as a miscalibrated real lidar would report it." Simulating the
*symptom* rather than the *cause* is what makes randomization transfer.

**Lidar-only mode fills the depth image with zeros in raw metres** (`:307-308`).
Zero metres is below `min_m = 0.3`, so `normalize_depth_raw` marks it invalid and
it becomes `invalid_fill = 1.0` — "far" — after preprocessing. The tensor still
exists so the exported graph's signature never changes between configurations.
This exact line was once a bug; see chapter 15.

## Privileged observations

`_privileged` at `:330` returns 8 floats the *critic* sees and the actor never
does:

```python
[geodesic_norm, clearance_norm, x_norm, y_norm, sin(yaw), cos(yaw), v_norm, ω_norm]
```

This is **asymmetric actor-critic**. `spec.py:143-144`:

> the critic gets ground truth the actor cannot have, which lowers value-function
> variance for free because the critic never ships.

The critic's job is to predict expected return. That is much easier when you know
the true geodesic distance to the goal. A better critic means lower-variance
advantages, which means faster and more stable PPO — and none of it costs
anything at deployment, because the critic is discarded before export.

## Try it

```bash
python -c "
import numpy as np
from sim2edge.envs.reference_env import NavEnv, EnvConfig

env = NavEnv(EnvConfig(use_depth=False), seed=0)
obs = env.reset(seed=0)
print('observation:')
for k, v in obs.items():
    print(f'  {k:12s} {str(v.shape):14s} [{v.min():.3f}, {v.max():.3f}]')
r = env.step((0.2, 0.1))
print()
print('reward:', r.reward)
print('info  :', {k: v for k, v in r.info.items() if k in ('goal_distance','geodesic','steps','success','collision')})
"
```

```
observation:
  depth        (1, 96, 160)   [1.000, 1.000]
  lidar        (180,)         [0.040, 0.819]
  goal         (3,)           [-0.055, 0.999]
  last_action  (2,)           [-1.000, 0.000]
  privileged   (8,)           [0.000, 0.882]

reward: -0.09411111111111115
```

Three things to read:

- **Depth is all 1.0** — lidar-only mode, the "no return" fill, exactly as
  described above.
- **`last_action` starts at `[-1, 0]`**, not `[0, 0]`. Recall from chapter 02 that
  `v = 0` normalizes to `-1`. A zero-initialized `last_action` would tell the
  policy the robot was previously driving at 0.15 m/s.
- **The first reward is negative.** The robot barely moved (motor lag plus a
  pre-filled delay queue), so progress ≈ 0, while the time penalty and the
  smoothness cost of going from `(0,0)` to `(0.2, 0.1)` both bite. Early negative
  reward is normal and not a bug.

### Run a whole episode with the oracle

```bash
python -c "
import numpy as np
from sim2edge.envs.reference_env import NavEnv, EnvConfig
from sim2edge.oracle.expert import PurePursuitExpert

env = NavEnv(EnvConfig(use_depth=False), seed=0)
obs = env.reset(seed=0)
ex = PurePursuitExpert(env.room)
total = 0.0
for t in range(1200):
    cmd = ex.act(*env.pose)
    r = env.step((cmd.v, cmd.omega))
    total += r.reward
    if r.terminated or r.truncated:
        print(f'ended at step {t+1}: success={r.info[\"success\"]} collision={r.info[\"collision\"]}')
        print(f'return {total:.2f}, path {r.info[\"path_length\"]:.2f} m, optimal {r.info[\"optimal_path_length\"]:.2f} m')
        break
"
```

```
ended at step 626: success=True collision=False
return 22.59, path 5.47 m, optimal 5.92 m
```

Note `ex.act(*env.pose)` — the expert reads **ground-truth** pose, not the
observation. That is what makes it an oracle.

Also note `path 5.47 < optimal 5.92`. That is not a contradiction: `optimal` is the
geodesic distance measured over the grid at 5 cm resolution, and the robot drives a
smoothed continuous path that cuts inside the grid's octile staircase. It is why
`_spl` (`reference_env.py:364`) wraps the ratio in `min(1.0, ...)` — otherwise the
oracle would score SPL above 1.

Episode return 22.59 decomposes roughly as: `3.0 × 5.92` of geodesic progress
(≈17.8), plus the `+10` goal bonus, minus `626 × 0.005` of time penalty (≈3.1),
minus smoothness. That accounting is worth doing by hand once — it is the fastest
way to confirm you have read the reward correctly.

### Feel the randomization

```bash
python -c "
from sim2edge.envs.reference_env import NavEnv, EnvConfig
from sim2edge.spec import DEFAULT_SPEC as S

for scale in (0.0, 1.0, 2.0):
    cfg = EnvConfig(use_depth=False, randomization=S.randomization.at_scale(scale))
    env = NavEnv(cfg, seed=0); env.reset(seed=0)
    d = env.dr
    print(f'scale {scale}: delay={d.action_delay_steps} steps, lidar_sigma={d.lidar_sigma_m:.4f} m, gain={d.velocity_gain:.3f}')
"
```

```
scale 0.0: delay=0 steps, lidar_sigma=0.0000 m, gain=1.000
scale 1.0: delay=1 steps, lidar_sigma=0.0060 m, gain=0.985
scale 2.0: delay=1 steps, lidar_sigma=0.0120 m, gain=0.970
```

Same seed, same room, three different robots. Individual draws are noisy — `delay`
happened to land on 1 at both the 1× and 2× tiers here — so read the trend
(`lidar_sigma` doubling, `gain` drifting further from 1.0), not any single value.

One thing that *is* deterministic: at scale 0 the delay is **0**, not the `(2, 2)`
range chapter 02 showed. `at_scale(0)` sets `enabled=False` (`spec.py:320`), and
`sample_randomization` then returns `EpisodeRandomization.none()` — a perfect robot
with zero latency, bypassing the ranges entirely. The clean tier is genuinely
clean.

This is the mechanism behind the noise-tier evaluation in chapter 11.

## What this env is not

`reference_env.py:13-16`:

> It is deliberately **not** the training environment for the deployed policy —
> Isaac Lab is. Physics here is a kinematic differential-drive model with
> first-order motor lag, which is a good match for a Create3 on flat floor but is
> not a rigid-body simulation.

No contacts, no friction cones, no rigid-body solver. A robot that "collides" here
simply has its episode terminated; it does not get pushed. For a wheeled robot on
a flat floor commanding body velocity, that is a defensible approximation — but it
is an approximation, and it is why the deployed policy is supposed to train in
Isaac Lab.

## Check yourself

- Why is `last_action` the pre-safety, pre-lag command?
- The progress reward telescopes. What term actually makes the policy prefer a
  short path over a long one?
- What would go wrong if `_privileged` were fed to the actor?
- The delay queue is pre-filled with zeros. What does the policy learn if you
  pre-fill it with the first commanded action instead?

→ [07 — Sensors and randomization](07-sensors-and-randomization.md)
