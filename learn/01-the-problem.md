# 01 — The problem

## The task

> Drive to a coordinate specified relative to where you started, without hitting
> anything, using only your own sensors.

That is *point-goal navigation*. It sounds trivial and is not, for one reason:
the robot does not have a map. It knows where the goal is relative to itself
(from odometry, which drifts), and it can see roughly 8 metres of obstacle
geometry around itself (from lidar). It cannot see around a corner. It must
commit to a direction using local information and hope that direction is not a
dead end.

## Why this is a learning problem at all

You could solve navigation classically: build an occupancy grid from the lidar,
run A* on it, follow the path with a pure-pursuit controller. **This repository
contains exactly that** — `sim2edge/oracle/` — and it achieves a 100% success
rate.

So why train a network?

Because the classical solver is *given the map*. Read
`sim2edge/oracle/expert.py:42`:

```python
def plan_path(room: Room, start_xy=None, ...)
```

It takes a `Room` — the ground-truth layout, including obstacles the lidar cannot
see. That is why it is called an **oracle**, not a controller. It cheats. It
exists to generate training data and to be an upper bound, not to be deployed.

The policy gets only what the robot gets. The interesting claim of the project is
that a small network, given 180 lidar bins and a goal vector, can approach what
the oracle achieves with full knowledge — and can then run on a 5-watt NPU at
20 Hz.

## The formal setup

Navigation is framed as a Markov Decision Process:

| MDP element | Here | Defined in |
|---|---|---|
| **State** | robot pose, velocity, room layout | `NavEnv` internals, `envs/reference_env.py:96-108` |
| **Observation** | depth + lidar + goal + last action | `ObservationSpec`, `spec.py:160` |
| **Action** | `(v, ω)` — linear and angular velocity | `ActionSpec`, `spec.py:188` |
| **Reward** | progress, goal bonus, collision penalty, … | `RewardSpec`, `spec.py:234` |
| **Transition** | unicycle kinematics with motor lag | `_integrate`, `envs/reference_env.py:173` |
| **Episode end** | reached, collided, or 1200 steps | `_finish_step`, `envs/reference_env.py:248-250` |

The observation is *not* the state — the policy cannot see the room layout, only
what the sensors return. This makes it formally a POMDP, and it is the source of
every hard failure mode in the project: the policy can be in two very different
situations that look identical through a 180-bin lidar scan.

### Why `(v, ω)` and not wheel speeds

`spec.py:188-206` sets `v_max = 0.30 m/s`, `omega_max = 1.50 rad/s`, control at
20 Hz. These are the Create3's limits minus a margin. The robot's firmware
accepts a `geometry_msgs/Twist` on `/cmd_vel` and closes its own wheel loop, so
commanding body velocity *is* the deployed interface. Nothing in the policy ever
needs to know about wheels.

The docstring makes the important point:

> if the expert demonstrates commands the robot cannot track, behaviour cloning
> faithfully learns to issue commands the robot cannot track.

Note `v_min = 0.0` — no reverse. The policy turns instead. This halves the action
space the network has to explore and matches how the lidar is mounted (360°, so
turning in place never blinds it).

## The shape of the whole pipeline

Read this diagram twice; every chapter after this one is a box in it.

```
     ┌───────────────────────────────────────────┐
     │  sim2edge/spec.py — the contract          │   every constant, one file
     └───────────────────────────────────────────┘
            │                          │
   ┌────────▼─────────┐      ┌─────────▼──────────┐
   │  SIMULATION      │      │  DEPLOYMENT        │
   │                  │      │                    │
   │  rooms.py        │      │  ROS 2 node        │
   │  sensors.py      │      │  /scan, /odom      │
   │  reference_env   │      │       │            │
   │  isaaclab_env    │      │       ▼            │
   │       │          │      │  preprocess (C++)  │
   │       ▼          │      │       │            │
   │  preprocess.py   │      │       ▼            │
   │       │          │      │  policy.pte (NPU)  │
   └───────┼──────────┘      │       │            │
           │                 │       ▼            │
   ┌───────▼──────────┐      │  safety layer      │
   │  TRAINING        │      │       │            │
   │  oracle -> BC    │      │       ▼            │
   │  DAgger          │      │  /cmd_vel          │
   │  PPO             │      └────────────────────┘
   │       │          │              ▲
   │       ▼          │              │
   │  nav_policy.py   │──────────────┘
   │  (torch model)   │   export + INT8 quantize
   └──────────────────┘
```

Three things are worth noticing now:

1. **Everything hangs off one contract file.** The simulator, the training code,
   the quantizer, and the C++ node all read their constants from `spec.py`. This
   is the single most important structural decision in the repo, and chapter 02
   is entirely about it.

2. **Preprocessing appears twice** — once in Python for the simulator, once in
   C++ for the robot. Keeping those two in sync is the project's stated #1 risk.
   Chapter 14 covers the machinery built to enforce it.

3. **Training is two stages.** An oracle generates demonstrations, a network
   clones them (chapter 09), and then PPO improves on them (chapter 10). Both
   stages exist because each alone is fragile in a different way.

## The deployment constraint that shapes everything

The policy must run on a Hexagon HTP V75 at 8-bit integer precision, in under
50 ms, at 20 Hz. This is not a detail bolted on at the end — it reaches back and
constrains the model architecture, the observation encoding, and even the reward.

Concrete consequences you will meet later:

| Constraint | Consequence | Chapter |
|---|---|---|
| INT8 quantization is sensitive to discontinuities | goal bearing encoded as `(sin, cos)`, never a raw angle | 03 |
| Only some ops delegate to HTP | no adaptive pooling, no recurrence, no dynamic shapes | 08 |
| Quantization calibration must see the deployment distribution | calibration data comes from the *same* preprocessing function the robot runs | 13 |
| A quantized net can be confidently wrong | a classical safety layer holds final authority over `/cmd_vel` | 14 |

`sim2edge/models/nav_policy.py:11-15` states the design rule directly:

> Designed backward from the NPU. Every operator below is one the reference
> models already proved fully delegated on HTP V75 at 8a8w.

## Try it

Confirm the task constants are what this chapter claims. From the repo root:

```bash
python -c "
from sim2edge.spec import DEFAULT_SPEC as S
a, r = S.action, S.reward
print(f'v:     {a.v_min} .. {a.v_max} m/s')
print(f'omega: +/-{a.omega_max} rad/s')
print(f'rate:  {a.control_rate_hz} Hz  (dt = {a.dt} s)')
print(f'episode cap: {r.max_episode_steps} steps = {r.max_episode_steps * a.dt} s')
print(f'goal radius: {r.goal_radius_m} m')
"
```

```
v:     0.0 .. 0.3 m/s
omega: +/-1.5 rad/s
rate:  20.0 Hz  (dt = 0.05 s)
episode cap: 1200 steps = 60.0 s
goal radius: 0.35 m
```

A 60-second budget at 0.3 m/s is 18 metres of driving. Goals are capped at 8 m
(`spec.py:118`). So the policy has roughly a 2× path-length margin over a
straight line — enough to go around obstacles, not enough to wander.

## Check yourself

- Why can't the oracle be deployed on the robot?
- The observation includes the *last action*. Why would a policy need to know
  what it just did? (`spec.py:126-132` answers it; the answer is about the reward
  function, not about dynamics.)
- If `v_min` were `-0.30` instead of `0.0`, what would get harder?

→ [02 — The contract](02-the-contract.md)
