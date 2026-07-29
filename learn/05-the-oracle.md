# 05 — The oracle

The oracle is a classical navigation stack that solves the task perfectly by
cheating: it is handed the ground-truth room. It exists for two reasons — to
generate demonstrations for behaviour cloning (chapter 09), and to be the upper
bound every learned policy is measured against (chapter 11).

Two files, 620 lines: `sim2edge/oracle/grid.py` (distance fields) and
`sim2edge/oracle/expert.py` (planning and control).

## Part 1 — The geodesic distance field

### What it is

Given the goal cell and a grid of free/occupied cells, compute for *every* free
cell the length of the shortest obstacle-avoiding path to the goal. The result is
an array the same shape as the grid.

`distance_field` at `grid.py:191`. Movement is 8-connected with octile costs
(`grid.py:219`, `_neighbor_offsets`): orthogonal steps cost `resolution`, diagonal
steps cost `resolution·√2`. Using cost 1 for diagonals would make the field
systematically underestimate distance along diagonals and bias every path toward
staircase motion.

### Two implementations, one answer

```python
if _HAVE_SCIPY_CSGRAPH:
    return _distance_field_scipy(...)   # grid.py:227 — sparse matrix + Dijkstra
return _distance_field_heapq(...)       # grid.py:260 — hand-rolled priority queue
```

The SciPy path builds a sparse adjacency matrix over free cells and calls
`scipy.sparse.csgraph.dijkstra`. The fallback is a plain binary heap. Both are
*exact*, not approximations — `grid.py:28-29` is explicit that results must not
depend on which path ran.

`obstacle_distance` at `grid.py:133` is the same story for the Euclidean distance
transform: SciPy's `distance_transform_edt` if available, otherwise
`_edt_separable` at `grid.py:149` (the classic separable two-pass algorithm).

### Why not A*?

This is the most interesting design note in the file. `expert.py:12-16`:

> Planning descends the geodesic distance field rather than running a separate A*.
> The field already exists — it is needed for the PPO progress reward — and
> steepest descent on it *is* the shortest path under the field's octile metric.
> Running an independent A* would introduce a second notion of "optimal path"
> that could disagree with the reward, which is a subtle way to make PPO fight BC.

Think about what that would mean. BC teaches the policy to follow A*'s path. PPO
rewards progress along the geodesic field. If those disagree even slightly, PPO
spends its first thousand iterations *unlearning* what BC taught. Deriving both
from one field makes the disagreement impossible by construction.

This is the same principle as `spec.py`, applied to algorithms instead of
constants: **don't compute the same quantity twice.**

## Part 2 — Planning by steepest descent

`plan_path` at `expert.py:42` is 40 lines and does exactly what its name says:

```python
for _ in range(max_steps):
    if field[r, c] <= 0.0:
        break
    best, best_val = None, field[r, c]
    for dr, dc in _NEIGHBORS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and field[nr, nc] < best_val:
            best, best_val = (nr, nc), field[nr, nc]
    if best is None or best in seen:
        break  # local minimum (should not happen on an exact field)
    r, c = best
```

Stand on a cell, step to whichever of the eight neighbours has the lowest value,
repeat. On an exact distance field there are no local minima, so this always
reaches the goal — the guard at `:70-71` is defensive, not load-bearing.

Unreachable is detected up front at `:57` by testing the start against the
`UNREACHABLE` sentinel, and returns an empty array rather than looping.

### Smoothing: string-pulling

A raw descent path is staircased at 5 cm resolution. `smooth_path` at `expert.py:94`
fixes it, and the docstring explains why it matters more than it looks:

> Pure-pursuit over a staircase produces a visibly wobbling `omega` command, and
> behaviour cloning would learn to reproduce the wobble — a smoothness penalty in
> PPO then has to unlearn what BC was explicitly taught.

The algorithm is greedy shortcutting: from each waypoint, jump to the furthest
later waypoint still in line of sight (`_line_of_sight`, `:84`), twice, then
resample at uniform 5 cm spacing (`_densify`, `:128`).

The critical detail is at `expert.py:109`:

```python
grid = room.inflated
```

Shortcuts are checked against the **inflated** grid, not the raw collision grid,
so a shortcut can never cut a corner the robot's footprint would clip. This is
where the `planning_margin_m = 0.06` from chapter 04 earns its keep.

## Part 3 — Pure pursuit

`PurePursuitExpert` at `expert.py:164` turns a path into `(v, ω)` commands. The
core geometry is four lines (`:229-243`): transform the lookahead point into the
robot frame, and the arc through it has curvature `2·y_left / d²`.

Everything else in `act()` is the parts that make the demonstrations *usable as
training data*.

### Adaptive lookahead

```python
lookahead = float(np.clip(
    self.min_lookahead_m + 0.6 * (clearance - self.spec.robot.radius_m),
    self.min_lookahead_m, self.lookahead_m))
```

Lookahead shrinks in tight spaces (`:215-217`): a long lookahead in a doorway cuts
the corner, and the inflated-grid guarantee does not extend to the controller's
tracking error.

### Rotate in place on large heading error

```python
if abs(heading_err) > self.rotate_in_place_rad:   # 0.7 rad ≈ 40°
    omega = math.copysign(a.omega_max * 0.8, heading_err)
    return self._limit(0.0, omega, target_i, done=False)
```

`:236-238`: driving forward through a large heading error wastes distance and,
near obstacles, sweeps the footprint into them. The real Create3 turns in place
too — so the demonstrations match the deployed behaviour.

### The speed profile

Four multiplicative reductions (`:245-257`), each for a different reason:

| Factor | Line | Reason |
|---|---|---|
| heading error | `:247` | don't drive fast while misaligned |
| clearance | `:248` | slow down in tight spaces |
| goal deceleration | `:252` | `v ≤ √(2·a·d)` — stop *at* the goal, not past it |
| curvature | `:256` | `v ≤ ω_max/‖κ‖`, so `ω` never saturates |

That last one is subtle and worth understanding: pure pursuit commands
`ω = v·κ`. If `v·κ` exceeds `ω_max`, the clamp silently changes the arc the
controller thought it was driving, and it stops tracking the path. Capping `v`
instead preserves the geometry.

### Real limits, always

`_limit` at `:261` clamps every command to the base's velocity *and* acceleration
limits, read from the same `ActionSpec` the simulator and the C++ node use. The
module docstring at `expert.py:5-10` says why this is non-negotiable:

> An oracle that demonstrates commands the Create3 cannot track produces a policy
> that faithfully imitates commands the Create3 cannot track, and the resulting
> sim-to-real gap looks like a perception problem while actually being a dynamics
> problem.

That sentence is worth rereading. It describes a failure that would send you
debugging the wrong subsystem for days.

## Try it

### Plan a path

```bash
python -c "
import numpy as np
from sim2edge.rooms import sample_room
from sim2edge.oracle.expert import plan_path, smooth_path
from sim2edge.spec import DEFAULT_SPEC as S

room = sample_room(np.random.default_rng(0), S)
raw = plan_path(room)
sm  = smooth_path(room, raw)
print('raw waypoints   :', raw.shape)
print('smoothed        :', sm.shape)
print('raw length      : %.3f m' % np.linalg.norm(np.diff(raw, axis=0), axis=1).sum())
print('smoothed length : %.3f m' % np.linalg.norm(np.diff(sm, axis=0), axis=1).sum())
print('geodesic optimal: %.3f m' % room.optimal_path_length)
"
```

```
raw waypoints   : (98, 2)
smoothed        : (117, 2)
raw length      : 5.918 m
smoothed length : 5.830 m
geodesic optimal: 5.918 m
```

Two things: the smoothed path is *shorter* (5.830 vs 5.918) because string-pulling
removed the staircase overhead, but it has *more* points (117 vs 98) because
`_densify` resamples at uniform 5 cm spacing afterwards. Fewer corners, more
samples along them — which is exactly what a pure-pursuit controller wants.

The raw descent length equals the geodesic optimum to three decimals, which is the
expected result: steepest descent on an exact field *is* the shortest path under
that field's metric.

### Drive with it

```bash
python -c "
import numpy as np
from sim2edge.rooms import sample_room
from sim2edge.oracle.expert import PurePursuitExpert
from sim2edge.spec import DEFAULT_SPEC as S

room = sample_room(np.random.default_rng(0), S)
ex = PurePursuitExpert(room, S)
print('has_path:', ex.has_path, ' path:', ex.path.shape)
x, y, yaw = room.start
for t in range(6):
    cmd = ex.act(x, y, yaw)
    print(f't={t}  v={cmd.v:.3f}  omega={cmd.omega:+.3f}  target={cmd.target_index}')
"
```

```
has_path: True  path: (117, 2)
t=0  v=0.030  omega=+0.150  target=9
t=1  v=0.060  omega=+0.255  target=9
t=2  v=0.090  omega=+0.255  target=9
t=3  v=0.108  omega=+0.255  target=9
```

Because `_limit` enforces `accel_max = 0.6 m/s²` at `dt = 0.05 s`, `v` rises by at
most 0.03 m/s per tick — and it does exactly that for the first three ticks. The
expert ramps; it does not jump. So does the robot.

(The pose is not advancing here — `act()` is being called repeatedly from the same
start pose — so `target` stays at waypoint 9. In a real episode the robot moves and
the target index advances.)

### Measure the upper bound

The evaluation harness can drive with the oracle directly (`eval/harness.py:215`,
`oracle_policy`):

```bash
python -c "
from sim2edge.envs.reference_env import NavEnv, EnvConfig
from sim2edge.eval.harness import evaluate, oracle_policy

env = NavEnv(EnvConfig(use_depth=False), seed=0)
m = evaluate(oracle_policy(env), n_episodes=10, env=env)
print(m.summary())
"
```

```
SR 100.0%  SPL 1.000  coll  0.0%  timeout  0.0%  t_goal  16.9s  jerk  0.41  interv  0.00
```

**100% success, SPL 1.000.** That is the ceiling. Any learned policy is measured
against it — and note that the oracle achieves it *with domain randomization on*,
because it reads ground-truth pose from the environment rather than the noisy
observation.

`oracle_policy` needs privileged access, which is why `evaluate` accepts an `env`
argument (`harness.py:95-97`): the policy and the harness must step the same
instance.

## The cost of having an oracle

Everything in this chapter exists so that BC has labels. `REBUILD_PLAN.md:25`
proposes deleting all 620 lines:

> Only existed to generate BC data.

and `:27` deletes the geodesic reward with it, arguing Euclidean progress is
nearly identical in convex-ish rooms. If you skip BC and train PPO from scratch
with a dense progress reward, the oracle has no other customer.

The counter-argument is in this chapter: the geodesic field is also the reward
signal, the SPL denominator, and the feasibility test. You would be replacing all
four with approximations. Whether that trade is right depends on whether PPO
alone actually solves the task — which, as chapter 15 notes, nobody here has
tested yet.

## Check yourself

- Why does steepest descent on an exact distance field never get stuck?
- What breaks if `smooth_path` checks line-of-sight against `room.collision`
  instead of `room.inflated`?
- The speed profile caps `v` at `ω_max/‖κ‖`. What visibly goes wrong without it?
- The oracle scores SPL 1.000. Given SPL's definition (chapter 11), what does that
  tell you about the paths it drives?

→ [06 — The environment](06-the-environment.md)
