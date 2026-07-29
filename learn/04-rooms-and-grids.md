# 04 — Rooms and grids

The policy needs thousands of different rooms to train in. Hand-authoring them is
out of the question, so `sim2edge/rooms.py` (369 lines) generates them
procedurally, then rasterizes each one into several grids that the reward, the
oracle, and the sensor simulation all read.

This chapter is about representation: how a continuous 2D world becomes arrays.

## Obstacles have a vertical extent, and that is the point

`Obstacle` at `rooms.py:47`:

```python
@dataclass
class Obstacle:
    kind: str      # "box" | "cylinder"
    x: float
    y: float
    sx: float      # box: half-extent. cylinder: radius.
    sy: float
    yaw: float
    z_min: float   # underside height
    z_max: float   # top
```

`z_min` and `z_max` are the interesting fields. The module docstring
(`rooms.py:11-23`) explains why three classes of obstacle exist:

- **tall** — both the lidar and the camera see it;
- **low** — below the lidar plane, so the lidar *cannot see it at all*;
- **overhanging** — underside above the robot, so it is visible to both sensors
  but is not actually an obstacle.

> A lidar-only policy is structurally unable to solve the first two cases
> correctly, which is the whole reason depth is in the observation.

This is the repository's justification for having a depth camera. It is worth
noticing now, because `REBUILD_PLAN.md:161-164` disagrees with it — arguing that
with a 360° lidar, depth adds little navigation capability and is mainly a better
*demo*. Both positions can't be right; chapter 15 returns to it.

## Rasterization: one layout, four grids

`Room._rasterize` at `rooms.py:134` walks the obstacle list once and produces four
layers. The height logic is the whole story:

```python
# Collidable only if it intrudes into the robot's vertical envelope.
if obs.spans(0.0, robot_h):
    collision |= m
# Visible to the lidar only if it crosses the scan plane exactly.
if obs.z_min <= lidar_z <= obs.z_max:
    lidar_occ |= m
# The camera sees everything, so the depth layers take all of it.
np.maximum(z_max_map, np.float32(obs.z_max), out=z_max_map, where=m)
np.minimum(z_min_map, np.float32(obs.z_min), out=z_min_map, where=m)
```

| Layer | Meaning | Consumer |
|---|---|---|
| `collision` | intrudes into `[0, 0.31] m` — the robot's envelope | reward, termination, oracle |
| `lidar_occ` | crosses the scan plane at `0.175 m` | simulated lidar |
| `z_max_map` | height of the tallest surface per cell | 2.5D depth renderer |
| `z_min_map` | height of the lowest surface per cell | 2.5D depth renderer |

Robot geometry comes from `RobotSpec` (`spec.py:352`): radius `0.175 m`, height
`0.31 m`, lidar plane `0.175 m`, camera at `0.19 m`. The comment at `spec.py:358-362`
warns about the consequence of getting this wrong:

> an obstacle whose underside is above it (a table top, a shelf) is driven under,
> and must NOT be rasterized into the collision grid or the oracle refuses to plan
> through gaps the robot actually fits under.

The three grids can genuinely disagree about the same object, and that disagreement
*is* the sim-to-real difficulty the depth channel is supposed to resolve.

Arena walls are added last (`rooms.py:159-171`) at `WALL_THICKNESS = 0.10 m`,
collidable and lidar-visible on all four sides.

## Occupancy grids

`OccupancyGrid` (`sim2edge/oracle/grid.py:53`) is a boolean array plus a
resolution, with world↔cell conversion:

```python
def world_to_cell(self, x, y) -> Tuple[int, int]      # grid.py:77
def cell_to_world(self, r, c) -> Tuple[float, float]  # grid.py:84
def inflate(self, radius_m) -> "OccupancyGrid"        # grid.py:106
```

Resolution is `0.05 m` (`RoomSpec.grid_resolution_m`, `spec.py:386`), so an 8×6 m
room is a 160×120 grid — 19,200 cells. Small enough to run a distance transform on
per episode.

### Inflation, and why the margin must be strictly positive

`_build_fields` at `rooms.py:180`:

```python
pad = self.spec.robot.radius_m + self.spec.room.planning_margin_m
self.inflated = self.collision.inflate(pad)
```

`planning_margin_m = 0.06` (`spec.py:396`). The docstring there is one of the more
useful lessons in the repository:

> Must be > 0: planning at exactly the collision radius makes every path tangent
> to the collision boundary, so any tracking error at all becomes a collision, and
> the oracle's demonstrations become a dataset of near-misses.

Inflating by the robot radius lets you plan for a *point* robot in the inflated
space — a standard trick. But inflating by *exactly* the radius produces paths
that graze every corner. Since a real controller has tracking error, and since
those paths become the behaviour-cloning dataset, you would be teaching the policy
that grazing corners is correct. Six centimetres of slack fixes it.

### Clearance

`clearance` (`rooms.py:181`) is a Euclidean distance transform: for every cell, the
distance to the nearest collidable obstacle.

It does triple duty:
- `in_collision(x, y)` is just `clearance_at(x, y) < radius` (`rooms.py:200-208`);
- it is a critic input (chapter 08);
- it is how start and goal positions are rejection-sampled to have room around
  them.

One array, three uses — a good example of picking a representation that answers
more than one question.

## Sampling a room

`sample_room` at `rooms.py:285`, with ranges from `RoomSpec` (`spec.py:376`):

| Parameter | Range |
|---|---|
| arena size | 4.0 – 10.0 m |
| obstacle count | 5 – 20 |
| obstacle size | 0.15 – 1.20 m |
| goal distance | 2.0 – 8.0 m |
| spawn clearance | robot radius + 0.15 m |

```bash
python -c "
import numpy as np
from sim2edge.rooms import sample_room
from sim2edge.spec import DEFAULT_SPEC as S

room = sample_room(np.random.default_rng(0), S)
print(f'arena      : {room.arena_w:.2f} x {room.arena_h:.2f} m')
print(f'obstacles  : {len(room.obstacles)}')
print(f'start      : {room.start}')
print(f'goal       : {room.goal}')
print(f'grid shape : {room.collision.shape}')
print(f'feasible   : {room.is_feasible}')
print(f'optimal    : {room.optimal_path_length:.3f} m')
print()
for o in room.obstacles[:4]:
    print(f'  {o.kind:9s} at ({o.x:5.2f},{o.y:5.2f})  z {o.z_min:.2f}-{o.z_max:.2f}')
"
```

```
arena      : 7.80 x 5.60 m
obstacles  : 8
start      : (1.259925652225463, 0.5545376522809562, 1.0806756900855436)
goal       : (3.924999952316284, 4.925000190734863)
grid shape : (112, 156)
feasible   : True
optimal    : 5.918 m

  cylinder  at ( 6.16, 4.86)  z 0.00-1.13
  box       at ( 6.17, 0.31)  z 0.00-0.05
  box       at ( 2.46, 2.41)  z 0.00-0.52
  box       at ( 7.48, 5.20)  z 0.00-1.05
```

Look at the second obstacle: a box spanning `z 0.00-0.05` — five centimetres tall.
The lidar plane is at 0.175 m, so `lidar_occ` does **not** contain it, while
`collision` does (it intrudes into `[0, 0.31]`). That is a real instance of the
"low obstacle" class, generated on the very first seed: a thing the robot will hit
and the lidar cannot see. The depth camera is the only sensor that can save it.

Seeded, so you get the same room. Every episode in training gets a fresh seed and
therefore a fresh layout — this is the cheapest and most effective form of
generalization pressure in the whole project.

## The geodesic field

`Room.geodesic` (`rooms.py:99`) is the payoff of all the rasterization. For every
free cell, the shortest *obstacle-avoiding* distance to the goal. It is computed
lazily and cached, because it is the most expensive part of building a room.

Euclidean distance to the goal says "the goal is 4 m that way," even if a wall is
in between. Geodesic distance says "the goal is 7 m of driving away, and the first
metre of it is to your left." That difference is what the reward function is built
on (chapter 06).

```bash
python -c "
import numpy as np
from sim2edge.rooms import sample_room
from sim2edge.spec import DEFAULT_SPEC as S
import math

room = sample_room(np.random.default_rng(0), S)
sx, sy, _ = room.start
gx, gy = room.goal
print(f'euclidean start->goal : {math.hypot(gx-sx, gy-sy):.3f} m')
print(f'geodesic  start->goal : {room.geodesic_at(sx, sy):.3f} m')
print(f'clearance at start    : {room.clearance_at(sx, sy):.3f} m')
"
```

```
euclidean start->goal : 5.119 m
geodesic  start->goal : 5.918 m
clearance at start    : 0.500 m
```

Geodesic exceeds Euclidean by 0.80 m — that gap is the obstacle detour, and it is
exactly the information a Euclidean reward would hide from the policy.

Clearance at the start is 0.50 m, comfortably above the robot's 0.175 m radius plus
the 0.15 m spawn clearance — the rejection sampling did its job.

Unreachable cells get a finite sentinel, `UNREACHABLE = 1.0e6` (`grid.py:47-49`):

> Finite so it survives arithmetic and comparisons without producing NaNs in
> reward differences.

Using `inf` would make one subtraction poison an entire training batch with NaN.
Chapter 06 shows the guard that handles this sentinel in the reward.

## Try: see a room as ASCII art

```bash
python -c "
import numpy as np
from sim2edge.rooms import sample_room
from sim2edge.spec import DEFAULT_SPEC as S

room = sample_room(np.random.default_rng(3), S)
occ = room.collision.occupied
sr, sc = room.collision.world_to_cell(room.start[0], room.start[1])
gr, gc = room.collision.world_to_cell(*room.goal)
step = 3
for r in range(0, occ.shape[0], step):
    line = ''
    for c in range(0, occ.shape[1], step):
        if abs(r-sr) < step and abs(c-sc) < step: line += 'S'
        elif abs(r-gr) < step and abs(c-gc) < step: line += 'G'
        else: line += '#' if occ[r:r+step, c:c+step].any() else '.'
    print(line)
"
```

Try seeds 0 through 5. Look for rooms where a naive straight line from `S` to `G`
passes through a `#` — those are the episodes that separate a good policy from a
bad one.

## Cost

Building a room means rasterizing every obstacle, running a distance transform,
inflating, and (lazily) computing a Dijkstra field over ~20k cells. That is
milliseconds in numpy, which is fine at one environment.

It is *not* fine at 4096 parallel GPU environments, and this is the seam where
`isaaclab_env/nav_task.py` gets awkward: it pre-generates one room per environment
at startup and uploads padded geodesic fields to the GPU (`nav_task.py:215-232`).
`REBUILD_PLAN.md:29` proposes dropping the whole thing — randomize obstacle poses
on reset instead of baking a layout per environment. Chapter 12 looks at the
actual code.

## Check yourself

- A table 0.5 m tall with a 0.4 m clearance underneath. Which of `collision`,
  `lidar_occ`, `z_max_map` include it? (Robot height 0.31 m, lidar plane 0.175 m.)
- Why is `clearance` computed on `collision` rather than on `lidar_occ`?
- Why is `UNREACHABLE` a large finite number rather than `inf`?
- The geodesic field is computed toward the goal, not from the start. Why does
  that direction matter for how the reward uses it?

→ [05 — The oracle](05-the-oracle.md)
