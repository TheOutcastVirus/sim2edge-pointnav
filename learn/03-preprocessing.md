# 03 — Preprocessing

Preprocessing is the function that turns messy sensor readings into the fixed-size
float arrays a neural network can consume. In this project it is also the single
most dangerous piece of code, because it exists twice — once in Python for the
simulator, once in C++ for the robot — and a disagreement between the two is
invisible.

`sim2edge/preprocess.py` (273 lines) is the Python half. Read its docstring at
lines 1-21 before anything else; it explains that every algorithm is written so it
can be transcribed to C++ *literally*, with no dependence on library behaviour.

## The four channels

| Channel | In | Out | Function |
|---|---|---|---|
| Depth | `(H, W)` metric float image | `(1, 96, 160)` in `[0,1]` | `preprocess_depth` `:111` |
| Lidar | `LaserScan.ranges` + angles | `(180,)` in `[0,1]` | `preprocess_lidar` `:122` |
| Goal | `(x, y)` in robot frame | `(3,)` | `preprocess_goal` `:170` |
| Last action | `(v, ω)` physical | `(2,)` in `[-1,1]` | `preprocess_last_action` `:192` |

`Preprocessor` at `:207` bundles all four. It is used identically by the sim env,
the calibration-set builder, and the parity test — which is the whole point:
calibration data is guaranteed to pass through the same code path the deployed
policy's inputs do.

## Lidar: the interesting one

Start here, because three separate design decisions are visible in twenty lines.

### Decision 1 — bin by angle, not by index

`preprocess.py:132-135`:

> Beams are binned by their *actual* angle rather than by index, so the result
> does not depend on scan direction, start angle, or beam count. This matters:
> the RPLIDAR's scan start angle is not stable across driver versions, and
> binning by index would silently rotate the policy's world view.

A policy whose world view is rotated 15° will drive into walls with great
confidence. Binning by angle costs one modulo and makes the failure impossible.

### Decision 2 — min-pool, don't average

`preprocess.py:164-166`:

```python
out = np.full(ls.bins, ls.invalid_fill, dtype=np.float32)
np.minimum.at(out, idx, norm)
```

Averaging a beam that hit a chair leg at 0.4 m with a beam that hit the far wall
at 6 m gives 3.2 m — a reading that describes empty space where a chair leg is.
Min-pooling is the only reduction that cannot delete a thin obstacle. `spec.py:83-84`
says so directly.

### Decision 3 — invalid becomes *far*, and is normalized before pooling

`preprocess.py:158-161`:

```python
valid = np.isfinite(r) & (r >= lo) & (r <= hi)
clipped = np.clip(r, ls.min_m, ls.max_m)
norm = (clipped - ls.min_m) / (ls.max_m - ls.min_m)
norm = np.where(valid, norm, np.float32(ls.invalid_fill)).astype(np.float32)
```

Invalid returns (`inf`, `nan`, out of range) become `1.0`, meaning "far". They are
normalized *before* pooling, so "invalid" and "far" reduce identically and there
is no separate mask to keep in sync between Python and C++.

The choice of `1.0` rather than `0.0` is optimistic — a dropout is treated as free
space. That is safe here only because the classical safety layer (chapter 14)
covers the pessimistic case. `0.0` would denormalize to `min_m = 0.15 m`: an
obstacle 15 cm dead ahead, in every dropped beam.

### See it

```bash
python -c "
import numpy as np, math
from sim2edge.preprocess import preprocess_lidar
inc = 2*math.pi/360
scan = np.full(360, 4.0, np.float32)   # a room roughly 4 m in every direction
scan[178:183] = 0.8                    # something close, straight ahead
scan[90] = float('inf')                # one dropped beam, off to the right
lid = preprocess_lidar(scan, -math.pi, inc)
print('background bin :', lid[0])
print('forward bin 90 :', lid[90])
print('bin 45         :', lid[45])
print('argmin, min    :', int(lid.argmin()), float(lid.min()))
"
```

```
background bin : 0.49044585
forward bin 90 : 0.08280255
bin 45         : 0.49044585
argmin, min    : 89 0.08280254900455475
```

Three things to read out of that:

- **Background is 0.49, not 4.0.** Everything is normalized:
  `(4.0 - 0.15) / (8.0 - 0.15) = 0.4904`.
- **Bin 90 is forward.** Angles run `-π` to `π` across 180 bins, so index 90 is
  angle 0. The near obstacle at beams 178-182 landed there:
  `(0.8 - 0.15) / 7.85 = 0.0828`.
- **The dropped beam vanished.** Beam 90 (`inf`) is at angle
  `-π + 90·(2π/360) = -π/2`, i.e. bin 45 — and bin 45 still reads 0.49, because
  its *other* beam was a valid 4.0 m return and min-pooling took the smaller
  normalized value. Two raw beams per output bin means a single dropout is
  usually absorbed.

Sanity-check the geometry yourself: 360 raw beams into 180 bins, so bin `k`
covers raw beams `2k` and `2k+1`.

## Depth: min-pooling in two dimensions

`preprocess_depth` at `:111` is three lines because the work is in
`block_min_pool` at `:65` and `normalize_depth_raw` at `:87`.

The reason it is min-pooling and not `cv::resize` is spelled out at
`preprocess.py:9-13`:

> Bilinear or area interpolation across a depth discontinuity invents a surface
> that is in neither the near object nor the far background, and INTER_AREA in
> OpenCV would not match anything reasonable written in numpy.

That second clause is the C++-parity argument: `cv::resize` has version-dependent
edge behaviour, so no numpy implementation would match it byte-for-byte forever.
An explicit integer-block min-pool has exactly one correct answer.

`_block_bounds` at `:49` computes the block edges with integer arithmetic:

```python
edges = (np.arange(n_out + 1, dtype=np.int64) * n_in) // n_out
starts = edges[:-1]
ends = np.maximum(edges[1:], starts + 1)
```

The `np.maximum(..., starts + 1)` guard means upsampling degenerates to
nearest-neighbour rather than producing empty bins. The C++ side computes the same
integers with the same integer arithmetic — that is why it is written with `//`
rather than anything float-valued.

## Goal: sin/cos, never a raw angle

`preprocess_goal` at `:170`:

```python
dist = math.hypot(goal_x, goal_y)
d_norm = min(dist, max_d) / max_d
sin_b, cos_b = goal_y / dist, goal_x / dist
return np.array([d_norm, sin_b, cos_b], dtype=np.float32)
```

```bash
python -c "
from sim2edge.preprocess import preprocess_goal
for g in [(3.0, 4.0), (0.0, 0.0), (-1.0, 0.0), (20.0, 0.0)]:
    print(g, '->', preprocess_goal(*g))
"
```

```
(3.0, 4.0) -> [0.625 0.8   0.6  ]
(0.0, 0.0) -> [0. 0. 1.]
(-1.0, 0.0) -> [0.125 0.   -1.  ]
(20.0, 0.0) -> [1. 0. 1.]
```

- `(3, 4)`: distance 5, normalized `5/8 = 0.625`; bearing `(sin, cos) = (0.8, 0.6)`.
- `(0, 0)`: the degenerate case is handled explicitly at `:185-186`, returning
  bearing "straight ahead" rather than dividing by zero.
- `(-1, 0)`: goal directly *behind*, `cos = -1`.
- `(20, 0)`: clipped to the 8 m maximum, so distance saturates at `1.0`.

Why not just feed the bearing angle? `spec.py:110-113`:

> The angle is encoded as sin/cos rather than a raw angle so the input is
> continuous across the ±π wrap. A wrap discontinuity in an INT8-quantized input
> is a nasty, hard-to-debug failure right where the robot most needs to turn
> around.

A raw angle jumps from `+3.14` to `-3.14` for an infinitesimal change in heading.
A network has to learn that those two inputs mean the same thing. At INT8, where
the whole range might be 256 levels, that discontinuity sits exactly at "the goal
is behind me" — the moment the robot most needs to act correctly.

## Last action

`preprocess_last_action` at `:192` is a thin wrapper over `ActionSpec.normalize`.
It is in the observation because the reward has a smoothness term
(`spec.py:246`) penalizing `‖aₜ − aₜ₋₁‖²`. Without `aₜ₋₁` in the observation, that
penalty depends on information the policy cannot see, so the optimal policy is not
representable. `spec.py:129-131` states this.

This is a general principle worth carrying away: **if your reward depends on
something, your observation must contain it.**

## Putting it together

```bash
python -c "
import numpy as np, math
from sim2edge.preprocess import Preprocessor
from sim2edge.spec import DEFAULT_SPEC as S

pre = Preprocessor(S)
obs = pre.build(
    depth_m=np.full((240, 320), 2.5, np.float32),
    scan_ranges=np.full(360, 4.0, np.float32),
    scan_angle_min=-math.pi,
    scan_angle_increment=2*math.pi/360,
    goal_xy_robot=(2.0, 1.0),
    last_action=(0.1, 0.2),
)
for k, v in obs.items():
    print(f'{k:12s} {str(v.shape):14s} min={v.min():.3f} max={v.max():.3f}')
print()
print('as_tuple order:', [a.shape for a in pre.as_tuple(obs)])
"
```

`as_tuple` at `:243` orders the dict into the positional order the exported graph
expects — `spec.obs.actor_inputs`. That single method is what keeps the argument
order consistent between training, calibration, and the C++ node's tensor
binding.

## Coordinate conventions

Stated at `preprocess.py:20-22`:

> Units in: metres and radians, robot frame, x forward, y left, yaw CCW.
> Units out: float32 arrays normalized to [0, 1] (sensors) or [-1, 1] (actions).

`world_goal_to_robot_frame` at `:253` does the rotation, and `wrap_angle` at `:271`
is the `[-π, π)` normalizer used everywhere. Frame bugs are the second most common
source of "the policy learned nothing" — when you write your own env, test this
function first.

## The C++ twin

`ros_ws/src/nav_policy_node/src/preprocess.cpp` (188 lines) implements the same
four functions. `tools/parity_check.py` (316 lines) runs both on identical raw
inputs — including NaN depth, all-invalid scans, and out-of-range values — and
requires agreement to 1e-5. The README reports it currently agrees **bit-exactly**.

```bash
python -m pytest tests/test_cpp_parity.py -q
```

Chapter 14 covers how that gate works. `REBUILD_PLAN.md:28` argues the entire
apparatus should be deleted in favour of writing the ROS node in `rclpy` and
importing this very file — making the bug class non-existent rather than
tested-for. Hold that thought.

## Check yourself

- A driver update changes the RPLIDAR's `angle_min` from `0` to `-π`. Does
  `preprocess_lidar` care? Why not?
- Why are invalid beams normalized *before* min-pooling rather than masked after?
- You have 180 raw beams and 180 bins. Trace `_block_bounds(180, 180)`. Now try
  90 raw beams into 180 bins — what does the guard at `:57` do?

→ [04 — Rooms and grids](04-rooms-and-grids.md)
