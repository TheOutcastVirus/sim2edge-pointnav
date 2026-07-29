# 08 — The policy network

`sim2edge/models/nav_policy.py` (417 lines) is the only file in the repository
that is *not* allowed to import anything but torch. Its docstring at `:4-8`
explains why:

> **This file must not import anything but torch.** It is used identically for
> training on the 4090, for quantization calibration, and for lowering to
> ExecuTorch on the Ventuno, and the board's export venv is pinned to a specific
> torch with no torchvision available. The reference project hit exactly this wall
> (`ventuno-collision-avoidance/tools/collision_model.py:8-15`); the fix is to
> never let a training-side dependency into the model definition.

A single `import torchvision` at the top of a model file, added casually during
training, will fail at export time on the board. That is a real wall a sibling
project hit.

## The architecture

```
depth (1,96,160) ──> DepthEncoder  (4x Conv2d s2, 32→64→96→128) ──> Linear 256 ─┐
lidar (180,)     ──> LidarEncoder  (3x Conv1d s2, 32→64→64)      ──> Linear 128 ─┤
goal  (3,)       ──> Linear 32 ──────────────────────────────────────────────────┤
last_action (2,) ──> Linear 16 ──────────────────────────────────────────────────┤
                                                                                 │
                                          concat (432) ──> trunk MLP (512, 256) ─┘
                                                                │
                                                     actor_head Linear(256, 2)
                                                                │
                                                          tanh + scale
                                                                │
                                                         (v, omega) in SI units
```

Plus a separate critic that never ships:

```
lidar ──> critic_lidar (its own LidarEncoder) ──┐
goal, last_action, privileged (8) ──────────────┴──> MLP (256, 256) ──> value
```

```bash
python -c "
from sim2edge.models.nav_policy import build_policy
from sim2edge.spec import DEFAULT_SPEC as S
p = build_policy(S)
print('total params :', f'{p.num_parameters:,}')
print('actor params :', f'{p.num_actor_parameters:,}')
"
```

```
total params : 3,030,805
actor params : 2,716,850
```

2.72 M parameters ship. The other 314 k are the critic, discarded at export.
Note both are **properties**, not methods.

## Why a 1D conv over the lidar

`LidarEncoder` at `:127`:

> Convolutional rather than an MLP so the encoder is translation-equivariant in
> bearing: an obstacle 30 degrees to the left and the same obstacle 30 degrees to
> the right are the same local pattern, and a plain MLP would have to learn each
> bearing independently.

This is the same argument as using a CNN on images. "A wall segment" looks the
same whichever direction it is in; a convolution learns that pattern once and
applies it at all 180 bearings. An MLP with 180 inputs has to learn it 180 times,
from 180 times less data each.

### The padding decision

`:135-139`:

> Padding is zeros, not circular. Circular padding would be the geometrically
> correct choice for a 360-degree scan, but it lowers to a pad operator that is
> not among those proven delegated on HTP V75 by the reference models, and an
> unsupported op silently splits the graph and costs far more than the wrap-around
> seam is worth.

This is the deployment constraint reaching back into the architecture. Circular
padding is *correct* — bin 179 really is adjacent to bin 0. But an op that doesn't
delegate splits the graph into NPU and CPU partitions with a memory copy between
them, and that costs far more than the small error at one seam. `REBUILD_PLAN.md:150-152`
independently arrives at the same recommendation.

**Zero padding on normalized lidar means the encoder sees `0.0` at the edges —
which denormalizes to `min_m`, "obstacle at 15 cm."** The seam is not neutral, it
is pessimistic. That is arguably the safer direction to be wrong in, but it is
worth knowing.

## Why no pretrained backbone

`DepthEncoder` at `:92-94`:

> at 96x160 single-channel there is nothing an ImageNet backbone would transfer,
> and it would drag in torchvision, which the export environment cannot import.

Both halves matter. ImageNet features are about texture and colour semantics; a
depth image has neither. And torchvision is unavailable where this model must run.

### Static shapes, computed not hardcoded

`:110-117`:

```python
h, w = cfg.depth_height, cfg.depth_width
for _ in cfg.depth_channels:
    h = (h + 1) // 2
    w = (w + 1) // 2
self.flat_dim = cfg.depth_channels[-1] * h * w
```

> Fixed spatial size after the strided stack. Computed rather than hardcoded, but
> static: no adaptive pooling, so the exported graph has no shape-dependent
> operators.

`nn.AdaptiveAvgPool2d((1,1))` is the idiomatic PyTorch way to make a CNN
resolution-agnostic. It is also a dynamic-shape operator, which is a problem for a
graph that must be fully static to delegate. Computing the size in Python at
construction time gives you the flexibility with none of the graph cost.

## The asymmetric critic

`NavPolicy` docstring at `:183-187`:

> The actor sees only what the robot can sense. The critic additionally sees
> ground-truth geodesic distance, obstacle clearance, and true pose — free
> accuracy, because the critic is never exported and never runs on the robot.

And the critic deliberately has **no depth encoder** (`:211-216`):

> the privileged observation already contains the geodesic distance and the true
> obstacle clearance, which is most of what a value function needs, and a second
> depth CNN would roughly double the cost of every training step to re-derive
> information the critic is simply handed.

Good reasoning: the critic's question is "how much return do I expect from here?"
Given true geodesic distance to the goal, that is nearly answered. Re-deriving it
from pixels would be pure waste.

## The squash, and why it is in the graph

`squash` at `:247`:

```python
t = torch.tanh(pre)
v = cfg.v_min + (cfg.v_max - cfg.v_min) * 0.5 * (t[..., :1] + 1.0)
omega = cfg.omega_max * t[..., 1:]
return torch.cat([v, omega], dim=-1)
```

> In-graph on purpose: tanh is cheap and proven on HTP, and doing the scaling here
> means the C++ node receives SI units directly and cannot apply a different scale
> than training did.

This is a contract-mismatch defence expressed as an architecture choice. If the
network output were `[-1, 1]` and the C++ node multiplied by `v_max`, then `v_max`
would live in two places. Putting the scaling inside the exported graph makes the
disagreement impossible.

`nav_policy.py:20-23` states the boundary explicitly:

> the action squash and scaling are in-graph because tanh is cheap and proven on
> HTP, but every clamp, rate limit, and safety decision is in C++ where it can be
> reasoned about and unit-tested.

## State-independent log_std

`:204-206`:

```python
self.log_std = nn.Parameter(torch.full((cfg.action_dim,), float(cfg.init_log_std)))
```

> the standard choice for continuous control: a state-dependent std tends to
> collapse early in PPO and makes the BC-anchored fine-tune much harder to keep
> stable.

The exploration noise is a learned constant per action dimension, not a function
of the observation. Simpler, more stable, and — since it is not part of the actor
mean path — it never gets exported.

## Distribution before the squash

`distribution` at `:273`. This is a subtle choice worth reading twice:

> Placing the distribution before the tanh, and omitting the change-of-variables
> correction, is the same convention `rsl_rl` uses. The squash then behaves as a
> deterministic output transform. The cost is that log-probabilities are of `u`,
> not of the physical action; the benefit is that the entropy term stays
> well-conditioned instead of collapsing as tanh saturates.

Formally, if `a = tanh(u)` and `u ~ N(μ,σ)`, the density of `a` needs a Jacobian
correction `log(1 - tanh²(u))`. Omitting it means the PPO objective is over `u`,
not `a`. That is technically "wrong" — and it is what rsl_rl, and most working
locomotion code, does, because the correction makes entropy blow up as `tanh`
saturates. Matching rsl_rl's convention matters here because chapter 12 trains
with rsl_rl.

## The exported module

`ActorMean` at `:342`, produced by `policy.export_module()` (`:327`). This is the
*only* thing that ships:

- no Gaussian head
- no critic
- no `log_std`
- no privileged input

`freeze_norm()` at `:309` puts BatchNorm into eval mode with frozen statistics
before export, so the exported graph has no running-stat updates.

```bash
python -c "
import torch
from sim2edge.models.nav_policy import build_policy
from sim2edge.spec import DEFAULT_SPEC as S

p = build_policy(S)
actor = p.export_module()
ex = tuple(torch.zeros((1,) + s) for s in S.actor_input_shapes)
out = actor(*ex)
print('output shape:', out.shape)
print('output      :', out.tolist())
"
```

```
output shape: torch.Size([1, 2])
output      : [[0.15000000596046448, 0.0]]
```

`v = 0.15` from a zero input is exactly right: the actor head is initialized near
zero (`:226-229`), so `pre ≈ 0`, `tanh(0) = 0`, and the squash maps `0` to the
midpoint of `[0, 0.30]`. An untrained policy drives forward at half speed with no
turn.

## Verify it will delegate

The whole architecture is a bet that every operator lands on the HTP. You can
check the bet before ever touching the board:

```bash
python -c "
import torch
from sim2edge.models.nav_policy import build_policy
from sim2edge.spec import DEFAULT_SPEC as S

actor = build_policy(S).export_module()
ex = tuple(torch.zeros((1,) + s) for s in S.actor_input_shapes)
ep = torch.export.export(actor, ex)
ops = sorted({str(n.target) for n in ep.graph.nodes if n.op == 'call_function'})
print(len(ops), 'distinct ops:')
for o in ops: print('  ', o)
"
```

```
12 distinct ops:
   aten.add.Tensor
   aten.batch_norm.default
   aten.cat.default
   aten.conv1d.default
   aten.conv2d.default
   aten.flatten.using_ints
   aten.linear.default
   aten.mul.Tensor
   aten.relu_.default
   aten.slice.Tensor
   aten.tanh.default
   aten.unsqueeze.default
```

Twelve operators, all of them on the list `nav_policy.py:11-15` names as proven
delegated on HTP V75 at 8a8w. No `adaptive_avg_pool`, no `pad`, no control flow,
no dynamic shapes. This is what "designed backward from the NPU" produces, and
this one command is the cheapest possible check that the design held.

Chapter 13 takes this graph the rest of the way.

## Config, mirrored not imported

`NavPolicyConfig` at `:35` duplicates the spec's numbers as defaults rather than
importing them — because importing `sim2edge.spec` would break the torch-only
guarantee. `build_policy` at `:396` passes the real spec values in, and
`assert_matches_spec` verifies they agree. Duplication with an assertion, rather
than duplication and hope.

## Check yourself

- Why is the critic's lidar encoder a *separate* module rather than shared with
  the actor's?
- What is the practical consequence of zero-padding rather than circular-padding
  the lidar convolution, given that `0.0` normalizes to 15 cm?
- Why does an untrained policy output `v = 0.15` rather than `v = 0`?
- If you added `nn.AdaptiveAvgPool2d` to the depth encoder, what would break, and
  at which stage would you find out?

→ [09 — Behaviour cloning](09-behaviour-cloning.md)
