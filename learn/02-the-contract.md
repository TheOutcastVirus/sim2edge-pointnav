# 02 — The contract

## The failure this file exists to prevent

Two sibling projects on the same board lost more time to one bug than to anything
else, and it is worth stating precisely because it is not a crash:

> The preprocessing used to calibrate the quantized model did not exactly match
> the preprocessing performed at runtime.

Say the calibration set was built by normalizing lidar with `max_m = 10.0`, and
the C++ node on the robot normalizes with `max_m = 8.0`. The quantizer picks INT8
scales for one distribution; the robot feeds it another. Nothing throws. Nothing
logs. The network produces a confident, plausible, wrong action — and you spend
three days suspecting the reward function.

This class of bug has a name in this repo: a **contract mismatch**. `spec.py` is
the structural answer to it.

## One file owns every number

`sim2edge/spec.py` (546 lines) is a tree of frozen dataclasses. Nothing else in
the repository is allowed to write down a task constant.

```
NavSpec                                      spec.py:432
├── obs: ObservationSpec                     spec.py:160
│   ├── depth: DepthSpec                     spec.py:46   96x160, 0.3-6.0 m
│   ├── lidar: LidarSpec                     spec.py:80   180 bins, 0.15-8.0 m
│   ├── goal: GoalSpec                       spec.py:107  3 floats, max 8 m
│   ├── last_action: LastActionSpec          spec.py:126  2 floats
│   └── privileged: PrivilegedSpec           spec.py:140  8 floats, critic only
├── action: ActionSpec                       spec.py:188  v, omega, rate limits
├── reward: RewardSpec                       spec.py:234  five coefficients
├── randomization: RandomizationSpec         spec.py:264  ~18 ranges
├── robot: RobotSpec                         spec.py:352
├── room: RoomSpec                           spec.py:376
└── safety: SafetySpec                       spec.py:405
```

Read the module docstring at `spec.py:1-28`. The key constraint:

> **This module is deliberately stdlib-only.** No numpy, no torch, no simulator
> imports.

That is not fastidiousness. The file has to import cleanly inside the Isaac Lab
CUDA stack, inside the board's pinned ExecuTorch venv (which has no torchvision
and a minimal dependency set), and inside a standalone code generator. Any import
that is heavy or platform-specific would break one of those three.

## Frozen dataclasses, and why

```python
@dataclass(frozen=True)
class LidarSpec:
    bins: int = 180
    min_m: float = 0.15
    max_m: float = 8.0
    invalid_fill: float = 1.0
    angle_min: float = -math.pi
    angle_max: float = math.pi
```

`frozen=True` means assignment raises. You cannot write
`spec.obs.lidar.max_m = 10.0` halfway through a training script and have the
change silently escape into a checkpoint. To vary something you construct a new
spec, which makes the variation explicit and — as you'll see next — visible in
the fingerprint.

## The fingerprint

`NavSpec.fingerprint()` at `spec.py:469` hashes the entire contract to a short
hex string.

```bash
python -c "from sim2edge.spec import DEFAULT_SPEC; print(DEFAULT_SPEC.fingerprint())"
```

```
da7943e44905b1c5
```

That string is the load-bearing part of the whole design. It gets:

1. **written into every training checkpoint.** `scripts/train.py:74-78` refuses
   to initialize PPO from a BC checkpoint whose fingerprint doesn't match:

   ```python
   got = ckpt.get("spec_fingerprint")
   if got and got != spec.fingerprint():
       raise SystemExit(
           f"ERROR: {args.init} was trained under spec fingerprint {got!r}, "
           f"current spec is {spec.fingerprint()!r}.")
   ```

2. **written into every exported model's JSON sidecar.** `tools/export_common.py:151`,
   `write_sidecar()`.

3. **compiled into the C++ ROS node**, which refuses to load a `.pte` whose
   sidecar disagrees.

So a change to any constant propagates as a *loud* failure at three checkpoints,
rather than as a silent behavioural change.

### Watch it work

```bash
python -c "
from sim2edge.spec import NavSpec, LidarSpec, ObservationSpec, DEFAULT_SPEC
print('default: ', DEFAULT_SPEC.fingerprint())
tweaked = NavSpec(obs=ObservationSpec(lidar=LidarSpec(max_m=10.0)))
print('max_m=10:', tweaked.fingerprint())
"
```

The two differ. One constant moved and the identity of the whole contract
changed, which is exactly the point.

## The observation layout

`ObservationSpec` at `spec.py:160` also fixes the *order* of the actor's inputs:

```python
actor_inputs: Tuple[str, ...] = ("depth", "lidar", "goal", "last_action")
```

with the comment at `spec.py:168-170`:

> Actor input order. This is also the positional order of the exported graph's
> `forward()` arguments, and therefore the order the C++ node must bind input
> tensors in. Changing it changes the fingerprint.

```bash
python -c "
from sim2edge.spec import DEFAULT_SPEC as S
for name, shape in S.obs.shapes().items():
    print(f'{name:12s} {shape}')
print()
print('actor inputs, in graph order:', S.obs.actor_inputs)
print('actor input shapes:          ', S.actor_input_shapes)
"
```

```
depth        (1, 96, 160)
lidar        (180,)
goal         (3,)
last_action  (2,)
privileged   (8,)

actor inputs, in graph order: ('depth', 'lidar', 'goal', 'last_action')
actor input shapes:           ((1, 96, 160), (180,), (3,), (2,))
```

Note `actor_input_shapes` is a **property**, not a method — `S.actor_input_shapes`,
no parentheses. (`spec.py:490`.)

`privileged` is absent from `actor_inputs`. That is deliberate and is the subject
of chapter 08: the critic gets ground-truth information the actor cannot have,
because the critic never ships.

## Normalize / denormalize

`ActionSpec` carries the conversion between physical units and the `[-1, 1]` range
the network works in (`spec.py:216-226`):

```python
def normalize(self, v, omega):
    v_n = 2.0 * (v - self.v_min) / (self.v_max - self.v_min) - 1.0
    w_n = omega / self.omega_max
    return (_clip(v_n, -1.0, 1.0), _clip(w_n, -1.0, 1.0))
```

```bash
python -c "
from sim2edge.spec import DEFAULT_SPEC as S
a = S.action
for v, w in [(0.0, 0.0), (0.15, 0.0), (0.30, -1.5), (0.30, 1.5)]:
    n = a.normalize(v, w)
    print(f'({v:4.2f}, {w:5.2f}) -> ({n[0]:5.2f}, {n[1]:5.2f}) -> {a.denormalize(*n)}')
"
```

```
(0.00,  0.00) -> (-1.00,  0.00) -> (0.0, 0.0)
(0.15,  0.00) -> ( 0.00,  0.00) -> (0.15, 0.0)
(0.30, -1.50) -> ( 1.00, -1.00) -> (0.3, -1.5)
(0.30,  1.50) -> ( 1.00,  1.00) -> (0.3, 1.5)
```

Notice that `v = 0` maps to `-1`, not `0`. Linear velocity has no negative range,
so the full `[-1, 1]` output of a `tanh` is used for `[0, v_max]`. This matters at
INT8: wasting half the output range would halve the effective resolution on the
only axis that moves the robot forward.

## The noise-tier trick

`RandomizationSpec.at_scale()` at `spec.py:313` returns a copy with every range
rescaled, so the evaluation harness can run one policy at clean / deployment /
2×-deployment noise (chapter 11) without duplicating a single constant.

```bash
python -c "
from sim2edge.spec import DEFAULT_SPEC as S
r = S.randomization
for s in (0.0, 1.0, 2.0):
    t = r.at_scale(s)
    print(f'scale {s}: lidar noise {t.lidar_range_noise_m}, delay {t.action_delay_steps}, enabled={t.enabled}')
"
```

```
scale 0.0: lidar noise (0.0, 0.0), delay (2, 2), enabled=False
scale 1.0: lidar noise (0.0, 0.03), delay (1, 3), enabled=True
scale 2.0: lidar noise (0.0, 0.06), delay (0, 4), enabled=True
```

The scaling rule is not uniform, and `_scale_range` at `spec.py:330` is worth
reading because the class docstring above it describes it loosely as scaling
"about their low end," which is only true for one of three cases:

```python
if lo < 0.0 < hi:      out = (lo * scale, hi * scale)          # about zero
elif lo == 0.0:        out = (0.0, hi * scale)                 # about the low end
else:                  mid = 0.5 * (lo + hi)                   # about the nominal
                       out = (mid + (lo-mid)*scale, mid + (hi-mid)*scale)
```

That is why lidar noise `(0.0, 0.03)` collapses to `(0.0, 0.0)` at scale 0 — it is
one-sided — while action delay `(1, 3)` collapses to `(2, 2)`, its nominal
midpoint, not to 1. A "clean" robot still has 2 steps of latency, because latency
is a property of the system, not a defect. Symmetric ranges like a `±1°` mount
offset scale about zero.

`enabled` also goes `False` at scale 0 (`spec.py:320`).

## Serialization

`NavSpec` round-trips through JSON (`spec.py:444-467`), which is how
`configs/spec.json` is produced and how the C++ header generator reads the
contract.

```bash
python -c "
from sim2edge.spec import NavSpec, DEFAULT_SPEC
j = DEFAULT_SPEC.to_json()
print(j[:200], '...')
print('round-trips:', NavSpec.from_json(j).fingerprint() == DEFAULT_SPEC.fingerprint())
"
```

The C++ header is generated from it, and CI can assert the generated file is not
stale:

```bash
python tools/gen_spec_header.py --check
```

```
Generated files are up to date.
```

That command is the third enforcement mechanism: a contract change cannot land in
the training code and be forgotten on the robot, because the header check fails.

## What the rebuild plan says about this file

`REBUILD_PLAN.md:30` proposes cutting `spec.py` entirely:

> Keep the *numbers* — they're reasonable. Throw away the schema layer. A
> dataclass or a plain module of constants is fine.

That is a fair critique of the *size* — 546 lines of dataclasses, `from_dict`
reconstruction, and type-hint walking is a lot of machinery. But notice what it
keeps: the numbers, and (at `REBUILD_PLAN.md:32-36`) the export sidecar with a
hash checked at load. The idea survives the simplification; the schema layer is
what goes.

Worth forming your own view on by the end of chapter 13.

## Check yourself

- Why must `spec.py` avoid importing numpy?
- If you change `LidarSpec.bins` from 180 to 360, name three things that break
  and one thing that catches the breakage.
- `invalid_fill = 1.0` for both lidar and depth. What would go wrong with `0.0`?
  (The answer is in `spec.py:92-93` and again in `spec.py:60-63` — and it is a bug
  that actually happened here; see chapter 15.)

→ [03 — Preprocessing](03-preprocessing.md)
