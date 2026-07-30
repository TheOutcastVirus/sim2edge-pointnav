# Runbook

How to run things. Rationale and history live in `docs/daily/`.

---

## Setup

Two Python environments, deliberately separate — Isaac Sim pins Python 3.11 and
torch 2.7, and the rest of the project has no reason to.

| | Path | Python | Used for |
|---|---|---|---|
| Project | `.venv` | 3.12 | Tests, layout pools, dataset, training |
| Isaac | `~/isaac-venv` | 3.11 | Anything importing `isaaclab` |

```bash
uv sync --extra dev                      # project venv
~/isaac-venv/bin/python -m pip install -e .   # make sim2edge importable from Isaac
sudo apt install ros-jazzy-turtlebot4-description ros-jazzy-xacro
```

Isaac Sim 5.1.0 / Isaac Lab 0.54.4, source install at
`~/Documents/Qualcomm/IsaacLab`. Moving that directory breaks all six editable
packages — re-run `./isaaclab.sh --install` from the new location, or symlink
instead.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

105 pass, 2 skipped without Isaac. No GPU or simulator required — keep it that way.

---

# Phase 1 — Behaviour cloning data

## 1. Build the robot asset (one-time)

```bash
~/isaac-venv/bin/python tools/convert_tb4.py --output assets/turtlebot4_lite.usd
```

~90 s, mostly Isaac booting. Skip if `assets/turtlebot4_lite.usd` exists. Expect
`Unresolved reference .../wheel_drop_*/visuals` warnings; they are cosmetic.

Add `--urdf-only` to expand the xacro without Isaac (fast, and activates two more
tests).

## 2. Warm the warp cache (one-time per warp version)

```bash
ls ~/.cache/warp/     # want a dir matching: python -c 'import warp; print(warp.config.version)'
```

If absent:

```bash
~/isaac-venv/bin/python scripts/collect_bc.py --num_envs 4 --num_layouts 8 --check
```

> Several minutes, **no output**, ~100% CPU on one core. This is warp compiling
> ray-cast kernels, not a hang — `SIGINT` gives no traceback because it is inside
> native code. **Do not interrupt:** a killed compile is discarded and the next
> run starts over.

## 3. Bring-up check

```bash
~/isaac-venv/bin/python scripts/collect_bc.py --num_envs 4 --gui --check
```

Asserts rays hit geometry, bins carry returns, robots spawn in free space,
layouts differ per environment, and the oracle is not pinned at max spin.

GUI: right-drag orbits, middle-drag pans, `F` frames a selected prim. Add
`--dump-scene` for env origins, camera pose and prim world positions when the
viewport is unreadable.

## 4. Collect

```bash
~/isaac-venv/bin/python scripts/collect_bc.py \
    --num_envs 512 --episodes 500000 --num_layouts 512 --seed 24680 \
    --sigma 0.15 --out data/overnight
```

Appends after any existing `bc_*.npz` in `--out`. Pre-collision prefixes are retained by default; add `--filter-failures` only for the legacy progress filter. `Ctrl-C` writes buffered
episodes and exits cleanly; a second `Ctrl-C` abandons them.

Progress line:

```
step  1200  kept  4120 (full 1840 trunc 1990 part 290)  dropped 31  | 33.4 ms/step (sim 30.9  rec 2.5)  | 7665 env-steps/s
```

- `full` reached the goal, `trunc` was cut before a graze, `part` ran out of clock
- `sim` is GPU, `rec` is the host recorder
- **yield is only meaningful once `kept + dropped` is several times `num_envs`** —
  below that only fast episodes have finished and it always reads high

Watch at startup: the recorder buffer ceiling is printed and RSS should plateau
there; `ms/step` should stay flat; no `foundLostPairsCapacity` error in the first
30 s.

### Parameters worth knowing

| Flag | Default | Notes |
|---|---|---|
| `--num_envs` | 256 | Raises throughput; **lowers** GPU utilisation, which is expected — the ceiling is a serial CPU section, not the GPU. Do not run multiple collectors on one GPU. |
| `--episodes` | 2000 | Retained episode prefixes, not attempts. Overshoots by at most one episode. |
| `--num_layouts` | 512 | Pool size. ~21 ms each, built once. |
| `--sigma` | 0.15 | Execution noise as a fraction of each axis limit. `0` collects pure on-policy expert data — the distribution BC fails on. |
| `--shard_steps` | 200000 | ~160 MB per shard. |
| `--filter-failures` | off | Legacy mode: drops short or low-progress prefixes. |

Memory scales as `num_envs × 900 × 720 × 2 B`: **5.4 GB at 4096, 10.9 GB at
8192**, plus a transient spike when many environments finish on the same step.

If PhysX reports `foundLostPairsCapacity`, raise
`gpu_found_lost_pairs_capacity` in `NavEnvCfg.sim` — the message states the
number it wants. Do not ignore it: dropped contacts mean robots pass through
geometry while the run still looks healthy.

## 5. Verify before training

```bash
.venv/bin/python scripts/verify_shard.py data/bc_000.npz
```

Runs without Isaac. Checks structure, that storage is raw metres and not
pre-binned, that labels sit inside the Create3's envelope, that perturbation is
live, that observations assemble correctly, and that noise is fresh per epoch.

Progress is reported as a quality warning rather than a hard failure: collision-truncated prefixes still contain valid oracle labels, but a high warning count means the drive dynamics need more work.

---

# Phase 2 — BC training

```python
from sim2edge.dataset import load_shard

shard = load_shard("data/bc_000.npz", device="cuda")
for obs, act in shard.batches(512, noise_scale=1.0):
    ...
```

Noise and binning happen at load, so two epochs over the same states show two
different corruptions. `noise_scale=0.0` is exactly clean and reproducible, which
is what evaluation baselines need. A fingerprint mismatch raises rather than
warns.

*(Training script not yet written.)*

---

# Phase 3 — PPO fine-tune

*(Not yet built. `_get_rewards` returns zeros; needs the geodesic-progress
reward.)*

---

# Phase 4 — Export and deploy

*(Not yet built. ExecuTorch export, INT8 calibration, C++ node, and the
preprocessing parity test.)*

---

## Common failures

| Symptom | Fix |
|---|---|
| Silent, minutes, ~100% CPU on one core | Warp JIT. One-time per version; let it finish. |
| `No module named 'pxr'` | Something under `isaaclab.sim` imported before `AppLauncher` ran. Import `isaaclab.app`, launch, *then* the rest. |
| `cannot import name 'AppLauncher' ... circular import` | A local file shadows a stdlib module Isaac imports — e.g. a scratch `inspect.py`. Rename it. |
| `PxGpuDynamicsMemoryConfig::foundLostPairsCapacity` | Raise `gpu_found_lost_pairs_capacity` in `NavEnvCfg.sim` to the number reported. |
| RSS climbs and `ms/step` creeps up | Recorder buffers should be a fixed ceiling printed at startup. If not, something reintroduced per-step allocation. |
| Every lidar bin reads 1.0 | Rays hit nothing — sensor targets not matching obstacle prims, or obstacles parked below the floor. |
| Viewport looks empty | Rooms span `(0,0)`–`(size)` from the env origin, so the camera must aim at the middle. `--gui` frames env 0 after reset; `--dump-scene` gives coordinates. |
| `ValueError: shard ... was collected under spec X` | `spec.py` changed after collection. Recollect, or check out the spec that produced the shard. |
| Yield looks great early, then collapses | Expected. Fast episodes finish first; wait until `kept + dropped` ≫ `num_envs`. |

## Repository layout

```
sim2edge/        simulator-free core: spec, grids, rooms, oracle,
                 preprocess, noise, dataset
isaaclab_env/    Isaac Lab adapter -- owns no task definition
scripts/         collect_bc.py, verify_shard.py
tools/           convert_tb4.py
tests/           100 tests, no GPU or simulator required
docs/daily/      engineering log
assets/, data/   generated; not in git
```

`archive/v1-numpy-pipeline` holds the previous build — numpy reference env, PPO,
eval harness, ExecuTorch export, ROS 2 C++ node — worth consulting when those
phases come back.
