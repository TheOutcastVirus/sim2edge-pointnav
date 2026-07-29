# 13 — Export and quantization

A trained `NavPolicy` is a Python object holding float32 tensors. The robot needs
a self-contained binary running INT8 arithmetic on a Hexagon NPU. This chapter is
the bridge.

Four files: `tools/export_common.py` (218), `tools/export_cpu.py` (124),
`tools/export_qnn.py` (216), `tools/smoke_export.py` (173).

## The stages

```
NavPolicy (training)
     │  .export_module()          — actor mean path only
     ▼
ActorMean (nn.Module)
     │  torch.export.export()     — traced, static graph
     ▼
ExportedProgram
     │  quantize (PTQ, 8a8w)      — calibrated on real observations
     ▼
Quantized graph
     │  to_edge_transform_and_lower(QnnPartitioner)
     ▼
policy.pte  +  policy.json (sidecar)
```

## Stage 1 — export only the actor mean

`export_module()` (`nav_policy.py:327`) returns `ActorMean` (`:342`). What is left
behind matters as much as what ships:

- **the Gaussian head** — exploration noise is a training device; the deployed
  policy is deterministic
- **the critic** — 314 k parameters that answer a question nobody asks on the robot
- **`log_std`** — part of the distribution, not the mean path
- **the privileged input** — does not exist on a robot

`freeze_norm()` (`:309`) puts BatchNorm into eval mode with frozen statistics
first, so no running-stat update sneaks into the graph.

`REBUILD_PLAN.md:209-214` adds a warning that applies the moment you use rsl_rl:

> If `rsl_rl` wraps your policy with an observation normalizer, either fold its
> running mean/std into the first layer or disable it and normalize in
> `preprocess.py`. Pick one; a mismatched normalizer is the exact silent-wrong-action
> failure the old README warns about.

Two normalizations in a pipeline is the same bug as two preprocessing
implementations, wearing a different hat.

## Stage 2 — `torch.export`

`torch.export.export` traces the module into a static graph with no Python left in
it. It is strict: data-dependent control flow, dynamic shapes, and side effects
all fail here. Chapter 08's architecture was designed so that they don't.

You verified this in chapter 08 — 12 operators, all on the proven list. `smoke_export.py`
formalizes it. `PROVEN_OPS` at `smoke_export.py:47`:

```python
PROVEN_OPS = {
    "aten::conv1d", "aten::conv2d", "aten::linear", "aten::batch_norm",
    "aten::relu", "aten::relu_", "aten::tanh", "aten::cat", "aten::flatten",
    "aten::slice", "aten::add", "aten::mul", "aten::unsqueeze",
}
```

Stage 1 of the smoke test (`:91-99`) also checks that the *exported* graph produces
the same output as the eager module to `atol=1e-6` — catching tracing bugs before
quantization can obscure them.

```bash
ssh vkommera@dronelab12-4
cd ~/Documents/Qualcomm/sim2edge-ventuno
.venv/bin/python tools/smoke_export.py --skip-qnn --out-dir /tmp/smoke
```

```
actor input shapes [(1, 96, 160), (180,), (3,), (2,)]
QNN_SDK_ROOT       (unset)

── 1. torch.export with 4 inputs ───────────────────────────────
   4-input signature accepted; output (1, 2) = (v=0.1500, omega=+0.0000)
   PASS
── 2. operator set vs HTP-proven ops ───────────────────────────
   12 distinct operators, all previously delegated
   PASS
── 3. XNNPACK lowering (Gate-A fallback) ───────────────────────
   FAIL: ModuleNotFoundError: No module named 'executorch'
── 5. latency sanity ───────────────────────────────────────────
   torch eager fp32: 0.62 ms/inference (budget 50 ms)
   PASS
```

Stages 1, 2 and 5 run anywhere; 3 needs ExecuTorch (not installed on the training
host — it is a board-side dependency); 4 needs QNN on the board. The `(v=0.1500,
omega=+0.0000)` is chapter 08's untrained-policy output, confirming the exported
graph matches the eager module.

**0.62 ms eager float32 on CPU**, against a 50 ms budget. Before any quantization,
before any NPU. That number is the strongest available evidence that the C++
rewrite argument in chapter 14 is unnecessary — and note the script's own caution
that eager torch is a sanity floor, not the real measurement.

## Stage 3 — quantization

**Post-training quantization at 8a8w**: 8-bit activations, 8-bit weights. Every
float32 tensor becomes `int8 × scale + zero_point`.

The compression is 4×; the speedup on an NPU is much more than 4×, because integer
MAC units are what the hardware is built out of. Running float on an HTP is
possible and slow.

The whole difficulty is choosing the scales. For weights it is easy — you have
them, take the range. For **activations** you need to know the distribution of
values flowing through each layer, and that depends on the input data. So you run
a few hundred representative inputs through the model and record the ranges. That
is **calibration**.

### Calibration is where projects die

`REBUILD_PLAN.md:218-219`:

> **Calibration data must come from the same `preprocess.py` the robot runs.**
> Dump a few thousand observations from a sim rollout and calibrate on those.

Suppose you calibrate on observations where lidar values span `[0.3, 0.9]`, and on
the robot they span `[0.0, 1.0]`. The quantizer allocated all 256 levels to
`[0.3, 0.9]`. Real inputs clip at both ends. The network sees a flattened world and
outputs confident nonsense.

Nothing errors. Nothing logs. The robot drives into a wall with total conviction.

`tools/calib_dataset.py` (168 lines) builds the set:

```bash
.venv/bin/python tools/calib_dataset.py \
    --weights runs/tutorial_bc/bc_final.pt \
    --episodes 40 --stride 5 --lidar-only \
    --output calib/nav_calib.npz
```

`collect_sim` at `:46` rolls out episodes and pulls `env.raw_sensors()` — the
**raw** sensor dict (`reference_env.py:354`) — then runs it through the same
`Preprocessor` the robot uses. Not the observation the env already built: the raw
messages, reprocessed. That is what makes the guarantee real.

`--stride 5` samples every fifth step, because consecutive steps at 20 Hz are
nearly identical and a calibration set of duplicates covers less of the
distribution than its size suggests.

`--real-fraction 0.35` (`:111`) mixes in observations from a real rosbag via
`tools/bag_to_obs.py`. Sim observations cover the geometry; real ones cover the
sensor artifacts sim does not model. Best available answer short of a full data
collection.

## Stage 4 — lowering to QNN

`tools/export_qnn.py` runs **on the board**, inside the pinned ExecuTorch venv.
Nothing is cross-compiled.

Its docstring (`:13-17`) contains the kind of institutional knowledge that only
comes from having done it:

> `--soc-model QCS8300` is not a typo. ExecuTorch has no QCS8275 entry, so the
> board bring-up patches a fake `QCS8300` chipset enum that maps to HTP V75. That
> patch lives in `/opt/executorch` and will break on an ExecuTorch upgrade — pin
> the working commit and do not upgrade mid-project.

Two lowering flows (`:18-27`): a helper that mirrors the reference scripts'
one-call `build_executorch_binary`, and an explicit
`torch.export → to_edge_transform_and_lower` path. The explicit flow exists because
the reference exports were all single-tensor while this policy takes four:

> If the helper fights the 4-input signature, switch flows rather than reshaping
> the model — a model contorted to fit a helper function is a model that no longer
> matches the C++ node.

That is a good rule generally. Never let tooling reshape your contract.

### Partial delegation is the failure to watch for

The QNN partitioner assigns subgraphs to the HTP. Anything it cannot handle stays
on CPU, with a memory copy at each boundary. A single unsupported op in the middle
of your network can split it into three partitions and cost more than the op saves.

This is why chapter 08 rejected circular padding and adaptive pooling. Verify:
confirm the whole graph lands on HTP and measure latency
(`REBUILD_PLAN.md:229-230`). Budget is 50 ms at 20 Hz; expect single-digit
milliseconds.

> **Note on the 5 Hz figure.** Reference repos on this board quote ~5 Hz for NPU
> inference. That number is wrong — the NPU achieves considerably more. Do not use
> it to justify architecture compromises or a C++ rewrite.

There is also a CPU path, `tools/export_cpu.py`, targeting XNNPACK. It is the
Gate-A fallback: if QNN lowering fights back, ship the XNNPACK `.pte` — the policy
may hold 20 Hz on CPU alone — and debug QNN in parallel rather than blocking
training on it.

## The sidecar

`write_sidecar` at `export_common.py:151` — the piece `REBUILD_PLAN.md:32-36`
explicitly keeps even while cutting almost everything else:

```json
{
  "model": "nav_policy_qnn.pte",
  "backend": "qnn",
  "quantization": "8a8w",
  "spec_version": "1.0.0",
  "spec_fingerprint": "da7943e44905b1c5",
  "actor_inputs": ["depth", "lidar", "goal", "last_action"],
  "input_shapes": [[1, 96, 160], [180], [3], [2]],
  "output": ["v_mps", "omega_radps"],
  "created": "..."
}
```

Twenty lines of code. The ROS node reads it at load time and refuses a `.pte`
whose fingerprint does not match the one compiled into it. The silent failure
becomes a loud one.

Note `"output": ["v_mps", "omega_radps"]` — SI units, because the squash is
in-graph (chapter 08). The node does not scale anything.

## Verifying the quantized model

Two levels, and you need both.

### Tensor level

`report_parity` at `export_common.py:185` compares action batches **relative to
each channel's range** (`:191-195`):

> Absolute MSE on `(v, omega)` is close to meaningless because the two channels
> have ranges differing by 10x. What matters operationally is the error as a
> fraction of each channel's usable range.

`v` spans 0.30 and `ω` spans 3.0. An MAE of 0.01 is 3% of the linear range and
0.3% of the angular one. Reporting one number for both hides which channel broke.

### Behavioural level

Tensor agreement is necessary, not sufficient. The gate that matters
(`IMPLEMENTATION_PLAN.md` §5):

> require action MSE small relative to action range and **sim SR drop <2%** when
> the quantized actions are replayed through the sim eval.

`ExecuTorchPolicy` (`eval/torch_policy.py:64`) implements the same `PolicyFn`
interface as `TorchPolicy`, so the chapter-11 harness runs the `.pte` directly:

```bash
.venv/bin/python tools/parity_check.py \
    --pte models/nav_policy_qnn.pte \
    --weights runs/tutorial_bc/bc_final.pt \
    --calibration calib/nav_calib.npz \
    --sim-episodes 200 --max-sr-drop 0.02
```

`check_quantization` at `parity_check.py:215`. A 1% success-rate drop is fine; 15%
means quantization ate the policy and you need QAT or a wider calibration set.

## Latency

```bash
.venv/bin/python tools/bench_latency.py --pte models/nav_policy_qnn.pte --iters 500
```

`tools/bench_latency.py` reports mean and p99. **Watch p99, not the mean.** At
20 Hz you have a 50 ms budget every tick; a 6 ms mean with a 60 ms tail means you
miss a control cycle occasionally, and on a robot that is a lurch.

## Try what you can today

Everything before QNN runs on the training host:

```bash
ssh vkommera@dronelab12-4
cd ~/Documents/Qualcomm/sim2edge-ventuno
.venv/bin/python tools/smoke_export.py --skip-qnn --out-dir /tmp/smoke
```

And the op check from chapter 08 is still the fastest way to see whether an
architecture change broke delegability — worth running after *every* model edit,
because finding out on the board costs an afternoon.

## Status

Per the README: export, calibration, parity and benchmark tooling are **complete
but untested against real QNN hardware**. The scripts encode real institutional
knowledge from two sibling projects on the same board, but no `.pte` from this
repository has ever run on a Hexagon NPU.

## Check yourself

- Why must calibration data come from `preprocess.py` rather than from the
  simulator's internal observation?
- Why does `report_parity` normalize by each channel's range?
- What is the practical cost of one unsupported operator in the middle of the
  graph?
- The `.pte` matches FP32 to 1e-4 on 1000 samples, but sim success rate drops from
  88% to 71%. What is your first hypothesis?
- Why is the action squash inside the exported graph rather than in the C++ node?

→ [14 — On the robot](14-on-the-robot.md)
