# 00 — Start here

This is a guided tour of `sim2edge-ventuno`, built so that each idea rests on the
one before it. It does not start from "deploy a quantized policy to an NPU." It
starts from "what is the robot actually being asked to do," and gets there.

## What the project does, in one paragraph

A Turtlebot 4 Lite has had its compute swapped for an Arduino Ventuno Q — a board
with a Qualcomm QCS8275 and a Hexagon HTP V75 neural accelerator. We want the
robot to drive to a coordinate while avoiding obstacles, using only its own lidar,
depth camera, and wheel odometry. The policy that decides where to drive is a
neural network trained entirely in simulation, quantized to 8-bit integers, and
executed on the NPU through ExecuTorch's QNN backend. Nothing about the robot's
behaviour is programmed by hand except a small classical safety layer that can
override it.

## How to read this

In order. Each chapter has three parts:

- **Concepts** — the idea, and why this project made the choice it made.
- **In the code** — where it lives, with file and line references you can open.
- **Try it** — commands and snippets that actually run, with the output I got.

The chapters are cumulative. Chapter 9 assumes you did chapter 5.

| # | Chapter | What you'll understand after it |
|---|---|---|
| 01 | [The problem](01-the-problem.md) | Why this is an RL problem, and the shape of the whole pipeline |
| 02 | [The contract](02-the-contract.md) | `spec.py`, fingerprints, and why one file owns every constant |
| 03 | [Preprocessing](03-preprocessing.md) | Turning raw sensor messages into a fixed-size float vector |
| 04 | [Rooms and grids](04-rooms-and-grids.md) | Procedural layouts, occupancy grids, rasterization |
| 05 | [The oracle](05-the-oracle.md) | Distance fields, shortest paths, pure pursuit — the classical baseline |
| 06 | [The environment](06-the-environment.md) | The RL loop: observation, action, reward, termination |
| 07 | [Sensors and randomization](07-sensors-and-randomization.md) | Ray casting, and how sim-to-real transfer is bought |
| 08 | [The policy network](08-the-policy-network.md) | Encoders, actor-critic, and designing backward from the NPU |
| 09 | [Behaviour cloning](09-behaviour-cloning.md) | Learning from the oracle, and why plain BC fails |
| 10 | [PPO](10-ppo.md) | Policy gradients, GAE, and the BC→PPO handoff |
| 11 | [Evaluation](11-evaluation.md) | Success rate, SPL, noise tiers, safety interventions |
| 12 | [Isaac Lab](12-isaac-lab.md) | Scaling from 1 environment to 4096 on a GPU |
| 13 | [Export and quantization](13-export-and-quantization.md) | `torch.export`, INT8 PTQ, ExecuTorch, QNN delegation |
| 14 | [On the robot](14-on-the-robot.md) | The ROS 2 node, the safety layer, and the parity gate |
| 15 | [Where it breaks](15-where-it-breaks.md) | Honest status, known defects, and what to build next |

## Setup

Everything in chapters 01–11 and 13 runs on CPU with numpy and torch. Chapter 12
needs Isaac Lab and an NVIDIA GPU. Chapter 14 needs the board.

Run commands from the repository root:

```bash
cd ~/Documents/Qualcomm/sim2edge-ventuno
python -m pytest -q          # 75 tests, ~1 minute, no GPU
```

`python -c "..."` snippets work from the repo root because Python puts the
current directory on `sys.path`. If you run a *script* from elsewhere, set
`PYTHONPATH` to the repo root instead.

> **Where to run things.** Training and evaluation belong on the GPU host
> (`dronelab12-4`), not on a laptop. The chapters flag which commands are cheap
> enough to run anywhere.

### A note on the local Python environment

On a machine where SciPy was built against NumPy 1.x but NumPy 2.x is installed,
you will see a wall of `_ARRAY_API not found` warnings. This is not a bug in this
repo — `sim2edge/oracle/grid.py:25-43` catches it deliberately and falls back to
pure-numpy implementations of the distance transform and Dijkstra. The comment
there explains why the `except` is broad:

> SciPy is optional. It is a large speedup where it works, but it is caught
> broadly rather than on ImportError alone: a SciPy built against a different
> NumPy ABI raises AttributeError at import time, and dataset generation must
> degrade to the numpy fallbacks instead of dying. The fallbacks below are
> exact, not approximations, so results do not depend on which path ran.

Results are identical either way; only speed changes.

## The one-minute map of the repository

```
sim2edge/            the platform-agnostic core — numpy and torch only
├── spec.py          every constant. the contract. read this first.
├── preprocess.py    raw sensor messages -> policy input
├── rooms.py         procedural layout generator
├── sensors.py       simulated lidar + depth
├── randomization.py per-episode domain randomization
├── safety.py        classical safety layer (Python twin of the C++ node)
├── oracle/          occupancy grid, geodesic distance field, pure-pursuit expert
├── envs/            the portable numpy environment
├── models/          the policy network
├── training/        behaviour cloning, DAgger, PPO
└── eval/            evaluation harness

isaaclab_env/        the GPU training environment (Isaac Lab)
tools/               export, calibration, parity, benchmarking
ros_ws/              the C++ ROS 2 nodes that run on the robot
tests/               75 tests, no GPU or ROS required
docs/                deployment, architecture, domain randomization, host setup
```

## Two honest warnings before you start

1. **Most of this code has never run against real hardware or a real simulator.**
   The numpy pipeline works and is tested. Isaac Lab has never been executed. The
   `.pte` export has never touched a QNN device. Chapter 15 is a full inventory.
   Treat the code as a well-documented design you are auditing, not as a working
   system you are learning to operate.

2. **The repository is larger than the problem requires.** `REBUILD_PLAN.md`
   argues the same demo fits in ~800 lines instead of ~7,000, and it is mostly
   right. Understanding why each piece exists is exactly what tells you which
   pieces you would keep. Chapter 15 revisits this with the whole picture in
   view.

→ [01 — The problem](01-the-problem.md)
