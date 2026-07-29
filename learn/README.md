# learn/ — a guided tour of sim2edge-ventuno

A 16-chapter curriculum that builds up to the deployment story instead of starting
from it. Concepts first, then where they live in this code, then commands you can
actually run.

**Start with [`00-start-here.md`](00-start-here.md).**

| # | Chapter |
|---|---|
| 00 | [Start here](00-start-here.md) — orientation, setup, repo map |
| 01 | [The problem](01-the-problem.md) — point-goal nav, the MDP, the pipeline shape |
| 02 | [The contract](02-the-contract.md) — `spec.py` and the fingerprint |
| 03 | [Preprocessing](03-preprocessing.md) — sensors → float vectors |
| 04 | [Rooms and grids](04-rooms-and-grids.md) — layouts, occupancy, geodesic fields |
| 05 | [The oracle](05-the-oracle.md) — planning and pure pursuit |
| 06 | [The environment](06-the-environment.md) — the RL loop and the reward |
| 07 | [Sensors and randomization](07-sensors-and-randomization.md) — ray casting, sim-to-real |
| 08 | [The policy network](08-the-policy-network.md) — designed backward from the NPU |
| 09 | [Behaviour cloning](09-behaviour-cloning.md) — learning from the oracle, and why it fails |
| 10 | [PPO](10-ppo.md) — policy gradients and the BC handoff |
| 11 | [Evaluation](11-evaluation.md) — SR, SPL, noise tiers, interventions |
| 12 | [Isaac Lab](12-isaac-lab.md) — scaling to thousands of environments |
| 13 | [Export and quantization](13-export-and-quantization.md) — `torch.export`, INT8, QNN |
| 14 | [On the robot](14-on-the-robot.md) — ROS 2, the safety layer, parity |
| 15 | [Where it breaks](15-where-it-breaks.md) — honest status and what to do next |

## Running the examples

Short snippets (chapters 01–08, 11, 13) run from the repo root on any machine with
numpy and torch. Training and evaluation belong on the GPU host:

```bash
ssh vkommera@dronelab12-4
cd ~/Documents/Qualcomm/sim2edge-ventuno
.venv/bin/python -m pytest -q
```

Every command and output block in these chapters was executed, not invented. Where
a component has never been run — Isaac Lab, the QNN export, the robot — the
chapter says so plainly rather than presenting a plausible transcript.
