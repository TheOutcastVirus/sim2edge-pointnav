# sim2edge-ventuno

Goal-conditioned robot navigation trained in simulation and deployed to the
edge: **Isaac Lab → BC/PPO → ExecuTorch on a Qualcomm QCS8275 (Ventuno)**,
running a TurtleBot4-class robot with a spinning lidar and IMU, no odometry.

## Pipeline

1. **Simulate & collect** — Isaac Lab environments (`isaaclab_env/`) generate
   diverse layouts and an oracle policy produces behaviour-cloning data
   (`scripts/collect_bc.py`).
2. **Train** — behaviour cloning, PPO fine-tuning, and a lidar/IMU motion
   belief for odometry-free navigation (`scripts/train_bc.py`,
   `train_ppo.py`, `train_belief.py`, `train_belief_ppo.py`).
3. **Export & lower** — the trained actor is exported to a portable program
   and quantized/lowered to run on the QCS8275's Hexagon NPU via QNN/ExecuTorch
   (`scripts/export_actor.py`, `scripts/lower_qnn.py`).
4. **Deploy** — a C++ ROS 2 stack (`ros2_ws/`) runs the lowered model plus a
   classical CPU localizer on the robot, closing the loop with lidar and IMU.

## Layout

```
sim2edge/        simulator-free core: spec, grids, rooms, oracle,
                 preprocess, noise, dataset, localization
isaaclab_env/    Isaac Lab adapter: batch training env + interactive viewer
scripts/         collect / train / eval / export entry points
tools/           asset conversion, replay export, diagnostic probes
ros2_ws/         the deployed robot: C++ ROS 2 workspace (shared with the board)
tests/           simulator-free unit tests
docs/RUNBOOK.md  exact commands for every phase above
docs/daily/      engineering log
```

## Getting started

See `docs/RUNBOOK.md` for setup and the full command sequence from data
collection through running the physical robot.
