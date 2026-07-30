"""The expert that drives during collection. Batched torch, no Python loop over
environments.

Two stages, both branch-free on device:

1. **Where to go** -- walk the precomputed flow field ``LOOKAHEAD_CELLS`` steps.
   All the geometry (octile costs, the corner-cut rule, unreachable regions) was
   resolved once per layout in :func:`sim2edge.grids.flow_field`, so this is a
   chain of gathers, not a search.
2. **How to get there** -- a lookahead controller producing physical
   ``(v, omega)``, clamped to the Create3's real limits and rate-limited against
   the previous command.

The expert is held to the robot's actual limits on purpose. An oracle that
demonstrates commands the base cannot track produces a policy that faithfully
imitates commands the base cannot track, and the resulting sim-to-real error
looks like a perception problem while being a dynamics one.

The labels this produces are the **commanded** ``(v, omega)`` -- what the Twist
carries -- never the achieved velocity read back from physics. Training on
achieved velocity teaches the policy to output a value the actuator will lag a
second time, a systematic underactuation that is invisible in sim.
"""

from __future__ import annotations

import torch

from .spec import ACCEL_MAX, ALPHA_MAX, DT, OMEGA_MAX, V_MAX
from .preprocess import wrap_angle

__all__ = ["LayoutPool", "oracle_action", "perturb_action"]

#: How far ahead on the path to aim, in metres. Short enough to track corners,
#: long enough that grid quantization does not make the command chatter.
LOOKAHEAD_M = 0.55

#: Above this bearing error the robot turns in place. A differential drive can
#: spin on the spot, and arcing toward a target behind you is both slower and a
#: much worse demonstration than simply facing it first.
TURN_IN_PLACE_RAD = 0.7

#: Proportional gain on bearing error. With the speed taper below this is a
#: well-damped pursuit controller; a full curvature formulation buys smoother
#: arcs and costs a special case at alpha near pi.
K_ANGULAR = 2.2

#: Start braking this far from the goal so the final command is not a step to
#: zero, which the rate limiter would stretch over several ticks anyway.
BRAKE_DIST_M = 0.6


class LayoutPool:
    """Layouts stacked into device tensors, built once and indexed thereafter.

    Rooms have different arena sizes, so every grid is zero-padded to the
    largest. Padding is *blocked* space with a self-looping flow, so an
    out-of-bounds index is inert rather than wrapping into another room.

    A pool of a few hundred layouts tiled across thousands of environments is
    the right trade: diversity comes from pool size times start/goal sampling,
    which is cheap, not from unique geometry per environment, which costs a
    proportional number of USD prims.
    """

    def __init__(self, rooms, device="cpu"):
        h = max(r.free.shape[0] for r in rooms)
        w = max(r.free.shape[1] for r in rooms)
        n = len(rooms)

        flow = torch.zeros((n, h * w), dtype=torch.long)
        free = torch.zeros((n, h, w), dtype=torch.bool)
        body = torch.zeros((n, h, w), dtype=torch.bool)
        for i, room in enumerate(rooms):
            rh, rw = room.free.shape
            # Re-index the flow field from the room's own width to the padded
            # width. Copying the raw values would point every step into the
            # wrong row of the padded grid.
            src = torch.from_numpy(room.flow.astype("int64"))
            rows, cols = src // rw, src % rw
            padded = torch.arange(h * w).reshape(h, w)
            padded[:rh, :rw] = rows * w + cols
            flow[i] = padded.reshape(-1)
            free[i, :rh, :rw] = torch.from_numpy(room.free)
            body[i, :rh, :rw] = torch.from_numpy(room.body_free)

        self.h, self.w, self.res = h, w, float(rooms[0].res)
        self.flow = flow.to(device)
        self.free = free.to(device)
        self.body_free = body.to(device)
        self.goal = torch.stack(
            [torch.from_numpy(r.goal) for r in rooms]).to(device)

    def is_free(self, layout: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """(N,) True where the robot centre is legal, i.e. not in collision.

        Thresholded at the **body** radius, not the planning radius. The two
        differ by 16 mm of deliberate margin, and testing against the planning
        radius reported collisions where nothing was touching -- which, with a
        sticky per-episode flag, discarded a large share of good episodes.
        """
        col = torch.clamp((xy[:, 0] / self.res).long(), 0, self.w - 1)
        row = torch.clamp((xy[:, 1] / self.res).long(), 0, self.h - 1)
        return self.body_free[layout, row, col]

    def to(self, device) -> "LayoutPool":
        self.flow = self.flow.to(device)
        self.free = self.free.to(device)
        self.body_free = self.body_free.to(device)
        self.goal = self.goal.to(device)
        return self


def _lookahead_point(pool: LayoutPool, layout: torch.Tensor,
                     xy: torch.Tensor, steps: int) -> torch.Tensor:
    """Walk the flow field ``steps`` times; return the world point landed on.

    Self-loops at the goal and in unreachable regions make this safe to run for
    a fixed number of iterations with no termination check -- which is exactly
    what lets it stay branch-free and batched.
    """
    col = torch.clamp((xy[:, 0] / pool.res).long(), 0, pool.w - 1)
    row = torch.clamp((xy[:, 1] / pool.res).long(), 0, pool.h - 1)
    idx = row * pool.w + col

    # Index both axes at once. Slicing `pool.flow[layout]` first would
    # materialize a full (N, H*W) copy of the fields -- hundreds of megabytes
    # for a few thousand environments -- to then read one element per row.
    for _ in range(steps):
        idx = pool.flow[layout, idx]

    r, c = idx // pool.w, idx % pool.w
    return torch.stack([(c.to(xy.dtype) + 0.5) * pool.res,
                        (r.to(xy.dtype) + 0.5) * pool.res], dim=1)


def oracle_action(pool: LayoutPool, layout: torch.Tensor, pose: torch.Tensor,
                  prev_action: torch.Tensor) -> torch.Tensor:
    """``(N, 2)`` physical ``(v, omega)`` for one control tick.

    ``pose`` is ``(N, 3)`` world x, y, yaw. ``prev_action`` is the command
    issued last tick, which the rate limiter needs -- pass zeros on reset.
    """
    steps = max(1, int(round(LOOKAHEAD_M / pool.res)))
    target = _lookahead_point(pool, layout, pose[:, :2], steps)

    d = target - pose[:, :2]
    alpha = wrap_angle(torch.atan2(d[:, 1], d[:, 0]) - pose[:, 2])

    goal_dist = torch.linalg.vector_norm(pool.goal[layout] - pose[:, :2], dim=1)

    # Forward speed falls off with bearing error and again near the goal. The
    # cos taper is what makes a single proportional controller behave like
    # pursuit: hard turns automatically slow down instead of overshooting.
    v = V_MAX * torch.cos(alpha).clamp(min=0.0)
    v = torch.minimum(v, V_MAX * (goal_dist / BRAKE_DIST_M).clamp(max=1.0))
    omega = (K_ANGULAR * alpha).clamp(-OMEGA_MAX, OMEGA_MAX)

    # Facing away from the target: spin first, drive after. Arcing backwards
    # around is both slower and a demonstration worth not imitating.
    spin = alpha.abs() > TURN_IN_PLACE_RAD
    v = torch.where(spin, torch.zeros_like(v), v)

    # Rate limits last, so nothing above can emit a step the base cannot follow.
    v = torch.clamp(v, prev_action[:, 0] - ACCEL_MAX * DT,
                    prev_action[:, 0] + ACCEL_MAX * DT)
    omega = torch.clamp(omega, prev_action[:, 1] - ALPHA_MAX * DT,
                        prev_action[:, 1] + ALPHA_MAX * DT)
    return torch.stack([v.clamp(0.0, V_MAX), omega.clamp(-OMEGA_MAX, OMEGA_MAX)],
                       dim=1)


def perturb_action(action: torch.Tensor, sigma: float,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Add exploration noise to the command that gets *executed*.

    The label recorded for a state stays the oracle's unperturbed command; only
    what the robot actually does is perturbed. That is the whole trick: it walks
    the robot into a tube of states around the expert path and labels each one
    with what the expert would do *there*.

    Without it, every state in the dataset is one the oracle put itself in, the
    policy has no idea what to do once it drifts, and drift is guaranteed at
    deployment. More expert data does not fix that -- it is a distribution
    problem, not a sample-count one.

    Scaled by each axis's own limit so one sigma means the same thing for both.
    """
    if sigma <= 0.0:
        return action
    scale = torch.tensor([V_MAX, OMEGA_MAX], device=action.device,
                         dtype=action.dtype)
    noise = torch.randn(action.shape, generator=generator, device=action.device,
                        dtype=action.dtype) * sigma * scale
    out = action + noise
    return torch.stack([out[:, 0].clamp(0.0, V_MAX),
                        out[:, 1].clamp(-OMEGA_MAX, OMEGA_MAX)], dim=1)
