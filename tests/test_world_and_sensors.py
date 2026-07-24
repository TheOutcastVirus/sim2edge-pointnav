"""Tests for layout generation, sensor simulation, and the oracle expert.

The theme here is the vertical structure of the world. The depth channel exists
in the observation because a 2D lidar cannot see an obstacle below its scan
plane and cannot tell that an overhang is not an obstacle. If the simulator did
not model that distinction, the depth channel would look useless in evaluation
and the Gate-C comparison would be measuring nothing.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim2edge.oracle.expert import PurePursuitExpert, plan_path, smooth_path  # noqa: E402
from sim2edge.oracle.grid import (  # noqa: E402
    UNREACHABLE,
    distance_field,
    obstacle_distance,
)
from sim2edge.rooms import Obstacle, Room, sample_room  # noqa: E402
from sim2edge.sensors import render_depth, simulate_lidar  # noqa: E402
from sim2edge.spec import DEFAULT_SPEC  # noqa: E402

S = DEFAULT_SPEC


def _scene(obstacles, arena=8.0):
    return Room(arena, arena, obstacles, (2.0, 4.0, 0.0), (7.0, 4.0), 0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Grid primitives
# ─────────────────────────────────────────────────────────────────────────────


def test_obstacle_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    occ = rng.random((60, 80)) < 0.02
    occ[0, 0] = True
    res = 0.05

    got = obstacle_distance(occ, res)
    rr, cc = np.nonzero(occ)
    yy, xx = np.mgrid[0:60, 0:80]
    ref = np.min(np.sqrt((yy[..., None] - rr) ** 2 + (xx[..., None] - cc) ** 2),
                 axis=-1) * res
    assert np.abs(ref - got).max() < 1e-4


def test_distance_field_is_zero_at_goal_and_grows_outward():
    free = np.ones((40, 40), dtype=bool)
    field = distance_field(free, (20, 20), 0.05)
    assert field[20, 20] == pytest.approx(0.0)
    assert field[20, 25] == pytest.approx(5 * 0.05, abs=1e-5)
    # Octile metric: a diagonal costs sqrt(2) per step.
    assert field[25, 25] == pytest.approx(5 * 0.05 * math.sqrt(2), abs=1e-5)


def test_distance_field_routes_around_a_wall():
    free = np.ones((40, 40), dtype=bool)
    free[10:30, 20] = False  # a wall with gaps at both ends
    field = distance_field(free, (20, 30), 0.05)
    detour = field[20, 10]
    straight = abs(30 - 10) * 0.05
    assert detour > straight  # it must go around, not through
    assert detour < UNREACHABLE * 0.5


def test_unreachable_cells_are_marked():
    free = np.ones((30, 30), dtype=bool)
    free[10, :] = False  # seal the arena in two
    field = distance_field(free, (20, 15), 0.05)
    assert field[5, 15] >= UNREACHABLE * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Room generation
# ─────────────────────────────────────────────────────────────────────────────


def test_sampled_rooms_are_feasible_and_in_range():
    for seed in range(25):
        room = sample_room(seed)
        assert room.is_feasible, f"seed {seed} produced an unreachable goal"
        assert room.optimal_path_length < UNREACHABLE * 0.5
        assert not room.in_collision(*room.start[:2]), "start is in collision"


def test_room_generation_is_deterministic_by_seed():
    a, b = sample_room(42), sample_room(42)
    assert a.arena_w == b.arena_w and a.start == b.start and a.goal == b.goal
    assert len(a.obstacles) == len(b.obstacles)


def test_layout_distribution_contains_all_three_height_classes():
    """Tall, low (under the lidar plane), and overhanging obstacles must all
    occur, or the depth-vs-lidar comparison measures nothing."""
    classes = {"tall": 0, "low": 0, "overhang": 0}
    for seed in range(30):
        for o in sample_room(seed).obstacles:
            if o.z_min > S.robot.height_m:
                classes["overhang"] += 1
            elif o.z_max < S.robot.lidar_height_m:
                classes["low"] += 1
            else:
                classes["tall"] += 1
    assert all(v > 0 for v in classes.values()), classes


def test_collision_semantics_respect_obstacle_height():
    low = _scene([Obstacle("box", 3.5, 4.0, 0.4, 0.4, 0.0, 0.0, 0.12)])
    over = _scene([Obstacle("box", 3.5, 4.0, 0.4, 0.4, 0.0, 0.45, 1.0)])
    # A 12 cm box is invisible to the lidar but is still solidly in the way.
    assert low.in_collision(3.5, 4.0)
    # An overhang starting above the robot is driven under, not around.
    assert not over.in_collision(3.5, 4.0)


# ─────────────────────────────────────────────────────────────────────────────
# Sensors
# ─────────────────────────────────────────────────────────────────────────────


def test_lidar_measures_the_expected_range():
    room = _scene([Obstacle("box", 4.0, 4.0, 0.3, 0.3, 0.0, 0.0, 1.0)])
    ranges, angle_min, inc = simulate_lidar(room, 2.0, 4.0, 0.0)
    forward = ranges[180]  # angle_min = -pi, so index 180 is straight ahead
    assert forward == pytest.approx(1.7, abs=0.06)  # face at x=3.7


def test_lidar_cannot_see_below_or_above_its_plane():
    for obstacle in (Obstacle("box", 3.5, 4.0, 0.4, 0.4, 0.0, 0.0, 0.12),
                     Obstacle("box", 3.5, 4.0, 0.4, 0.4, 0.0, 0.45, 1.0)):
        room = _scene([obstacle])
        ranges, _, _ = simulate_lidar(room, 2.0, 4.0, 0.0)
        # It should see straight through to the far wall (~5.9 m), not the box.
        assert ranges[180] > 4.0, f"lidar wrongly saw {obstacle.kind} at z=" \
                                  f"[{obstacle.z_min}, {obstacle.z_max}]"


def test_depth_sees_what_the_lidar_cannot():
    empty = render_depth(_scene([]), 2.0, 4.0, 0.0)
    low = render_depth(
        _scene([Obstacle("box", 3.5, 4.0, 0.4, 0.4, 0.0, 0.0, 0.12)]),
        2.0, 4.0, 0.0)
    over = render_depth(
        _scene([Obstacle("box", 3.5, 4.0, 0.4, 0.4, 0.0, 0.45, 1.0)]),
        2.0, 4.0, 0.0)

    for name, img in (("low", low), ("overhang", over)):
        changed = np.abs(img - empty) > 0.05
        assert changed.any(), f"depth failed to see the {name} obstacle"
        assert img[changed].min() == pytest.approx(1.1, abs=0.1)

    # And they appear in the correct part of the image: low obstacles low,
    # overhangs high.
    low_rows = np.nonzero((np.abs(low - empty) > 0.05).any(axis=1))[0]
    over_rows = np.nonzero((np.abs(over - empty) > 0.05).any(axis=1))[0]
    assert low_rows.min() > over_rows.max()


def test_depth_renders_the_floor_rather_than_nothing():
    """Downward rays must terminate on the floor. If the lower half of every
    image read 'invalid', the policy would learn to ignore that region."""
    img = render_depth(_scene([]), 4.0, 4.0, 0.0)
    bottom = img[-8:, :]
    assert (bottom > 0.0).mean() > 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Oracle
# ─────────────────────────────────────────────────────────────────────────────


def test_planned_path_reaches_the_goal_and_avoids_obstacles():
    room = sample_room(5)
    path = smooth_path(room, plan_path(room))
    assert len(path) >= 2
    end = np.array(room.goal)
    assert np.linalg.norm(path[-1] - end) < 0.2
    for x, y in path:
        assert not room.in_collision(float(x), float(y)), \
            "oracle planned through a collision"


def test_expert_respects_the_robots_limits():
    """An expert that demonstrates untrackable commands teaches a policy to
    issue untrackable commands."""
    room = sample_room(7)
    expert = PurePursuitExpert(room, S)
    a = S.action
    prev = (0.0, 0.0)
    x, y, yaw = room.start
    for _ in range(300):
        cmd = expert.act(x, y, yaw)
        assert a.v_min - 1e-6 <= cmd.v <= a.v_max + 1e-6
        assert abs(cmd.omega) <= a.omega_max + 1e-6
        assert abs(cmd.v - prev[0]) <= a.accel_max * a.dt + 1e-6
        assert abs(cmd.omega - prev[1]) <= a.alpha_max * a.dt + 1e-6
        prev = (cmd.v, cmd.omega)
        # Integrate forward so the expert sees a moving robot.
        x += cmd.v * math.cos(yaw) * a.dt
        y += cmd.v * math.sin(yaw) * a.dt
        yaw += cmd.omega * a.dt
        if cmd.done:
            break
