"""Pure helpers behind the interactive viewer. No simulator, no GUI."""

import math

import numpy as np
import pytest

from isaaclab_env.interactive import frame_camera, snap_goal
from scripts.interactive_nav import body_offset, pacing_delay, world_offset


def _room(res=0.05, size=(4.0, 3.0)):
    """Free space everywhere except a 0.5 m border and a box at (2, 1.5)."""
    rows, cols = int(size[1] / res), int(size[0] / res)
    free = np.ones((rows, cols), dtype=bool)
    border = int(0.5 / res)
    free[:border], free[-border:], free[:, :border], free[:, -border:] = (
        False, False, False, False)
    box = (slice(int(1.2 / res), int(1.8 / res)),
           slice(int(1.7 / res), int(2.3 / res)))
    free[box] = False
    return free, res


def test_snap_goal_returns_the_clicked_cell_when_it_is_already_free():
    free, res = _room()
    goal = snap_goal((3.0, 0.9), free, res)
    assert goal is not None
    assert goal == pytest.approx([3.025, 0.925], abs=res)


def test_snap_goal_accepts_a_click_that_lands_just_inside_a_wall():
    free, res = _room()
    # 8 cm into the border: a click a person plainly meant for the corridor.
    goal = snap_goal((2.9, 0.42), free, res)
    assert goal is not None
    row, col = int(goal[1] / res), int(goal[0] / res)
    assert free[row, col]
    assert np.linalg.norm(goal - np.array([2.9, 0.42])) < 0.15


def test_snap_goal_refuses_a_click_deep_inside_an_obstacle():
    free, res = _room()
    assert snap_goal((2.0, 1.5), free, res, max_search_m=0.10) is None


def test_snap_goal_never_returns_an_occupied_cell():
    free, res = _room()
    rng = np.random.default_rng(0)
    for _ in range(200):
        point = rng.uniform([0.0, 0.0], [4.0, 3.0])
        goal = snap_goal(point, free, res)
        if goal is None:
            continue
        assert free[int(goal[1] / res), int(goal[0] / res)]


def test_frame_camera_looks_at_the_arena_centre_not_its_corner():
    # The documented failure: a room spans (0,0)-(size) from its env origin, so
    # a camera aimed at the origin frames the corner and the viewport reads as
    # empty.
    _, lookat = frame_camera((10.0, 6.0))
    assert lookat == pytest.approx((5.0, 3.0, 0.0))


def test_frame_camera_scales_its_height_with_the_arena():
    small, _ = frame_camera((4.0, 4.0))
    large, _ = frame_camera((10.0, 10.0))
    assert large[2] > small[2]
    assert large[2] / small[2] == pytest.approx(10.0 / 4.0)


def test_body_and_world_offsets_are_inverse():
    rng = np.random.default_rng(7)
    for _ in range(100):
        offset = rng.uniform(-5.0, 5.0, size=2)
        yaw = float(rng.uniform(-math.pi, math.pi))
        assert world_offset(body_offset(offset, yaw), yaw) == pytest.approx(
            offset, abs=1e-9)


def test_body_offset_puts_a_goal_dead_ahead_on_the_x_axis():
    ahead = body_offset(np.array([0.0, 2.0]), math.pi / 2)
    assert ahead == pytest.approx([2.0, 0.0], abs=1e-9)


def test_pacing_delay_holds_the_control_period_and_never_goes_negative():
    assert pacing_delay(0.01, 0.05, 1.0) == pytest.approx(0.04)
    assert pacing_delay(0.09, 0.05, 1.0) == 0.0
    # Twice real time asks for half the period.
    assert pacing_delay(0.0, 0.05, 2.0) == pytest.approx(0.025)
    # Free-running is no delay, not an unbounded one.
    assert pacing_delay(0.0, 0.05, 0.0) == 0.0
