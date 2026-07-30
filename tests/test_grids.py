"""Grid primitives. These are the oracle's ground truth -- if they are wrong,
every demonstration is wrong in a way that looks like a policy problem."""

import numpy as np
import pytest

from sim2edge import grids
from sim2edge.grids import INF, UNREACHABLE


# ── shifting ─────────────────────────────────────────────────────────────────

def test_shift_reads_the_neighbour_and_does_not_wrap():
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    # out[r, c] == a[r, c + 1]; the last column has no right neighbour.
    out = grids._shift(a, 0, 1, INF)
    assert np.array_equal(out[:, :2], a[:, 1:])
    assert np.all(out[:, 2] == INF), "wrapped around the edge"


# ── geodesic ─────────────────────────────────────────────────────────────────

RES = 0.1


def test_geodesic_is_metres_not_cells():
    free = np.ones((5, 5), dtype=bool)
    d = grids.geodesic(free, (0, 0), RES)
    assert d[0, 4] == pytest.approx(4 * RES)          # four cardinal steps
    assert d[0, 0] == 0.0


def test_geodesic_uses_the_octile_metric():
    free = np.ones((5, 5), dtype=bool)
    d = grids.geodesic(free, (0, 0), RES)
    assert d[3, 3] == pytest.approx(3 * np.sqrt(2) * RES, rel=1e-5)
    # Mixed leg: 4 across, 2 down -> two diagonals plus two straights.
    assert d[2, 4] == pytest.approx((2 * np.sqrt(2) + 2) * RES, rel=1e-5)


def test_geodesic_routes_around_a_wall_rather_than_through_it():
    free = np.ones((5, 7), dtype=bool)
    free[0:4, 3] = False                      # wall with a gap on the bottom row
    d = grids.geodesic(free, (0, 0), RES)
    straight = 6 * RES
    assert d[0, 6] > straight, "tunnelled through the wall"
    assert d[0, 6] < UNREACHABLE, "should still be reachable via the gap"


def test_sealed_region_is_unreachable():
    free = np.ones((5, 7), dtype=bool)
    free[:, 3] = False                        # full-height wall, no gap
    d = grids.geodesic(free, (0, 0), RES)
    assert d[0, 6] >= UNREACHABLE
    assert np.all(d[~free] >= UNREACHABLE), "blocked cells must stay unreachable"


def test_blocked_cells_are_never_assigned_a_finite_distance():
    free = np.ones((4, 4), dtype=bool)
    free[2, 2] = False
    d = grids.geodesic(free, (0, 0), RES)
    assert d[2, 2] >= UNREACHABLE


def test_goal_inside_an_obstacle_yields_an_empty_field():
    free = np.ones((4, 4), dtype=bool)
    free[1, 1] = False
    d = grids.geodesic(free, (1, 1), RES)
    assert np.all(d >= UNREACHABLE)


def test_diagonal_may_not_cut_a_zero_width_corner():
    """Two obstacles touching at a corner leave a gap of exactly zero width.

    The grid says the diagonal move is free; the robot says otherwise. Without
    the corner rule the oracle would demonstrate a path through solid geometry.
    """
    free = np.ones((3, 3), dtype=bool)
    free[0, 1] = False
    free[1, 0] = False
    d = grids.geodesic(free, (0, 0), RES)
    assert d[1, 1] >= UNREACHABLE, "squeezed diagonally between two corners"


def test_diagonal_is_forbidden_when_either_side_is_blocked():
    """Strict no-corner-cutting: grazing one obstacle corner is also illegal.

    ``free`` is already inflated by the robot radius, so a cell boundary is a
    surface the robot is touching. A diagonal past a single blocked corner
    clears it by exactly zero. The route must go around, at cardinal cost.
    """
    free = np.ones((3, 3), dtype=bool)
    free[0, 1] = False                        # (1, 0) is still open
    d = grids.geodesic(free, (0, 0), RES)
    assert d[1, 1] == pytest.approx(2 * RES), "cut the corner of a blocked cell"


def test_diagonal_is_allowed_through_genuinely_open_space():
    free = np.ones((3, 3), dtype=bool)
    d = grids.geodesic(free, (0, 0), RES)
    assert d[1, 1] == pytest.approx(np.sqrt(2) * RES, rel=1e-5)


# ── rasterize / clearance ────────────────────────────────────────────────────

def _empty(n=4):
    """kind, pos, half, yaw, valid -- the fixed-slot arrays rasterize takes."""
    return (np.zeros(n, np.int8), np.zeros((n, 2), np.float32),
            np.zeros((n, 2), np.float32), np.zeros(n, np.float32),
            np.zeros(n, bool))


def test_walls_occupy_the_border():
    kind, pos, half, yaw, valid = _empty()
    occ = grids.rasterize(kind, pos, half, yaw, valid, (2.0, 2.0), 0.5,
                          wall_thickness=0.3)
    assert occ[0].all() and occ[-1].all()
    assert occ[:, 0].all() and occ[:, -1].all()
    assert not occ[1:3, 1:3].any(), "interior should be free"


def test_box_occupies_exactly_its_extent():
    kind, pos, half, yaw, valid = _empty()
    pos[0], half[0], valid[0] = (1.0, 1.0), (0.3, 0.3), True
    occ = grids.rasterize(kind, pos, half, yaw, valid, (2.0, 2.0), 0.5,
                          wall_thickness=0.0)
    expected = np.zeros((4, 4), bool)
    expected[1:3, 1:3] = True                 # cell centres 0.75 and 1.25
    assert np.array_equal(occ, expected)


def test_cylinder_is_round_not_square():
    kind, pos, half, yaw, valid = _empty()
    kind[0], pos[0], half[0], valid[0] = 1, (1.0, 1.0), (0.4, 0.0), True
    occ = grids.rasterize(kind, pos, half, yaw, valid, (2.0, 2.0), 0.5,
                          wall_thickness=0.0)
    assert occ.sum() == 4, "radius 0.4 reaches the four centre cells only"


def test_invalid_slots_are_not_drawn():
    """The whole point of the fixed-length array: unused slots are inert."""
    kind, pos, half, yaw, valid = _empty()
    pos[0], half[0], valid[0] = (1.0, 1.0), (0.3, 0.3), False
    occ = grids.rasterize(kind, pos, half, yaw, valid, (2.0, 2.0), 0.5,
                          wall_thickness=0.0)
    assert not occ.any()


def test_clearance_is_euclidean_distance_in_metres():
    occ = np.zeros((5, 5), bool)
    occ[2, 2] = True
    c = grids.clearance(occ, 0.1)
    assert c[2, 2] == 0.0
    assert c[2, 3] == pytest.approx(0.1)
    assert c[3, 3] == pytest.approx(0.1 * np.sqrt(2), rel=1e-5)


# ── flow field ───────────────────────────────────────────────────────────────

def _walk(flow, start_rc, w, limit=500):
    """Follow the flow field from a cell; returns the list of cells visited."""
    i = start_rc[0] * w + start_rc[1]
    seen = [i]
    for _ in range(limit):
        j = int(flow.flat[i])
        if j == i:
            break
        i = j
        seen.append(i)
    return [divmod(i, w) for i in seen]


def test_flow_leads_to_the_goal_from_everywhere_reachable():
    free = np.ones((7, 9), dtype=bool)
    free[0:5, 4] = False                      # wall with a gap along the bottom
    D = grids.geodesic(free, (0, 0), RES)
    flow = grids.flow_field(free, D, RES)
    for r in range(7):
        for c in range(9):
            if not free[r, c] or D[r, c] >= UNREACHABLE:
                continue
            assert _walk(flow, (r, c), 9)[-1] == (0, 0), f"stalled from {(r, c)}"


def test_every_flow_step_strictly_decreases_distance():
    """A step that does not decrease D is a cycle waiting to happen, and on
    device the oracle would sit still while the episode timed out."""
    free = np.ones((6, 6), dtype=bool)
    free[2, 2] = free[3, 3] = False
    D = grids.geodesic(free, (0, 0), RES)
    flow = grids.flow_field(free, D, RES)
    for r in range(6):
        for c in range(6):
            i = r * 6 + c
            j = int(flow[r, c])
            if j == i:
                continue
            assert D.flat[j] < D[r, c], f"{(r, c)} -> {divmod(j, 6)} does not descend"


def test_the_goal_is_a_fixed_point():
    free = np.ones((4, 4), dtype=bool)
    D = grids.geodesic(free, (0, 0), RES)
    flow = grids.flow_field(free, D, RES)
    assert flow[0, 0] == 0


def test_blocked_cells_point_back_to_the_nearest_free_cell():
    """Recovery, not a fixed point. The robot leaves the free grid constantly --
    tracking error, physics, and deliberate action perturbation all push it off
    -- and a self-loop there gives the oracle a lookahead point equal to the
    robot's own position, i.e. a meaningless bearing and a permanent spin.
    """
    free = np.ones((4, 4), dtype=bool)
    free[1, 1] = False
    D = grids.geodesic(free, (0, 0), RES)
    flow = grids.flow_field(free, D, RES)

    dest = int(flow[1, 1])
    assert dest != 1 * 4 + 1, "stranded: no way back onto the path"
    assert free.flat[dest], "recovery must land on free space"
    assert _walk(flow, (1, 1), 4)[-1] == (0, 0), "recovery must reach the goal"


def test_recovery_only_applies_off_the_free_grid():
    """Free cells must still follow the true shortest path, not be redirected."""
    free = np.ones((5, 5), dtype=bool)
    free[2, 2] = False
    D = grids.geodesic(free, (0, 0), RES)
    flow = grids.flow_field(free, D, RES)
    for r in range(5):
        for c in range(5):
            if free[r, c] and D[r, c] > 0:
                assert D.flat[int(flow[r, c])] < D[r, c]


def test_flow_never_steps_into_a_blocked_cell():
    free = np.ones((6, 6), dtype=bool)
    free[2, 2] = free[2, 3] = free[3, 2] = False
    D = grids.geodesic(free, (0, 0), RES)
    flow = grids.flow_field(free, D, RES)
    dest_free = free.flat[flow.ravel()].reshape(6, 6)
    assert dest_free[free].all(), "a free cell points into an obstacle"


def test_flow_obeys_the_same_corner_rule_as_the_field():
    """If the walk could cut a corner the field forbade, the oracle would
    demonstrate a path the distance field says is impossible."""
    free = np.ones((3, 3), dtype=bool)
    free[0, 1] = False
    D = grids.geodesic(free, (0, 0), RES)
    flow = grids.flow_field(free, D, RES)
    assert _walk(flow, (1, 1), 3) == [(1, 1), (1, 0), (0, 0)]


def test_a_rotated_box_covers_a_different_footprint():
    """Yaw is the diversity that survives the frozen-scale constraint, so the
    grid has to honour it -- otherwise the oracle plans against geometry the
    simulator is not showing the lidar."""
    kind, pos, half, yaw, valid = _empty()
    pos[0], half[0], valid[0] = (1.0, 1.0), (0.7, 0.2), True
    flat = grids.rasterize(kind, pos, half, yaw, valid, (2.0, 2.0), 0.1,
                           wall_thickness=0.0)
    yaw[0] = np.pi / 2
    turned = grids.rasterize(kind, pos, half, yaw, valid, (2.0, 2.0), 0.1,
                             wall_thickness=0.0)
    assert flat.sum() == turned.sum(), "rotation must preserve area"
    assert not np.array_equal(flat, turned), "yaw was ignored"
    assert np.array_equal(flat, turned.T), "a quarter turn is a transpose here"


def test_cylinders_ignore_yaw():
    kind, pos, half, yaw, valid = _empty()
    kind[0], pos[0], half[0], valid[0] = 1, (1.0, 1.0), (0.4, 0.4), True
    a = grids.rasterize(kind, pos, half, yaw, valid, (2.0, 2.0), 0.1, 0.0)
    yaw[0] = 0.9
    b = grids.rasterize(kind, pos, half, yaw, valid, (2.0, 2.0), 0.1, 0.0)
    assert np.array_equal(a, b)
