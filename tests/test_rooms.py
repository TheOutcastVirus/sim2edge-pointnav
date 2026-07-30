"""Layout sampling. Every property here is one the Isaac adapter assumes
without checking, because checking it per-reset on 2048 environments is not
free."""

import numpy as np
import pytest

from sim2edge import grids, rooms
from sim2edge.grids import UNREACHABLE
from sim2edge.rooms import (MAX_OBSTACLES, N_OBSTACLE_SLOTS,
                           N_WALL_SLOTS, sample_room, wall_placement)
from sim2edge.spec import GOAL_MAX_M, ROBOT_RADIUS_M


@pytest.fixture(scope="module")
def sampled():
    """Eight layouts, shared across tests -- each costs a geodesic solve."""
    rng = np.random.default_rng(0)
    return [sample_room(rng) for _ in range(8)]


# ── the fixed-length contract Isaac depends on ───────────────────────────────

def test_obstacle_arrays_are_always_max_length(sampled):
    """Isaac pre-allocates this many prims once. A variable length would mean
    spawning at reset, which dominates step time."""
    for room in sampled:
        assert room.kind.shape == (MAX_OBSTACLES,)
        assert room.pos.shape == (MAX_OBSTACLES, 2)
        assert room.half.shape == (MAX_OBSTACLES, 2)
        assert room.valid.shape == (MAX_OBSTACLES,)


def test_slot_index_is_a_stable_prim_identity(sampled):
    """Entry i always describes prim i at its baked size, used or not. Isaac
    spawns each slot once at that scale and only ever moves it, because
    MultiMeshRayCaster bakes scale at spawn and tracks only pose."""
    from sim2edge.rooms import _slot_table
    table = _slot_table()
    for room in sampled:
        for i, (k, hx, hy) in enumerate(table):
            assert room.kind[i] == k
            assert room.half[i, 0] == pytest.approx(hx)
            assert room.half[i, 1] == pytest.approx(hy)


def test_every_layout_shares_one_prim_inventory(sampled):
    """Two layouts must differ only in which slots are used and where they sit;
    if their sizes differed, one set of prims could not serve both."""
    a, b = sampled[0], sampled[1]
    assert np.array_equal(a.half, b.half) and np.array_equal(a.kind, b.kind)
    assert not np.array_equal(a.valid, b.valid) or not np.array_equal(a.pos, b.pos)


def test_unused_slots_are_marked_invalid_not_resized(sampled):
    for room in sampled:
        idle = ~room.valid
        assert np.all(room.half[idle] > 0.0), "an idle prim still has its size"


def test_obstacle_count_is_in_range(sampled):
    for room in sampled:
        n = int(room.valid[:N_OBSTACLE_SLOTS].sum())
        assert 1 <= n <= N_OBSTACLE_SLOTS


# ── the arena boundary is real geometry, not just occupied cells ─────────────

def test_walls_are_always_present():
    """When walls existed only in the grid, the oracle planned against a
    boundary the lidar could not see -- rays left the arena and came back at max
    range, and the robot could drive straight out."""
    for room in sampled_rooms(4):
        assert room.valid[N_OBSTACLE_SLOTS:].all()


def test_wall_prims_cover_the_grid_border():
    """The slabs must coincide with the border rasterize() draws, so the surface
    the lidar sees and the boundary the oracle plans against are the same one."""
    from sim2edge import grids
    for room in sampled_rooms(4):
        walls_only = room.valid.copy()
        walls_only[:N_OBSTACLE_SLOTS] = False
        drawn = grids.rasterize(room.kind, room.pos, room.half, room.yaw,
                                walls_only, room.size_xy, room.res,
                                wall_thickness=0.0)
        border = np.zeros_like(drawn)
        border[0], border[-1], border[:, 0], border[:, -1] = True, True, True, True
        assert (drawn & border).sum() == border.sum(), \
            "wall slabs leave part of the arena border uncovered"


def test_wall_slabs_span_any_arena():
    """A slab is positioned, never scaled -- MultiMeshRayCaster bakes scale at
    spawn. So one fixed length has to cover the widest arena."""
    from sim2edge.rooms import ARENA_MAX_M, WALL_HALF
    assert 2 * WALL_HALF[0] >= ARENA_MAX_M


def test_wall_placement_is_four_slabs_in_a_rectangle():
    placed = wall_placement((8.0, 6.0))
    assert len(placed) == N_WALL_SLOTS
    yaws = sorted(a for _, a in placed)
    assert yaws[:2] == [0.0, 0.0], "two slabs run along x"
    assert all(abs(a - np.pi / 2) < 1e-9 for a in yaws[2:]), "two run along y"


def sampled_rooms(n):
    rng = np.random.default_rng(11)
    return [sample_room(rng) for _ in range(n)]


def test_arrays_are_the_dtypes_the_uploader_expects(sampled):
    room = sampled[0]
    assert room.pos.dtype == np.float32 and room.half.dtype == np.float32
    assert room.kind.dtype == np.int8 and room.valid.dtype == np.bool_
    assert room.yaw.dtype == np.float32 and room.yaw.shape == (MAX_OBSTACLES,)


# ── geometry ─────────────────────────────────────────────────────────────────

def test_obstacles_lie_inside_the_arena(sampled):
    """Bounding circle, because boxes are rotated and the corner sweeps out
    further than either half-extent."""
    for room in sampled:
        v = room.valid.copy()
        v[N_OBSTACLE_SLOTS:] = False      # walls protrude on purpose
        reach = np.hypot(room.half[v, 0], room.half[v, 1])
        assert np.all(room.pos[v, 0] - reach > 0)
        assert np.all(room.pos[v, 1] - reach > 0)
        assert np.all(room.pos[v, 0] + reach < room.size_xy[0])
        assert np.all(room.pos[v, 1] + reach < room.size_xy[1])


def test_obstacles_leave_room_to_drive_between_them(sampled):
    """Rejection sampling pads by a robot diameter, so no pair forms a gap the
    policy can see through but not fit through."""
    gap = 2.0 * (ROBOT_RADIUS_M + rooms.SPAWN_CLEARANCE_M)
    for room in sampled:
        idx = np.flatnonzero(room.valid[:N_OBSTACLE_SLOTS])
        reach = np.hypot(room.half[:, 0], room.half[:, 1])
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                need = reach[i] + reach[j] + gap
                dx = abs(room.pos[i, 0] - room.pos[j, 0]) - need
                dy = abs(room.pos[i, 1] - room.pos[j, 1]) - need
                assert dx >= 0 or dy >= 0, f"obstacles {i} and {j} are fused"


def test_the_arena_is_walled(sampled):
    for room in sampled:
        assert room.occ[0].all() and room.occ[-1].all()
        assert room.occ[:, 0].all() and room.occ[:, -1].all()


def test_free_space_is_the_grid_inflated_by_the_robot_radius(sampled):
    """`free` marks legal robot *centres*, not empty floor. Conflating the two
    is how a planner produces paths that scrape the wall."""
    for room in sampled:
        assert np.array_equal(room.free, room.clearance >= ROBOT_RADIUS_M)
        assert not room.free[room.occ].any()


# ── start / goal feasibility, guaranteed by construction ─────────────────────

def test_start_and_goal_sit_in_free_space(sampled):
    for room in sampled:
        for x, y in (room.start[:2], room.goal):
            r, c = grids.world_to_cell(x, y, room.res)
            assert room.free[r, c]
            assert room.clearance[r, c] >= ROBOT_RADIUS_M + rooms.SPAWN_CLEARANCE_M


def test_goal_is_reachable_from_the_start(sampled):
    for room in sampled:
        r, c = grids.world_to_cell(room.start[0], room.start[1], room.res)
        assert room.geodesic[r, c] < UNREACHABLE


def test_episode_difficulty_is_bounded_by_path_length_not_straight_line(sampled):
    """The sampler ranges over geodesic distance on purpose: a goal 3 m away
    through a wall is a harder episode than one 6 m down an open corridor."""
    for room in sampled:
        assert rooms.GOAL_MIN_DIST_M <= room.path_length_m <= GOAL_MAX_M


def test_the_goal_cell_is_the_zero_of_the_field(sampled):
    for room in sampled:
        r, c = grids.world_to_cell(room.goal[0], room.goal[1], room.res)
        assert room.geodesic[r, c] == pytest.approx(0.0)


def test_start_yaw_is_a_valid_heading(sampled):
    for room in sampled:
        assert 0.0 <= room.start[2] < 2 * np.pi


# ── reproducibility ──────────────────────────────────────────────────────────

def test_same_seed_gives_the_same_layout():
    """A layout pool is identified by its seed in the dataset shard. If this
    breaks, no run is reproducible from its metadata."""
    a = sample_room(np.random.default_rng(7))
    b = sample_room(np.random.default_rng(7))
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.valid, b.valid)
    assert np.array_equal(a.start, b.start) and np.array_equal(a.goal, b.goal)
    assert np.array_equal(a.geodesic, b.geodesic)


def test_different_seeds_give_different_layouts():
    a = sample_room(np.random.default_rng(1))
    b = sample_room(np.random.default_rng(2))
    assert not np.array_equal(a.pos, b.pos)
