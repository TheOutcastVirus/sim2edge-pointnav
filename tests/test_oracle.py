"""The oracle. Every label in the BC dataset comes from here, so a fault in
this file is not a training bug -- it is thousands of episodes of confidently
wrong supervision."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sim2edge import oracle as oracle_mod
from sim2edge.oracle import LayoutPool, oracle_action, perturb_action
from sim2edge.rooms import sample_room
from sim2edge.spec import ACCEL_MAX, ALPHA_MAX, DT, OMEGA_MAX, V_MAX


@pytest.fixture(scope="module")
def rooms():
    rng = np.random.default_rng(3)
    return [sample_room(rng) for _ in range(4)]


@pytest.fixture(scope="module")
def pool(rooms):
    return LayoutPool(rooms)


def _poses(rooms):
    return torch.tensor(np.stack([r.start for r in rooms]), dtype=torch.float32)


# ── the pool's padding contract ──────────────────────────────────────────────

def test_pool_pads_to_the_largest_room(pool, rooms):
    assert pool.h == max(r.free.shape[0] for r in rooms)
    assert pool.w == max(r.free.shape[1] for r in rooms)


def test_padding_is_inert_not_a_wormhole(pool, rooms):
    """Rooms are different sizes, so small ones have padding. A flow entry that
    still used the room's own width would step into another room's geometry."""
    for i, room in enumerate(rooms):
        rh, rw = room.free.shape
        flow = pool.flow[i].reshape(pool.h, pool.w)
        pad = torch.ones((pool.h, pool.w), dtype=torch.bool)
        pad[:rh, :rw] = False
        idx = torch.arange(pool.h * pool.w).reshape(pool.h, pool.w)
        assert torch.equal(flow[pad], idx[pad]), "padding must self-loop"


def test_flow_reindexing_survives_the_width_change(pool, rooms):
    """Repacking a (rh, rw) field into (h, w) has to renumber every entry.
    Copying the raw values would point each step into the wrong row."""
    for i, room in enumerate(rooms):
        rh, rw = room.free.shape
        flow = pool.flow[i].reshape(pool.h, pool.w)[:rh, :rw]
        src = torch.from_numpy(room.flow.astype("int64"))
        expected = (src // rw) * pool.w + (src % rw)
        assert torch.equal(flow, expected)


# ── the controller respects the robot ────────────────────────────────────────

def test_commands_are_inside_the_create3_envelope(pool, rooms):
    pose = _poses(rooms)
    prev = torch.zeros(len(rooms), 2)
    for _ in range(50):
        a = oracle_action(pool, torch.arange(len(rooms)), pose, prev)
        assert torch.all(a[:, 0] >= 0.0) and torch.all(a[:, 0] <= V_MAX)
        assert torch.all(a[:, 1].abs() <= OMEGA_MAX)
        prev = a


def test_commands_obey_the_acceleration_limit(pool, rooms):
    """An oracle that steps faster than the base can accelerate teaches the
    policy to emit commands the hardware silently clips."""
    pose = _poses(rooms)
    prev = torch.zeros(len(rooms), 2)
    for _ in range(50):
        a = oracle_action(pool, torch.arange(len(rooms)), pose, prev)
        assert torch.all((a[:, 0] - prev[:, 0]).abs() <= ACCEL_MAX * DT + 1e-6)
        assert torch.all((a[:, 1] - prev[:, 1]).abs() <= ALPHA_MAX * DT + 1e-6)
        prev = a


def test_perturbed_execution_obeys_the_acceleration_limit():
    """Exploration must not create commands the physical base cannot make."""
    prev = torch.tensor([[0.2, -0.3], [0.1, 0.4]])
    label = torch.tensor([[V_MAX, OMEGA_MAX], [0.0, -OMEGA_MAX]])
    out = perturb_action(label, sigma=2.0, prev_action=prev,
                         generator=torch.Generator().manual_seed(4))

    assert torch.all((out[:, 0] - prev[:, 0]).abs() <= ACCEL_MAX * DT + 1e-6)
    assert torch.all((out[:, 1] - prev[:, 1]).abs() <= ALPHA_MAX * DT + 1e-6)


def test_zero_perturbation_keeps_the_oracle_command():
    action = torch.tensor([[0.2, -0.3]])
    assert perturb_action(action, 0.0, prev_action=torch.zeros_like(action)) is action


def test_it_turns_in_place_rather_than_arcing_backwards(pool, rooms):
    """Facing away from the path, a differential drive should spin, not sweep a
    wide arc into whatever is beside it."""
    from sim2edge.oracle import _lookahead_point

    room = rooms[0]
    p = LayoutPool([room])
    idx = torch.zeros(1, dtype=torch.long)
    xy = torch.tensor([[float(room.start[0]), float(room.start[1])]])

    # Aim the robot *at* the lookahead point, and then exactly 180 degrees off
    # it. Perturbing the start yaw by pi would only invert whatever error it
    # already had, which is not the same thing.
    target = _lookahead_point(p, idx, xy, 11)
    d = target - xy
    at = float(torch.atan2(d[0, 1], d[0, 0]))

    facing = torch.tensor([[xy[0, 0], xy[0, 1], at]])
    away = torch.tensor([[xy[0, 0], xy[0, 1], at + math.pi]])

    forward = oracle_action(p, idx, facing, torch.zeros(1, 2))
    spun = oracle_action(p, idx, away, torch.zeros(1, 2))

    assert spun[0, 0] == pytest.approx(0.0), "drove forward while facing away"
    assert spun[0, 1].abs() > 0.0, "did not turn"
    assert forward[0, 0] > spun[0, 0], "aimed at the path should be the faster one"


def test_it_slows_down_near_the_goal(pool, rooms):
    room = rooms[0]
    p = LayoutPool([room])
    at_goal = torch.tensor([[room.goal[0], room.goal[1], 0.0]])
    prev = torch.tensor([[V_MAX, 0.0]])
    a = oracle_action(p, torch.zeros(1, dtype=torch.long), at_goal, prev)
    assert a[0, 0] < V_MAX


# ── it actually solves the task ──────────────────────────────────────────────

def _rollout(pool, room, idx, steps=900):
    """Kinematic differential-drive integration -- enough to check the expert
    closes the loop. It is not a physics model and is not used for training."""
    pose = torch.tensor(np.asarray([room.start]), dtype=torch.float32)
    prev = torch.zeros(1, 2)
    goal = torch.tensor(room.goal, dtype=torch.float32)
    for _ in range(steps):
        a = oracle_action(pool, torch.tensor([idx]), pose, prev)
        v, w = a[0, 0], a[0, 1]
        pose = pose.clone()
        pose[0, 0] += v * DT * torch.cos(pose[0, 2])
        pose[0, 1] += v * DT * torch.sin(pose[0, 2])
        pose[0, 2] += w * DT
        prev = a
        if torch.linalg.vector_norm(pose[0, :2] - goal) < 0.25:
            return True, pose
    return False, pose


def test_the_oracle_reaches_the_goal(pool, rooms):
    for i, room in enumerate(rooms):
        reached, pose = _rollout(pool, room, i)
        assert reached, (f"room {i} not solved; ended at {pose[0, :2].tolist()} "
                         f"vs goal {room.goal.tolist()}")


def test_the_oracle_stays_in_free_space(pool, rooms):
    """A demonstration that clips an obstacle is a demonstration of clipping
    obstacles."""
    from sim2edge import grids
    room = rooms[0]
    pose = torch.tensor(np.asarray([room.start]), dtype=torch.float32)
    prev = torch.zeros(1, 2)
    for _ in range(400):
        a = oracle_action(pool, torch.tensor([0]), pose, prev)
        pose = pose.clone()
        pose[0, 0] += a[0, 0] * DT * torch.cos(pose[0, 2])
        pose[0, 1] += a[0, 0] * DT * torch.sin(pose[0, 2])
        pose[0, 2] += a[0, 1] * DT
        prev = a
        r, c = grids.world_to_cell(float(pose[0, 0]), float(pose[0, 1]), room.res)
        assert room.clearance[r, c] > 0.0, "drove into an obstacle"


# ── batching must not change the answer ──────────────────────────────────────

def test_batched_matches_one_at_a_time(pool, rooms):
    """The entire performance argument rests on batching. If the batched path
    disagreed with the scalar one, every speedup would be a silent behaviour
    change."""
    pose = _poses(rooms)
    prev = torch.rand(len(rooms), 2) * 0.2
    idx = torch.arange(len(rooms))

    together = oracle_action(pool, idx, pose, prev)
    apart = torch.cat([oracle_action(pool, idx[i:i + 1], pose[i:i + 1],
                                     prev[i:i + 1]) for i in range(len(rooms))])
    assert torch.allclose(together, apart, atol=1e-6)


def test_lookahead_distance_is_honoured(pool, rooms):
    """The walk length is derived from LOOKAHEAD_M and the grid resolution; if
    that arithmetic drifts the controller silently retunes itself."""
    steps = max(1, int(round(oracle_mod.LOOKAHEAD_M / pool.res)))
    assert steps == 11, "0.55 m at 5 cm resolution"


# ── recovery, which is what makes perturbed collection viable ────────────────

def test_it_recovers_when_pushed_off_the_free_grid(pool, rooms):
    """DART-style collection deliberately perturbs the executed command so the
    dataset contains recovery states. That only produces useful labels if the
    oracle still knows what to do once the robot is somewhere it should not be.
    """
    from sim2edge import grids
    room = rooms[0]
    p = LayoutPool([room])

    # Find a cell the planner considers illegal but that is not solid geometry:
    # the inflated band around an obstacle, which is exactly where tracking
    # error puts the robot.
    band = (~room.free) & (room.clearance > 0.0)
    assert band.any()
    r, c = divmod(int(np.flatnonzero(band.ravel())[len(np.flatnonzero(band.ravel())) // 2]),
                  room.free.shape[1])
    x, y = grids.cell_to_world(r, c, room.res)

    pose = torch.tensor([[float(x), float(y), 0.0]], dtype=torch.float32)
    a = oracle_action(p, torch.zeros(1, dtype=torch.long), pose,
                      torch.zeros(1, 2))
    assert torch.isfinite(a).all()

    # The command must actually move it: a stranded robot emits pure spin.
    reached, _ = _rollout_from(p, room, pose)
    assert reached, "could not recover onto the path"


def _rollout_from(pool, room, pose, steps=900):
    prev = torch.zeros(1, 2)
    goal = torch.tensor(room.goal, dtype=torch.float32)
    for _ in range(steps):
        a = oracle_action(pool, torch.zeros(1, dtype=torch.long), pose, prev)
        pose = pose.clone()
        pose[0, 0] += a[0, 0] * DT * torch.cos(pose[0, 2])
        pose[0, 1] += a[0, 0] * DT * torch.sin(pose[0, 2])
        pose[0, 2] += a[0, 1] * DT
        prev = a
        if torch.linalg.vector_norm(pose[0, :2] - goal) < 0.25:
            return True, pose
    return False, pose


def test_a_stranded_robot_would_have_shown_up_as_pure_spin(pool, rooms):
    """Guards the regression directly: the old failure was v == 0 with omega
    pinned at the limit, forever."""
    room = rooms[0]
    p = LayoutPool([room])
    pose = torch.tensor(np.asarray([room.start]), dtype=torch.float32)
    prev = torch.zeros(1, 2)
    spun = 0
    for _ in range(300):
        a = oracle_action(p, torch.zeros(1, dtype=torch.long), pose, prev)
        spun += int(a[0, 0] == 0.0 and a[0, 1].abs() >= OMEGA_MAX - 1e-6)
        pose = pose.clone()
        pose[0, 0] += a[0, 0] * DT * torch.cos(pose[0, 2])
        pose[0, 1] += a[0, 0] * DT * torch.sin(pose[0, 2])
        pose[0, 2] += a[0, 1] * DT
        prev = a
    assert spun < 40, f"spent {spun} ticks spinning at the limit"
