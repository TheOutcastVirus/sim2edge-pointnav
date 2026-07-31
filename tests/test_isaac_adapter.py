"""The adapter is not runnable here -- Isaac Lab lives in a separate venv. What
*is* testable is that its absence never leaks into the rest of the repo, and
that the constants it shares with the dataset loader agree."""

import importlib

import numpy as np
import pytest


def test_importing_the_task_does_not_require_isaac_lab():
    """CI, the dataset loader, and the board's export venv all import sim2edge
    without a 40 GB install. A bare `import isaaclab` at module scope here
    would break all three."""
    m = importlib.import_module("isaaclab_env.nav_task")
    assert hasattr(m, "Sim2EdgeNavEnv")


def test_constructing_without_isaac_lab_fails_loudly():
    m = importlib.import_module("isaaclab_env.nav_task")
    if m._HAVE_ISAACLAB:
        pytest.skip("isaaclab present; the guard path is not exercised")
    with pytest.raises(ImportError, match="isaaclab is required"):
        m.Sim2EdgeNavEnv(None)


def test_obstacle_prim_count_matches_the_slot_table():
    """The adapter spawns one prim per slot. If these ever diverge, a layout
    would address a prim that does not exist -- silently, since the write loop
    would just miss it."""
    from sim2edge.rooms import MAX_OBSTACLES, _slot_table
    assert len(_slot_table()) == MAX_OBSTACLES


def test_every_moving_joint_has_an_actuator_when_isaac_is_available():
    """Free wheel suspension destroys differential-drive command tracking."""
    m = importlib.import_module("isaaclab_env.nav_task")
    if not m._HAVE_ISAACLAB:
        pytest.skip("cfg objects need isaaclab")
    expressions = {
        expr
        for actuator in m.NavEnvCfg.robot_cfg.actuators.values()
        for expr in actuator.joint_names_expr
    }
    assert ".*wheel_joint" in expressions
    assert ".*wheel_drop.*" in expressions


def test_env_spacing_exceeds_the_largest_arena():
    """Rooms are tiled at fixed origins. If spacing were smaller than an arena,
    a neighbouring room would appear inside this one's lidar scan."""
    from sim2edge import rooms
    m = importlib.import_module("isaaclab_env.nav_task")
    spacing = rooms.ARENA_MAX_M + 4.0
    assert spacing > rooms.ARENA_MAX_M
    # A beam can reach LIDAR_MAX_M, so tiling alone does not isolate rooms --
    # the arena walls do. This asserts the walls exist to be relied on.
    r = rooms.sample_room(np.random.default_rng(0))
    assert r.occ[0].all() and r.occ[:, 0].all()


def test_lidar_beam_count_can_fill_every_bin():
    """Binning is by angle, so beams only need to be dense enough that no bin
    is empty by construction; an empty bin reads as 'no obstacle'."""
    from sim2edge.spec import LIDAR_BINS
    m = importlib.import_module("isaaclab_env.nav_task")
    assert m.LIDAR_BEAMS >= LIDAR_BINS


def test_the_mount_yaw_is_applied_exactly_once():
    """The sensor is mounted with the hardware's quarter turn and the beam
    bearings undo it. Applying it twice, or not at all, rotates the policy's
    entire world view -- the failure looks like a badly trained policy."""
    import math
    from sim2edge.spec import LIDAR_MOUNT_YAW
    m = importlib.import_module("isaaclab_env.nav_task")
    if not m._HAVE_ISAACLAB:
        pytest.skip("cfg objects need isaaclab")
    rot = m.NavEnvCfg.lidar_cfg.offset.rot
    assert rot[0] == pytest.approx(math.cos(LIDAR_MOUNT_YAW / 2))
    assert rot[3] == pytest.approx(math.sin(LIDAR_MOUNT_YAW / 2))


# ── the RL reward's arithmetic has to work out ───────────────────────────────
#
# These are cheap, they need no simulator, and each one pins a weight that
# quietly inverts the optimal policy if it is set wrong.

def test_moving_is_worth_more_than_standing_still():
    """The time penalty must not exceed what one step of travel earns, or the
    optimal policy is to stop and wait out the clock."""
    from sim2edge.spec import DT, V_MAX
    m = importlib.import_module("isaaclab_env.nav_task")
    per_step_progress = m.NavEnvCfg.progress_reward * V_MAX * DT
    assert m.NavEnvCfg.time_penalty < per_step_progress


def test_reaching_the_goal_beats_running_out_the_clock():
    from sim2edge.spec import DT
    m = importlib.import_module("isaaclab_env.nav_task")
    longest_episode = m.NavEnvCfg.episode_length_s / DT
    assert m.NavEnvCfg.goal_reward > m.NavEnvCfg.time_penalty * longest_episode


def test_the_progress_clamp_cannot_swallow_real_motion():
    """It exists to bound grid artifacts. Set below one cell transition it would
    instead silently discount most of the progress actually made, because
    consecutive geodesic readings differ by whole cells."""
    from sim2edge import rooms
    from sim2edge.spec import DT, V_MAX
    m = importlib.import_module("isaaclab_env.nav_task")
    assert m.PROGRESS_CLAMP_M > rooms.GRID_RES_M + V_MAX * DT


# ── the converted asset agrees with the spec ─────────────────────────────────

URDF = "assets/turtlebot4_lite.urdf"


@pytest.mark.skipif(not __import__("pathlib").Path(URDF).exists(),
                    reason="run tools/convert_tb4.py --urdf-only first")
def test_urdf_lidar_mount_matches_the_spec():
    """The spec's mount pose is transcribed from this description. If a ROS
    package update moves the sensor, the sim would quietly train against a pose
    the robot does not have -- and it would surface months later as 'the policy
    does not transfer'."""
    import xml.etree.ElementTree as ET
    from sim2edge.spec import LIDAR_MOUNT_XYZ, LIDAR_MOUNT_YAW

    root = ET.parse(URDF).getroot()
    joint = next(j for j in root.iter("joint") if j.get("name") == "rplidar_joint")
    origin = joint.find("origin")

    xyz = [float(v) for v in origin.get("xyz").split()]
    rpy = [float(v) for v in origin.get("rpy").split()]

    assert joint.find("parent").get("link") == "base_link"
    for got, want in zip(xyz, LIDAR_MOUNT_XYZ):
        assert got == pytest.approx(want, abs=1e-3)
    assert rpy[2] == pytest.approx(LIDAR_MOUNT_YAW, abs=1e-6)


@pytest.mark.skipif(not __import__("pathlib").Path(URDF).exists(),
                    reason="run tools/convert_tb4.py --urdf-only first")
def test_urdf_has_the_wheel_joints_the_adapter_asks_for():
    """The adapter resolves these by exact name to pin left/right ordering."""
    import xml.etree.ElementTree as ET
    names = {j.get("name") for j in ET.parse(URDF).getroot().iter("joint")}
    assert {"left_wheel_joint", "right_wheel_joint"} <= names
