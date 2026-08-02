"""The closed-loop belief evaluator, exercised against a stand-in environment.

``scripts/eval_belief.py`` cannot be imported without Isaac, and an Isaac run
costs minutes. Everything that is easy to get wrong in it, though, is ordinary
tensor bookkeeping: the sweep ring buffer, the gyro history window, when the
belief is allowed to integrate, and whether zeroing the odometry head really
reproduces dead reckoning. All of that is testable on CPU in seconds against a
fake environment, and none of it involves a simulator.

The stand-in integrates commands kinematically with a slip factor, which makes
dead reckoning inexact in the same direction the corrected simulator does -- the
property the whole design exists to exploit.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import types

import pytest
import torch

from sim2edge.models import BeliefNavPolicy, BeliefNavPolicyConfig
from sim2edge.spec import DT, GOAL_TOLERANCE_M

ROOT = Path(__file__).resolve().parents[1]

#: Mirrors ``isaaclab_env.nav_task.LIDAR_BEAMS``, which cannot be imported here
#: because that module is stubbed to keep this test free of Isaac. The raw beam
#: count only has to be plausible for the binning to exercise, but keeping it
#: equal to the real one means the sweep indexes the same shapes it will in
#: anger.
LIDAR_BEAMS = 720


#: Modules this file replaces with stand-ins, and must put back.
_STUBBED = ("isaaclab_env", "isaaclab_env.nav_task", "eval_belief")


def _load_evaluator():
    """Import the script with its Isaac-only dependency stubbed out.

    ``normalize_action`` is imported inside ``_evaluate_tier`` from the Isaac
    task module, so the stub has to stay installed for the duration of the
    tests, not merely while the script is being imported. Stubbing rather than
    refactoring the script keeps the tested code byte-identical to the deployed
    code.

    The stub is genuinely global state, and ``test_isaac_adapter.py`` asserts on
    what happens when Isaac is *absent* -- a leaked stub makes it believe Isaac
    is present and fails it. Hence the paired restore in the fixture below;
    this loader must never be called outside it.
    """
    stub = types.ModuleType("isaaclab_env.nav_task")
    stub.normalize_action = lambda command: command
    stub.NavEnvCfg = object
    stub.Sim2EdgeNavEnv = object
    package = types.ModuleType("isaaclab_env")
    package.nav_task = stub
    sys.modules["isaaclab_env"] = package
    sys.modules["isaaclab_env.nav_task"] = stub

    spec = importlib.util.spec_from_file_location(
        "eval_belief", ROOT / "scripts" / "eval_belief.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEnv:
    """Kinematic stand-in with the exact surface ``_evaluate_tier`` touches."""

    def __init__(self, n=8, max_steps=40, slip=0.03, seed=0):
        self.num_envs = n
        self.device = "cpu"
        self.max_steps = max_steps
        self.slip = slip
        self._seed = seed
        self.scene = types.SimpleNamespace(env_origins=torch.zeros(n, 2))
        self._beam_angles = (torch.arange(LIDAR_BEAMS, dtype=torch.float32)
                             * (2.0 * math.pi / LIDAR_BEAMS)) - math.pi
        self.reset(seed)

    def reset(self, seed=0):
        n = self.num_envs
        g = torch.Generator().manual_seed(seed)
        self._pose = torch.zeros(n, 3)
        # Goals a couple of metres out, at assorted bearings.
        angle = torch.rand(n, generator=g) * 2.0 * math.pi
        radius = 1.5 + torch.rand(n, generator=g)
        self._goal_w = torch.stack(
            [radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)
        self._action = torch.zeros(n, 2)
        self._layout = torch.arange(n)
        self._optimal_path_m = torch.linalg.vector_norm(self._goal_w, dim=1)
        self._reached = torch.zeros(n, dtype=torch.bool)
        self._collided = torch.zeros(n, dtype=torch.bool)
        self._steps = 0
        # A per-episode systematic slip, as the corrected simulator applies.
        self._slip_factor = 1.0 - self.slip * torch.rand(n, generator=g)
        return None

    def _robot_pose(self):
        return self._pose.clone()

    def raw_scan(self):
        # A circular room: range to the wall along each beam, which does move
        # with the robot, so the odometry head has something to key on.
        n = self.num_envs
        bearing = self._beam_angles[None, :] + self._pose[:, 2:3]
        radius = 4.0
        px, py = self._pose[:, 0:1], self._pose[:, 1:2]
        cos, sin = torch.cos(bearing), torch.sin(bearing)
        b = px * cos + py * sin
        c = px ** 2 + py ** 2 - radius ** 2
        ranges = -b + torch.sqrt((b ** 2 - c).clamp_min(0.0))
        return ranges.expand(n, LIDAR_BEAMS).contiguous(), self._beam_angles

    def step(self, command):
        v = command[:, 0] * self._slip_factor
        w = command[:, 1]
        yaw = self._pose[:, 2]
        self._pose[:, 0] += v * torch.cos(yaw) * DT
        self._pose[:, 1] += v * torch.sin(yaw) * DT
        self._pose[:, 2] = torch.atan2(torch.sin(yaw + w * DT),
                                       torch.cos(yaw + w * DT))
        self._action = command.clone()
        self._steps += 1

        remaining = torch.linalg.vector_norm(
            self._goal_w - self._pose[:, :2], dim=1)
        self._reached = remaining < GOAL_TOLERANCE_M
        terminated = self._reached.clone()
        truncated = torch.full((self.num_envs,), self._steps >= self.max_steps)
        return None, None, terminated, truncated, {}


@pytest.fixture(scope="module")
def evaluator():
    """Install the Isaac stubs, and put ``sys.modules`` back exactly as found."""
    saved = {name: sys.modules.get(name) for name in _STUBBED}
    try:
        yield _load_evaluator()
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture(scope="module")
def policy():
    torch.manual_seed(0)
    return BeliefNavPolicy(BeliefNavPolicyConfig()).eval()


def _run(evaluator, policy, mode, scale=0.0, n=8, steps=40, slip=0.03):
    from sim2edge import noise as noise_mod
    env = FakeEnv(n=n, max_steps=steps, slip=slip)
    report, layouts = evaluator._evaluate_tier(
        env, policy, policy.actor, mode, scale, 0, torch, noise_mod)
    return report, layouts


@pytest.mark.parametrize("mode", ["belief", "dead-reckoning", "pose",
                                  "reference"])
def test_tier_runs_and_reports_every_field(evaluator, policy, mode):
    report, layouts = _run(evaluator, policy, mode)
    assert report["mode"] == mode
    assert layouts.numel() == 8
    for key in ("success_rate", "spl", "collision_rate", "timeout_rate",
                "mean_jerk", "mean_steps", "mean_path_m"):
        assert math.isfinite(report[key]), key
    assert 0.0 <= report["success_rate"] <= 1.0


@pytest.mark.parametrize("mode", ["pose", "reference"])
def test_pose_modes_report_no_belief_diagnostics(evaluator, policy, mode):
    """A goal read from a pose has no belief, and must not fake one."""
    report, _ = _run(evaluator, policy, mode)
    for key in ("belief_error_m", "belief_error_final_m", "drift_per_m",
                "belief_arrival_rate"):
        assert math.isnan(report[key]), key


@pytest.mark.parametrize("mode", ["belief", "dead-reckoning"])
def test_belief_modes_report_finite_diagnostics(evaluator, policy, mode):
    report, _ = _run(evaluator, policy, mode)
    for key in ("belief_error_m", "belief_error_final_m", "drift_per_m",
                "belief_error_max_m", "belief_arrival_rate"):
        assert math.isfinite(report[key]), key
    assert report["belief_error_final_m"] >= 0.0


def test_drift_is_aggregated_over_distance_not_mean_of_episode_ratios(evaluator):
    """A near-zero path must not dominate the portable drift metric."""
    final = torch.tensor([0.01, 1.0])
    distance = torch.tensor([0.001, 100.0])
    got = evaluator._aggregate_drift(final, distance)
    assert got == pytest.approx(1.01 / 100.001)


def test_dead_reckoning_restores_the_odometry_head(evaluator, policy):
    """The baseline must not be destructive: it borrows the head, not owns it."""
    before = policy.odometry.head.weight.detach().clone()
    _run(evaluator, policy, "dead-reckoning")
    assert torch.equal(policy.odometry.head.weight, before)


def test_dead_reckoning_drift_tracks_the_slip_it_cannot_see(evaluator, policy):
    """With no slip, commanded dead reckoning is exact and the belief is too.

    This is the measurement from section 7 of the 2026-07-31 log, turned into a
    test: an environment that tracks commands perfectly leaves the odometry head
    with nothing to learn *and* leaves dead reckoning with nothing to get wrong.
    A belief error that does not grow when slip is removed is the evidence that
    the evaluator is actually integrating rather than peeking at a pose.
    """
    clean, _ = _run(evaluator, policy, "dead-reckoning", slip=0.0)
    slipping, _ = _run(evaluator, policy, "dead-reckoning", slip=0.10)
    assert clean["belief_error_final_m"] < 1.0e-3
    assert slipping["belief_error_final_m"] > 10.0 * clean["belief_error_final_m"]


def test_belief_never_reads_the_pose(evaluator, policy, monkeypatch):
    """Move the world under the robot; a belief-driven run must not notice.

    The strongest available statement that no pose leaks into the belief path:
    the same commands against the same sensor stream produce the same actions
    even when the true pose is translated to somewhere else entirely. The
    reported belief error changes, because truth moved; the driving does not.
    """
    from sim2edge import noise as noise_mod

    commands = {}
    for tag, offset in (("origin", 0.0), ("shifted", 25.0)):
        env = FakeEnv(n=4, max_steps=12)
        env._pose[:, :2] += offset
        env._goal_w += offset
        seen = []
        original = env.step

        def record(command, seen=seen, original=original):
            seen.append(command.clone())
            return original(command)

        env.step = record
        evaluator._evaluate_tier(env, policy, policy.actor, "belief", 0.0, 0,
                                 torch, noise_mod)
        commands[tag] = torch.stack(seen)

    torch.testing.assert_close(commands["origin"], commands["shifted"],
                               atol=1.0e-5, rtol=1.0e-4)


def test_first_tick_does_not_integrate(evaluator, policy):
    """belief0 is correct *at* step zero; integrating there double-counts.

    ``rollout`` skips the update at ``t == 0`` for this reason, and the
    evaluator has to make the identical choice or the closed-loop number and
    the training number describe different models.
    """
    from sim2edge import noise as noise_mod
    env = FakeEnv(n=4, max_steps=1)
    report, _ = evaluator._evaluate_tier(env, policy, policy.actor, "belief",
                                         0.0, 0, torch, noise_mod)
    # One tick only, so the reported error is entirely the initialization.
    assert report["belief_error_final_m"] < 1.0e-6


def test_belief_arrival_is_measured_against_the_true_distance(evaluator, policy):
    """Arrival is the robot's own call; the error recorded is the true one."""
    from sim2edge import noise as noise_mod
    env = FakeEnv(n=8, max_steps=60)
    report, _ = evaluator._evaluate_tier(env, policy, policy.actor, "belief",
                                         0.0, 0, torch, noise_mod)
    if report["belief_arrival_rate"] > 0.0:
        assert math.isfinite(report["belief_arrival_error_m"])
        assert report["belief_arrival_error_m"] >= 0.0
    else:
        assert math.isnan(report["belief_arrival_error_m"])


def test_layouts_are_identical_across_modes(evaluator, policy):
    """Tiers are only comparable if they ran the same rooms."""
    _, first = _run(evaluator, policy, "belief")
    _, second = _run(evaluator, policy, "dead-reckoning")
    assert torch.equal(first, second)
