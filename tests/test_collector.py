"""BC recorder policy without launching Isaac."""

import numpy as np
import pytest
import torch

from scripts.collect_bc import BeliefDriver, Collector
from scripts.collect_bc import _assert_tracks, _measure_drive_response
from sim2edge.spec import DT


@pytest.mark.parametrize(
    ("achieved", "commanded"),
    [(0.8, 1.0), (1.2, 1.0), (0.24, 0.3), (0.36, 0.3), (-0.8, -1.0)],
)
def test_drive_response_accepts_twenty_percent_error(achieved, commanded):
    _assert_tracks("test axis", achieved, commanded)


@pytest.mark.parametrize(
    ("achieved", "commanded"),
    [(0.79, 1.0), (1.21, 1.0), (0.23, 0.3), (-0.79, -1.0)],
)
def test_drive_response_rejects_more_than_twenty_percent_error(
        achieved, commanded):
    with pytest.raises(AssertionError, match="does not track"):
        _assert_tracks("test axis", achieved, commanded)


class _TrackingEnv:
    """Minimal kinematic stand-in for the Isaac drive-response measurement."""

    device = "cpu"
    num_envs = 2

    def __init__(self):
        self.pose = torch.zeros(self.num_envs, 3)

    def _robot_pose(self):
        return self.pose

    def step(self, normalized_action):
        from isaaclab_env.nav_task import denormalize_action
        command = denormalize_action(normalized_action)
        heading = self.pose[:, 2].clone()
        self.pose[:, 0] += command[:, 0] * torch.cos(heading) * DT
        self.pose[:, 1] += command[:, 0] * torch.sin(heading) * DT
        self.pose[:, 2] += command[:, 1] * DT
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return {}, torch.zeros(self.num_envs), done, done, {}


@pytest.mark.parametrize("command", [(0.0, 1.0), (0.3, 0.0)])
def test_drive_response_measurement_recovers_kinematic_command(command):
    result = _measure_drive_response(_TrackingEnv(), torch, command,
                                     settle_steps=1, command_steps=3,
                                     measure_steps=2)
    assert result["forward_speed"] == pytest.approx(command[0], abs=1e-5)
    assert result["lateral_speed"] == pytest.approx(0.0, abs=1e-5)
    assert result["yaw_rate"] == pytest.approx(command[1], abs=1e-5)


def _record(c, steps, *, moving=True, layout=0, start_id=0,
            path_length=2.0):
    for step in range(steps):
        ranges = torch.full((c.num_envs, 3), 1.0 + step)
        x = float(step) * (0.1 if moving else 0.0)
        pose = torch.tensor([[x, 0.0, 0.0]] * c.num_envs)
        goal = torch.tensor([[2.0, 0.0]] * c.num_envs)
        action = torch.tensor([[0.2, 0.0]] * c.num_envs)
        c.record(
            ranges, pose, goal, action, action,
            torch.full((c.num_envs,), layout),
            torch.full((c.num_envs,), start_id),
            torch.full((c.num_envs,), path_length),
        )


def test_default_retains_short_collision_prefix(tmp_path):
    c = Collector(tmp_path, 100, 1, max_steps=10, n_beams=3, device="cpu")
    _record(c, 3)
    c.flush(torch.tensor([True]), torch.tensor([False]), torch.tensor([True]),
            torch.zeros(3), filter_failures=False)

    assert c.kept_truncated == 1
    assert c.dropped_short == 0
    assert c.steps_kept == 3
    assert c.pending[0]["outcome"].tolist() == [1, 1, 1]
    c.close(torch.zeros(3))


def test_opt_in_filter_reports_discarded_samples(tmp_path):
    c = Collector(tmp_path, 100, 1, max_steps=10, n_beams=3, device="cpu")
    _record(c, 3)
    c.flush(torch.tensor([True]), torch.tensor([False]), torch.tensor([True]),
            torch.zeros(3), filter_failures=True)

    assert c.episodes_kept == 0
    assert c.dropped_short == 1
    assert c.steps_dropped == 3
    c.close(torch.zeros(3))


def test_shards_are_capped_to_one_episode_of_overshoot(tmp_path):
    c = Collector(tmp_path, 4, 2, max_steps=10, n_beams=3, device="cpu")
    _record(c, 3)
    c.flush(torch.tensor([True, True]), torch.tensor([True, True]),
            torch.tensor([False, False]), torch.zeros(3),
            filter_failures=False)
    c.close(torch.zeros(3))

    assert c.shards_written == 1
    with np.load(tmp_path / "bc_000.npz") as shard:
        assert len(shard["episode"]) == 6
        assert len(shard["episode"]) < c.shard_steps + c.max_steps
        assert shard["outcome"].tolist() == [0] * 6
        assert shard["layout"].tolist() == [0] * 6


def test_layout_split_routes_whole_episode_and_stamps_provenance(tmp_path):
    c = Collector(tmp_path, 100, 1, max_steps=10, n_beams=3, device="cpu",
                  layout_seed=17, num_layouts=20, starts_per_layout=8,
                  sigma=0.15, validation_fraction=0.2)
    validation_layout = int(c.validation_layout_ids[0])
    _record(c, 3, layout=validation_layout, start_id=5, path_length=3.25)
    c.flush(torch.tensor([True]), torch.tensor([True]), torch.tensor([False]),
            torch.zeros(3), filter_failures=False)
    c.close(torch.zeros(3))

    assert not list((tmp_path / "train").glob("*.npz"))
    path = tmp_path / "validation" / "bc_000.npz"
    with np.load(path) as shard:
        assert shard["layout"].tolist() == [validation_layout] * 3
        assert shard["start_id"].tolist() == [5] * 3
        assert shard["path_length"].tolist() == pytest.approx([3.25] * 3)
        assert int(shard["layout_seed"]) == 17
        assert int(shard["starts_per_layout"]) == 8
        assert str(shard["split"]) == "validation"


def test_policy_driver_is_stamped_for_dagger_weighting(tmp_path):
    c = Collector(tmp_path, 100, 1, max_steps=10, n_beams=3, device="cpu",
                  driver="policy")
    _record(c, 3)
    c.flush(torch.tensor([True]), torch.tensor([True]), torch.tensor([False]),
            torch.zeros(3), filter_failures=False)
    c.close(torch.zeros(3))
    with np.load(tmp_path / "bc_000.npz") as shard:
        assert str(shard["driver"]) == "policy"


def test_belief_driver_initializes_only_reset_envs_and_carries_state():
    from types import SimpleNamespace

    class FakePolicy:
        def initial_state(self, goal):
            n = len(goal)
            return (goal.clone(), torch.zeros(n, 2, 2),
                    torch.zeros(n, dtype=torch.long))

        @staticmethod
        def observation(lidar, belief, previous_action):
            return belief

        @staticmethod
        def actor(observation):
            return observation

        @staticmethod
        def step_state(belief, covariance, count, lidar, raw, valid, gyro,
                       previous_action):
            next_belief = belief + 1.0
            arrived = torch.ones(len(belief), dtype=torch.bool)
            return (next_belief, arrived, next_belief, covariance,
                    count + 1, torch.ones(len(belief)))

    class FakeEnv:
        num_envs = 2
        device = "cpu"

        def __init__(self):
            self.pose = torch.tensor([[0.0, 0.0, 0.0],
                                      [1.0, 1.0, 0.0]])
            self._goal_w = torch.tensor([[2.0, 0.0], [1.0, 4.0]])
            self.scene = SimpleNamespace(env_origins=torch.zeros(2, 3))
            self._observation_history = SimpleNamespace(
                lidar=torch.zeros(2, 4, 180))
            self.actions = torch.zeros(2, 4, 2)
            self.raw = torch.zeros(2, 4, 720)
            self.valid = torch.ones(2, 4, 720, dtype=torch.bool)
            self.gyro = torch.zeros(2, 4)

        def _robot_pose(self):
            return self.pose

        def motion_action(self):
            return self.actions

        def motion_scan(self):
            return self.raw, self.valid

        def gyro_increment(self):
            return self.gyro

    env = FakeEnv()
    driver = BeliefDriver(FakePolicy(), env, torch)

    first, arrived = driver.action()
    torch.testing.assert_close(first, torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    assert not arrived.any()  # reset never double-integrates the opening tick

    second, arrived = driver.action()
    torch.testing.assert_close(second, torch.tensor([[3.0, 1.0], [1.0, 4.0]]))
    assert arrived.all()

    driver.reset(torch.tensor([True, False]))
    env.pose[0] = torch.tensor([1.0, 0.0, 0.0])
    env._goal_w[0] = torch.tensor([1.5, 0.0])
    third, arrived = driver.action()
    torch.testing.assert_close(third, torch.tensor([[0.5, 0.0], [2.0, 5.0]]))
    assert not arrived[0] and arrived[1]


def test_compression_runs_asynchronously(tmp_path, monkeypatch):
    import threading
    from sim2edge import dataset

    started = threading.Event()
    release = threading.Event()
    real_save = dataset.save_shard

    def slow_save(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return real_save(*args, **kwargs)

    monkeypatch.setattr(dataset, "save_shard", slow_save)
    c = Collector(tmp_path, 2, 1, max_steps=10, n_beams=3, device="cpu")
    _record(c, 3)
    c.flush(torch.tensor([True]), torch.tensor([True]), torch.tensor([False]),
            torch.zeros(3), filter_failures=False)
    assert started.wait(timeout=2)
    # flush returned even though the writer is deliberately still blocked.
    assert not (tmp_path / "bc_000.npz").exists()
    release.set()
    c.close(torch.zeros(3))
    assert (tmp_path / "bc_000.npz").is_file()
