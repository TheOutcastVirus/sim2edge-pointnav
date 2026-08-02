"""The goal-belief policy: composition, causal ordering, and warm-start transfer.

The composition test is the important one. Every other part of this design can
be wrong by a tolerable amount; a sign error in ``compose_belief`` produces a
robot that circles its goal forever, and it is invisible in any loss curve.
"""

from __future__ import annotations

import math

import pytest
import torch

from sim2edge.models import (BeliefNavPolicy, BeliefNavPolicyConfig, NavPolicy,
                             NavPolicyConfig, compose_belief)
from sim2edge.preprocess import goal_features, relative_goal_features
from sim2edge.spec import (DT, GOAL_MAX_M, LIDAR_MAX_M, V_MAX, fingerprint,
                           observation_fingerprint)
from sim2edge.training import load_policy_checkpoint


def _body_frame(pose: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    delta = goal - pose[:, :2]
    cos, sin = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
    return torch.stack([cos * delta[:, 0] + sin * delta[:, 1],
                        -sin * delta[:, 0] + cos * delta[:, 1]], dim=1)


def test_composition_reproduces_the_true_belief_over_a_trajectory():
    """Exact inputs must give exact output for 200 steps, or the algebra is wrong."""
    torch.manual_seed(0)
    steps, envs = 200, 32
    goal = torch.randn(envs, 2) * 4.0
    pose = torch.zeros(steps + 1, envs, 3)
    pose[0, :, 2] = torch.rand(envs) * 2 * math.pi
    speed = torch.rand(steps, envs) * V_MAX
    yaw_rate = (torch.rand(steps, envs) * 2 - 1) * 1.9

    for t in range(steps):
        yaw = pose[t, :, 2]
        pose[t + 1, :, 0] = pose[t, :, 0] + speed[t] * torch.cos(yaw) * DT
        pose[t + 1, :, 1] = pose[t, :, 1] + speed[t] * torch.sin(yaw) * DT
        pose[t + 1, :, 2] = yaw + yaw_rate[t] * DT

    belief = _body_frame(pose[0], goal)
    for t in range(steps):
        yaw = pose[t, :, 2]
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        world = pose[t + 1, :, :2] - pose[t, :, :2]
        body = torch.stack([cos * world[:, 0] + sin * world[:, 1],
                            -sin * world[:, 0] + cos * world[:, 1]], dim=1)
        belief = compose_belief(belief, body, pose[t + 1, :, 2] - pose[t, :, 2])

    error = (belief - _body_frame(pose[-1], goal)).norm(dim=1).max()
    assert float(error) < 1e-3


def test_composition_of_pure_rotation_rotates_the_goal_the_other_way():
    """Turning left must move the goal right in the body frame."""
    belief = torch.tensor([[2.0, 0.0]])
    turned = compose_belief(belief, torch.zeros(1, 2),
                            torch.tensor([math.pi / 2]))
    assert torch.allclose(turned, torch.tensor([[0.0, -2.0]]), atol=1e-6)


def test_composition_of_pure_translation_subtracts_it():
    belief = torch.tensor([[3.0, 1.0]])
    moved = compose_belief(belief, torch.tensor([[0.5, 0.0]]), torch.zeros(1))
    assert torch.allclose(moved, torch.tensor([[2.5, 1.0]]), atol=1e-6)


def test_relative_and_pose_derived_goal_features_agree():
    """One encoder, two ways in. If these drift, a checkpoint changes meaning."""
    generator = torch.Generator().manual_seed(5)
    pose = torch.randn(256, 3, generator=generator) * 2.0
    goal = torch.randn(256, 2, generator=generator) * 3.0
    assert torch.allclose(goal_features(pose, goal),
                          relative_goal_features(_body_frame(pose, goal)),
                          atol=1e-6)


def test_goal_features_survive_standing_on_the_goal():
    assert torch.isfinite(relative_goal_features(torch.zeros(4, 2))).all()


def test_untrained_head_is_exactly_dead_reckoning():
    """Option A must be recoverable from option B, or the baseline is not nested."""
    torch.manual_seed(1)
    policy = BeliefNavPolicy()
    previous = torch.zeros(16, 4, 2)
    previous[:, -1, 0] = 0.7                       # normalized forward command
    with torch.no_grad():
        step = policy.translation(torch.rand(16, 4, 180), torch.zeros(16, 4),
                                  previous)
    expected = 0.7 * V_MAX * DT
    assert float((step[:, 0] - expected).abs().max()) < 1e-3
    assert float(step[:, 1].abs().max()) < 1e-3


def test_gradients_reach_the_convolutional_stack():
    """A zero-initialized head would starve everything upstream on step one."""
    torch.manual_seed(2)
    policy = BeliefNavPolicy()
    actions, beliefs = policy.rollout(
        torch.rand(8, 4, 4, 180), torch.randn(8, 4, 4) * 0.02,
        torch.zeros(8, 4, 4, 2), torch.randn(4, 2))
    (actions.square().mean() + beliefs.square().mean()).backward()
    assert float(policy.odometry.conv[0].weight.grad.abs().sum()) > 0.0


def test_rollout_does_not_integrate_on_the_first_step():
    """``belief0`` is already correct at step zero; updating there double-counts."""
    torch.manual_seed(3)
    policy = BeliefNavPolicy()
    initial = torch.randn(6, 2)
    _, beliefs = policy.rollout(torch.rand(5, 6, 4, 180), torch.randn(5, 6, 4),
                                torch.zeros(5, 6, 4, 2), initial)
    assert torch.allclose(beliefs[0], initial)


def test_rollout_matches_stepping_by_hand():
    """The training path and the deployed path must be one implementation."""
    torch.manual_seed(4)
    policy = BeliefNavPolicy().eval()
    lidar = torch.rand(6, 3, 4, 180)
    gyro = torch.randn(6, 3, 4) * 0.02
    previous = torch.randn(6, 3, 4, 2) * 0.1
    initial = torch.randn(3, 2)

    with torch.no_grad():
        actions, beliefs = policy.rollout(lidar, gyro, previous, initial)
        belief = initial
        for t in range(6):
            if t == 0:
                action = policy.actor(
                    policy.observation(lidar[0], belief, previous[0]))
            else:
                action, belief = policy.step(belief, lidar[t], gyro[t],
                                             previous[t])
            assert torch.allclose(actions[t], action, atol=1e-6)
            assert torch.allclose(beliefs[t], belief, atol=1e-6)


def test_actor_transfers_whole_from_a_nav_policy_checkpoint():
    """The warm start is the reason the actor observation was left untouched."""
    torch.manual_seed(6)
    reference = NavPolicy(NavPolicyConfig()).eval()
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(actor=NavPolicyConfig()))
    policy.load_actor(reference.state_dict())            # strict by design
    observation = torch.randn(12, reference.config.observation_dim)
    with torch.no_grad():
        assert torch.allclose(reference(observation),
                              policy.actor(observation), atol=1e-7)


def test_initial_belief_clamps_to_the_trained_range():
    policy = BeliefNavPolicy()
    far = policy.initial_belief(torch.tensor([[40.0, 0.0], [0.0, -3.0]]))
    assert float(far[0].norm()) == pytest.approx(GOAL_MAX_M, abs=1e-4)
    assert float(far[1].norm()) == pytest.approx(3.0, abs=1e-4)


def test_persistent_calibration_filters_forward_scale_measurements():
    config = BeliefNavPolicyConfig(
        persistent_calibration=True, calibration_update_rate=0.2)
    policy = BeliefNavPolicy(config).eval()
    with torch.no_grad():
        policy.odometry.proposal_head.weight.zero_()
        policy.odometry.proposal_head.bias[:] = torch.tensor([
            torch.atanh(torch.tensor(0.2)), 0.0])
        policy.odometry.confidence_head[-1].weight.zero_()
        policy.odometry.confidence_head[-1].bias.fill_(20.0)

    lidar = torch.zeros(1, 4, 180)
    gyro = torch.zeros(1, 4)
    previous = torch.zeros(1, 4, 2)
    previous[:, :, 0] = 1.0
    raw = torch.ones(1, 4, 720)
    valid = torch.ones_like(raw, dtype=torch.bool)
    calibration = policy.initial_calibration(torch.zeros(1, 2))

    first, _, confidence, calibration = policy.calibrated_motion_estimate(
        calibration, lidar, gyro, previous, raw, valid)
    assert float(confidence.detach()) == pytest.approx(1.0, abs=1e-5)
    assert float(calibration.detach()) == pytest.approx(0.02, abs=1e-5)
    assert float(first[0, 0].detach()) == pytest.approx(
        V_MAX * DT * 1.02, abs=1e-6)

    second, _, _, calibration = policy.calibrated_motion_estimate(
        calibration, lidar, gyro, previous, raw, valid)
    assert float(calibration.detach()) == pytest.approx(0.036, abs=1e-5)
    assert float(second[0, 0].detach()) == pytest.approx(
        V_MAX * DT * 1.036, abs=1e-6)


def test_persistent_calibration_requires_explicit_live_state():
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(
        persistent_calibration=True)).eval()
    belief, covariance, count = policy.initial_state(torch.zeros(1, 2))
    with pytest.raises(ValueError, match="requires carried state"):
        policy.step_state(
            belief, covariance, count, torch.zeros(1, 4, 180),
            torch.zeros(1, 4, 720), torch.ones(1, 4, 720, dtype=torch.bool),
            torch.zeros(1, 4), torch.zeros(1, 4, 2))


def test_learned_motion_state_starts_from_verified_stateless_translation():
    torch.manual_seed(19)
    baseline = BeliefNavPolicy().eval()
    recurrent = BeliefNavPolicy(BeliefNavPolicyConfig(
        motion_state_size=16)).eval()
    incompatible = recurrent.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith(("odometry.motion_memory.",
                               "odometry.motion_scale_head."))
               for key in incompatible.missing_keys)

    lidar = torch.rand(3, 4, 180)
    gyro = torch.randn(3, 4) * 0.01
    previous = torch.rand(3, 4, 2) * 2.0 - 1.0
    previous[:, -1, 0] = 1.0
    raw = torch.rand(3, 4, 720) * LIDAR_MAX_M
    valid = torch.ones_like(raw, dtype=torch.bool)
    expected = baseline.motion_estimate(
        lidar, gyro, previous, raw, valid)[0]
    state = recurrent.initial_calibration(torch.zeros(3, 2))
    actual, _, _, next_state = recurrent.calibrated_motion_estimate(
        state, lidar, gyro, previous, raw, valid)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert next_state.shape == (3, 16)
    assert not torch.equal(next_state, state)


def test_motion_rollout_backpropagates_into_learned_memory():
    torch.manual_seed(20)
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(motion_state_size=8))
    steps, envs = 5, 2
    lidar = torch.rand(steps, envs, 4, 180)
    gyro = torch.randn(steps, envs, 4) * 0.01
    previous = torch.rand(steps, envs, 4, 2) * 2.0 - 1.0
    raw = torch.rand(steps, envs, 4, 720) * LIDAR_MAX_M
    valid = torch.ones_like(raw, dtype=torch.bool)
    translation, _, _ = policy.motion_rollout(
        lidar, gyro, previous, raw, valid)
    translation[1:, :, 0].sum().backward()
    gradient = policy.odometry.motion_scale_head.weight.grad
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0


def test_learned_motion_memory_is_exact_dead_reckoning_without_returns():
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(motion_state_size=8)).eval()
    lidar = torch.zeros(2, 4, 180)
    gyro = torch.zeros(2, 4)
    previous = torch.zeros(2, 4, 2)
    previous[:, -1, 0] = torch.tensor([0.5, 1.0])
    raw = torch.full((2, 4, 720), LIDAR_MAX_M)
    valid = torch.zeros_like(raw, dtype=torch.bool)
    state = policy.initial_calibration(torch.zeros(2, 2))
    translation, _, confidence, next_state = (
        policy.calibrated_motion_estimate(
            state, lidar, gyro, previous, raw, valid))
    expected = torch.stack([
        previous[:, -1, 0] * V_MAX * DT,
        torch.zeros(2),
    ], dim=1)
    torch.testing.assert_close(translation, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(next_state, state, atol=0.0, rtol=0.0)
    assert not confidence.any()


def test_belief_policy_reads_no_pose_anywhere():
    """The point of the design, asserted against the signature.

    ``step`` takes a belief, a scan, a gyro reading and the last command. If a
    pose or a goal in world coordinates ever appears here, the odometry input
    has come back in through a side door.
    """
    import inspect
    parameters = set(inspect.signature(BeliefNavPolicy.step).parameters)
    assert parameters == {"self", "belief", "lidar", "gyro", "previous_action"}


def test_belief_checkpoint_actor_loads_for_pose_driven_dagger(tmp_path):
    """DAgger executes the actor on exact goal features while retaining labels."""
    policy = BeliefNavPolicy().eval()
    checkpoint = {
        "algorithm": "belief-ppo",
        "spec_fingerprint": fingerprint(),
        "observation_fingerprint": observation_fingerprint(
            policy.config.actor.history_frames, belief=True),
        "model_config": policy.config.to_dict(),
        "model": policy.state_dict(),
    }
    path = tmp_path / "belief.pt"
    torch.save(checkpoint, path)
    actor = load_policy_checkpoint(path)
    observation = torch.randn(5, actor.config.observation_dim)
    torch.testing.assert_close(actor(observation), policy.actor(observation))


def test_scan_motion_solver_recovers_translation_in_a_circular_room():
    """The estimator contains an actual scan-motion measurement."""
    policy = BeliefNavPolicy().eval()
    beams = policy.config.raw_lidar_bins
    angle = ((torch.arange(beams) + 0.5) * (2 * math.pi / beams) - math.pi)
    pose0 = torch.tensor([0.2, -0.3, 0.4])
    distance, delta_yaw = 0.02, 0.03
    pose1 = pose0.clone()
    pose1[:2] += distance * torch.tensor([math.cos(0.4), math.sin(0.4)])
    pose1[2] += delta_yaw

    def scan(pose):
        bearing = pose[2] + angle
        b = pose[0] * torch.cos(bearing) + pose[1] * torch.sin(bearing)
        c = pose[0].square() + pose[1].square() - 16.0
        return -b + torch.sqrt(b.square() - c)

    first, second = scan(pose0), scan(pose1)
    raw = torch.stack([first, first, first, second])[None]
    valid = torch.ones_like(raw, dtype=torch.bool)
    lidar = torch.zeros(1, 4, 180)
    gyro = torch.tensor([[0.0, 0.0, 0.0, delta_yaw]])
    previous = torch.zeros(1, 4, 2)
    previous[0, -1, 0] = distance / (V_MAX * DT)
    with torch.no_grad():
        policy.odometry.head.bias.fill_(10.0)
        translation, _, confidence = policy.motion_estimate(
            lidar, gyro, previous, raw, valid)
    expected = torch.tensor([[distance, 0.0]])
    torch.testing.assert_close(translation, expected, atol=2e-4, rtol=0.0)
    assert float(confidence) > 0.99


def test_invalid_scan_falls_back_to_command_dead_reckoning():
    policy = BeliefNavPolicy().eval()
    previous = torch.zeros(3, 4, 2)
    previous[:, -1, 0] = 0.6
    raw = torch.full((3, 4, 720), LIDAR_MAX_M)
    valid = torch.zeros_like(raw, dtype=torch.bool)
    with torch.no_grad():
        translation, _, confidence = policy.motion_estimate(
            torch.ones(3, 4, 180), torch.zeros(3, 4), previous, raw, valid)
    expected = torch.full((3,), 0.6 * V_MAX * DT)
    torch.testing.assert_close(translation[:, 0], expected, atol=1e-4, rtol=0.0)
    assert float(translation[:, 1].abs().max()) < 1e-4
    assert float(confidence.max()) < 0.01


def test_uncertainty_aware_arrival_requires_three_quiet_ticks():
    policy = BeliefNavPolicy().eval()
    with torch.no_grad():
        policy.actor.action_head.weight.zero_()
        policy.actor.action_head.bias.copy_(torch.tensor([-10.0, 0.0]))
    belief, covariance, count = policy.initial_state(torch.zeros(2, 2))
    lidar = torch.ones(2, 4, 180)
    raw = torch.full((2, 4, 720), LIDAR_MAX_M)
    valid = torch.zeros_like(raw, dtype=torch.bool)
    gyro = torch.zeros(2, 4)
    previous = torch.zeros(2, 4, 2)
    for tick in range(3):
        _, arrived, belief, covariance, count, _ = policy.step_state(
            belief, covariance, count, lidar, raw, valid, gyro, previous)
        assert bool(arrived.all()) == (tick == 2)


def test_arrival_stops_a_moving_actor_after_confirmation():
    """A stop gate cannot require an actor with no stop state to stop first."""
    policy = BeliefNavPolicy().eval()
    with torch.no_grad():
        policy.actor.action_head.weight.zero_()
        policy.actor.action_head.bias.copy_(torch.tensor([10.0, 0.0]))
    belief, covariance, count = policy.initial_state(torch.zeros(1, 2))
    lidar = torch.ones(1, 4, 180)
    raw = torch.full((1, 4, 720), LIDAR_MAX_M)
    valid = torch.zeros_like(raw, dtype=torch.bool)
    gyro = torch.zeros(1, 4)
    previous = torch.zeros(1, 4, 2)
    for _ in range(policy.config.arrival_confirm_ticks):
        command, arrived, belief, covariance, count, _ = policy.step_state(
            belief, covariance, count, lidar, raw, valid, gyro, previous)
    assert bool(arrived)
    torch.testing.assert_close(command, torch.zeros_like(command))
