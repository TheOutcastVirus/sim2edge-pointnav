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
from sim2edge.spec import DT, GOAL_MAX_M, V_MAX


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


def test_belief_policy_reads_no_pose_anywhere():
    """The point of the design, asserted against the signature.

    ``step`` takes a belief, a scan, a gyro reading and the last command. If a
    pose or a goal in world coordinates ever appears here, the odometry input
    has come back in through a side door.
    """
    import inspect
    parameters = set(inspect.signature(BeliefNavPolicy.step).parameters)
    assert parameters == {"self", "belief", "lidar", "gyro", "previous_action"}
