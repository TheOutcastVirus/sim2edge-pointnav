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
                             ScanVote,
                             OdometryHead,
                             NavPolicyConfig, compose_belief)
from sim2edge.preprocess import goal_features, relative_goal_features
from sim2edge.spec import (DT, GOAL_MAX_M, LIDAR_MAX_M, MOTION_HISTORY_FRAMES,
                           SENSOR_CONTRACT_VERSION, V_MAX, fingerprint,
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
        "sensor_contract_version": SENSOR_CONTRACT_VERSION,
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


# ── gyro derotation of the scan history ──────────────────────────────────────
# Measured on `belief_onpolicy_v15`, yaw across the one-second motion window
# shifts scan content by 44 of 720 bins on average and 137 at p95, against an
# encoder receptive field of about 28. These cover the alignment that removes it.


def _offset_circular_room(bearing: torch.Tensor, x: float, y: float
                          ) -> torch.Tensor:
    """Range to a radius-4 wall from a point off its centre. Not rotationally
    symmetric, so an alignment that got the sign or scale wrong cannot pass."""
    b = x * torch.cos(bearing) + y * torch.sin(bearing)
    c = x * x + y * y - 16.0
    return -b + torch.sqrt(b.square() - c)


def _spun_history(frames: int, beams: int, yaw_rate: float
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """A static scene observed while turning in place, plus the gyro that saw it."""
    angle = (torch.arange(beams) + 0.5) * (2 * math.pi / beams) - math.pi
    yaw = torch.arange(frames, dtype=torch.float32) * yaw_rate
    scans = torch.stack([
        _offset_circular_room(yaw[i] + angle, 0.9, -1.3) for i in range(frames)
    ])[None]
    gyro = torch.full((1, frames), yaw_rate)
    gyro[:, 0] = 0.0                    # nothing precedes the oldest frame
    return scans, gyro


def test_derotation_makes_a_pure_spin_look_stationary():
    """The decisive geometric check: rotation is known, so it must cancel exactly.

    Turning in place changes every bearing's range but moves the robot nowhere.
    After alignment each frame must reproduce the newest one; whatever survives
    is translation parallax, which is the signal the head is asked to measure.
    """
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(derotate_scan_history=True))
    frames = policy.config.motion_history_frames
    beams = policy.config.raw_lidar_bins
    scans, gyro = _spun_history(frames, beams, yaw_rate=0.06)
    valid = torch.ones_like(scans, dtype=torch.bool)

    aligned, usable = policy.odometry._align_history(scans, valid, gyro)

    newest = scans[:, -1]
    before = (scans - newest[:, None]).abs().max()
    after = (aligned - newest[:, None]).abs().max()
    # 1.1 rad of accumulated yaw moves the wall metres in the raw history.
    assert float(before) > 0.5
    # What is left is linear-interpolation error on a 0.5-degree grid.
    assert float(after) < 5e-3
    assert bool(usable.all())


def test_derotation_leaves_the_newest_frame_bit_exact():
    """Frame -1 is the reference frame; resampling it would corrupt the anchor."""
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(derotate_scan_history=True))
    frames = policy.config.motion_history_frames
    beams = policy.config.raw_lidar_bins
    torch.manual_seed(5)
    scans = torch.rand(2, frames, beams) * LIDAR_MAX_M
    valid = torch.rand(2, frames, beams) > 0.3
    gyro = torch.randn(2, frames) * 0.05

    aligned, usable = policy.odometry._align_history(scans, valid, gyro)

    torch.testing.assert_close(aligned[:, -1], scans[:, -1], atol=0.0, rtol=0.0)
    assert bool((usable[:, -1] == valid[:, -1]).all())


def test_derotation_is_identity_when_the_robot_does_not_turn():
    """Zero gyro must reproduce the unrotated encoder exactly, flag or not."""
    torch.manual_seed(6)
    plain = BeliefNavPolicy().eval()
    aligned = BeliefNavPolicy(
        BeliefNavPolicyConfig(derotate_scan_history=True)).eval()
    aligned.load_state_dict(plain.state_dict())
    frames = plain.config.motion_history_frames
    raw = torch.rand(4, frames, plain.config.raw_lidar_bins) * LIDAR_MAX_M
    valid = torch.rand_like(raw) > 0.2
    previous = torch.rand(4, frames, 2)
    lidar = torch.rand(4, frames, 180)

    with torch.no_grad():
        quiet = [p.motion_estimate(lidar, torch.zeros(4, frames), previous,
                                  raw, valid)[0] for p in (plain, aligned)]
        turning = [p.motion_estimate(lidar, torch.full((4, frames), 0.05),
                                     previous, raw, valid)[0]
                   for p in (plain, aligned)]

    torch.testing.assert_close(quiet[0], quiet[1], atol=1e-6, rtol=0.0)
    # And it must actually be doing something once the robot turns, or the flag
    # is silently inert and the test above proves nothing.
    assert float((turning[0] - turning[1]).abs().max()) > 0.0


def test_derotation_keeps_the_zero_return_fallback_exact():
    """The deployable guarantee survives the new input path: no scan, no motion."""
    policy = BeliefNavPolicy(
        BeliefNavPolicyConfig(derotate_scan_history=True)).eval()
    frames = policy.config.motion_history_frames
    previous = torch.zeros(3, frames, 2)
    previous[:, -1, 0] = 0.6
    raw = torch.full((3, frames, policy.config.raw_lidar_bins), LIDAR_MAX_M)
    valid = torch.zeros_like(raw, dtype=torch.bool)
    with torch.no_grad():
        translation, _, confidence = policy.motion_estimate(
            torch.ones(3, frames, 180), torch.randn(3, frames) * 0.05,
            previous, raw, valid)
    expected = torch.full((3,), 0.6 * V_MAX * DT)
    torch.testing.assert_close(translation[:, 0], expected, atol=1e-4, rtol=0.0)
    assert float(translation[:, 1].abs().max()) < 1e-4
    assert float(confidence.max()) < 0.01


def test_derotation_passes_gradient_to_the_convolutional_stack():
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(derotate_scan_history=True))
    frames = policy.config.motion_history_frames
    torch.manual_seed(7)
    raw = torch.rand(2, 3, frames, policy.config.raw_lidar_bins) * LIDAR_MAX_M
    valid = torch.ones_like(raw, dtype=torch.bool)
    # The actor keeps its four-frame observation; only the estimator sees 20.
    actions, beliefs = policy.rollout(
        torch.rand(2, 3, 4, 180), torch.randn(2, 3, frames) * 0.05,
        torch.zeros(2, 3, frames, 2), torch.randn(3, 2), raw, valid)
    (actions.square().mean() + beliefs.square().mean()).backward()
    assert float(policy.odometry.conv[0].weight.grad.abs().sum()) > 0.0


def test_scaled_difference_ignores_bearings_missing_from_either_frame():
    """A no-return filled at max range must not read as a nine-metre closure."""
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(
        derotate_scan_history=True, scaled_scan_difference=True)).eval()
    frames = policy.config.motion_history_frames
    beams = policy.config.raw_lidar_bins
    # Identical geometry in every frame, but the oldest frame lost half its
    # returns. With the mask absent, those bearings fabricate a huge difference.
    raw = torch.full((1, frames, beams), 3.0)
    valid = torch.ones_like(raw, dtype=torch.bool)
    valid[0, 0, : beams // 2] = False
    gyro = torch.zeros(1, frames)
    previous = torch.zeros(1, frames, 2)
    with torch.no_grad():
        intact = policy.motion_estimate(
            torch.ones(1, frames, 180), gyro, previous,
            raw, torch.ones_like(valid))[0]
        holed = policy.motion_estimate(
            torch.ones(1, frames, 180), gyro, previous, raw, valid)[0]
    # The remaining difference is only what the validity channel itself carries,
    # not an amplified 9 m artefact.
    assert float((intact - holed).abs().max()) < 2e-3


def test_scaled_difference_lifts_a_centimetre_signal_off_the_floor():
    """The channel the head must read cannot sit three orders below its neighbour."""
    frames = MOTION_HISTORY_FRAMES
    beams = 720
    plain = OdometryHead(BeliefNavPolicyConfig())
    scaled = OdometryHead(BeliefNavPolicyConfig(scaled_scan_difference=True))
    # A uniform 2 cm range closure: the size of one full-speed tick.
    metric = torch.full((1, frames, beams), 3.0)
    metric[:, -1] -= 0.02
    valid = torch.ones_like(metric, dtype=torch.bool)

    captured: dict[str, torch.Tensor] = {}

    def grab(name, head):
        head.conv.register_forward_pre_hook(
            lambda _module, args: captured.__setitem__(name, args[0]))

    grab("plain", plain)
    grab("scaled", scaled)
    args = (torch.ones(1, frames, 180), torch.zeros(1, frames),
            torch.zeros(1, frames, 2), metric, valid)
    with torch.no_grad():
        plain._features(*args)
        scaled._features(*args)

    magnitude = lambda name: float(
        captured[name][:, frames:2 * frames].abs().max())
    assert magnitude("plain") < 2.0e-3
    assert magnitude("scaled") > 4.0e-2


def test_confidence_gate_bias_default_reproduces_existing_checkpoints():
    """The stuck-gate fix must be opt-in; -2.0 is what every checkpoint holds."""
    head = OdometryHead(BeliefNavPolicyConfig())
    assert float(head.confidence_head[-1].bias.detach()) == pytest.approx(-2.0)


def test_raising_the_gate_bias_leaves_the_correction_at_dead_reckoning():
    """Mid-range gate, near-zero proposal: the gate gains gradient without
    moving the initial translation off the command prior."""
    policy = BeliefNavPolicy(
        BeliefNavPolicyConfig(confidence_gate_bias=0.0)).eval()
    frames = policy.config.motion_history_frames
    previous = torch.zeros(8, frames, 2)
    previous[:, -1, 0] = 0.7
    raw = torch.rand(8, frames, policy.config.raw_lidar_bins) * LIDAR_MAX_M
    valid = torch.ones_like(raw, dtype=torch.bool)
    with torch.no_grad():
        step = policy.translation(torch.rand(8, frames, 180),
                                  torch.zeros(8, frames), previous, raw, valid)
        _, _, confidence = policy.motion_estimate(
            torch.rand(8, frames, 180), torch.zeros(8, frames), previous,
            raw, valid)
    assert float(confidence.mean()) == pytest.approx(0.5, abs=0.02)
    assert float((step[:, 0] - 0.7 * V_MAX * DT).abs().max()) < 1e-3


def test_the_belief_loss_alone_starves_the_confidence_gate():
    """Documents the defect the ranking loss exists to fix.

    Every geometric loss reaches the gate through ``confidence * proposal``, so
    its gradient carries a factor of a proposal initialized at ~1e-3. Raising the
    gate's bias does not help: it lifts the proposal's gradient by the same
    sigmoid factor. Both biases leave the gate three orders of magnitude behind,
    which is why every trained checkpoint reports a gate frozen at its init.
    """
    def gradients(bias):
        torch.manual_seed(11)
        policy = BeliefNavPolicy(BeliefNavPolicyConfig(
            derotate_scan_history=True, confidence_gate_bias=bias))
        frames = policy.config.motion_history_frames
        raw = torch.rand(4, 6, frames, policy.config.raw_lidar_bins) * LIDAR_MAX_M
        valid = torch.ones_like(raw, dtype=torch.bool)
        _, beliefs = policy.rollout(
            torch.rand(4, 6, 4, 180), torch.randn(4, 6, frames) * 0.05,
            torch.zeros(4, 6, frames, 2), torch.randn(6, 2), raw, valid)
        beliefs.square().mean().backward()
        total = lambda module: float(module.weight.grad.abs().sum())
        return (total(policy.odometry.confidence_head[-1]),
                total(policy.odometry.proposal_head))

    for bias in (-2.0, 0.0):
        gate, proposal = gradients(bias)
        assert gate < proposal / 1000.0, f"gate not starved at bias {bias}"


def _synthetic_sequence_batch(steps=6, envs=8):
    """A minimal well-formed window; the loss only needs shapes and finiteness."""
    from sim2edge.dataset import SequenceBatch
    from sim2edge.spec import HISTORY_FRAMES
    torch.manual_seed(12)
    motion = MOTION_HISTORY_FRAMES
    return SequenceBatch(
        lidar=torch.rand(steps, envs, HISTORY_FRAMES, 180),
        raw_lidar=torch.rand(steps, envs, motion, 720) * LIDAR_MAX_M,
        raw_valid=torch.ones(steps, envs, motion, 720, dtype=torch.bool),
        gyro=torch.randn(steps, envs, motion) * 0.05,
        previous_action=torch.rand(steps, envs, motion, 2) * 0.5,
        action=torch.rand(steps, envs, 2) * 0.3,
        belief=torch.randn(steps, envs, 2),
        initial_belief=torch.randn(envs, 2),
        translation=torch.rand(steps, envs, 2) * 0.02,
        valid_steps=torch.ones(steps, envs, dtype=torch.bool),
    )


def test_confidence_ranking_loss_unstarves_the_gate():
    """The point of the term: an O(1) gate gradient that skips the proposal.

    Without it the gate's gradient is under a thousandth of the proposal head's
    (see ``test_the_belief_loss_alone_starves_the_confidence_gate``). With it the
    gate must become the better-fed of the two.
    """
    from sim2edge.training import BeliefConfig, belief_losses

    def gate_and_proposal_gradients(confidence_coef):
        torch.manual_seed(13)
        policy = BeliefNavPolicy(BeliefNavPolicyConfig(
            derotate_scan_history=True, scaled_scan_difference=True))
        config = BeliefConfig(batch_size=8, seq_len=6, amp="off",
                              confidence_coef=confidence_coef)
        loss, _ = belief_losses(policy, _synthetic_sequence_batch(), config)
        loss.backward()
        total = lambda module: float(module.weight.grad.abs().sum())
        return (total(policy.odometry.confidence_head[-1]),
                total(policy.odometry.proposal_head))

    off_gate, off_proposal = gate_and_proposal_gradients(0.0)
    on_gate, on_proposal = gate_and_proposal_gradients(20.0)
    # Left alone, the gate trails the proposal head by nearly three orders of
    # magnitude even with the mismatched-lidar term already in the total.
    assert off_gate < off_proposal / 500.0
    # The ranking term is the only path to the gate that does not carry a factor
    # of the proposal, so it scales the gate's gradient directly.
    assert on_gate > 50.0 * off_gate
    assert on_gate > on_proposal / 20.0


def test_confidence_ranking_loss_is_zero_once_the_gate_separates():
    """A hinge, not a penalty: a gate that already discriminates is left alone."""
    from sim2edge.training import BeliefConfig, belief_losses
    torch.manual_seed(14)
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(
        derotate_scan_history=True, scaled_scan_difference=True))
    # A gate that is saturated high everywhere cannot separate matched from
    # mismatched, so the hinge must be active...
    with torch.no_grad():
        policy.odometry.confidence_head[-1].bias.fill_(20.0)
    config = BeliefConfig(batch_size=8, seq_len=6, amp="off",
                          confidence_coef=1.0)
    _, flat = belief_losses(policy, _synthetic_sequence_batch(), config)
    assert float(flat["confidence_loss"]) == pytest.approx(
        config.confidence_margin, abs=1e-3)


# ── the scan vote ────────────────────────────────────────────────────────────
#
# These tests pin the *reason* the vote exists rather than its shapes. The
# design claim is specifically that agreement-counting survives the gross
# outliers that defeat least squares, so the outlier test below runs both
# estimators on identical residuals and requires them to disagree. If someone
# ever swaps the vote for a solve because it "does the same thing more
# directly", that test is what stops them.


def _circular_room_scan(position: torch.Tensor, beams: int,
                        radius: float = 4.0) -> torch.Tensor:
    """Exact ranges from ``position`` inside a circle centred on the origin."""
    angle = -math.pi + (torch.arange(beams, dtype=torch.float64) + 0.5) * (
        2 * math.pi / beams)
    b = position[0] * torch.cos(angle) + position[1] * torch.sin(angle)
    c = position.square().sum() - radius ** 2
    return (-b + torch.sqrt(b.square() - c)).float()


def _vote_inputs(translation, beams=720, outlier_fraction=0.0, seed=0):
    """Two analytic scans separated by ``translation``, optionally corrupted."""
    start = torch.tensor([0.2, -0.3], dtype=torch.float64)
    first = _circular_room_scan(start, beams)
    second = _circular_room_scan(start + translation.double(), beams)
    difference = (second - first)[None]
    weight = torch.ones_like(difference)
    if outlier_fraction > 0.0:
        generator = torch.Generator().manual_seed(seed)
        corrupt = torch.rand(beams, generator=generator) < outlier_fraction
        # Depth edges: a surface visible in one frame and occluded in the next
        # produces a range jump of order the room, not of order the motion.
        jump = (torch.rand(beams, generator=generator) - 0.5) * 2.0
        difference = torch.where(corrupt[None], jump[None], difference)
    return difference, weight


def _scan_vote(bins=41, sigma=0.01, radius=3):
    return ScanVote(720, bins, 1.5 * V_MAX * DT, sigma, radius)


def test_scan_vote_recovers_a_known_translation_in_a_clean_room():
    truth = torch.tensor([0.019, 0.004])
    estimate, _, sharpness = _scan_vote()(*_vote_inputs(truth))
    torch.testing.assert_close(estimate, truth[None], atol=1.5e-3, rtol=0.0)
    assert float(sharpness) > 0.05


def test_scan_vote_survives_outliers_that_defeat_least_squares():
    """The whole reason this is a vote and not a solve.

    Thirty per cent of bearings are replaced with metre-scale jumps, which is
    the depth-edge rate measured on ``belief_onpolicy_v15``. A squared residual
    is unbounded, so those bearings relocate a least-squares minimum; a bounded
    agreement kernel lets them abstain instead.
    """
    truth = torch.tensor([0.019, 0.004])
    angle = -math.pi + (torch.arange(720, dtype=torch.float32) + 0.5) * (
        2 * math.pi / 720)
    direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
    vote = _scan_vote()

    # Averaged over seeds: which bearings happen to be corrupted moves the
    # least-squares answer a long way, and the point being pinned is the
    # systematic difference between the estimators, not one draw of it.
    vote_errors, solve_errors = [], []
    for seed in range(5):
        difference, weight = _vote_inputs(truth, outlier_fraction=0.3, seed=seed)
        estimate, _, _ = vote(difference, weight)
        solved = torch.linalg.lstsq(direction, -difference[0]).solution
        vote_errors.append(float((estimate[0] - truth).norm()))
        solve_errors.append(float((solved - truth).norm()))

    worst_vote = max(vote_errors)
    typical_solve = sorted(solve_errors)[len(solve_errors) // 2]
    # The vote stays well inside a grid cell of the truth on every draw...
    assert worst_vote < 5.0e-4, vote_errors
    # ...while least squares is off by most of the signal it is measuring.
    assert typical_solve > 5.0e-3, solve_errors
    assert typical_solve > 20.0 * worst_vote


def test_scan_vote_sharpness_falls_when_geometry_stops_constraining():
    """Sharpness is an observability claim, so it must fail where geometry does.

    What it detects is *how much bearing-diverse evidence there is*, which is
    the quantity that decides whether the surface has an isolated peak. It does
    not detect corruption: outliers add a pedestal that dividing by the peak
    normalizes away, and 30% of them move sharpness by under 2%. Measured on
    ``belief_onpolicy_v15`` it still correlates -0.46 with the vote's error,
    because sparse returns are what actually go wrong in a real room.
    """
    truth = torch.tensor([0.015, 0.0])
    vote = _scan_vote()
    _, _, dense = vote(*_vote_inputs(truth))

    difference, weight = _vote_inputs(truth)
    weight = weight.clone()
    weight[:, 60:] = 0.0            # one small arc still returning
    _, _, sparse = vote(difference, weight)
    assert float(sparse) < 0.5 * float(dense), (float(dense), float(sparse))


def test_scan_vote_is_absent_by_default_and_costs_nothing():
    """Every existing checkpoint keeps its exact architecture."""
    assert BeliefNavPolicyConfig().scan_vote_bins == 0
    assert BeliefNavPolicy(BeliefNavPolicyConfig()).odometry.vote is None
    voting = BeliefNavPolicy(BeliefNavPolicyConfig(scan_vote_bins=21))
    assert voting.odometry.vote is not None
    assert (voting.odometry_parameter_count
            > BeliefNavPolicy().odometry_parameter_count)


def test_scan_vote_carries_no_parameters_and_no_gradient():
    """It is fixed sensor processing, like the deskew, not a learned layer."""
    vote = _scan_vote()
    assert list(vote.parameters()) == []
    difference, weight = _vote_inputs(torch.tensor([0.019, 0.0]))
    difference.requires_grad_(True)
    estimate, surface, sharpness = vote(difference, weight)
    assert not estimate.requires_grad
    assert not surface.requires_grad
    assert not sharpness.requires_grad


def test_vote_head_still_falls_back_to_dead_reckoning_without_returns():
    """The fallback guarantee must survive the new branch."""
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(scan_vote_bins=21)).eval()
    frames = policy.config.motion_history_frames
    previous = torch.zeros(2, frames, 2)
    previous[:, -1, 0] = 0.5
    raw = torch.full((2, frames, 720), LIDAR_MAX_M)
    valid = torch.zeros(2, frames, 720, dtype=torch.bool)
    with torch.no_grad():
        translation = policy.translation(
            torch.zeros(2, 4, 180), torch.zeros(2, frames), previous, raw, valid)
    expected = torch.tensor([[0.5 * V_MAX * DT, 0.0]]).repeat(2, 1)
    torch.testing.assert_close(translation, expected, atol=1e-7, rtol=0.0)


def test_vote_gradients_reach_the_trunk_through_the_surface_encoder():
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(scan_vote_bins=21))
    frames = policy.config.motion_history_frames
    torch.manual_seed(5)
    raw = torch.rand(2, frames, 720) * 4.0 + 0.5
    valid = torch.ones(2, frames, 720, dtype=torch.bool)
    policy.translation(torch.rand(2, 4, 180), torch.randn(2, frames) * 0.02,
                       torch.rand(2, frames, 2), raw, valid).sum().backward()
    gradient = policy.odometry.vote_encoder[0].weight.grad
    assert gradient is not None and float(gradient.abs().sum()) > 0.0


# ── honest uncertainty ───────────────────────────────────────────────────────


def test_drift_variance_accumulates_linearly_in_distance():
    """The property the random-walk term cannot have.

    A scale error is systematic, so its standard deviation must grow in
    proportion to distance travelled. Summing independent per-step variances
    grows as the square root instead, which is precisely how the declared bound
    fell 8.8x behind the true error.
    """
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(
        drift_scale_uncertainty=0.03))
    step, steps = 0.02, 50
    travelled = torch.zeros(1)
    variance = torch.zeros(1)
    for _ in range(steps):
        variance = variance + policy._drift_variance(
            travelled, torch.full((1,), step))
        travelled = travelled + step
    # sigma == eps * distance, exactly.
    expected = 0.03 * step * steps
    torch.testing.assert_close(variance.sqrt(), torch.tensor([expected]),
                               atol=1e-7, rtol=0.0)
    # And it is genuinely superlinear against a random walk of the same steps.
    random_walk = (0.03 * step) * math.sqrt(steps)
    assert float(variance.sqrt()) > 5.0 * random_walk


def test_drift_term_is_absent_by_default_and_changes_nothing():
    assert BeliefNavPolicyConfig().drift_scale_uncertainty == 0.0
    policy = BeliefNavPolicy().eval()
    belief, covariance, count = policy.initial_state(torch.zeros(2, 2))
    variance = torch.full((2, 2), 1e-4)
    delta_yaw = torch.tensor([0.1, -0.2])
    with_none = policy._propagate_covariance(covariance, variance, delta_yaw)
    with_zero = policy._propagate_covariance(
        covariance, variance, delta_yaw, torch.zeros(2))
    torch.testing.assert_close(with_none, with_zero, atol=0.0, rtol=0.0)


def test_drift_aware_checkpoint_refuses_to_run_without_carried_distance():
    """Silently dropping the state would restore the old, dishonest bound."""
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(
        drift_scale_uncertainty=0.03)).eval()
    belief, covariance, count = policy.initial_state(torch.zeros(1, 2))
    frames = policy.config.motion_history_frames
    with pytest.raises(ValueError, match="travelled"):
        policy.step_state(
            belief, covariance, count, torch.zeros(1, 4, 180),
            torch.full((1, frames, 720), LIDAR_MAX_M),
            torch.zeros(1, frames, 720, dtype=torch.bool),
            torch.zeros(1, frames), torch.zeros(1, frames, 2))


def test_carried_distance_widens_the_arrival_bound_with_travel():
    """Two robots, same belief, different history: the further one is less sure."""
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(
        drift_scale_uncertainty=0.03)).eval()
    frames = policy.config.motion_history_frames
    belief, covariance, count = policy.initial_state(torch.zeros(2, 2))
    previous = torch.zeros(2, frames, 2)
    previous[:, -1, 0] = 0.5
    travelled = torch.tensor([0.0, 5.0])
    with torch.no_grad():
        result = policy.step_state(
            belief, covariance, count, torch.zeros(2, 4, 180),
            torch.full((2, frames, 720), LIDAR_MAX_M),
            torch.zeros(2, frames, 720, dtype=torch.bool),
            torch.zeros(2, frames), previous, travelled=travelled)
    covariance = result[3]
    sigma = policy._largest_covariance_eigenvalue(covariance).sqrt()
    # At zero travel the bound is the variance head's floor (~0.25 cm); five
    # metres in, the systematic term dominates it several times over. The ratio
    # is not larger only because a single step is being compared -- the point is
    # that identical sensors and an identical belief produce different
    # confidence purely because of how far the robot has come.
    assert float(sigma[1]) > 3.5 * float(sigma[0])
    # The carried distance advances by the step actually taken. The tolerance
    # is float32 accumulation against a five-metre base, not model error.
    torch.testing.assert_close(result[-1] - travelled,
                               torch.full((2,), 0.5 * V_MAX * DT),
                               atol=1e-6, rtol=0.0)


def test_step_state_return_arity_is_exactly_the_state_it_was_given():
    """The evaluator unpacks this tuple positionally, so its length is a contract.

    Six values always, then calibration if calibration went in, then travelled
    if travelled did. Getting the order wrong assigns a distance to a
    calibration state and drifts silently.
    """
    frames = BeliefNavPolicyConfig().motion_history_frames

    def step(policy, **carried):
        belief, covariance, count = policy.initial_state(torch.zeros(1, 2))
        with torch.no_grad():
            return policy.step_state(
                belief, covariance, count, torch.zeros(1, 4, 180),
                torch.full((1, frames, 720), LIDAR_MAX_M),
                torch.zeros(1, frames, 720, dtype=torch.bool),
                torch.zeros(1, frames), torch.zeros(1, frames, 2), **carried)

    plain = BeliefNavPolicy().eval()
    assert len(step(plain)) == 6

    drifting = BeliefNavPolicy(BeliefNavPolicyConfig(
        drift_scale_uncertainty=0.05)).eval()
    result = step(drifting, travelled=drifting.initial_travel(torch.zeros(1, 2)))
    assert len(result) == 7
    assert result[-1].shape == (1,)

    recurrent = BeliefNavPolicy(BeliefNavPolicyConfig(
        persistent_calibration=True, drift_scale_uncertainty=0.05)).eval()
    goal = torch.zeros(1, 2)
    result = step(recurrent, calibration=recurrent.initial_calibration(goal),
                  travelled=recurrent.initial_travel(goal))
    assert len(result) == 8
    # calibration is a scale correction and travelled is a distance; both are
    # (1,) here, so only the order distinguishes them.
    assert float(result[-1]) == pytest.approx(0.0, abs=1e-6)


def test_scan_vote_reports_nothing_rather_than_a_grid_corner():
    """No usable bearings must read as no evidence, not as maximum displacement.

    Every candidate scores zero, so an unguarded argmax returns index 0, which
    is the corner of the candidate grid and a ~3 cm displacement in both axes.
    """
    vote = _scan_vote()
    difference = torch.zeros(2, 720)
    weight = torch.zeros(2, 720)
    estimate, surface, sharpness = vote(difference, weight)
    torch.testing.assert_close(estimate, torch.zeros(2, 2), atol=0.0, rtol=0.0)
    assert float(sharpness.abs().max()) == 0.0
    assert float(surface.abs().max()) == 0.0


def test_vote_residual_is_expressed_in_the_frame_the_head_answers_in():
    """A frame error here is a bias that appears only while turning.

    The vote compares scans aligned into the *newest* frame, but the head's
    output is added to the commanded displacement in the body frame at the
    *start* of the tick, and the two differ by one tick of yaw. A robot that
    drives straight forward while turning must produce a purely forward
    residual: leaving it in the newest frame instead would rotate it and
    manufacture a lateral correction proportional to sin(yaw rate).
    """
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(scan_vote_bins=41)).eval()
    frames = policy.config.motion_history_frames
    beams = policy.config.raw_lidar_bins
    distance, turn, radius = 0.02, 0.09, 4.0

    grid = -math.pi + (torch.arange(beams, dtype=torch.float64) + 0.5) * (
        2 * math.pi / beams)

    def scan(position, yaw):
        """Exact ranges from ``position`` at ``yaw`` inside a centred circle."""
        world = yaw + grid
        b = position[0] * torch.cos(world) + position[1] * torch.sin(world)
        c = position.square().sum() - radius ** 2
        return (-b + torch.sqrt(b.square() - c)).float()

    start = torch.tensor([0.35, -0.2], dtype=torch.float64)
    # Forward along the *starting* heading, which is yaw zero, then turn.
    finish = start + torch.tensor([distance, 0.0], dtype=torch.float64)

    before, after = scan(start, 0.0), scan(finish, turn)
    metric = torch.cat([before.expand(frames - 1, beams), after[None]])[None]
    valid = torch.ones(1, frames, beams, dtype=torch.bool)
    gyro = torch.zeros(1, frames)
    gyro[:, -1] = turn
    previous = torch.zeros(1, frames, 2)          # nothing was commanded

    aligned, aligned_valid = policy.odometry._align_history(metric, valid, gyro)
    with torch.no_grad():
        _, scalars, _ = policy.odometry._vote_surfaces(
            aligned, aligned_valid, previous, gyro)
    residual = scalars[:, :2]                     # (residual_x, residual_y)

    limit = policy.config.forward_correction_scale * V_MAX * DT
    recovered = residual * limit
    # Forward component is the motion; lateral is zero because the residual was
    # rotated back into the frame the robot started the tick in.
    assert abs(float(recovered[0, 0]) - distance) < 2.0e-3, recovered
    assert abs(float(recovered[0, 1])) < 1.0e-3, recovered
    # The test must be able to see the bug it is guarding against. Undoing the
    # rotation reproduces what leaving the residual in the newest frame would
    # have handed the head, and its lateral component is outside the tolerance
    # above. It is about 2 mm at one tick of full-rate turn -- small per tick,
    # systematic across every turning step, and it integrates.
    unrotated = compose_belief(recovered, torch.zeros_like(recovered),
                               torch.full((1,), turn))
    assert abs(float(unrotated[0, 1])) > 1.0e-3, unrotated


def test_precomputed_vote_is_bit_identical_to_computing_it_inline():
    """The deployment split must not be a second implementation.

    On the robot the vote runs ahead of the network, in float on the CPU, and
    its outputs are handed to the quantized graph as ordinary inputs. That is
    only safe if the two paths are the same arithmetic -- a divergence here is
    the exact bug class ``rollout`` versus ``step`` was built to avoid, and it
    would appear as a sim-to-real gap with no other symptom.
    """
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(
        scan_vote_bins=21, derotate_scan_history=True)).eval()
    frames = policy.config.motion_history_frames
    torch.manual_seed(23)
    raw = torch.rand(3, frames, 720) * 4.0 + 0.4
    valid = torch.rand(3, frames, 720) > 0.2
    raw = torch.where(valid, raw, torch.full_like(raw, LIDAR_MAX_M))
    lidar = torch.rand(3, 4, 180)
    gyro = torch.randn(3, frames) * 0.04
    previous = torch.rand(3, frames, 2) * 2.0 - 1.0

    with torch.no_grad():
        inline = policy.translation(lidar, gyro, previous, raw, valid)
        vote = policy.odometry.compute_vote(lidar, gyro, previous, raw, valid)
        commanded = previous[:, -1, 0] * V_MAX
        split = torch.stack(
            [commanded * DT, torch.zeros_like(commanded)], dim=1
        ) + policy.odometry(lidar, gyro, previous, raw, valid, vote=vote)

    torch.testing.assert_close(split, inline, atol=0.0, rtol=0.0)


def test_precomputed_vote_carries_no_learned_weights():
    """Only the geometry leaves the graph; the convolution that reads it stays.

    If a parameter ended up on the preprocessing side it would run in float on
    the CPU, wasting the NPU and quietly splitting the model across the
    quantization boundary.
    """
    policy = BeliefNavPolicy(BeliefNavPolicyConfig(scan_vote_bins=21)).eval()
    assert list(policy.odometry.vote.parameters()) == []
    frames = policy.config.motion_history_frames
    surfaces, scalars, sharpness = policy.odometry.compute_vote(
        torch.zeros(1, 4, 180), torch.zeros(1, frames),
        torch.zeros(1, frames, 2), torch.rand(1, frames, 720) * 3.0 + 0.5,
        torch.ones(1, frames, 720, dtype=torch.bool))
    for tensor in (surfaces, scalars, sharpness):
        assert not tensor.requires_grad
    # The surface arrives as an image for the in-graph convolution to read.
    assert surfaces.shape == (1, len(policy.config.scan_vote_gaps), 21, 21)
