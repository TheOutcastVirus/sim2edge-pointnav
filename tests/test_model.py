"""The policy graph is small, bounded and exportable from its first version."""

import pytest

torch = pytest.importorskip("torch")

from sim2edge.models import NavPolicy, NavPolicyConfig
from sim2edge.spec import LIDAR_BINS, OBS_DIM, OMEGA_MAX, V_MAX


def test_lidar_convolutions_use_circular_padding():
    policy = NavPolicy()
    conv = [layer for layer in policy.lidar_conv
            if isinstance(layer, torch.nn.Conv1d)]
    assert conv and all(layer.padding_mode == "circular" for layer in conv)


def test_policy_emits_physical_bounded_actions():
    policy = NavPolicy()
    output = policy(torch.randn(7, OBS_DIM))
    assert output.shape == (7, 2)
    assert torch.all((output[:, 0] >= 0.0) & (output[:, 0] <= V_MAX))
    assert torch.all(output[:, 1].abs() <= OMEGA_MAX)


def test_initial_policy_is_neutral_and_finite():
    policy = NavPolicy()
    output = policy(torch.zeros(4, OBS_DIM))
    assert torch.isfinite(output).all()
    assert torch.allclose(output[:, 0], torch.full((4,), V_MAX / 2), atol=1e-5)
    assert torch.allclose(output[:, 1], torch.zeros(4), atol=1e-5)


def test_graph_exports_with_static_deployment_signature():
    policy = NavPolicy().eval()
    example = torch.randn(2, OBS_DIM)
    eager = policy(example)
    exported = torch.export.export(policy, (example,)).module()(example)
    assert torch.allclose(eager, exported, atol=1e-6)


def test_parameter_budget_stays_edge_sized():
    policy = NavPolicy()
    assert 100_000 < policy.parameter_count < 500_000


def test_legacy_one_frame_configuration_remains_loadable():
    config = NavPolicyConfig.from_dict({"history_frames": 1})
    policy = NavPolicy(config)
    assert policy.example_input(2).shape == (2, LIDAR_BINS + 5)
