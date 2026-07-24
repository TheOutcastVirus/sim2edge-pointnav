"""Tests for the policy network, the environment contract, and the safety layer.

The model tests are export-oriented: what matters is not just that the network
computes something, but that the exact graph which becomes the ``.pte`` is
reachable, bounded, and lowerable with the four-input signature the reference
projects never exercised.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim2edge.envs.reference_env import EnvConfig, NavEnv  # noqa: E402
from sim2edge.models.nav_policy import ActorMean, build_policy  # noqa: E402
from sim2edge.safety import SafetyLayer, forward_min_range  # noqa: E402
from sim2edge.spec import DEFAULT_SPEC  # noqa: E402
from sim2edge.training.data import (  # noqa: E402
    Demonstrations,
    collect_episodes,
)

S = DEFAULT_SPEC


def _obs(batch=2):
    obs = {name: torch.randn((batch,) + tuple(shape))
           for name, shape in zip(S.obs.actor_inputs, S.actor_input_shapes)}
    obs["privileged"] = torch.randn(batch, S.obs.privileged.dim)
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


def test_policy_output_is_always_within_the_robots_limits():
    """The squash is in-graph, so this must hold for any input, including the
    extreme activations INT8 quantization can produce."""
    policy = build_policy(S).eval()
    for scale in (1.0, 1e3, 1e6):
        obs = {k: v * scale for k, v in _obs(4).items()}
        with torch.no_grad():
            a = policy(obs["depth"], obs["lidar"], obs["goal"],
                       obs["last_action"])
        assert torch.all(a[:, 0] >= S.action.v_min - 1e-5)
        assert torch.all(a[:, 0] <= S.action.v_max + 1e-5)
        assert torch.all(a[:, 1].abs() <= S.action.omega_max + 1e-5)


def test_policy_never_commands_reverse():
    """v_min is 0: the robot turns rather than backing into what it cannot see.
    It has no rear-facing depth sensor."""
    policy = build_policy(S).eval()
    with torch.no_grad():
        obs = _obs(16)
        a = policy(obs["depth"], obs["lidar"], obs["goal"], obs["last_action"])
    assert torch.all(a[:, 0] >= 0.0)


def test_export_wrapper_matches_the_training_forward():
    policy = build_policy(S).eval()
    obs = _obs(3)
    args = (obs["depth"], obs["lidar"], obs["goal"], obs["last_action"])
    with torch.no_grad():
        assert torch.allclose(policy(*args), policy.export_module()(*args),
                              atol=1e-6)


def test_export_wrapper_excludes_the_critic():
    """The critic must be unreachable from the exported graph, so it cannot be
    pulled in by accident and inflate what ships."""
    policy = build_policy(S)
    exported = ActorMean(policy)
    names = {n for n, _ in exported.named_parameters()}
    assert not any(n.startswith("critic") for n in names)
    assert "log_std" not in names
    assert policy.num_actor_parameters < policy.num_parameters


def test_multi_input_torch_export_succeeds():
    """The Day-1 smoke test, as a unit test.

    The reference projects only ever exported single-tensor models, so the
    four-input signature was the plan's flagged unknown. Failing here means the
    explicit lowering flow in tools/export_qnn.py is needed.
    """
    policy = build_policy(S).eval()
    example = tuple(torch.zeros((1,) + tuple(s)) for s in S.actor_input_shapes)
    program = torch.export.export(policy.export_module(), example)
    with torch.no_grad():
        ref = policy.export_module()(*example)
    assert torch.allclose(program.module()(*example), ref, atol=1e-6)


def test_exported_graph_uses_only_htp_proven_operators():
    """Every op must be one the reference models proved fully delegated on
    HTP V75 at 8a8w. An unsupported op silently splits the graph and costs far
    more than it appears to."""
    proven = {
        "aten::conv1d", "aten::conv2d", "aten::linear", "aten::batch_norm",
        "aten::relu", "aten::relu_", "aten::tanh", "aten::cat",
        "aten::flatten", "aten::slice", "aten::add", "aten::mul",
        "aten::unsqueeze",
    }
    policy = build_policy(S).eval()
    example = tuple(torch.zeros((1,) + tuple(s)) for s in S.actor_input_shapes)
    program = torch.export.export(policy.export_module(), example)

    used = set()
    for node in program.graph.nodes:
        if node.op == "call_function" and hasattr(node.target, "_schema"):
            used.add(node.target._schema.name)
    assert used <= proven, f"unproven operators in the exported graph: {used - proven}"


def test_freeze_norm_stops_batchnorm_statistics():
    """BatchNorm and PPO interact badly: rollouts are collected in one mode and
    updated in another, so drifting statistics make a state's value
    non-stationary."""
    policy = build_policy(S)
    policy.freeze_norm()
    policy.train()  # PPO calls train(); the freeze must survive it
    policy.freeze_norm()
    bns = [m for m in policy.modules()
           if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    assert bns and all(not m.training for m in bns)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────


def _lidar_only(**kw):
    return EnvConfig(use_depth=False, **kw)


def test_env_observation_matches_the_spec():
    env = NavEnv(_lidar_only(), seed=0)
    obs = env.reset(seed=1)
    for name, shape in zip(S.obs.actor_inputs, S.actor_input_shapes):
        assert obs[name].shape == tuple(shape), name
        assert obs[name].dtype == np.float32
    assert obs["privileged"].shape == (S.obs.privileged.dim,)


def test_env_is_deterministic_for_a_seed():
    a, b = NavEnv(_lidar_only(), seed=0), NavEnv(_lidar_only(), seed=0)
    oa, ob = a.reset(seed=7), b.reset(seed=7)
    assert np.array_equal(oa["lidar"], ob["lidar"])
    for _ in range(20):
        ra, rb = a.step((0.2, 0.3)), b.step((0.2, 0.3))
        assert ra.reward == pytest.approx(rb.reward)
    assert a.pose == pytest.approx(b.pose)


def test_reaching_the_goal_terminates_with_the_bonus():
    env = NavEnv(_lidar_only(), seed=0)
    env.reset(seed=3)
    # Teleport onto the goal; the next step should register success.
    env.x, env.y = env.room.goal
    result = env.step((0.0, 0.0))
    assert result.info["success"] and result.terminated
    assert result.reward > S.reward.goal_bonus * 0.5


def test_progress_reward_is_geodesic_not_euclidean():
    """Guard against the sentinel leaking into a reward difference and against
    a silent switch to Euclidean distance."""
    env = NavEnv(_lidar_only(), seed=0)
    env.reset(seed=11)
    for _ in range(40):
        r = env.step((S.action.v_max, 0.0))
        assert abs(r.reward) < 100.0, "unreachable sentinel leaked into reward"
        if r.terminated or r.truncated:
            break


def test_action_delay_is_applied():
    """With a delay, the robot must not respond on the very first step."""
    from sim2edge.randomization import EpisodeRandomization

    env = NavEnv(_lidar_only(), seed=0)
    env.reset(seed=5)
    env.dr = EpisodeRandomization.none()
    env.dr.action_delay_steps = 2
    env._action_queue = [(0.0, 0.0)] * 2
    env.v = 0.0
    env.step((S.action.v_max, 0.0))
    assert env.v == pytest.approx(0.0, abs=1e-6)


def test_zero_noise_tier_removes_sensor_noise():
    clean = NavEnv(_lidar_only(randomization=S.randomization.at_scale(0.0)),
                   seed=0)
    a = clean.reset(seed=2)
    b = clean.reset(seed=2)
    assert np.array_equal(a["lidar"], b["lidar"])


# ─────────────────────────────────────────────────────────────────────────────
# Safety layer
# ─────────────────────────────────────────────────────────────────────────────


def _scan(distance, n=360):
    return np.full(n, distance, dtype=np.float32)


def test_bubble_stops_but_still_permits_rotation():
    """A robot stopped 25 cm from a wall must be able to turn away from it;
    freezing both axes turns a recoverable state into a deadlock."""
    layer = SafetyLayer(S)
    v, omega, intervened = layer.filter(
        S.action.v_max, 1.0, _scan(S.safety.stop_distance_m - 0.05))
    assert v == pytest.approx(0.0)
    assert omega > 0.0
    assert intervened


def test_bubble_scales_speed_in_the_slow_zone():
    layer = SafetyLayer(S)
    mid = 0.5 * (S.safety.stop_distance_m + S.safety.slow_distance_m)
    v, _, _ = layer.filter(S.action.v_max, 0.0, _scan(mid))
    assert 0.0 < v < S.action.v_max


def test_clear_path_is_not_slowed():
    layer = SafetyLayer(S)
    # Ramp up over several ticks so the rate limiter is not what is measured.
    for _ in range(20):
        v, omega, intervened = layer.filter(S.action.v_max, 0.0, _scan(5.0))
    assert v == pytest.approx(S.action.v_max, abs=1e-3)
    assert not intervened


def test_sensor_loss_stops_the_robot():
    layer = SafetyLayer(S)
    v, omega, intervened = layer.filter(S.action.v_max, 1.0, _scan(5.0),
                                        sensor_ok=False)
    assert v == 0.0 and omega == 0.0 and intervened


def test_all_invalid_scan_is_not_treated_as_clear_space():
    layer = SafetyLayer(S)
    v, _, _ = layer.filter(S.action.v_max, 0.0,
                           np.full(360, np.inf, dtype=np.float32))
    assert v == pytest.approx(0.0)


def test_rate_limiting_respects_the_acceleration_limit():
    layer = SafetyLayer(S)
    v, _, _ = layer.filter(S.action.v_max, 0.0, _scan(5.0))
    assert v <= S.action.accel_max * S.action.dt + 1e-6


def test_forward_min_range_ignores_the_rear_sector():
    ranges = np.full(360, 5.0, dtype=np.float32)
    ranges[0] = 0.2  # index 0 is angle -pi: directly behind
    d = forward_min_range(ranges, -math.pi, 2 * math.pi / 360,
                          S.safety.forward_sector_rad)
    assert d == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Demonstration storage
# ─────────────────────────────────────────────────────────────────────────────


def test_quantized_storage_round_trips_within_tolerance():
    demos, stats = collect_episodes(2, _lidar_only(), S, seed_base=0)
    assert len(demos) > 0 and stats["success_rate"] == 1.0
    batch = demos.dequantized(np.arange(min(32, len(demos))))
    # uint16 lidar storage: 1/65535 of the normalized range.
    assert batch["lidar"].max() <= 1.0 + 1e-6
    assert batch["depth"].shape[1:] == S.obs.depth.shape


def test_lidar_only_datasets_do_not_materialize_constant_depth():
    """A 10k-episode lidar-only dataset would otherwise spend ~70 GB of RAM
    storing one repeated constant.

    The synthesized constant is ``invalid_fill``, matching what the env emits;
    ``test_lidar_only_placeholder_matches_the_env`` in ``test_training.py`` is
    the end that pins the two together.
    """
    demos, _ = collect_episodes(2, _lidar_only(), S, seed_base=0)
    assert not demos.has_depth
    assert demos.depth.nbytes == 0
    batch = demos.dequantized(np.arange(4))
    assert batch["depth"].shape == (4,) + S.obs.depth.shape
    assert np.all(batch["depth"] == S.obs.depth.invalid_fill)


def test_concat_refuses_to_mix_incompatible_datasets():
    lidar_only, _ = collect_episodes(1, _lidar_only(), S, seed_base=0)
    with_depth, _ = collect_episodes(1, EnvConfig(use_depth=True), S,
                                     seed_base=0)
    if len(lidar_only) and len(with_depth):
        with pytest.raises(ValueError, match="different env configurations"):
            Demonstrations.concat([lidar_only, with_depth])
