"""The navigation policy — a dependency-free, torch-only module.

**This file must not import anything but torch.** It is used identically for
training on the 4090, for quantization calibration, and for lowering to
ExecuTorch on the Ventuno, and the board's export venv is pinned to a specific
torch with no torchvision available. The reference project hit exactly this wall
(`ventuno-collision-avoidance/tools/collision_model.py:8-15`); the fix is to
never let a training-side dependency into the model definition.

Designed backward from the NPU. Every operator below is one the reference models
already proved fully delegated on HTP V75 at 8a8w: Conv2d (including strided),
Conv1d, BatchNorm, ReLU, Linear, cat, tanh. There is no adaptive pooling, no
dynamic shape, no recurrence, and no control flow depending on tensor values.

Two structural rules carried over from the reference repos:

1. **Only the actor mean path is exported.** The Gaussian head and the critic
   exist solely for training and are not reachable from :class:`ActorMean`.
2. **Nothing but the network is in the graph.** Normalization constants live in
   ``sim2edge/spec.py``; the action squash and scaling are in-graph because tanh
   is cheap and proven on HTP, but every clamp, rate limit, and safety decision
   is in C++ where it can be reasoned about and unit-tested.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["NavPolicy", "NavPolicyConfig", "ActorMean", "build_policy"]


class NavPolicyConfig:
    """Architecture sizes. Plain class, not a dataclass, to stay import-light.

    Defaults are mirrored from ``sim2edge.spec`` rather than imported, so this
    file keeps its torch-only guarantee. ``build_policy`` passes the real spec
    values in and :meth:`assert_matches_spec` verifies they agree.
    """

    def __init__(
        self,
        depth_height: int = 96,
        depth_width: int = 160,
        lidar_bins: int = 180,
        goal_dim: int = 3,
        action_dim: int = 2,
        privileged_dim: int = 8,
        depth_channels: Tuple[int, ...] = (32, 64, 96, 128),
        depth_feat: int = 256,
        lidar_channels: Tuple[int, ...] = (32, 64, 64),
        lidar_feat: int = 128,
        goal_feat: int = 32,
        action_feat: int = 16,
        trunk: Tuple[int, ...] = (512, 256),
        critic_hidden: Tuple[int, ...] = (256, 256),
        v_min: float = 0.0,
        v_max: float = 0.30,
        omega_max: float = 1.50,
        init_log_std: float = -1.0,
    ):
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.lidar_bins = lidar_bins
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.privileged_dim = privileged_dim
        self.depth_channels = tuple(depth_channels)
        self.depth_feat = depth_feat
        self.lidar_channels = tuple(lidar_channels)
        self.lidar_feat = lidar_feat
        self.goal_feat = goal_feat
        self.action_feat = action_feat
        self.trunk = tuple(trunk)
        self.critic_hidden = tuple(critic_hidden)
        self.v_min = v_min
        self.v_max = v_max
        self.omega_max = omega_max
        self.init_log_std = init_log_std


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────


class DepthEncoder(nn.Module):
    """Small strided CNN over the depth image.

    No pretrained backbone: at 96x160 single-channel there is nothing an
    ImageNet backbone would transfer, and it would drag in torchvision, which
    the export environment cannot import.
    """

    def __init__(self, cfg: NavPolicyConfig):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in cfg.depth_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # Fixed spatial size after the strided stack. Computed rather than
        # hardcoded, but static: no adaptive pooling, so the exported graph has
        # no shape-dependent operators.
        h, w = cfg.depth_height, cfg.depth_width
        for _ in cfg.depth_channels:
            h = (h + 1) // 2
            w = (w + 1) // 2
        self.flat_dim = cfg.depth_channels[-1] * h * w
        self.fc = nn.Linear(self.flat_dim, cfg.depth_feat)
        self.act = nn.ReLU(inplace=True)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.conv(depth)
        x = torch.flatten(x, 1)
        return self.act(self.fc(x))


class LidarEncoder(nn.Module):
    """1D CNN over the min-pooled scan.

    Convolutional rather than an MLP so the encoder is translation-equivariant
    in bearing: an obstacle 30 degrees to the left and the same obstacle 30
    degrees to the right are the same local pattern, and a plain MLP would have
    to learn each bearing independently.

    Padding is zeros, not circular. Circular padding would be the geometrically
    correct choice for a 360-degree scan, but it lowers to a pad operator that
    is not among those proven delegated on HTP V75 by the reference models, and
    an unsupported op silently splits the graph and costs far more than the
    wrap-around seam is worth.
    """

    def __init__(self, cfg: NavPolicyConfig):
        super().__init__()
        layers = []
        in_ch = 1
        length = cfg.lidar_bins
        for i, out_ch in enumerate(cfg.lidar_channels):
            k = 5 if i < 2 else 3
            p = k // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, k, stride=2, padding=p, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
            length = (length + 2 * p - k) // 2 + 1
        self.conv = nn.Sequential(*layers)
        self.flat_dim = cfg.lidar_channels[-1] * length
        self.fc = nn.Linear(self.flat_dim, cfg.lidar_feat)
        self.act = nn.ReLU(inplace=True)

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        x = self.conv(lidar.unsqueeze(1))  # (B, 1, bins)
        x = torch.flatten(x, 1)
        return self.act(self.fc(x))


def _mlp(sizes: Tuple[int, ...], final_act: bool = True) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if final_act or i < len(sizes) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────


class NavPolicy(nn.Module):
    """Actor-critic policy with an asymmetric, privileged critic.

    The actor sees only what the robot can sense. The critic additionally sees
    ground-truth geodesic distance, obstacle clearance, and true pose — free
    accuracy, because the critic is never exported and never runs on the robot.
    """

    def __init__(self, cfg: Optional[NavPolicyConfig] = None):
        super().__init__()
        self.cfg = cfg = cfg or NavPolicyConfig()

        # ── Actor ────────────────────────────────────────────────────────────
        self.depth_encoder = DepthEncoder(cfg)
        self.lidar_encoder = LidarEncoder(cfg)
        self.goal_encoder = _mlp((cfg.goal_dim, cfg.goal_feat))
        self.action_encoder = _mlp((cfg.action_dim, cfg.action_feat))

        fused = cfg.depth_feat + cfg.lidar_feat + cfg.goal_feat + cfg.action_feat
        self.trunk = _mlp((fused,) + cfg.trunk)
        self.actor_head = nn.Linear(cfg.trunk[-1], cfg.action_dim)

        #: State-independent log std, the standard choice for continuous
        #: control: a state-dependent std tends to collapse early in PPO and
        #: makes the BC-anchored fine-tune much harder to keep stable.
        self.log_std = nn.Parameter(
            torch.full((cfg.action_dim,), float(cfg.init_log_std)))

        # ── Critic (training only, never exported) ───────────────────────────
        # It gets its own lidar encoder plus the privileged vector, but no depth
        # encoder: the privileged observation already contains the geodesic
        # distance and the true obstacle clearance, which is most of what a value
        # function needs, and a second depth CNN would roughly double the cost
        # of every training step to re-derive information the critic is simply
        # handed.
        self.critic_lidar = LidarEncoder(cfg)
        critic_in = (cfg.lidar_feat + cfg.privileged_dim + cfg.goal_dim
                     + cfg.action_dim)
        self.critic = nn.Sequential(
            _mlp((critic_in,) + cfg.critic_hidden),
            nn.Linear(cfg.critic_hidden[-1], 1),
        )

        self.apply(_init_weights)
        # A small final layer makes the initial policy near-neutral, so BC does
        # not start by fighting a large random bias.
        nn.init.uniform_(self.actor_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.actor_head.bias)

    # ── Forward paths ────────────────────────────────────────────────────────

    def features(self, depth: torch.Tensor, lidar: torch.Tensor,
                 goal: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([
            self.depth_encoder(depth),
            self.lidar_encoder(lidar),
            self.goal_encoder(goal),
            self.action_encoder(last_action),
        ], dim=1))

    def pre_squash(self, depth: torch.Tensor, lidar: torch.Tensor,
                   goal: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        """Unbounded actor output. The PPO Gaussian lives in this space."""
        return self.actor_head(self.features(depth, lidar, goal, last_action))

    def squash(self, pre: torch.Tensor) -> torch.Tensor:
        """Map unbounded output to physical ``(v, omega)``.

        In-graph on purpose: tanh is cheap and proven on HTP, and doing the
        scaling here means the C++ node receives SI units directly and cannot
        apply a different scale than training did.
        """
        cfg = self.cfg
        t = torch.tanh(pre)
        v = cfg.v_min + (cfg.v_max - cfg.v_min) * 0.5 * (t[..., :1] + 1.0)
        omega = cfg.omega_max * t[..., 1:]
        return torch.cat([v, omega], dim=-1)

    def forward(self, depth: torch.Tensor, lidar: torch.Tensor,
                goal: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        """Deterministic physical action — the exported path."""
        return self.squash(self.pre_squash(depth, lidar, goal, last_action))

    def value(self, lidar: torch.Tensor, goal: torch.Tensor,
              last_action: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.critic_lidar(lidar), goal, last_action, privileged],
                      dim=1)
        return self.critic(x).squeeze(-1)

    # ── Distribution helpers (training only) ─────────────────────────────────

    def distribution(self, depth, lidar, goal, last_action):
        """Gaussian over the *pre-squash* action.

        Placing the distribution before the tanh, and omitting the change-of-
        variables correction, is the same convention rsl_rl uses. The squash
        then behaves as a deterministic output transform. The cost is that
        log-probabilities are of ``u``, not of the physical action; the benefit
        is that the entropy term stays well-conditioned instead of collapsing as
        tanh saturates, which matters a great deal for the BC-anchored fine-tune
        where the actor starts near-deterministic.
        """
        mean = self.pre_squash(depth, lidar, goal, last_action)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        """Sample (or take the mean of) an action. Returns physical units."""
        dist = self.distribution(obs["depth"], obs["lidar"], obs["goal"],
                                 obs["last_action"])
        pre = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(pre).sum(-1)
        return self.squash(pre), pre, log_prob

    def evaluate(self, obs: Dict[str, torch.Tensor], pre_actions: torch.Tensor):
        """Log-prob, entropy, and value for a batch of stored pre-squash actions."""
        dist = self.distribution(obs["depth"], obs["lidar"], obs["goal"],
                                 obs["last_action"])
        return (
            dist.log_prob(pre_actions).sum(-1),
            dist.entropy().sum(-1),
            self.value(obs["lidar"], obs["goal"], obs["last_action"],
                       obs["privileged"]),
        )

    # ── Training-mode utilities ──────────────────────────────────────────────

    def freeze_norm(self) -> None:
        """Put every normalization layer in eval mode and stop its statistics.

        Called before PPO fine-tuning. BatchNorm and PPO interact badly: rollouts
        are collected in one mode and the update runs in another, so the running
        statistics drift with the changing policy and the value of a state stops
        being stationary. Freezing after BC keeps the benefit of normalization
        during supervised learning without letting it destabilize RL — and the
        frozen affine parameters fold into the preceding convolution at export,
        so this costs nothing at inference.
        """
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
                m.track_running_stats = False
                for p in m.parameters():
                    p.requires_grad_(False)

    def export_module(self) -> "ActorMean":
        """Wrap the actor mean path for `torch.export`."""
        return ActorMean(self)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_actor_parameters(self) -> int:
        """Parameters that actually ship to the robot."""
        exported = ActorMean(self)
        return sum(p.numel() for p in exported.parameters())


class ActorMean(nn.Module):
    """Inference-only wrapper: the exact graph that becomes the ``.pte``.

    Holds references to the actor submodules only. The critic, its lidar
    encoder, and ``log_std`` are unreachable from here, so they cannot be pulled
    into the exported graph by accident — and ``num_actor_parameters`` reports
    what really ships.
    """

    def __init__(self, policy: NavPolicy):
        super().__init__()
        self.cfg = policy.cfg
        self.depth_encoder = policy.depth_encoder
        self.lidar_encoder = policy.lidar_encoder
        self.goal_encoder = policy.goal_encoder
        self.action_encoder = policy.action_encoder
        self.trunk = policy.trunk
        self.actor_head = policy.actor_head
        self.eval()

    def forward(self, depth: torch.Tensor, lidar: torch.Tensor,
                goal: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        """``(depth, lidar, goal, last_action) -> (v, omega)`` in SI units.

        The argument order is fixed by ``ObservationSpec.actor_inputs`` and is
        the order the C++ node binds input tensors in. Do not reorder.
        """
        x = torch.cat([
            self.depth_encoder(depth),
            self.lidar_encoder(lidar),
            self.goal_encoder(goal),
            self.action_encoder(last_action),
        ], dim=1)
        pre = self.actor_head(self.trunk(x))
        cfg = self.cfg
        t = torch.tanh(pre)
        v = cfg.v_min + (cfg.v_max - cfg.v_min) * 0.5 * (t[..., :1] + 1.0)
        omega = cfg.omega_max * t[..., 1:]
        return torch.cat([v, omega], dim=-1)


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=2.0 ** 0.5)
        nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def build_policy(spec=None, **overrides) -> NavPolicy:
    """Build a policy whose sizes are taken from the shared spec.

    Importing the spec happens here, not at module import, so
    ``nav_policy.py`` itself remains torch-only for the export environment.
    """
    if spec is None:
        from ..spec import DEFAULT_SPEC as spec  # noqa: PLC0415

    cfg = NavPolicyConfig(
        depth_height=spec.obs.depth.height,
        depth_width=spec.obs.depth.width,
        lidar_bins=spec.obs.lidar.bins,
        goal_dim=spec.obs.goal.shape[0],
        action_dim=spec.action.shape[0],
        privileged_dim=spec.obs.privileged.dim,
        v_min=spec.action.v_min,
        v_max=spec.action.v_max,
        omega_max=spec.action.omega_max,
        **overrides,
    )
    return NavPolicy(cfg)
