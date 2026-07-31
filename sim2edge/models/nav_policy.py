"""Small, export-friendly policy for lidar goal navigation.

The architecture is deliberately built backward from the deployed graph. It
accepts the exact flat observation assembled by :mod:`sim2edge.preprocess` and
returns a physical ``(v, omega)`` command; no caller is allowed to maintain a
second copy of the action scaling convention.

The lidar branch uses strided 1-D convolutions. Nearby beams describe one local
piece of geometry, so sharing filters over bearing is both more data-efficient
and much smaller than teaching a dense layer every possible obstacle position.
The final flatten preserves absolute bearing, which the goal-conditioned trunk
needs in order to distinguish "pass left" from "pass right".

Only common Conv1d, Linear, ReLU, concatenation and tanh operators are used.
They have static shapes and straightforward ExecuTorch/QNN lowerings; export
compatibility is tested now rather than discovered after training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from ..spec import HISTORY_FRAMES, LIDAR_BINS, OMEGA_MAX, V_MAX

__all__ = ["NavPolicy", "NavPolicyConfig"]


@dataclass(frozen=True)
class NavPolicyConfig:
    """Architecture only; training hyperparameters live with the trainer."""

    lidar_bins: int = LIDAR_BINS
    history_frames: int = HISTORY_FRAMES
    lidar_channels: tuple[int, ...] = (32, 64, 64)
    lidar_features: int = 128
    goal_features: int = 32
    action_features: int = 16
    trunk: tuple[int, ...] = (256, 128)
    v_max: float = V_MAX
    omega_max: float = OMEGA_MAX

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NavPolicyConfig":
        # JSON/checkpoints turn tuples into lists; normalize them here so the
        # reconstructed config compares equal to the original.
        value = dict(value)
        for key in ("lidar_channels", "trunk"):
            if key in value:
                value[key] = tuple(value[key])
        # Checkpoints produced before temporal stacking remain usable for
        # controlled comparisons and DAgger collection.
        value.setdefault("history_frames", 1)
        return cls(**value)

    @property
    def observation_dim(self) -> int:
        return self.history_frames * self.lidar_bins + 3 + self.history_frames * 2


def _mlp(sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for input_size, output_size in zip(sizes, sizes[1:]):
        layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
    return nn.Sequential(*layers)


class NavPolicy(nn.Module):
    """``(N, OBS_DIM) -> (N, 2)`` deterministic physical command policy."""

    def __init__(self, config: NavPolicyConfig | None = None):
        super().__init__()
        self.config = config or NavPolicyConfig()
        cfg = self.config

        conv: list[nn.Module] = []
        channels = cfg.history_frames
        length = cfg.lidar_bins
        for index, output_channels in enumerate(cfg.lidar_channels):
            kernel = 5 if index < 2 else 3
            padding = kernel // 2
            conv.extend((
                nn.Conv1d(channels, output_channels, kernel, stride=2,
                          padding=padding),
                nn.ReLU(),
            ))
            channels = output_channels
            length = (length + 2 * padding - kernel) // 2 + 1
        self.lidar_conv = nn.Sequential(*conv)
        self.lidar_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * length, cfg.lidar_features),
            nn.ReLU(),
        )
        self.goal_encoder = _mlp((3, cfg.goal_features))
        self.action_encoder = _mlp((2 * cfg.history_frames, cfg.action_features))

        fused = cfg.lidar_features + cfg.goal_features + cfg.action_features
        self.trunk = _mlp((fused,) + cfg.trunk)
        self.action_head = nn.Linear(cfg.trunk[-1], 2)

        self.apply(_initialize)
        # Begin near a neutral midpoint command instead of allowing a random
        # final layer to saturate tanh before the first optimizer step.
        nn.init.uniform_(self.action_head.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        lidar_width = self.config.history_frames * self.config.lidar_bins
        lidar = observation[:, :lidar_width].reshape(
            -1, self.config.history_frames, self.config.lidar_bins)
        goal_start = lidar_width
        goal = observation[:, goal_start:goal_start + 3]
        last_action = observation[:, goal_start + 3:]

        features = torch.cat((
            self.lidar_projection(self.lidar_conv(lidar)),
            self.goal_encoder(goal),
            self.action_encoder(last_action),
        ), dim=1)
        raw = self.action_head(self.trunk(features))

        # The graph emits SI units. Keeping this squash in-model makes training,
        # Isaac evaluation, ExecuTorch and the future C++ node share one scale.
        squashed = torch.tanh(raw)
        v = 0.5 * self.config.v_max * (squashed[:, :1] + 1.0)
        omega = self.config.omega_max * squashed[:, 1:]
        return torch.cat((v, omega), dim=1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def example_input(self, batch_size: int = 1, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.observation_dim, device=device)


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=2.0 ** 0.5)
        nn.init.zeros_(module.bias)
