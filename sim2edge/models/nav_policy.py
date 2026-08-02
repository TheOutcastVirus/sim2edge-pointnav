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

__all__ = ["NavPolicy", "NavPolicyConfig", "build_encoder", "encode",
           "initialize_layer", "physical_action"]


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


def build_encoder(config: NavPolicyConfig) -> tuple[nn.Module, ...]:
    """The five stacks between the flat observation and the trunk output.

    A free function rather than a shared base class, because the actor's
    submodule *names* are its checkpoint keys: every BC run to date is stored
    under ``lidar_conv.*``, ``trunk.*`` and so on, and wrapping them in a
    container would orphan those files. The PPO critic wants the identical
    architecture with independent weights, so it calls this too and assigns the
    results to the same names.
    """
    conv: list[nn.Module] = []
    channels = config.history_frames
    length = config.lidar_bins
    for index, output_channels in enumerate(config.lidar_channels):
        kernel = 5 if index < 2 else 3
        padding = kernel // 2
        conv.extend((
            nn.Conv1d(channels, output_channels, kernel, stride=2,
                      padding=padding),
            nn.ReLU(),
        ))
        channels = output_channels
        length = (length + 2 * padding - kernel) // 2 + 1

    lidar_conv = nn.Sequential(*conv)
    lidar_projection = nn.Sequential(
        nn.Flatten(),
        nn.Linear(channels * length, config.lidar_features),
        nn.ReLU(),
    )
    goal_encoder = _mlp((3, config.goal_features))
    action_encoder = _mlp((2 * config.history_frames, config.action_features))
    fused = (config.lidar_features + config.goal_features
             + config.action_features)
    trunk = _mlp((fused,) + config.trunk)
    return lidar_conv, lidar_projection, goal_encoder, action_encoder, trunk


def encode(module: nn.Module, observation: torch.Tensor) -> torch.Tensor:
    """Flat observation -> trunk features, for anything built by ``build_encoder``.

    ``module`` supplies the layers and a ``config``; it is passed in rather than
    inherited from for the checkpoint-key reason above.
    """
    config = module.config
    lidar_width = config.history_frames * config.lidar_bins
    lidar = observation[:, :lidar_width].reshape(
        -1, config.history_frames, config.lidar_bins)
    goal = observation[:, lidar_width:lidar_width + 3]
    last_action = observation[:, lidar_width + 3:]

    fused = torch.cat((
        module.lidar_projection(module.lidar_conv(lidar)),
        module.goal_encoder(goal),
        module.action_encoder(last_action),
    ), dim=1)
    return module.trunk(fused)


def physical_action(raw: torch.Tensor, v_max: float,
                    omega_max: float) -> torch.Tensor:
    """Unbounded head output -> bounded SI ``(v, omega)``.

    Keeping this squash in-model makes training, Isaac evaluation, ExecuTorch
    and the future C++ node share one scale. ``tanh(raw)`` is exactly the
    normalized action ``isaaclab_env.nav_task.denormalize_action`` expects, so
    the simulator and the deployed graph apply the same two constants.
    """
    squashed = torch.tanh(raw)
    v = 0.5 * v_max * (squashed[:, :1] + 1.0)
    omega = omega_max * squashed[:, 1:]
    return torch.cat((v, omega), dim=1)


class NavPolicy(nn.Module):
    """``(N, OBS_DIM) -> (N, 2)`` deterministic physical command policy."""

    def __init__(self, config: NavPolicyConfig | None = None):
        super().__init__()
        self.config = config or NavPolicyConfig()
        cfg = self.config

        (self.lidar_conv, self.lidar_projection, self.goal_encoder,
         self.action_encoder, self.trunk) = build_encoder(cfg)
        self.action_head = nn.Linear(cfg.trunk[-1], 2)

        self.apply(initialize_layer)
        # Begin near a neutral midpoint command instead of allowing a random
        # final layer to saturate tanh before the first optimizer step.
        nn.init.uniform_(self.action_head.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.action_head.bias)

    def features(self, observation: torch.Tensor) -> torch.Tensor:
        return encode(self, observation)

    def pre_squash(self, observation: torch.Tensor) -> torch.Tensor:
        """The head output *before* tanh -- the coordinates PPO explores in.

        RL adds Gaussian noise here rather than to the physical command: a
        Gaussian on a bounded quantity puts probability mass outside the
        Create3's envelope, and clipping it back makes the density the ratio is
        computed from a fiction. This is also the space the BC anchor compares
        in, where a fixed distance means the same thing at every command.
        """
        return self.action_head(self.features(observation))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return physical_action(self.pre_squash(observation),
                               self.config.v_max, self.config.omega_max)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def example_input(self, batch_size: int = 1, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.observation_dim, device=device)


def initialize_layer(module: nn.Module) -> None:
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=2.0 ** 0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
