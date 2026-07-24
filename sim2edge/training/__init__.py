"""Training pipelines: behaviour cloning with DAgger, then PPO fine-tuning."""

from .bc import BCConfig, run_bc_pipeline, train_bc
from .data import Demonstrations, DemoDataset, collect_episodes
from .ppo import PPOConfig, PPOTrainer

__all__ = [
    "BCConfig",
    "train_bc",
    "run_bc_pipeline",
    "PPOConfig",
    "PPOTrainer",
    "Demonstrations",
    "DemoDataset",
    "collect_episodes",
]
