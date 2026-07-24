"""Adapters that present a trained policy to the evaluation harness.

The harness only knows about ``obs dict -> (v, omega)``. Everything that can
produce that — a torch checkpoint, an ExecuTorch ``.pte`` running on the board,
a random baseline — plugs in here, which is what lets the *same* 500-episode
benchmark be run against FP32 and INT8 and have the numbers be comparable.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from ..spec import DEFAULT_SPEC, NavSpec

__all__ = ["TorchPolicy", "ExecuTorchPolicy", "random_policy"]


class TorchPolicy:
    """Run a :class:`NavPolicy` (or its exported actor) as a harness policy."""

    def __init__(
        self,
        model,
        spec: NavSpec = DEFAULT_SPEC,
        device: str = "cpu",
        deterministic: bool = True,
    ):
        self.model = model.to(device).eval()
        self.spec = spec
        self.device = device
        self.deterministic = deterministic

    @classmethod
    def from_checkpoint(cls, path: str, spec: NavSpec = DEFAULT_SPEC,
                        device: str = "cpu") -> "TorchPolicy":
        from ..models.nav_policy import build_policy

        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = build_policy(spec)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        return cls(model, spec, device)

    def _batch(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            k: torch.from_numpy(np.ascontiguousarray(obs[k], dtype=np.float32))
            .unsqueeze(0).to(self.device)
            for k in ("depth", "lidar", "goal", "last_action")
        }

    @torch.no_grad()
    def __call__(self, obs: Dict[str, np.ndarray]) -> Sequence[float]:
        t = self._batch(obs)
        if self.deterministic:
            out = self.model(t["depth"], t["lidar"], t["goal"], t["last_action"])
        else:
            out, _, _ = self.model.act(t, deterministic=False)
        a = out[0].cpu().numpy()
        return (float(a[0]), float(a[1]))


class ExecuTorchPolicy:
    """Run a lowered ``.pte`` through the ExecuTorch runtime as a harness policy.

    This is what makes the §5 quantization parity gate a real gate rather than a
    hand-wave: the INT8 artifact that will run on the robot is replayed through
    the identical 500-episode benchmark the FP32 checkpoint was scored on, so
    "SR drop < 2%" is measured on the deployed artifact, not inferred from a
    weight-error norm.

    Requires the ``executorch`` python package (present in the board's export
    venv, and on any host with an ExecuTorch build).
    """

    def __init__(self, pte_path: str, spec: NavSpec = DEFAULT_SPEC):
        from executorch.runtime import Runtime  # noqa: PLC0415

        self.spec = spec
        self._runtime = Runtime.get()
        self._program = self._runtime.load_program(pte_path)
        self._method = self._program.load_method("forward")

    def __call__(self, obs: Dict[str, np.ndarray]) -> Sequence[float]:
        inputs = tuple(
            torch.from_numpy(np.ascontiguousarray(obs[name], dtype=np.float32))
            .unsqueeze(0)
            for name in self.spec.obs.actor_inputs
        )
        out = self._method.execute(inputs)[0]
        a = out[0].numpy() if hasattr(out[0], "numpy") else np.asarray(out[0])
        return (float(a[0]), float(a[1]))


def random_policy(spec: NavSpec = DEFAULT_SPEC, seed: Optional[int] = None):
    """Uniform random commands. The floor of the benchmark.

    Worth running once: a benchmark on which a random policy scores 30% is
    telling you the layouts are too easy, not that the policy is good.
    """
    rng = np.random.default_rng(seed)

    def act(_obs: Dict[str, np.ndarray]) -> Sequence[float]:
        return (float(rng.uniform(spec.action.v_min, spec.action.v_max)),
                float(rng.uniform(-spec.action.omega_max, spec.action.omega_max)))

    return act
