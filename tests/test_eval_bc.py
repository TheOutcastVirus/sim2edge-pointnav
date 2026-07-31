"""Pure helpers used by closed-loop evaluation."""

import math

import pytest

torch = pytest.importorskip("torch")

from scripts.eval_bc import _cap_forward_speed
from sim2edge.spec import V_MAX


def test_forward_speed_cap_is_linear_and_ignores_rear_returns():
    command = torch.tensor([[0.40, 0.3], [0.40, -0.2]])
    ranges = torch.tensor([[0.22, 0.10], [0.41, 0.10]])
    angles = torch.tensor([0.0, math.pi])
    capped = _cap_forward_speed(
        command, ranges, angles, 0.22, 0.60, V_MAX, torch)
    assert capped[0, 0] == 0.0
    assert capped[1, 0] == pytest.approx(V_MAX / 2, rel=1e-6)
    assert torch.equal(capped[:, 1], command[:, 1])
