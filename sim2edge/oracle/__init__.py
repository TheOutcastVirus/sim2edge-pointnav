"""Ground-truth planner and expert controller.

Training-side only — nothing here ships to the robot.

``grid`` is imported eagerly; ``expert`` is exposed lazily via ``__getattr__``.
That asymmetry is required, not stylistic: ``sim2edge.rooms`` depends on
``oracle.grid`` (a room owns a geodesic field), while ``oracle.expert`` depends
on ``sim2edge.rooms`` (the expert plans over a room). Importing ``expert``
eagerly here would turn ``import sim2edge.rooms`` into a circular import. Lazy
access keeps ``from sim2edge.oracle import PurePursuitExpert`` working without
the cycle.
"""

from typing import Any

from .grid import UNREACHABLE, OccupancyGrid, distance_field, obstacle_distance

__all__ = [
    "OccupancyGrid",
    "distance_field",
    "obstacle_distance",
    "UNREACHABLE",
    "PurePursuitExpert",
    "ExpertCommand",
    "plan_path",
    "smooth_path",
]

_LAZY = {"PurePursuitExpert", "ExpertCommand", "plan_path", "smooth_path"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import expert

        return getattr(expert, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
