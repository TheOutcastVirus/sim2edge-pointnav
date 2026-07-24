"""sim2edge: sim-to-real navigation policy pipeline for the Arduino Ventuno Q.

The public surface is deliberately small. `spec` is the task contract and the
single source of truth; everything else is an implementation of it.
"""

from .spec import DEFAULT_SPEC, NavSpec

__all__ = ["DEFAULT_SPEC", "NavSpec"]
__version__ = "0.1.0"
