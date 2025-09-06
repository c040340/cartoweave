"""CartoWeave public API.

The top level package only re-exports version and the SolvePack contract. The
compute pipeline is intentionally not imported to keep module side effects
minimal for schema tests.
"""

from .contracts.solvepack import SolvePack  # noqa: F401
from .version import __version__  # noqa: F401

__all__ = ["SolvePack", "__version__"]
