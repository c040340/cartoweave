"""Visualisation helpers for CartoWeave.

The :func:`interactive_view` function provides a lightweight interactive
viewer that mirrors the behaviour of the much larger original script while
remaining easy to understand.
"""

from typing import Any, Dict

from .view import interactive_view
from .defaults import VIZ_DEFAULTS, merge_defaults

# Backwards compatibility: expose a read-only copy of the minimal defaults as
# ``viz_config``.  Callers are expected to provide their own configuration via
# :func:`interactive_view`.
viz_config: Dict[str, Any] = VIZ_DEFAULTS

__all__ = ["interactive_view", "viz_config", "merge_defaults", "VIZ_DEFAULTS"]

