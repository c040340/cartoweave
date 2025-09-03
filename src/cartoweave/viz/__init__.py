"""Visualisation helpers for CartoWeave.

The :func:`interactive_view` function provides a lightweight interactive
viewer that mirrors the behaviour of the much larger original script while
remaining easy to understand.
"""

from .view import interactive_view
from ..config.viz import viz_config

__all__ = ["interactive_view", "viz_config"]

